"""Safe local persistence for canonical research protocol artifacts."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import (
    ConflictRecord,
    DedupCluster,
    EvidenceSnapshot,
    FigureSpec,
    ManifestRecord,
    PlanV1,
    RunConfig,
    SourceCandidate,
    SourceQuality,
    VerdictRecord,
)


class ArtifactSecurityError(ValueError):
    """Raised when an artifact request violates the closed storage policy."""


@dataclass(frozen=True)
class ArtifactReceipt:
    """Receipt for the exact bytes that were atomically persisted."""

    artifact_id: str
    artifact_type: str
    schema_version: str
    path_relative: str
    sha256: str
    byte_length: int
    created_at: str


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)

# This is deliberately closed: adding a type requires adding its validated model
# and its fixed directory in the same reviewed change.
_ARTIFACT_TYPES: dict[str, tuple[type[BaseModel], str]] = {
    "plan": (PlanV1, "plans"),
    "run_config": (RunConfig, "runs"),
    "manifest": (ManifestRecord, "manifests"),
    "source_candidate": (SourceCandidate, "sources"),
    "source_quality": (SourceQuality, "source_quality"),
    "evidence_snapshot": (EvidenceSnapshot, "evidence"),
    "conflict_record": (ConflictRecord, "conflicts"),
    "dedup_cluster": (DedupCluster, "dedup"),
    "verdict_record": (VerdictRecord, "verdicts"),
    "figure_spec": (FigureSpec, "figures"),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically as the bytes that will be hashed."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinks in an already-created path, including its root."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise ArtifactSecurityError("symlink path component is not allowed")


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ArtifactSecurityError(f"invalid {label}")
    if "/" in value or "\\" in value or ".." in value:
        raise ArtifactSecurityError(f"invalid {label}")
    return value


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _safe_close(fd: int | None) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _safe_unlink(name: str | None, directory_fd: int | None) -> None:
    if name is None or directory_fd is None:
        return
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


class ArtifactStore:
    """Persist validated artifacts below one explicitly configured root.

    Publication uses stable directory descriptors and hard-link no-clobber
    semantics. This implementation deliberately fails closed on platforms that
    cannot provide POSIX ``dir_fd`` operations.
    """

    def __init__(self, root: str | os.PathLike[str]):
        if root is None:
            raise ArtifactSecurityError("artifact root is required")
        if os.name != "posix":
            raise ArtifactSecurityError(
                "secure artifact storage requires POSIX dir_fd support"
            )
        raw_root = Path(root)
        if raw_root.exists() and raw_root.is_symlink():
            raise ArtifactSecurityError("artifact root must not be a symlink")
        raw_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _reject_symlink_components(raw_root)
        if not raw_root.is_dir():
            raise ArtifactSecurityError("artifact root must be a directory")
        self.root = Path(os.path.abspath(raw_root))

        root_fd = self._open_root()
        os.close(root_fd)

    @staticmethod
    def _artifact_definition(
        artifact_type: str,
    ) -> tuple[type[BaseModel], str]:
        try:
            return _ARTIFACT_TYPES[artifact_type]
        except (KeyError, TypeError) as exc:
            raise ArtifactSecurityError("unsupported artifact type") from exc

    @staticmethod
    def _validated_bytes(
        model: type[BaseModel], payload: Any
    ) -> tuple[bytes, BaseModel]:
        try:
            try:
                incoming_json = canonical_json_bytes(payload)
            except ValueError:
                validated = model.model_validate(payload)
            else:
                validated = model.model_validate_json(incoming_json, strict=True)
            encoded = canonical_json_bytes(validated.model_dump(mode="json"))
        except (ValidationError, ValueError, TypeError) as exc:
            raise ArtifactSecurityError(f"artifact validation failed: {exc}") from exc
        return encoded, validated

    def _open_root(self) -> int:
        try:
            root_fd = os.open(self.root, _OPEN_DIRECTORY_FLAGS)
        except OSError as exc:
            raise ArtifactSecurityError(
                "artifact root is not a safe directory"
            ) from exc
        try:
            self._assert_root_attached(root_fd)
        except Exception:
            os.close(root_fd)
            raise
        return root_fd

    def _assert_root_attached(self, root_fd: int) -> None:
        try:
            path_stat = os.stat(self.root, follow_symlinks=False)
            descriptor_stat = os.fstat(root_fd)
        except OSError as exc:
            raise ArtifactSecurityError(
                "artifact root changed during operation"
            ) from exc
        if not stat.S_ISDIR(path_stat.st_mode) or not _same_inode(
            path_stat, descriptor_stat
        ):
            raise ArtifactSecurityError("artifact root changed during operation")
        if stat.S_IMODE(descriptor_stat.st_mode) & 0o077:
            raise ArtifactSecurityError("artifact root must be private")
        if descriptor_stat.st_uid != os.geteuid():
            raise ArtifactSecurityError("artifact root must be owned by effective user")

    @staticmethod
    def _open_subdirectory(root_fd: int, name: str, *, create: bool) -> int:
        created = False
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=root_fd)
                created = True
            except FileExistsError:
                pass
        try:
            directory_fd = os.open(name, _OPEN_DIRECTORY_FLAGS, dir_fd=root_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ArtifactSecurityError("artifact directory is not safe") from exc
        try:
            ArtifactStore._assert_directory_attached(root_fd, name, directory_fd)
            if created:
                os.fsync(root_fd)
        except Exception:
            os.close(directory_fd)
            raise
        return directory_fd

    @staticmethod
    def _assert_directory_attached(
        root_fd: int,
        name: str,
        directory_fd: int,
    ) -> None:
        try:
            entry_stat = os.stat(
                name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            descriptor_stat = os.fstat(directory_fd)
        except OSError as exc:
            raise ArtifactSecurityError(
                "artifact directory changed during operation"
            ) from exc
        if not stat.S_ISDIR(entry_stat.st_mode) or not _same_inode(
            entry_stat, descriptor_stat
        ):
            raise ArtifactSecurityError("artifact directory changed during operation")

    @staticmethod
    def _ensure_absent(directory_fd: int, name: str, path_relative: str) -> None:
        try:
            target_stat = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if stat.S_ISLNK(target_stat.st_mode):
            raise ArtifactSecurityError("artifact target must not be a symlink")
        raise FileExistsError(f"artifact already exists: {path_relative}")

    @staticmethod
    def _write_temporary(directory_fd: int, stem: str, payload: bytes) -> str:
        name = f".{stem}.{secrets.token_hex(16)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
            os.fsync(fd)
        except Exception:
            _safe_close(fd)
            _safe_unlink(name, directory_fd)
            raise
        os.close(fd)
        return name

    @staticmethod
    def _publish_temporary(directory_fd: int, temporary: str, target: str) -> None:
        os.link(
            temporary,
            target,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        os.unlink(temporary, dir_fd=directory_fd)

    @staticmethod
    def _read_from_directory(directory_fd: int, name: str) -> bytes:
        try:
            fd = os.open(name, _OPEN_FILE_FLAGS, dir_fd=directory_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ArtifactSecurityError("artifact target is not a safe file") from exc
        try:
            target_stat = os.fstat(fd)
            if not stat.S_ISREG(target_stat.st_mode):
                raise ArtifactSecurityError("artifact target is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    @staticmethod
    def _acquire_lock(lock_directory_fd: int, name: str) -> int:
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            lock_fd = os.open(name, flags, 0o600, dir_fd=lock_directory_fd)
        except OSError as exc:
            raise ArtifactSecurityError("artifact lock is not safe") from exc
        try:
            lock_stat = os.fstat(lock_fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_mode & 0o077:
                raise ArtifactSecurityError("artifact lock has unsafe permissions")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return lock_fd
        except Exception:
            os.close(lock_fd)
            raise

    @staticmethod
    def _cleanup_temporaries(directory_fd: int, stem: str) -> None:
        prefix = f".{stem}."
        removed = False
        for name in os.listdir(directory_fd):
            if name.startswith(prefix) and name.endswith(".tmp"):
                _safe_unlink(name, directory_fd)
                removed = True
        if removed:
            os.fsync(directory_fd)

    @staticmethod
    def _entry_exists(directory_fd: int, name: str) -> bool:
        try:
            entry_stat = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            raise ArtifactSecurityError("artifact target is not a safe regular file")
        return True

    @classmethod
    def _recover_matching_receipt_orphan(
        cls,
        *,
        artifact_fd: int,
        target_name: str,
        receipt_fd: int,
        receipt_name: str,
        expected: ArtifactReceipt,
        path_relative: str,
    ) -> None:
        artifact_exists = cls._entry_exists(artifact_fd, target_name)
        receipt_exists = cls._entry_exists(receipt_fd, receipt_name)
        if artifact_exists:
            raise FileExistsError(f"artifact already exists: {path_relative}")
        if not receipt_exists:
            return

        try:
            raw = cls._read_from_directory(receipt_fd, receipt_name)
            existing = ArtifactReceipt(**json.loads(raw.decode("utf-8")))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactSecurityError("orphan artifact receipt is invalid") from exc

        stable_fields = (
            "artifact_id",
            "artifact_type",
            "schema_version",
            "path_relative",
            "sha256",
            "byte_length",
        )
        if any(
            getattr(existing, field) != getattr(expected, field)
            for field in stable_fields
        ):
            raise FileExistsError(
                f"artifact receipt already exists: receipts/{receipt_name}"
            )
        os.unlink(receipt_name, dir_fd=receipt_fd)
        os.fsync(receipt_fd)

    def persist(
        self,
        artifact_type: str,
        artifact_id: str,
        payload: Any,
    ) -> ArtifactReceipt:
        """Validate, publish without replacement, reread, and hash an artifact."""
        model, subdirectory = self._artifact_definition(artifact_type)
        safe_id = _validate_identifier(artifact_id, "artifact id")
        target_name = f"{safe_id}.json"
        path_relative = f"{subdirectory}/{target_name}"
        receipt_name = f"{artifact_type}-{safe_id}.receipt.json"
        encoded, validated = self._validated_bytes(model, payload)
        created_at = datetime.now(UTC).isoformat()
        digest = hashlib.sha256(encoded).hexdigest()
        receipt = ArtifactReceipt(
            artifact_id=safe_id,
            artifact_type=artifact_type,
            schema_version=str(getattr(validated, "schema_version")),
            path_relative=path_relative,
            sha256=digest,
            byte_length=len(encoded),
            created_at=created_at,
        )
        receipt_bytes = canonical_json_bytes(asdict(receipt))

        root_fd: int | None = None
        artifact_fd: int | None = None
        receipt_fd: int | None = None
        lock_directory_fd: int | None = None
        lock_fd: int | None = None
        artifact_temp: str | None = None
        receipt_temp: str | None = None
        artifact_published = False
        receipt_published = False
        try:
            root_fd = self._open_root()
            artifact_fd = self._open_subdirectory(root_fd, subdirectory, create=True)
            receipt_fd = self._open_subdirectory(root_fd, "receipts", create=True)
            lock_directory_fd = self._open_subdirectory(root_fd, ".locks", create=True)
            lock_fd = self._acquire_lock(
                lock_directory_fd,
                f"{artifact_type}-{safe_id}.lock",
            )
            self._cleanup_temporaries(artifact_fd, target_name)
            self._cleanup_temporaries(receipt_fd, receipt_name)
            self._recover_matching_receipt_orphan(
                artifact_fd=artifact_fd,
                target_name=target_name,
                receipt_fd=receipt_fd,
                receipt_name=receipt_name,
                expected=receipt,
                path_relative=path_relative,
            )
            artifact_temp = self._write_temporary(artifact_fd, target_name, encoded)
            receipt_temp = self._write_temporary(
                receipt_fd, receipt_name, receipt_bytes
            )

            # Receipt-first ordering makes a crash before artifact publication
            # fail closed: no artifact name is visible without its receipt.
            self._publish_temporary(receipt_fd, receipt_temp, receipt_name)
            receipt_temp = None
            receipt_published = True
            self._publish_temporary(artifact_fd, artifact_temp, target_name)
            artifact_temp = None
            artifact_published = True

            self._assert_root_attached(root_fd)
            self._assert_directory_attached(root_fd, subdirectory, artifact_fd)
            self._assert_directory_attached(root_fd, "receipts", receipt_fd)
            actual = self._read_from_directory(artifact_fd, target_name)
            actual_receipt = self._read_from_directory(receipt_fd, receipt_name)
            if actual_receipt != receipt_bytes:
                raise ArtifactSecurityError(
                    "persisted receipt bytes changed during operation"
                )
            if hashlib.sha256(actual).hexdigest() != digest or len(actual) != len(
                encoded
            ):
                raise ArtifactSecurityError(
                    "persisted artifact hash or length mismatch"
                )
            return receipt
        except Exception:
            if artifact_published:
                _safe_unlink(target_name, artifact_fd)
            if receipt_published:
                _safe_unlink(receipt_name, receipt_fd)
            if artifact_fd is not None:
                try:
                    os.fsync(artifact_fd)
                except OSError:
                    pass
            if receipt_fd is not None:
                try:
                    os.fsync(receipt_fd)
                except OSError:
                    pass
            raise
        finally:
            _safe_unlink(artifact_temp, artifact_fd)
            _safe_unlink(receipt_temp, receipt_fd)
            _safe_close(lock_fd)
            _safe_close(lock_directory_fd)
            _safe_close(receipt_fd)
            _safe_close(artifact_fd)
            _safe_close(root_fd)

    def read_bytes(self, artifact_type: str, artifact_id: str) -> bytes:
        _model, subdirectory = self._artifact_definition(artifact_type)
        safe_id = _validate_identifier(artifact_id, "artifact id")
        root_fd: int | None = None
        directory_fd: int | None = None
        try:
            root_fd = self._open_root()
            try:
                directory_fd = self._open_subdirectory(
                    root_fd,
                    subdirectory,
                    create=False,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"artifact not found: {safe_id}") from exc
            try:
                data = self._read_from_directory(directory_fd, f"{safe_id}.json")
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"artifact not found: {safe_id}") from exc
            self._assert_root_attached(root_fd)
            self._assert_directory_attached(root_fd, subdirectory, directory_fd)
            return data
        finally:
            _safe_close(directory_fd)
            _safe_close(root_fd)

    def _read_receipt(self, artifact_type: str, artifact_id: str) -> ArtifactReceipt:
        safe_id = _validate_identifier(artifact_id, "artifact id")
        receipt_name = f"{artifact_type}-{safe_id}.receipt.json"
        root_fd: int | None = None
        receipt_fd: int | None = None
        try:
            root_fd = self._open_root()
            try:
                receipt_fd = self._open_subdirectory(
                    root_fd,
                    "receipts",
                    create=False,
                )
                payload = self._read_from_directory(receipt_fd, receipt_name)
            except FileNotFoundError as exc:
                raise ArtifactSecurityError("artifact receipt is missing") from exc
            self._assert_root_attached(root_fd)
            self._assert_directory_attached(root_fd, "receipts", receipt_fd)
        finally:
            _safe_close(receipt_fd)
            _safe_close(root_fd)
        try:
            value = json.loads(payload.decode("utf-8"))
            return ArtifactReceipt(**value)
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactSecurityError("artifact receipt is invalid") from exc

    def read_verified(
        self,
        artifact_type: str,
        artifact_id: str,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        data = self.read_bytes(artifact_type, artifact_id)
        receipt = self._read_receipt(artifact_type, artifact_id)
        digest = hashlib.sha256(data).hexdigest()
        if receipt.artifact_type != artifact_type or receipt.artifact_id != artifact_id:
            raise ArtifactSecurityError("artifact receipt identity mismatch")
        if receipt.sha256 != digest or receipt.byte_length != len(data):
            raise ArtifactSecurityError(
                "artifact hash or length does not match receipt"
            )
        if expected_sha256 is not None and digest != expected_sha256:
            raise ArtifactSecurityError("artifact hash does not match expected hash")
        return data

    def read_payload(
        self,
        artifact_type: str,
        artifact_id: str,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        data = self.read_verified(
            artifact_type,
            artifact_id,
            expected_sha256=expected_sha256,
        )
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactSecurityError("artifact is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ArtifactSecurityError("artifact payload must be a JSON object")
        return value

    # Explicit aliases make the storage boundary readable to handlers and callers.
    write = persist
    read = read_payload


__all__ = [
    "ArtifactReceipt",
    "ArtifactSecurityError",
    "ArtifactStore",
    "canonical_json_bytes",
]
