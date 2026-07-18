#!/usr/bin/env python3
"""Fixed-command owner IAP transport for the owner-gate stage-zero streams.

There is deliberately no generic remote-command surface in this module.  It
materializes one locally pinned receiver, streams the exact stage-zero kit and
signed bundle over stdin, compares every remote receipt byte-for-byte with a
locally derived receipt, and stops before any service activation or Cloud
mutation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import selectors
import shlex
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_trust as release_trust


TRANSPORT_RECEIPT_SCHEMA = "muncho-owner-gate-iap-stage0-transport.v1"
MAX_STDOUT_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_SEALER_BYTES = 128 * 1024 * 1024
MAX_STREAM_BYTES = (
    len(outer.TREE_STREAM_MAGIC)
    + 8
    + outer.MAX_MANIFEST_BYTES
    + outer.MAX_TREE_BYTES
)
_SHA256 = launcher._SHA256
_REVISION = launcher._RELEASE_SHA
_FOLDER_RESOURCE = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION_RESOURCE = re.compile(
    r"^organizations/[1-9][0-9]{5,30}$"
)


class StableOuterSealer(Protocol):
    def snapshot(self) -> tuple[bytes, str]: ...


@dataclass(frozen=True)
class RawFoundationChainArtifacts:
    """Untrusted raw signed foundation-A inputs for the public IAP boundary.

    This carrier deliberately contains paths, not a Python validation
    capability or decoded projection.  Constructing it confers no authority;
    the IAP constructor consumes each immutable artifact, cryptographically
    re-decodes foundation A, and loads foundation B only from the fixed
    privileged apply journal.
    """

    pre_foundation_authority_path: Path
    owner_reauthentication_receipt_path: Path
    network_evidence_path: Path
    network_collector_public_key_path: Path
    project_ancestry_evidence_path: Path
    project_ancestry_collector_public_key_path: Path
    release_public_key_path: Path

    def __post_init__(self) -> None:
        paths = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
        )
        if (
            any(
                not isinstance(path, Path)
                or not path.is_absolute()
                or ".." in path.parts
                or str(path) != os.path.normpath(str(path))
                or os.path.realpath(path) != str(path)
                for path in paths
            )
            or len(paths) != len(set(paths))
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_raw_foundation_artifacts_invalid"
            )


@dataclass(frozen=True)
class _FoundationProjection:
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str
    project_ancestry_evidence_sha256: str
    project_ancestry_chain_sha256: str
    resource_ancestor_chain: tuple[str, ...]
    interpreter_sha256: str


def _read_foundation_artifact(path: Path, *, maximum: int) -> bytes:
    try:
        return release_trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except release_trust.OwnerGateTrustError as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None


def _load_collector_public_key(path: Path) -> Ed25519PublicKey:
    raw = _read_foundation_artifact(path, maximum=16 * 1024)
    try:
        key = (
            Ed25519PublicKey.from_public_bytes(raw)
            if len(raw) == 32
            else serialization.load_pem_public_key(raw)
        )
    except (TypeError, ValueError) as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None
    if not isinstance(key, Ed25519PublicKey):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    return key


def _load_foundation_projection(
    artifacts: RawFoundationChainArtifacts,
) -> _FoundationProjection:
    if type(artifacts) is not RawFoundationChainArtifacts:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    try:
        release_public_key = pre_foundation.load_pinned_public_key(
            artifacts.release_public_key_path,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
        now_unix = int(time.time())
        if now_unix <= 0:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_foundation_chain_invalid"
            )
        foundation_a = foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=_read_foundation_artifact(
                artifacts.pre_foundation_authority_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            owner_reauthentication_receipt_raw=_read_foundation_artifact(
                artifacts.owner_reauthentication_receipt_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            network_evidence_raw=_read_foundation_artifact(
                artifacts.network_evidence_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            project_ancestry_evidence_raw=_read_foundation_artifact(
                artifacts.project_ancestry_evidence_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            release_public_key=release_public_key,
            network_collector_public_key=_load_collector_public_key(
                artifacts.network_collector_public_key_path
            ),
            project_ancestry_collector_public_key=(
                _load_collector_public_key(
                    artifacts.project_ancestry_collector_public_key_path
                )
            ),
            now_unix=now_unix,
        )
        chain = foundation_apply.load_validated_foundation_apply_chain(
            foundation_a
        )
        authority = chain.foundation_a.authority
        interpreter = authority["interpreter_image"]
        ancestry_chain = tuple(
            item["resource_name"]
            for item in chain.foundation_a.ancestry_evidence.ordered_chain[1:]
        )
        projection = _FoundationProjection(
            pre_foundation_authority_sha256=(
                chain.pre_foundation_authority_sha256
            ),
            foundation_apply_receipt_sha256=(
                chain.foundation_apply_receipt_sha256
            ),
            project_ancestry_evidence_sha256=(
                chain.foundation_a.ancestry_evidence_sha256
            ),
            project_ancestry_chain_sha256=authority[
                "ancestry_chain_sha256"
            ],
            resource_ancestor_chain=ancestry_chain,
            interpreter_sha256=interpreter["interpreter_sha256"],
        )
    except launcher.OwnerLauncherError:
        raise
    except (
        KeyError,
        TypeError,
        foundation_apply.OwnerGateFoundationApplyError,
        pre_foundation.OwnerGatePreFoundationError,
    ) as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None
    if (
        _SHA256.fullmatch(projection.pre_foundation_authority_sha256) is None
        or _SHA256.fullmatch(projection.foundation_apply_receipt_sha256) is None
        or _SHA256.fullmatch(projection.project_ancestry_evidence_sha256) is None
        or _SHA256.fullmatch(projection.project_ancestry_chain_sha256) is None
        or _SHA256.fullmatch(projection.interpreter_sha256) is None
        or not projection.resource_ancestor_chain
        or len(projection.resource_ancestor_chain) > 31
        or len(projection.resource_ancestor_chain)
        != len(set(projection.resource_ancestor_chain))
        or _ORGANIZATION_RESOURCE.fullmatch(
            projection.resource_ancestor_chain[-1]
        )
        is None
        or any(
            _FOLDER_RESOURCE.fullmatch(item) is None
            for item in projection.resource_ancestor_chain[:-1]
        )
        or interpreter.get("python_version") != "3.11.2"
    ):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    return projection


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class _FixedOperation:
    name: str
    root_argv: tuple[str, ...]
    expected_stdout: bytes
    maximum_input_bytes: int
    timeout_seconds: float


def _canonical(value: Any) -> bytes:
    return outer.canonical_json_bytes(value)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _terminate(
    process: subprocess.Popen[bytes],
    *,
    kill_process_group: Callable[[int, int], None] = os.killpg,  # windows-footgun: ok — POSIX process boundary
) -> None:
    try:
        if process.poll() is None:
            try:
                kill_process_group(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.SubprocessError):
                pass
        if process.poll() is None:
            try:
                kill_process_group(process.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX process boundary
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.SubprocessError):
                pass
    finally:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass


def _bounded_process_exchange(
    argv: Sequence[str],
    environment: Mapping[str, str],
    input_source: BinaryIO,
    *,
    maximum_input_bytes: int,
    maximum_stdout_bytes: int = MAX_STDOUT_BYTES,
    maximum_stderr_bytes: int = MAX_STDERR_BYTES,
    timeout_seconds: float,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    process_terminator: Callable[[subprocess.Popen[bytes]], None] = _terminate,
) -> _ProcessResult:
    if (
        not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or isinstance(maximum_input_bytes, bool)
        or not isinstance(maximum_input_bytes, int)
        or maximum_input_bytes < 0
        or maximum_input_bytes > MAX_STREAM_BYTES
        or not 0 < maximum_stdout_bytes <= MAX_STDOUT_BYTES
        or not 0 < maximum_stderr_bytes <= MAX_STDERR_BYTES
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= 2_400
    ):
        raise launcher.OwnerLauncherError("owner_gate_stage0_iap_exchange_invalid")
    try:
        process = popen_factory(
            tuple(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            shell=False,
            start_new_session=True,
            bufsize=0,
        )
    except (OSError, subprocess.SubprocessError):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_iap_unavailable"
        ) from None
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process_terminator(process)
        raise launcher.OwnerLauncherError("owner_gate_stage0_iap_unavailable")
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    pending = memoryview(b"")
    input_count = 0
    input_eof = False
    input_open = True
    output_open = {"stdout": True, "stderr": True}
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        descriptors = {
            "stdin": process.stdin.fileno(),
            "stdout": process.stdout.fileno(),
            "stderr": process.stderr.fileno(),
        }
        for descriptor in descriptors.values():
            os.set_blocking(descriptor, False)
        selector.register(descriptors["stdin"], selectors.EVENT_WRITE, "stdin")
        selector.register(descriptors["stdout"], selectors.EVENT_READ, "stdout")
        selector.register(descriptors["stderr"], selectors.EVENT_READ, "stderr")
        while True:
            if input_open and not pending and not input_eof:
                chunk = input_source.read(64 * 1024)
                if not isinstance(chunk, bytes):
                    raise launcher.OwnerLauncherError(
                        "owner_gate_stage0_iap_input_invalid"
                    )
                if chunk:
                    input_count += len(chunk)
                    if input_count > maximum_input_bytes:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_input_oversized"
                        )
                    pending = memoryview(chunk)
                else:
                    input_eof = True
                    selector.unregister(descriptors["stdin"])
                    process.stdin.close()
                    input_open = False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_iap_timeout"
                )
            for key, _mask in selector.select(min(remaining, 0.25)):
                if key.data == "stdin":
                    if not pending:
                        continue
                    try:
                        written = os.write(key.fd, pending[: 64 * 1024])
                    except BlockingIOError:
                        continue
                    except OSError:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_stdin_failed"
                        ) from None
                    if written <= 0:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_stdin_failed"
                        )
                    pending = pending[written:]
                    continue
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                except OSError:
                    raise launcher.OwnerLauncherError(
                        "owner_gate_stage0_iap_output_failed"
                    ) from None
                target = stdout if key.data == "stdout" else stderr
                maximum = (
                    maximum_stdout_bytes
                    if key.data == "stdout"
                    else maximum_stderr_bytes
                )
                if chunk:
                    target.extend(chunk)
                    if len(target) > maximum:
                        raise launcher.OwnerLauncherError(
                            f"owner_gate_stage0_iap_{key.data}_oversized"
                        )
                else:
                    selector.unregister(key.fd)
                    getattr(process, key.data).close()
                    output_open[key.data] = False
            if process.poll() is not None and not any(output_open.values()):
                break
        if input_open or pending or not input_eof:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_iap_stdin_failed"
            )
        try:
            returncode = process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_iap_timeout"
            ) from None
        return _ProcessResult(returncode, bytes(stdout), bytes(stderr))
    except BaseException:
        process_terminator(process)
        raise
    finally:
        pending.release()
        selector.close()
        if process.poll() is not None:
            process_terminator(process)


class PinnedExactTreeStream:
    """Stable local stream identity checked before and after IAP transfer."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        purpose: str,
        release_id: str,
        expected_manifest_sha256: str,
    ) -> None:
        selected = os.path.abspath(os.fspath(path))
        if (
            os.path.realpath(selected) != selected
            or _SHA256.fullmatch(expected_manifest_sha256) is None
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_identity_invalid"
            )
        self.path = selected
        self.purpose = purpose
        self.release_id = release_id
        self.expected_manifest_sha256 = expected_manifest_sha256
        self._fingerprint, self.manifest, self.manifest_raw = self._capture()

    @staticmethod
    def _read_exact(descriptor: int, size: int) -> bytes:
        raw = bytearray()
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            raw.extend(chunk)
            remaining -= len(chunk)
        return bytes(raw)

    def _capture(self) -> tuple[tuple[Any, ...], Mapping[str, Any], bytes]:
        descriptor: int | None = None
        try:
            before = os.lstat(self.path)
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or opened.st_uid not in {0, os.getuid()}  # windows-footgun: ok — POSIX owner boundary
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o400
                or opened.st_size < len(outer.TREE_STREAM_MAGIC) + 9
                or opened.st_size > MAX_STREAM_BYTES
            ):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            magic = self._read_exact(descriptor, len(outer.TREE_STREAM_MAGIC))
            manifest_size = int.from_bytes(self._read_exact(descriptor, 8), "big")
            if magic != outer.TREE_STREAM_MAGIC or not 0 < manifest_size <= outer.MAX_MANIFEST_BYTES:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            manifest_raw = self._read_exact(descriptor, manifest_size)
            if _sha256(manifest_raw) != self.expected_manifest_sha256:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            try:
                decoded = json.loads(manifest_raw.decode("utf-8", errors="strict"))
                manifest = outer.validate_tree_stream_manifest(
                    decoded,
                    expected_purpose=self.purpose,
                    expected_release_id=self.release_id,
                )
            except (UnicodeError, ValueError, json.JSONDecodeError, outer.OwnerGateOuterStage0Error):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                ) from None
            expected_size = (
                len(outer.TREE_STREAM_MAGIC)
                + 8
                + manifest_size
                + sum(item["size"] for item in manifest["files"])
            )
            after = os.fstat(descriptor)
            fingerprint = (
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                self.expected_manifest_sha256,
            )
            if (
                expected_size != opened.st_size
                or _canonical(manifest) != manifest_raw
                or (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
            ):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            return fingerprint, manifest, manifest_raw
        except OSError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_identity_invalid"
            ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @property
    def size(self) -> int:
        return int(self._fingerprint[6])

    def open(self) -> BinaryIO:
        fingerprint, manifest, manifest_raw = self._capture()
        if (
            fingerprint != self._fingerprint
            or manifest != self.manifest
            or manifest_raw != self.manifest_raw
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            opened_fingerprint = (
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                self.expected_manifest_sha256,
            )
            if opened_fingerprint != self._fingerprint:
                os.close(descriptor)
                descriptor = -1
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_changed"
                )
            return os.fdopen(descriptor, "rb", closefd=True)
        except launcher.OwnerLauncherError:
            raise
        except OSError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            ) from None

    def assert_stable(self) -> None:
        fingerprint, manifest, manifest_raw = self._capture()
        if (
            fingerprint != self._fingerprint
            or manifest != self.manifest
            or manifest_raw != self.manifest_raw
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )

    def member(self, relative: str) -> bytes:
        offset = len(outer.TREE_STREAM_MAGIC) + 8 + len(self.manifest_raw)
        selected: Mapping[str, Any] | None = None
        for item in self.manifest["files"]:
            if item["path"] == relative:
                selected = item
                break
            offset += item["size"]
        if selected is None:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_member_missing"
            )
        with self.open() as stream:
            stream.seek(offset)
            raw = stream.read(selected["size"])
        if (
            not isinstance(raw, bytes)
            or len(raw) != selected["size"]
            or _sha256(raw) != selected["sha256"]
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )
        self.assert_stable()
        return raw


class TrustedOuterSealerSource:
    """Read the sealer only from the activated exact owner-support release."""

    def __init__(self, release_sha: str) -> None:
        if _REVISION.fullmatch(release_sha) is None:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        runtime = launcher.require_trusted_owner_runtime(release_sha)
        launcher.activate_trusted_owner_support(runtime, release_sha=release_sha)
        _root, source_root, _site = launcher._trusted_owner_support_paths(release_sha)
        self._release_sha = release_sha
        self._path = os.path.join(
            source_root,
            "scripts/canary/owner_gate_outer_stage0.py",
        )
        self._fingerprint, self._payload = self._capture()

    def _capture(self) -> tuple[tuple[Any, ...], bytes]:
        launcher.require_local_launcher_provenance(self._release_sha)
        fingerprint, payload = launcher._read_pinned_regular_file(
            self._path,
            maximum=MAX_SEALER_BYTES,
            unavailable_code="owner_gate_stage0_sealer_unavailable",
            invalid_code="owner_gate_stage0_sealer_invalid",
            changed_code="owner_gate_stage0_sealer_changed",
            allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok — POSIX owner boundary
        )
        if stat.S_IMODE(int(fingerprint[0])) not in {0o400, 0o444}:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        return fingerprint, payload

    def snapshot(self) -> tuple[bytes, str]:
        fingerprint, payload = self._capture()
        if fingerprint != self._fingerprint or payload != self._payload:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_changed")
        return payload, _sha256(payload)


def _tree_projection(manifest: Mapping[str, Any]) -> tuple[str, int]:
    directories = {item["path"]: item for item in manifest["directories"]}
    files = {item["path"]: item for item in manifest["files"]}
    projection: list[Mapping[str, Any]] = []
    for relative in sorted((*directories, *files)):
        if relative in directories:
            projection.append({
                "path": relative,
                "mode": "0555",
                "type": "directory",
            })
        else:
            item = files[relative]
            projection.append({
                "path": relative,
                "mode": item["mode"],
                "type": "file",
                "size": item["size"],
                "sha256": item["sha256"],
            })
    return outer.sha256_json(projection), len(projection)


def expected_tree_receipt(
    stream: PinnedExactTreeStream,
    *,
    receiver_self_sha256: str,
) -> Mapping[str, Any]:
    projection_sha256, projection_count = _tree_projection(stream.manifest)
    base = (
        outer.INCOMING_BASE
        if stream.purpose == "outer-stage0-kit"
        else outer.BUNDLE_INCOMING_BASE
    )
    unsigned = {
        "schema": outer.TREE_RECEIPT_SCHEMA,
        "purpose": stream.purpose,
        "release_id": stream.release_id,
        "stream_manifest_sha256": stream.expected_manifest_sha256,
        "transport_manifest_sha256": stream.manifest[
            "transport_manifest_sha256"
        ],
        "source_tree_projection_sha256": stream.manifest[
            "source_tree_projection_sha256"
        ],
        "receiver_self_sha256": receiver_self_sha256,
        "received_tree": {
            "path": str(base / stream.release_id),
            "uid": 0,
            "gid": 0,
            "mode": "0555",
            "projection_sha256": projection_sha256,
            "projection_count": projection_count,
        },
        "input_code_executed": False,
        "input_code_imported": False,
        "symlinks_received": False,
        "special_files_received": False,
        "extra_paths_received": False,
    }
    return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}


def expected_seal_receipt(
    kit_stream: PinnedExactTreeStream,
    *,
    receiver_self_sha256: str,
) -> Mapping[str, Any]:
    manifest_raw = kit_stream.member("outer-stage0-manifest.json")
    if _sha256(manifest_raw) != kit_stream.release_id:
        raise launcher.OwnerLauncherError("owner_gate_stage0_kit_authority_invalid")
    try:
        manifest = outer.validate_manifest(
            json.loads(manifest_raw.decode("utf-8", errors="strict"))
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, outer.OwnerGateOuterStage0Error):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_kit_authority_invalid"
        ) from None
    if _canonical(manifest) != manifest_raw:
        raise launcher.OwnerLauncherError("owner_gate_stage0_kit_authority_invalid")
    files = {
        **{item["path"]: item for item in manifest["files"]},
        "outer-stage0-manifest.json": {
            "sha256": _sha256(manifest_raw),
            "size": len(manifest_raw),
            "mode": "0444",
        },
    }
    directories = {
        str(parent)
        for relative in files
        for parent in Path(relative).parents
        if str(parent) != "."
    }
    projection: list[Mapping[str, Any]] = []
    for relative in sorted((*directories, *files)):
        if relative in directories:
            projection.append({
                "path": relative,
                "type": "directory",
                "mode": "0555",
            })
        else:
            item = files[relative]
            projection.append({
                "path": relative,
                "type": "file",
                "mode": item["mode"],
                "size": item["size"],
                "sha256": item["sha256"],
            })
    release = outer.RELEASE_BASE / kit_stream.release_id
    unsigned = {
        "schema": outer.RECEIPT_SCHEMA,
        "kit_manifest_sha256": kit_stream.release_id,
        "kit_self_hash": manifest["kit_manifest_sha256"],
        "source_release_revision": manifest["source_release_revision"],
        "source_tree_oid": manifest["source_tree_oid"],
        "outer_sealer_sha256": receiver_self_sha256,
        "trusted_runner": str(release / outer.TRUSTED_RUNNER),
        "release": {
            "path": str(release),
            "uid": 0,
            "gid": 0,
            "mode": "0555",
            "projection_sha256": outer.sha256_json(projection),
            "projection_count": len(projection),
        },
        "incoming_payload_code_executed": False,
        "incoming_payload_imported": False,
        "network_fetch_performed": False,
        "generic_shell_runtime_added": False,
    }
    return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}


class OwnerGateStage0IapTransport(launcher.OwnerGateIapTransport):
    """Fixed root bootstrap operations over the already pinned IAP identity."""

    _SEALER_ROOT = "/run/muncho-owner-gate-stage0-bootstrap"

    def __init__(
        self,
        *,
        release_sha: str,
        owner_identity: launcher.GcloudOwnerAccessToken,
        gcloud_executable: launcher.TrustedGcloudExecutable,
        gcloud_configuration: launcher.PinnedGcloudConfiguration,
        foundation_artifacts: RawFoundationChainArtifacts,
        sealer_source: StableOuterSealer | None = None,
        host_identity: launcher.StableOwnerGateHostIdentity | None = None,
        known_hosts: launcher.StableKnownHosts | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        timeout_seconds: float = 900.0,
        exchange: Callable[..., _ProcessResult] | None = None,
    ) -> None:
        projection = _load_foundation_projection(foundation_artifacts)
        super().__init__(
            release_sha=release_sha,
            owner_identity=owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            host_identity=host_identity,
            known_hosts=known_hosts,
            popen_factory=popen_factory,
            timeout_seconds=timeout_seconds,
        )
        self._foundation = projection
        self._stage0_sealer_source = sealer_source or TrustedOuterSealerSource(
            release_sha
        )
        self._stage0_exchange = exchange or _bounded_process_exchange

    def _attest_remote_interpreter(self) -> Mapping[str, Any]:
        digest = self._foundation.interpreter_sha256
        expected = (
            (
                "python_link",
                ("/usr/bin/readlink", "--", "/usr/bin/python3"),
                b"python3.11\n",
            ),
            (
                "python_link_identity",
                (
                    "/usr/bin/stat",
                    "--format=%F|%u|%g|%a|%h",
                    "--",
                    "/usr/bin/python3",
                ),
                b"symbolic link|0|0|777|1\n",
            ),
            (
                "python_target_identity",
                (
                    "/usr/bin/stat",
                    "--format=%F|%u|%g|%a|%h",
                    "--",
                    "/usr/bin/python3.11",
                ),
                b"regular file|0|0|755|1\n",
            ),
            (
                "python_target_digest",
                ("/usr/bin/sha256sum", "--", "/usr/bin/python3.11"),
                f"{digest}  /usr/bin/python3.11\n".encode("ascii"),
            ),
        )
        for name, argv, stdout in expected:
            self._execute_empty(
                _FixedOperation(
                    f"{name}_before",
                    argv,
                    stdout,
                    0,
                    60.0,
                )
            )
        self._execute_empty(
            _FixedOperation(
                "python_version_after_digest",
                ("/usr/bin/python3", "--version"),
                b"Python 3.11.2\n",
                0,
                60.0,
            )
        )
        for name, argv, stdout in expected:
            self._execute_empty(
                _FixedOperation(
                    f"{name}_after",
                    argv,
                    stdout,
                    0,
                    60.0,
                )
            )
        unsigned = {
            "schema": "muncho-owner-gate-host-interpreter-attestation.v1",
            "path": "/usr/bin/python3",
            "link_target": "python3.11",
            "resolved_path": "/usr/bin/python3.11",
            "target_sha256": digest,
            "python_version": "3.11.2",
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
            "identity_stable_before_after_version_probe": True,
            "python_executed_before_digest_match": False,
        }
        return {**unsigned, "attestation_sha256": outer.sha256_json(unsigned)}

    @staticmethod
    def _sealer_paths(sealer_sha256: str) -> tuple[str, str, str]:
        if _SHA256.fullmatch(sealer_sha256) is None:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        directory = f"{OwnerGateStage0IapTransport._SEALER_ROOT}/{sealer_sha256}"
        staging = f"{directory}/.owner_gate_outer_stage0.py.uploading"
        final = f"{directory}/owner_gate_outer_stage0.py"
        return directory, staging, final

    def _root_argv(
        self,
        snapshot: tuple[Any, ...],
        operation: _FixedOperation,
    ) -> tuple[str, ...]:
        (
            prefix,
            account,
            _launcher_sha256,
            known_hosts,
            private_key,
            _public_key,
            host_identity,
            _server_host_key,
        ) = snapshot
        if not isinstance(host_identity, launcher.OwnerGateHostIdentitySnapshot):
            raise launcher.OwnerLauncherError(
                "owner_gate_iap_identity_receipt_invalid"
            )
        root_command = (
            "/usr/bin/sudo",
            "--non-interactive",
            "--",
            *operation.root_argv,
        )
        remote_command = shlex.join(root_command)
        ssh_flags = self._sealed_ssh_flags(
            known_hosts,
            private_key,
            host_identity.vm_numeric_id,
        )
        expected = (
            *prefix,
            "compute",
            "ssh",
            f"{launcher.OS_LOGIN_USERNAME}@{self._VM_NAME}",
            f"--project={launcher.PROJECT}",
            f"--zone={launcher.ZONE}",
            f"--account={account}",
            "--plain",
            "--tunnel-through-iap",
            "--quiet",
            f"--command={remote_command}",
            *ssh_flags,
        )
        argv = tuple(expected)
        if (
            account != self._OWNER_ACCOUNT
            or argv != expected
            or argv[len(prefix) : len(prefix) + 9]
            != (
                "compute",
                "ssh",
                f"{launcher.OS_LOGIN_USERNAME}@{self._VM_NAME}",
                f"--project={launcher.PROJECT}",
                f"--zone={launcher.ZONE}",
                f"--account={self._OWNER_ACCOUNT}",
                "--plain",
                "--tunnel-through-iap",
                "--quiet",
            )
            or argv[-len(ssh_flags) :] != ssh_flags
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_iap_argv_invalid")
        return argv

    def _execute(
        self,
        operation: _FixedOperation,
        input_source: BinaryIO,
    ) -> bytes:
        before = self._authority_snapshot()
        argv = self._root_argv(before, operation)
        environment = self._environment(before[0])
        try:
            result = self._stage0_exchange(
                argv,
                environment,
                input_source,
                maximum_input_bytes=operation.maximum_input_bytes,
                maximum_stdout_bytes=MAX_STDOUT_BYTES,
                maximum_stderr_bytes=MAX_STDERR_BYTES,
                timeout_seconds=operation.timeout_seconds,
                popen_factory=self._popen_factory,
            )
        finally:
            after = self._authority_snapshot()
            if after != before:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_iap_authority_changed"
                )
        if (
            not isinstance(result, _ProcessResult)
            or result.returncode != 0
            or result.stderr != b""
            or result.stdout != operation.expected_stdout
        ):
            raise launcher.OwnerLauncherError(
                f"owner_gate_stage0_iap_{operation.name}_failed"
            )
        return result.stdout

    def _execute_empty(self, operation: _FixedOperation) -> bytes:
        return self._execute(operation, io.BytesIO(b""))

    def _materialize_sealer(self, payload: bytes, sha256: str) -> str:
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_SEALER_BYTES
            or _sha256(payload) != sha256
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        directory, staging, final = self._sealer_paths(sha256)
        operations = (
            _FixedOperation(
                "sealer_directory",
                (
                    "/usr/bin/install",
                    "-d",
                    "-o",
                    "root",
                    "-g",
                    "root",
                    "-m",
                    "0700",
                    directory,
                ),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stale_stage_remove",
                ("/bin/rm", "--force", "--", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage",
                (
                    "/usr/bin/dd",
                    f"of={staging}",
                    "bs=65536",
                    "conv=fsync",
                    "oflag=excl,nofollow",
                    "status=none",
                ),
                b"",
                len(payload),
                120.0,
            ),
            _FixedOperation(
                "sealer_stage_owner",
                ("/bin/chown", "root:root", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_mode",
                ("/bin/chmod", "0400", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_sync",
                ("/usr/bin/sync", "-f", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_digest",
                ("/usr/bin/sha256sum", staging),
                f"{sha256}  {staging}\n".encode("ascii"),
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_publish",
                ("/bin/cp", "--no-clobber", "--reflink=never", staging, final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_owner",
                ("/bin/chown", "root:root", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_mode",
                ("/bin/chmod", "0400", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_sync",
                ("/usr/bin/sync", "-f", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_digest",
                ("/usr/bin/sha256sum", final),
                f"{sha256}  {final}\n".encode("ascii"),
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_remove",
                ("/bin/rm", "--force", "--", staging),
                b"",
                0,
                60.0,
            ),
        )
        for operation in operations:
            source = io.BytesIO(payload) if operation.name == "sealer_stage" else io.BytesIO(b"")
            self._execute(operation, source)
        return final

    def _receive_stream(
        self,
        stream: PinnedExactTreeStream,
        *,
        sealer_path: str,
        sealer_sha256: str,
    ) -> Mapping[str, Any]:
        expected = expected_tree_receipt(
            stream,
            receiver_self_sha256=sealer_sha256,
        )
        operation = _FixedOperation(
            f"receive_{stream.purpose.replace('-', '_')}",
            (
                "/usr/bin/python3",
                "-I",
                "-B",
                sealer_path,
                "stream-receive",
                "--purpose",
                stream.purpose,
                "--release-id",
                stream.release_id,
                "--expected-stream-manifest-sha256",
                stream.expected_manifest_sha256,
                "--expected-self-sha256",
                sealer_sha256,
            ),
            _canonical(expected) + b"\n",
            stream.size,
            self._timeout_seconds,
        )
        with stream.open() as source:
            self._execute(operation, source)
        stream.assert_stable()
        return expected

    def _seal_kit(
        self,
        stream: PinnedExactTreeStream,
        *,
        sealer_path: str,
        sealer_sha256: str,
    ) -> Mapping[str, Any]:
        expected = expected_seal_receipt(
            stream,
            receiver_self_sha256=sealer_sha256,
        )
        operation = _FixedOperation(
            "seal_kit",
            (
                "/usr/bin/python3",
                "-I",
                "-B",
                sealer_path,
                "seal",
                "--incoming",
                str(outer.INCOMING_BASE / stream.release_id),
                "--expected-manifest-sha256",
                stream.release_id,
            ),
            _canonical(expected) + b"\n",
            0,
            self._timeout_seconds,
        )
        self._execute_empty(operation)
        return expected

    def transport_exact_stage0_and_bundle(
        self,
        *,
        kit_stream: PinnedExactTreeStream,
        bundle_stream: PinnedExactTreeStream,
    ) -> Mapping[str, Any]:
        if (
            not isinstance(kit_stream, PinnedExactTreeStream)
            or kit_stream.purpose != "outer-stage0-kit"
            or not isinstance(bundle_stream, PinnedExactTreeStream)
            or bundle_stream.purpose != "owner-gate-bundle"
            or kit_stream.manifest.get("release_id") != kit_stream.release_id
            or bundle_stream.manifest.get("release_id")
            != bundle_stream.release_id
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_pair_invalid"
            )
        interpreter = self._attest_remote_interpreter()
        sealer_payload, sealer_sha256 = self._stage0_sealer_source.snapshot()
        sealer_path = self._materialize_sealer(sealer_payload, sealer_sha256)
        kit_receiver = self._receive_stream(
            kit_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        seal = self._seal_kit(
            kit_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        bundle_receiver = self._receive_stream(
            bundle_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        final_sealer_payload, final_sealer_sha256 = (
            self._stage0_sealer_source.snapshot()
        )
        if (
            final_sealer_payload != sealer_payload
            or final_sealer_sha256 != sealer_sha256
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_changed")
        host_identity = self._host_identity.snapshot()
        unsigned = {
            "schema": TRANSPORT_RECEIPT_SCHEMA,
            "release_sha": self._release_sha,
            "project": launcher.PROJECT,
            "zone": launcher.ZONE,
            "vm_name": self._VM_NAME,
            "vm_numeric_id": host_identity.vm_numeric_id,
            "owner_account": self._OWNER_ACCOUNT,
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
            "interpreter_attestation_sha256": interpreter[
                "attestation_sha256"
            ],
            "sealer_sha256": sealer_sha256,
            "sealer_remote_path": sealer_path,
            "kit_receiver_receipt_sha256": kit_receiver["receipt_sha256"],
            "kit_seal_receipt_sha256": seal["receipt_sha256"],
            "bundle_receiver_receipt_sha256": bundle_receiver[
                "receipt_sha256"
            ],
            "recursive_scp_used": False,
            "caller_controlled_remote_command_used": False,
            "caller_controlled_remote_path_used": False,
            "cloud_control_plane_mutation_performed": False,
            "host_filesystem_materialization_performed": True,
            "host_filesystem_materialization_roots": [
                self._SEALER_ROOT,
                str(outer.INCOMING_BASE),
                str(outer.BUNDLE_INCOMING_BASE),
                str(outer.RELEASE_BASE),
                str(outer.TRANSPORT_RECEIPT_BASE),
                str(outer.RECEIPT_BASE),
            ],
            "service_activation_performed": False,
        }
        return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}


__all__ = [
    "MAX_STDERR_BYTES",
    "MAX_STDOUT_BYTES",
    "OwnerGateStage0IapTransport",
    "PinnedExactTreeStream",
    "RawFoundationChainArtifacts",
    "StableOuterSealer",
    "TRANSPORT_RECEIPT_SCHEMA",
    "TrustedOuterSealerSource",
    "expected_seal_receipt",
    "expected_tree_receipt",
]
