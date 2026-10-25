"""Launch one Claude worker only after an exact bounded Roots proof.

The authority service mints the credential. This bootstrap reads it once from
an inherited descriptor and verifies the principal and assignment. The Claude
worker process is the principal's trusted runtime; Roots enforces the token's
scope, partitions and expiry on every MCP operation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA = "hermes.roots-worker-assignment/v1"
RECEIPT_SCHEMA = "hermes.roots-worker-bootstrap-receipt/v1"
MAX_TOKEN_BYTES = 16_384
MAX_ASSIGNMENT_LIFETIME_SECONDS = 900
MAX_TASK_BYTES = 65_536
MAX_RESPONSE_BYTES = 1_048_576
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_READ_ONLY_ROOTS_TOOLS = {
    "mcp__roots__roots_whoami", "mcp__roots__roots_context", "mcp__roots__roots_decisions",
    "mcp__roots__roots_dependencies", "mcp__roots__roots_graph_stats", "mcp__roots__roots_holds",
    "mcp__roots__roots_neighbours", "mcp__roots__roots_node", "mcp__roots__roots_plans",
    "mcp__roots__roots_search", "mcp__roots__roots_tickets", "mcp__roots__roots_traverse",
    "mcp__roots__ticket_next", "mcp__roots__ticket_view",
}
_PASSTHROUGH_ENV = (
    "HOME", "PATH", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM", "USER", "LOGNAME",
)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "correlation_id": {"type": "string"},
        "status": {"type": "string", "enum": ["completed", "blocked"]},
        "answer": {"type": "string"},
    },
    "required": ["correlation_id", "status", "answer"],
    "additionalProperties": False,
}


def _default_claude_config_dir() -> Path:
    return Path.home() / ".claude"


class BootstrapRefusal(ValueError):
    """The worker must not start under the supplied assignment."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise BootstrapRefusal("Roots redirected the bootstrap probe; identity was not verified")


def _read_owned_file(path: Path, *, private: bool) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise BootstrapRefusal(f"cannot open reviewed file: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
            raise BootstrapRefusal(f"reviewed file is not an owned regular file: {path}")
        mode = stat.S_IMODE(info.st_mode)
        if (private and mode != 0o600) or (not private and mode & 0o022):
            raise BootstrapRefusal(f"reviewed file permissions are unsafe: {path}")
        chunks = bytearray()
        while len(chunks) <= 262_144:
            chunk = os.read(fd, min(65_536, 262_145 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > 262_144:
            raise BootstrapRefusal(f"reviewed file is too large: {path}")
        return bytes(chunks)
    finally:
        os.close(fd)


def _read_assignment(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_owned_file(path, private=True)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BootstrapRefusal("assignment is not JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal("assignment must be a JSON object")
    return value, raw


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise BootstrapRefusal(f"{name} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise BootstrapRefusal(f"{name} contains duplicates")
    return value


def _owned_executable(value: Any, name: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise BootstrapRefusal(f"{name} must be an absolute executable path")
    info = os.stat(value, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022 or not os.access(value, os.X_OK):
        raise BootstrapRefusal(f"{name} must be a non-writable executable regular file")
    return value


@dataclass
class _PinnedFile:
    fd: int
    read_fd: int
    path: str
    sha256: str

    def close(self) -> None:
        os.close(self.fd)
        os.close(self.read_fd)


def _pin_executable(path: str, name: str) -> _PinnedFile:
    execute_flag = getattr(os, "O_EXEC", os.O_RDONLY)
    try:
        fd = os.open(path, execute_flag | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise BootstrapRefusal(f"cannot pin {name}") from exc
    read_fd = -1
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o022:
            raise BootstrapRefusal(f"{name} changed or has unsafe permissions")
        read_fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        read_info = os.fstat(read_fd)
        if (read_info.st_dev, read_info.st_ino) != (info.st_dev, info.st_ino):
            raise BootstrapRefusal(f"{name} changed while it was pinned")
        digest = hashlib.sha256()
        while chunk := os.read(read_fd, 65_536):
            digest.update(chunk)
        pinned = _PinnedFile(fd=fd, read_fd=read_fd, path=f"/dev/fd/{fd}", sha256=digest.hexdigest())
        read_fd = -1
        return pinned
    except Exception:
        os.close(fd)
        raise
    finally:
        if read_fd >= 0:
            os.close(read_fd)


def _pin_directory(path: str, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BootstrapRefusal(f"cannot pin {name}") from exc
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        os.close(fd)
        raise BootstrapRefusal(f"{name} is not an owned directory")
    if stat.S_IMODE(info.st_mode) & 0o022:
        os.close(fd)
        raise BootstrapRefusal(f"{name} permissions are unsafe")
    return fd


def _verify_pinned_directory_path(path: str, fd: int, name: str) -> None:
    """Detect replacement of a reviewed directory while its descriptor is held."""
    try:
        path_info = os.stat(path, follow_symlinks=False)
        pinned_info = os.fstat(fd)
    except OSError as exc:
        raise BootstrapRefusal(f"cannot revalidate {name}") from exc
    if not stat.S_ISDIR(path_info.st_mode) or (
        path_info.st_dev, path_info.st_ino
    ) != (pinned_info.st_dev, pinned_info.st_ino):
        raise BootstrapRefusal(f"{name} changed after review")


def _stage_executable(source: _PinnedFile, directory: Path, name: str) -> tuple[str, str]:
    target = directory / name
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o500)
    try:
        source_before = os.fstat(source.read_fd)
        digest = hashlib.sha256()
        os.lseek(source.read_fd, 0, os.SEEK_SET)
        while chunk := os.read(source.read_fd, 65_536):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short staged executable write")
                view = view[written:]
        os.fsync(fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o500:
            raise BootstrapRefusal("staged executable permissions are unsafe")
        source_after = os.fstat(source.read_fd)
        watched = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(source_before, key) != getattr(source_after, key) for key in watched):
            raise BootstrapRefusal("executable changed while it was staged")
    finally:
        os.close(fd)
    staged_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if staged_digest != digest.hexdigest():
        raise BootstrapRefusal("staged executable digest does not match copied bytes")
    return str(target), staged_digest


def _create_launch_stage(
    parent: Path, claude: _PinnedFile, roots_mcp: _PinnedFile,
) -> tuple[Path, str, str, dict[str, str]]:
    directory = Path(tempfile.mkdtemp(prefix=".roots-worker-stage-", dir=parent))
    try:
        os.chmod(directory, 0o700)
        staged_claude, claude_digest = _stage_executable(claude, directory, "claude")
        staged_roots_mcp, roots_mcp_digest = _stage_executable(roots_mcp, directory, "roots-mcp")
        os.chmod(directory, 0o500)
        return directory, staged_claude, staged_roots_mcp, {
            "claude": claude_digest, "roots_mcp": roots_mcp_digest,
        }
    except BaseException:
        os.chmod(directory, 0o700)
        shutil.rmtree(directory)
        raise


def validate_assignment(value: dict[str, Any], *, now: int | None = None) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise BootstrapRefusal("unsupported assignment schema")
    required = (
        "principal", "role", "owner_identity", "ticket_ref", "ticket_updated_at", "roots_url",
        "assignment_id", "session_id", "correlation_id", "task", "claude_config_dir", "claim_dir",
    )
    for name in required:
        if not isinstance(value.get(name), str) or not value[name]:
            raise BootstrapRefusal(f"{name} must be a non-empty string")
    if value.get("authority_scope") != "assigned-ticket-with-shared-main-read":
        raise BootstrapRefusal(
            "authority_scope must be assigned-ticket-with-shared-main-read")
    capabilities = _string_list(value.get("capabilities"), "capabilities")
    partitions = _string_list(value.get("partitions"), "partitions")
    instructions = _string_list(value.get("instruction_files"), "instruction_files")
    allowed_tools = _string_list(value.get("allowed_tools"), "allowed_tools")
    if any(Path(item).name not in {"AGENTS.md", "CLAUDE.md"} for item in instructions):
        raise BootstrapRefusal("instruction_files may contain only AGENTS.md or CLAUDE.md")
    if any(item not in _READ_ONLY_ROOTS_TOOLS for item in allowed_tools):
        raise BootstrapRefusal("allowed_tools must name exact read-only Roots MCP tools")
    if not _CORRELATION_ID.fullmatch(value["correlation_id"]):
        raise BootstrapRefusal("correlation_id has an invalid shape")
    try:
        if str(uuid.UUID(value["assignment_id"])) != value["assignment_id"].lower():
            raise ValueError
    except ValueError as exc:
        raise BootstrapRefusal("assignment_id must be a canonical UUID") from exc
    try:
        if str(uuid.UUID(value["session_id"])) != value["session_id"].lower():
            raise ValueError
    except ValueError as exc:
        raise BootstrapRefusal("session_id must be a canonical UUID") from exc
    if not value["task"].strip():
        raise BootstrapRefusal("task must contain non-whitespace text")
    if len(value["task"].encode()) > MAX_TASK_BYTES:
        raise BootstrapRefusal("task is too large")
    if not isinstance(value.get("resume", False), bool):
        raise BootstrapRefusal("resume must be a boolean")
    use_default_config = value.get("use_default_claude_config", False)
    if not isinstance(use_default_config, bool):
        raise BootstrapRefusal("use_default_claude_config must be a boolean")
    previous = value.get("previous_assignment_id")
    if value.get("resume", False):
        try:
            if not isinstance(previous, str) or str(uuid.UUID(previous)) != previous.lower():
                raise ValueError
        except ValueError as exc:
            raise BootstrapRefusal(
                "resume requires a canonical previous_assignment_id",
            ) from exc
    elif previous is not None:
        raise BootstrapRefusal("previous_assignment_id is only valid with resume")
    issued_at, expires_at = value.get("issued_at"), value.get("expires_at")
    current = int(time.time()) if now is None else now
    if not isinstance(issued_at, int) or not isinstance(expires_at, int):
        raise BootstrapRefusal("issued_at and expires_at must be integer Unix seconds")
    if issued_at > current or expires_at <= current:
        raise BootstrapRefusal("assignment is not currently valid")
    if expires_at - issued_at > MAX_ASSIGNMENT_LIFETIME_SECONDS:
        raise BootstrapRefusal("assignment lifetime exceeds the delegated-token ceiling")
    root = urlsplit(value["roots_url"])
    try:
        loopback = bool(root.hostname and ipaddress.ip_address(root.hostname).is_loopback)
    except ValueError:
        loopback = False
    if root.scheme != "https" and not (root.scheme == "http" and loopback):
        raise BootstrapRefusal("roots_url must use HTTPS, except explicit loopback test endpoints")
    if not root.hostname or root.username or root.password or root.path not in {"", "/"} or root.query or root.fragment:
        raise BootstrapRefusal("roots_url must be an origin without credentials, path, query, or fragment")
    for key in ("workdir", "claude_config_dir", "claim_dir"):
        directory = Path(value.get(key, ""))
        if not directory.is_absolute() or not directory.resolve(strict=True).is_dir() or directory.is_symlink():
            raise BootstrapRefusal(f"{key} must be an existing non-symlink absolute directory")
    if (use_default_config
            and Path(value["claude_config_dir"]).resolve()
            != _default_claude_config_dir().resolve()):
        raise BootstrapRefusal(
            "default Claude config mode must pin the current user's default config directory")
    return {
        **value, "resume": value.get("resume", False),
        "use_default_claude_config": use_default_config,
        "capabilities": capabilities, "partitions": partitions, "allowed_tools": allowed_tools,
        "instruction_files": instructions,
        "claude_executable": _owned_executable(value.get("claude_executable"), "claude_executable"),
        "roots_mcp_executable": _owned_executable(value.get("roots_mcp_executable"), "roots_mcp_executable"),
    }


def read_inherited_token(fd: int) -> str:
    if fd < 3:
        raise BootstrapRefusal("credential descriptor must be inherited, not standard input/output")
    chunks = bytearray()
    try:
        while len(chunks) <= MAX_TOKEN_BYTES:
            chunk = os.read(fd, min(4096, MAX_TOKEN_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
    finally:
        os.close(fd)
    if not chunks or len(chunks) > MAX_TOKEN_BYTES or b"\x00" in chunks:
        raise BootstrapRefusal("inherited Roots credential is empty or oversized")
    try:
        token = chunks.decode().strip()
    except UnicodeDecodeError as exc:
        raise BootstrapRefusal("inherited Roots credential is not UTF-8") from exc
    if not token or any(char.isspace() for char in token):
        raise BootstrapRefusal("inherited Roots credential has an invalid shape")
    return token


def _get_json(url: str, token: str) -> dict[str, Any]:
    request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with build_opener(_NoRedirect).open(request, timeout=10) as response:
            body = response.read(262_145)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise BootstrapRefusal("Roots bootstrap probe failed") from exc
    if len(body) > 262_144:
        raise BootstrapRefusal("Roots bootstrap response is too large")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BootstrapRefusal("Roots bootstrap response is not JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapRefusal("Roots bootstrap response is not a JSON object")
    return value


def verify_proofs(assignment: dict[str, Any], whoami: dict[str, Any], ticket: dict[str, Any]) -> None:
    for key, expected in {
        "principal": assignment["principal"], "kind": "agent", "role": assignment["role"],
        "owner_identity": assignment["owner_identity"], "acting_for": None,
    }.items():
        if whoami.get(key) != expected:
            raise BootstrapRefusal(f"Roots identity mismatch: {key}")
    if set(whoami.get("capabilities", [])) != set(assignment["capabilities"]):
        raise BootstrapRefusal("Roots capability grant is missing or wider than requested")
    if set(whoami.get("partitions", [])) != set(assignment["partitions"]):
        raise BootstrapRefusal("Roots partition grant is missing or wider than requested")
    for key, expected in {
        "ticket_ref": assignment["ticket_ref"],
        "ticket_updated_at": assignment["ticket_updated_at"],
        "assignment_id": assignment["assignment_id"],
        "assignment_expires_at": assignment["expires_at"],
        "token_expires_at": assignment["expires_at"],
    }.items():
        if whoami.get(key) != expected:
            raise BootstrapRefusal(f"Roots credential binding mismatch: {key}")
    if not isinstance(whoami.get("token_jti"), str) or not whoami["token_jti"]:
        raise BootstrapRefusal("Roots credential binding lacks token_jti")
    for key, expected in {
        "ref": assignment["ticket_ref"], "status": "IN PROGRESS",
        "updated_at": assignment["ticket_updated_at"], "owner_identity": assignment["owner_identity"],
        "assigned_principal": assignment["principal"],
    }.items():
        if ticket.get(key) != expected:
            raise BootstrapRefusal(f"Roots ticket mismatch: {key}")
    if not ticket.get("why") or not ticket.get("definition_of_done"):
        raise BootstrapRefusal("Roots ticket lacks why or definition_of_done")


def _instruction_bytes(paths: list[str]) -> tuple[bytes, dict[str, str]]:
    blocks, digests = [], {}
    for value in paths:
        raw = _read_owned_file(Path(value), private=False)
        digests[value] = hashlib.sha256(raw).hexdigest()
        blocks.append(f"\n<!-- source: {value} -->\n".encode() + raw)
    return b"".join(blocks), digests


def _anonymous_file(content: bytes):
    handle = tempfile.TemporaryFile(mode="w+b")
    handle.write(content)
    handle.flush()
    handle.seek(0)
    return handle


def build_receipt(
    assignment_raw: bytes, assignment: dict[str, Any], whoami: dict[str, Any],
    instruction_digests: dict[str, str], *, launched_pid: int | None,
    executable_digests: dict[str, str], launch_stage_dir: str | None = None,
    result: dict[str, str] | None = None, response_sha256: str | None = None,
) -> dict[str, Any]:
    session_flag = "--resume" if assignment["resume"] else "--session-id"
    shape = [
        assignment["claude_executable"], "-p", session_flag, "<session-id>",
        "--output-format", "json", "--json-schema", "<response-schema>",
        "--permission-mode", "dontAsk", "--permission-prompts", "none", "--restricted",
        "--tools", "", "--allowedTools", "<exact-roots-tools>", "--setting-sources", "",
        "--strict-mcp-config", "--mcp-config", "<anonymous-fd>",
        "--append-system-prompt-file", "<anonymous-fd>", "--disable-slash-commands", "--no-chrome",
    ]
    return {
        "schema": RECEIPT_SCHEMA, "verified": True, "ready": result is not None,
        "state": "completed" if result is not None else "verified",
        "worker_pid": launched_pid, "principal": whoami["principal"], "role": whoami["role"],
        "launch_stage_dir": launch_stage_dir,
        "ticket_ref": assignment["ticket_ref"], "ticket_updated_at": assignment["ticket_updated_at"],
        "assignment_id": assignment["assignment_id"],
        "expires_at": assignment["expires_at"], "assignment_sha256": hashlib.sha256(assignment_raw).hexdigest(),
        "instruction_sha256": instruction_digests,
        "claude_executable_sha256": executable_digests["claude"],
        "roots_mcp_executable_sha256": executable_digests["roots_mcp"],
        "command_shape_sha256": hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest(),
        "session_id": assignment["session_id"], "correlation_id": assignment["correlation_id"],
        "resumed": assignment["resume"],
        "previous_assignment_id": assignment.get("previous_assignment_id"),
        "result": result, "response_sha256": response_sha256,
    }


def write_new_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise BootstrapRefusal("receipt must be a new absolute path")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)


class _ReceiptPublisher:
    def __init__(self, path: Path):
        if not path.is_absolute() or path.name in {"", ".", ".."}:
            raise BootstrapRefusal("receipt must be a new absolute path")
        self.name = path.name
        self.parent_fd = _pin_directory(str(path.parent), "receipt parent")
        try:
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            os.close(self.parent_fd)
            raise BootstrapRefusal("receipt must be a new absolute path")

    def publish(self, receipt: dict[str, Any]) -> None:
        temporary = f".{self.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
        fd = -1
        published = False
        try:
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600, dir_fd=self.parent_fd,
            )
            payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short receipt write")
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.link(
                temporary, self.name, src_dir_fd=self.parent_fd, dst_dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
            published = True
            os.unlink(temporary, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        except OSError as exc:
            if published:
                try:
                    os.unlink(self.name, dir_fd=self.parent_fd)
                except OSError:
                    pass
            try:
                os.unlink(temporary, dir_fd=self.parent_fd)
            except OSError:
                pass
            raise BootstrapRefusal("could not publish launch receipt atomically") from exc
        finally:
            if fd >= 0:
                os.close(fd)

    def close(self) -> None:
        os.close(self.parent_fd)


def _stop_and_reap(process: subprocess.Popen[Any]) -> None:
    pgid = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    try:
        os.killpg(pgid, 15)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 3
    while group_exists() and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.02)
    if group_exists():
        try:
            os.killpg(pgid, 9)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while group_exists() and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.02)
    try:
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise BootstrapRefusal("launched worker could not be proved dead") from exc
    if group_exists():
        raise BootstrapRefusal("credentialed worker process group still exists after SIGKILL")


def _write_claim_state(directory_fd: int, name: str, state: dict[str, Any]) -> None:
    temporary = f".{name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd,
    )
    try:
        os.write(fd, json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def _authority_context(
    assignment: dict[str, Any], instruction_digests: dict[str, str],
) -> dict[str, Any]:
    return {
        "principal": assignment["principal"], "role": assignment["role"],
        "owner_identity": assignment["owner_identity"], "ticket_ref": assignment["ticket_ref"],
        "authority_scope": assignment["authority_scope"], "roots_url": assignment["roots_url"],
        "capabilities": sorted(assignment["capabilities"]),
        "partitions": sorted(assignment["partitions"]),
        "allowed_tools": sorted(assignment["allowed_tools"]),
        "claude_config_dir": assignment["claude_config_dir"],
        "use_default_claude_config": assignment.get("use_default_claude_config", False),
        "instruction_sha256": dict(sorted(instruction_digests.items())),
    }


def _claim_assignment(
    directory_fd: int, assignment: dict[str, Any], assignment_raw: bytes,
    instruction_digests: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Atomically consume an authority-issued assignment before executing it."""
    previous = assignment.get("previous_assignment_id")
    if previous is not None:
        try:
            previous_raw = os.open(
                previous + ".json", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise BootstrapRefusal("previous assignment claim is unavailable") from exc
        try:
            previous_state = json.loads(os.read(previous_raw, 262_145))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BootstrapRefusal("previous assignment claim is invalid") from exc
        finally:
            os.close(previous_raw)
        previous_accepted = previous_state.get("accepted_at") if isinstance(previous_state, dict) else None
        previous_expires = previous_state.get("expires_at") if isinstance(previous_state, dict) else None
        if (
            not isinstance(previous_state, dict)
            or previous_state.get("assignment_id") != previous
            or previous_state.get("phase") != "completed"
            or previous_state.get("session_id") != assignment["session_id"]
            or not isinstance(previous_accepted, (int, float))
            or not isinstance(previous_expires, int)
            or previous_accepted >= previous_expires
            or previous_state.get("authority") != _authority_context(assignment, instruction_digests)
        ):
            raise BootstrapRefusal("resume predecessor is not a completed assignment")
    name = assignment["assignment_id"] + ".json"
    try:
        fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600, dir_fd=directory_fd,
        )
    except FileExistsError as exc:
        raise BootstrapRefusal("assignment_id has already been claimed") from exc
    try:
        state = {
            "schema": "hermes.roots-worker-claim/v1",
            "assignment_id": assignment["assignment_id"],
            "assignment_sha256": hashlib.sha256(assignment_raw).hexdigest(),
            "claimed_at": int(time.time()),
            "phase": "claimed", "session_id": assignment["session_id"],
            "correlation_id": assignment["correlation_id"],
            "previous_assignment_id": previous,
            "expires_at": assignment["expires_at"],
            "authority": _authority_context(assignment, instruction_digests),
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory_fd)
    return name, state


def _claim_phase(
    directory_fd: int, name: str, state: dict[str, Any], phase: str, **evidence: Any,
) -> dict[str, Any]:
    state = {**state, "phase": phase, "phase_at": int(time.time()), **evidence}
    _write_claim_state(directory_fd, name, state)
    return state


def _read_bounded(handle: Any, name: str) -> bytes:
    handle.flush()
    if os.fstat(handle.fileno()).st_size > MAX_RESPONSE_BYTES:
        raise BootstrapRefusal(f"Claude {name} exceeded the response limit")
    handle.seek(0)
    return handle.read(MAX_RESPONSE_BYTES + 1)


def _contains_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(key, secret) or _contains_secret(item, secret) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _validated_result(
    raw: bytes, assignment: dict[str, Any], token: str,
) -> tuple[dict[str, str], str]:
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapRefusal("Claude response is not JSON") from exc
    if _contains_secret(envelope, token):
        raise BootstrapRefusal("Claude response contained credential material")
    if not isinstance(envelope, dict) or envelope.get("session_id") != assignment["session_id"]:
        raise BootstrapRefusal("Claude response session_id does not match the assignment")
    result = envelope.get("structured_output")
    if not isinstance(result, dict) or set(result) != {"correlation_id", "status", "answer"}:
        raise BootstrapRefusal("Claude response does not match the required result shape")
    if result.get("correlation_id") != assignment["correlation_id"]:
        raise BootstrapRefusal("Claude response correlation_id does not match the assignment")
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise BootstrapRefusal("Claude response answer is empty")
    if result.get("status") != "completed":
        raise BootstrapRefusal("Claude reported that the assigned task is blocked")
    return result, hashlib.sha256(raw).hexdigest()


def _task_prompt(assignment: dict[str, Any]) -> bytes:
    return (json.dumps({
        "correlation_id": assignment["correlation_id"],
        "task": assignment["task"],
        "response_contract": {
            "correlation_id": "copy the supplied correlation_id exactly",
            "status": "completed only if the task succeeded; otherwise blocked",
            "answer": "the completed task response",
        },
    }, separators=(",", ":")) + "\n").encode()


def run(request_path: Path, token_fd: int, receipt_path: Path, *, verify_only: bool = False) -> None:
    raw_assignment, assignment_raw = _read_assignment(request_path)
    assignment = validate_assignment(raw_assignment)
    token = read_inherited_token(token_fd)
    base = assignment["roots_url"].rstrip("/")
    whoami = _get_json(base + "/whoami", token)
    ticket = _get_json(base + "/tickets/" + quote(assignment["ticket_ref"], safe=""), token)
    verify_proofs(assignment, whoami, ticket)
    instructions, instruction_digests = _instruction_bytes(assignment["instruction_files"])
    if int(time.time()) >= assignment["expires_at"]:
        raise BootstrapRefusal("assignment expired during bootstrap")
    with ExitStack() as resources:
        claude = _pin_executable(assignment["claude_executable"], "claude_executable")
        resources.callback(claude.close)
        roots_mcp = _pin_executable(assignment["roots_mcp_executable"], "roots_mcp_executable")
        resources.callback(roots_mcp.close)
        executable_digests = {"claude": claude.sha256, "roots_mcp": roots_mcp.sha256}
        publisher = _ReceiptPublisher(receipt_path)
        resources.callback(publisher.close)
        if verify_only:
            publisher.publish(build_receipt(
                assignment_raw, assignment, whoami, instruction_digests, launched_pid=None,
                executable_digests=executable_digests,
            ))
            return
        workdir_fd = _pin_directory(assignment["workdir"], "workdir")
        resources.callback(os.close, workdir_fd)
        config_fd = _pin_directory(assignment["claude_config_dir"], "claude_config_dir")
        resources.callback(os.close, config_fd)
        _verify_pinned_directory_path(assignment["claude_config_dir"], config_fd, "claude_config_dir")
        claim_dir_fd = _pin_directory(assignment["claim_dir"], "claim_dir")
        resources.callback(os.close, claim_dir_fd)
        claim_name, claim_state = _claim_assignment(
            claim_dir_fd, assignment, assignment_raw, instruction_digests,
        )
        process: subprocess.Popen[Any] | None = None
        stage_dir, staged_claude, staged_roots_mcp, staged_digests = _create_launch_stage(
            receipt_path.parent, claude, roots_mcp,
        )
        mcp_config = json.dumps({"mcpServers": {"roots": {
            "command": staged_roots_mcp,
            # Claude is the scoped agent principal and therefore part of the TCB.
            # The token is anonymous and ephemeral, but intentionally readable by
            # this process so it can start the approved Roots MCP server.
            "env": {"ROOTS_URL": assignment["roots_url"], "ROOTS_TOKEN": token},
        }}}, separators=(",", ":")).encode()
        try:
            with (
                _anonymous_file(mcp_config) as mcp_file,
                _anonymous_file(instructions) as instruction_file,
                tempfile.TemporaryFile(mode="w+b") as stdout_file,
                tempfile.TemporaryFile(mode="w+b") as stderr_file,
            ):
                session_flag = "--resume" if assignment["resume"] else "--session-id"
                argv = [
                    staged_claude, "-p", session_flag, assignment["session_id"],
                    "--output-format", "json", "--json-schema",
                    json.dumps(_RESPONSE_SCHEMA, separators=(",", ":")),
                    "--permission-mode", "dontAsk", "--permission-prompts", "none", "--restricted",
                    "--tools", "", "--allowedTools", *assignment["allowed_tools"],
                    "--setting-sources", "", "--strict-mcp-config",
                    "--mcp-config", f"/dev/fd/{mcp_file.fileno()}",
                    "--append-system-prompt-file", f"/dev/fd/{instruction_file.fileno()}",
                    "--disable-slash-commands", "--no-chrome",
                ]
                child_env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
                # Claude Code treats an explicitly configured ~/.claude as a
                # custom profile and does not use the logged-in default
                # credential store. Pin and replacement-check the directory in
                # both modes, but leave the variable absent for the default.
                if not assignment["use_default_claude_config"]:
                    child_env["CLAUDE_CONFIG_DIR"] = assignment["claude_config_dir"]

                def enter_pinned_workdir() -> None:
                    os.fchdir(workdir_fd)

                claim_state = _claim_phase(
                    claim_dir_fd, claim_name, claim_state, "launching",
                )
                process = subprocess.Popen(
                    argv, executable=staged_claude, env=child_env, preexec_fn=enter_pinned_workdir,
                    pass_fds=(workdir_fd, config_fd, mcp_file.fileno(), instruction_file.fileno()),
                    stdin=subprocess.PIPE, stdout=stdout_file, stderr=stderr_file, start_new_session=True,
                )
                launched_pid = process.pid
                claim_state = _claim_phase(
                    claim_dir_fd, claim_name, claim_state, "spawned", worker_pid=launched_pid,
                )
                _verify_pinned_directory_path(
                    assignment["claude_config_dir"], config_fd, "claude_config_dir",
                )
                try:
                    process.communicate(
                        input=_task_prompt(assignment),
                        timeout=max(0.01, assignment["expires_at"] - time.time()),
                    )
                except subprocess.TimeoutExpired as exc:
                    raise BootstrapRefusal("Claude task exceeded the assignment lifetime") from exc
                stdout = _read_bounded(stdout_file, "stdout")
                stderr = _read_bounded(stderr_file, "stderr")
                if process.returncode != 0:
                    # Both processes are inside the credential TCB and may echo
                    # the scoped token. Never carry child output into an error,
                    # receipt, terminal, or orchestrator log.
                    raise BootstrapRefusal("Claude task failed; child diagnostics withheld")
                result, response_sha256 = _validated_result(stdout, assignment, token)
            _stop_and_reap(process)
            process = None
            # Same-UID replacement is inside the declared local TCB, but detect
            # accidental profile replacement before accepting the result.
            _verify_pinned_directory_path(
                assignment["claude_config_dir"], config_fd, "claude_config_dir",
            )
            accepted_at = time.time()
            if accepted_at >= assignment["expires_at"]:
                raise BootstrapRefusal("assignment expired before result acceptance")
            claim_state = _claim_phase(
                claim_dir_fd, claim_name, claim_state, "completed",
                accepted_at=accepted_at, response_sha256=response_sha256, result=result,
            )
            if int(time.time()) >= assignment["expires_at"]:
                raise BootstrapRefusal("assignment expired before result publication")
            publisher.publish(build_receipt(
                assignment_raw, assignment, whoami, instruction_digests, launched_pid=launched_pid,
                executable_digests=staged_digests, launch_stage_dir=str(stage_dir),
                result=result, response_sha256=response_sha256,
            ))
        except BaseException:
            if process is not None:
                _stop_and_reap(process)
            os.chmod(stage_dir, 0o700)
            shutil.rmtree(stage_dir)
            raise
        os.chmod(stage_dir, 0o700)
        shutil.rmtree(stage_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--token-fd", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        run(args.request, args.token_fd, args.receipt, verify_only=args.verify_only)
    except BootstrapRefusal as exc:
        print(f"managed Roots worker refused: {exc}", file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
