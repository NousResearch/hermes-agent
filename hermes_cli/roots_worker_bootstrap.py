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
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA = "hermes.roots-worker-assignment/v1"
RECEIPT_SCHEMA = "hermes.roots-worker-bootstrap-receipt/v1"
MAX_TOKEN_BYTES = 16_384
MAX_ASSIGNMENT_LIFETIME_SECONDS = 900
_PASSTHROUGH_ENV = ("HOME", "PATH", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM")


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


def _stable_directory_path(fd: int, name: str) -> str:
    """Return an inode-bound directory path, or refuse on platforms without one."""
    candidate = f"/proc/self/fd/{fd}"
    try:
        if Path(candidate).is_dir() and os.path.samefile(candidate, fd):
            return candidate
    except OSError:
        pass
    raise BootstrapRefusal(
        f"this platform cannot bind {name} by descriptor without copying provider state"
    )


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
        "channel_plugin", "session_id", "claude_config_dir",
    )
    for name in required:
        if not isinstance(value.get(name), str) or not value[name]:
            raise BootstrapRefusal(f"{name} must be a non-empty string")
    if value.get("authority_scope") != "assigned-ticket-only":
        raise BootstrapRefusal("authority_scope must be assigned-ticket-only")
    capabilities = _string_list(value.get("capabilities"), "capabilities")
    partitions = _string_list(value.get("partitions"), "partitions")
    instructions = _string_list(value.get("instruction_files"), "instruction_files")
    if any(Path(item).name not in {"AGENTS.md", "CLAUDE.md"} for item in instructions):
        raise BootstrapRefusal("instruction_files may contain only AGENTS.md or CLAUDE.md")
    if not value["channel_plugin"].startswith("plugin:") or "@" not in value["channel_plugin"]:
        raise BootstrapRefusal("channel_plugin must be an approved marketplace plugin reference")
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
    for key in ("workdir", "claude_config_dir"):
        directory = Path(value.get(key, ""))
        if not directory.is_absolute() or not directory.resolve(strict=True).is_dir() or directory.is_symlink():
            raise BootstrapRefusal(f"{key} must be an existing non-symlink absolute directory")
    return {
        **value, "capabilities": capabilities, "partitions": partitions,
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
        "acting_for": assignment["owner_identity"],
    }.items():
        if whoami.get(key) != expected:
            raise BootstrapRefusal(f"Roots identity mismatch: {key}")
    if set(whoami.get("capabilities", [])) != set(assignment["capabilities"]):
        raise BootstrapRefusal("Roots capability grant is missing or wider than requested")
    if set(whoami.get("partitions", [])) != set(assignment["partitions"]):
        raise BootstrapRefusal("Roots partition grant is missing or wider than requested")
    if whoami.get("channel_plugins") != [assignment["channel_plugin"]]:
        raise BootstrapRefusal("Roots channel plugin grant is missing or wider than requested")
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
) -> dict[str, Any]:
    shape = [assignment["claude_executable"], "--mcp-config", "<anonymous-fd>",
             "--append-system-prompt-file", "<anonymous-fd>", "--channels", assignment["channel_plugin"]]
    return {
        "schema": RECEIPT_SCHEMA, "verified": True, "ready": False,
        "state": "launched" if launched_pid is not None else "verified",
        "worker_pid": launched_pid, "principal": whoami["principal"], "role": whoami["role"],
        "launch_stage_dir": launch_stage_dir,
        "ticket_ref": assignment["ticket_ref"], "ticket_updated_at": assignment["ticket_updated_at"],
        "expires_at": assignment["expires_at"], "assignment_sha256": hashlib.sha256(assignment_raw).hexdigest(),
        "instruction_sha256": instruction_digests,
        "claude_executable_sha256": executable_digests["claude"],
        "roots_mcp_executable_sha256": executable_digests["roots_mcp"],
        "command_shape_sha256": hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest(),
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
        stable_config_dir = _stable_directory_path(config_fd, "claude_config_dir")
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
            with _anonymous_file(mcp_config) as mcp_file, _anonymous_file(instructions) as instruction_file:
                argv = [
                    staged_claude, "--restricted", "--permission-mode", "dontAsk",
                    "--mcp-config", f"/dev/fd/{mcp_file.fileno()}",
                    "--append-system-prompt-file", f"/dev/fd/{instruction_file.fileno()}",
                    "--channels", assignment["channel_plugin"], "--session-id", assignment["session_id"],
                    "--disable-slash-commands", "--no-chrome",
                ]
                child_env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
                child_env["CLAUDE_CONFIG_DIR"] = stable_config_dir

                def enter_pinned_workdir() -> None:
                    os.fchdir(workdir_fd)

                process = subprocess.Popen(
                    argv, executable=staged_claude, env=child_env, preexec_fn=enter_pinned_workdir,
                    pass_fds=(workdir_fd, config_fd, mcp_file.fileno(), instruction_file.fileno()),
                    start_new_session=True,
                )
            if int(time.time()) >= assignment["expires_at"]:
                raise BootstrapRefusal("assignment expired before launch publication")
            publisher.publish(build_receipt(
                assignment_raw, assignment, whoami, instruction_digests, launched_pid=process.pid,
                executable_digests=staged_digests, launch_stage_dir=str(stage_dir),
            ))
        except BaseException:
            if process is not None:
                _stop_and_reap(process)
            os.chmod(stage_dir, 0o700)
            shutil.rmtree(stage_dir)
            raise


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
