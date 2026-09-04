"""Filesystem adapter for Desktop workspaces executed over SSH."""

from __future__ import annotations

import base64
import binascii
import posixpath
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.ssh import SSHEnvironment


class SshWorkspaceFsError(RuntimeError):
    """A filesystem error reported by the SSH execution target."""

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class _SshFsConfig:
    host: str
    user: str
    port: int
    key_path: str
    cwd: str
    timeout: int


_CACHE_LOCK = threading.RLock()
_BACKENDS: dict[str, tuple[_SshFsConfig, "SshWorkspaceFs"]] = {}


class SshWorkspaceFs:
    """Binary-safe workspace operations using the configured SSH transport."""

    def __init__(self, env: SSHEnvironment):
        self._env = env
        self._lock = threading.RLock()

    @property
    def cwd(self) -> str:
        return self._normalize(self._env.cwd)

    def _normalize(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            raise SshWorkspaceFsError("EINVAL", "Path is required")
        if "\0" in raw or raw.lower().startswith("file:"):
            raise SshWorkspaceFsError("EINVAL", "Invalid path")
        if raw == "~":
            raw = self._env._remote_home
        elif raw.startswith("~/"):
            raw = posixpath.join(self._env._remote_home, raw[2:])
        elif not raw.startswith("/"):
            raw = posixpath.join(self._env.cwd, raw)
        return posixpath.normpath(raw)

    def _execute(self, command: str, *, stdin_data: str | None = None, timeout: int | None = None) -> dict:
        with self._lock:
            return self._env.execute(
                command,
                cwd=self._env.cwd,
                stdin_data=stdin_data,
                timeout=timeout or self._env.timeout,
            )

    @staticmethod
    def _error_code(output: str, fallback: str = "EIO") -> str:
        marker = "__HERMES_FS_ERROR__:"
        for line in (output or "").splitlines():
            if line.startswith(marker):
                return line[len(marker) :].strip() or fallback
        return fallback

    def list_dir(self, path: str, hidden_names: set[str] | frozenset[str]) -> dict[str, Any]:
        target = self._normalize(path)
        quoted = shlex.quote(target)
        command = f"""
p={quoted}
if [ ! -e "$p" ] && [ ! -L "$p" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if [ ! -d "$p" ]; then printf '__HERMES_FS_ERROR__:ENOTDIR\\n'; exit 45; fi
if [ ! -r "$p" ]; then printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; fi
for child in "$p"/* "$p"/.[!.]* "$p"/..?*; do
  [ -e "$child" ] || [ -L "$child" ] || continue
  name=${{child##*/}}
  name64=$(printf '%s' "$name" | base64 | tr -d '\\r\\n')
  path64=$(printf '%s' "$child" | base64 | tr -d '\\r\\n')
  if [ -d "$child" ]; then kind=d; else kind=f; fi
  printf '%s\\t%s\\t%s\\n' "$name64" "$path64" "$kind"
done
"""
        result = self._execute(command)
        if result.get("returncode") != 0:
            return {"entries": [], "error": self._error_code(result.get("output", ""), "read-error")}

        entries = []
        try:
            for line in result.get("output", "").splitlines():
                if not line.strip():
                    continue
                name64, path64, kind = line.split("\t", 2)
                name = base64.b64decode(name64, validate=True).decode("utf-8", errors="replace")
                child_path = base64.b64decode(path64, validate=True).decode("utf-8", errors="replace")
                if name in hidden_names:
                    continue
                entries.append({"name": name, "path": child_path, "isDirectory": kind == "d"})
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise SshWorkspaceFsError("EIO", "SSH filesystem returned an invalid directory listing") from exc

        entries.sort(key=lambda item: (not item["isDirectory"], item["name"].lower(), item["name"]))
        return {"entries": entries}

    def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int,
        read_limit: int | None = None,
    ) -> tuple[bytes, int, str]:
        target = self._normalize(path)
        quoted = shlex.quote(target)
        read_command = (
            f"set -o pipefail; head -c {int(read_limit)} < \"$p\" | base64"
            if read_limit is not None
            else 'base64 < "$p"'
        )
        command = f"""
p={quoted}
if [ ! -e "$p" ] && [ ! -L "$p" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if [ ! -f "$p" ]; then printf '__HERMES_FS_ERROR__:ENOTREG\\n'; exit 45; fi
size=$(stat -c %s "$p" 2>/dev/null || stat -f %z "$p" 2>/dev/null) || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
if [ "$size" -gt {int(max_bytes)} ]; then printf '__HERMES_FS_ERROR__:EFBIG\\n'; exit 47; fi
printf '__HERMES_FS_SIZE__:%s\\n' "$size"
{{ {read_command}; }} || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
"""
        result = self._execute(command)
        output = result.get("output", "")
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(output))
        lines = output.splitlines()
        if not lines or not lines[0].startswith("__HERMES_FS_SIZE__:"):
            raise SshWorkspaceFsError("EIO", "SSH filesystem did not return a file size")
        try:
            size = int(lines[0].split(":", 1)[1])
        except ValueError as exc:
            raise SshWorkspaceFsError("EIO", "SSH filesystem returned an invalid file size") from exc
        encoded = "".join(lines[1:])
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SshWorkspaceFsError("EIO", "SSH filesystem returned invalid file data") from exc
        return data, size, target

    def inspect_file(self, path: str) -> tuple[str, int]:
        """Resolve a remote regular file without reading its contents."""
        target = self._normalize(path)
        quoted = shlex.quote(target)
        command = f"""
p={quoted}
if [ ! -e "$p" ] && [ ! -L "$p" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
target=$(realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null) || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
if [ ! -f "$target" ]; then printf '__HERMES_FS_ERROR__:ENOTREG\\n'; exit 45; fi
size=$(stat -c %s "$target" 2>/dev/null || stat -f %z "$target" 2>/dev/null) || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
path64=$(printf '%s' "$target" | base64 | tr -d '\\r\\n')
printf '__HERMES_FS_SIZE__:%s\\n' "$size"
printf '__HERMES_FS_PATH__:%s\\n' "$path64"
"""
        result = self._execute(command)
        output = result.get("output", "")
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(output))
        lines = output.splitlines()
        if len(lines) != 2 or not lines[0].startswith("__HERMES_FS_SIZE__:") or not lines[1].startswith("__HERMES_FS_PATH__:"):
            raise SshWorkspaceFsError("EIO", "SSH filesystem did not return file metadata")
        try:
            size = int(lines[0].split(":", 1)[1])
            resolved = base64.b64decode(lines[1].split(":", 1)[1], validate=True).decode("utf-8", errors="strict")
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise SshWorkspaceFsError("EIO", "SSH filesystem returned invalid file metadata") from exc
        return resolved, size

    def stream_file(self, path: str, *, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        """Yield a previously-authorized remote file without buffering it."""
        target = self._normalize(path)

        def _chunks() -> Iterator[bytes]:
            command = self._env._build_ssh_command()
            command.extend(["bash", "-c", shlex.quote(f"exec cat < {shlex.quote(target)}")])
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
            try:
                if process.stdout is None:
                    raise SshWorkspaceFsError("EIO", "SSH file stream has no stdout")
                while chunk := process.stdout.read(chunk_size):
                    yield chunk
                if process.wait(timeout=self._env.timeout) != 0:
                    raise SshWorkspaceFsError("EIO", "SSH file stream failed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                if process.stdout is not None:
                    process.stdout.close()

        return _chunks()

    def write_text(self, path: str, content: str, *, max_bytes: int) -> tuple[str, int]:
        target = self._normalize(path)
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > max_bytes:
            raise SshWorkspaceFsError("EFBIG")
        parent = posixpath.dirname(target) or "/"
        temp_path = posixpath.join(parent, f".{posixpath.basename(target)}.hermes-tmp-{uuid.uuid4().hex[:12]}")
        q_target = shlex.quote(target)
        q_parent = shlex.quote(parent)
        q_temp = shlex.quote(temp_path)
        command = f"""
target={q_target}
parent={q_parent}
tmp={q_temp}
if [ ! -d "$parent" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if [ -e "$target" ] && [ ! -f "$target" ]; then printf '__HERMES_FS_ERROR__:ENOTREG\\n'; exit 45; fi
trap 'rm -f "$tmp"' EXIT HUP INT TERM
cat > "$tmp" && mv -f "$tmp" "$target"
"""
        result = self._execute(command, stdin_data=content)
        if result.get("returncode") != 0 or result.get("stdin_error"):
            raise SshWorkspaceFsError(self._error_code(result.get("output", "")))
        return target, encoded_size

    def git_root(self, path: str) -> str | None:
        target = self._normalize(path)
        quoted = shlex.quote(target)
        command = f"""
p={quoted}
[ -d "$p" ] || p=${{p%/*}}
while [ -n "$p" ]; do
  if [ -e "$p/.git" ]; then printf '%s\\n' "$p"; exit 0; fi
  [ "$p" = / ] && break
  p=${{p%/*}}; [ -n "$p" ] || p=/
done
exit 1
"""
        result = self._execute(command)
        if result.get("returncode") != 0:
            return None
        return (result.get("output") or "").strip() or None

    def git_branch(self, path: str) -> str:
        target = self._normalize(path)
        result = self._execute(f"git -C {shlex.quote(target)} branch --show-current 2>/dev/null")
        return (result.get("output") or "").strip() if result.get("returncode") == 0 else ""


def get_ssh_workspace_fs(profile_key: str, terminal_config: dict[str, Any]) -> SshWorkspaceFs | None:
    """Return a cached adapter when this profile selects the SSH backend."""
    if str(terminal_config.get("backend") or "local").strip().lower() != "ssh":
        return None

    config = _SshFsConfig(
        host=str(terminal_config.get("ssh_host") or "").strip(),
        user=str(terminal_config.get("ssh_user") or "").strip(),
        port=int(terminal_config.get("ssh_port") or 22),
        key_path=str(terminal_config.get("ssh_key") or "").strip(),
        cwd=str(terminal_config.get("cwd") or "~").strip() or "~",
        timeout=int(terminal_config.get("timeout") or 180),
    )
    if not config.host or not config.user:
        raise SshWorkspaceFsError("ECONN", "SSH host and user are required")

    with _CACHE_LOCK:
        cached = _BACKENDS.get(profile_key)
        if cached and cached[0] == config:
            return cached[1]
        if cached:
            cached[1]._env.cleanup()
        try:
            env = SSHEnvironment(
                host=config.host,
                user=config.user,
                cwd=config.cwd,
                timeout=config.timeout,
                port=config.port,
                key_path=config.key_path,
                sync_files=False,
            )
        except Exception as exc:
            raise SshWorkspaceFsError("ECONN", "Could not connect to the SSH workspace") from exc
        backend = SshWorkspaceFs(env)
        _BACKENDS[profile_key] = (config, backend)
        return backend
