"""On-demand filesystem reads from an already running terminal environment.

The registry is process-local. In particular, a serve process cannot recover a
compute-host child's environment by invoking a provider's creation factory.
"""

import base64
import binascii
import shlex
import uuid

from hermes_cli.ssh_workspace_fs import SshWorkspaceFs, SshWorkspaceFsError
from tools.environments.base import BaseEnvironment


class TerminalWorkspaceFsError(SshWorkspaceFsError):
    workspace_kind = "Terminal"


class TerminalWorkspaceFs(SshWorkspaceFs):
    """Reuse the shell metadata/base64 protocol, not SSH's subprocess transport."""

    def __init__(self, env: BaseEnvironment, *, env_key=None):
        super().__init__(env)
        self._env_key = env_key
        if not getattr(env, "_hermes_file_read_token", None):
            env._hermes_file_read_token = uuid.uuid4().hex
        self._generation = env._hermes_file_read_token

    def _validate_generation(self):
        from tools import terminal_tool
        from tools.terminal_file_handles import resolve
        with terminal_tool._env_lock:
            if self._env_key is not None and resolve(self._env_key) is not self._env:
                raise TerminalWorkspaceFsError("ECONN", "Terminal environment is unavailable")
        if self._env._hermes_file_read_token != self._generation:
            raise TerminalWorkspaceFsError("ECONN", "Terminal environment is unavailable")

    def _execute(self, command, **kwargs):
        from tools import terminal_tool

        try:
            with terminal_tool._call_in_flight(self._env_key):
                kwargs["timeout"] = min(kwargs.get("timeout") or self._env.timeout, 10)
                self._validate_generation()
                result = self._env.execute(command, **kwargs)
                self._validate_generation()
                return result
        except Exception as exc:
            raise TerminalWorkspaceFsError("ECONN", "Terminal environment is unavailable") from exc

    def _normalize(self, path):
        if str(path).strip().startswith("~") and not getattr(self._env, "_remote_home", None):
            raise SshWorkspaceFsError("EINVAL", "Remote home is unknown; use an absolute path")
        return super()._normalize(path)

    def inspect_file(self, path):
        from gateway.media_fetch import remote_path_is_denied
        from hermes_cli.web_routers.files import _is_sensitive_path
        from pathlib import PurePosixPath

        target = self._normalize(path)
        # realpath may succeed for a nonexistent final component. Preserve ENOENT,
        # instead of treating a deleted file as a non-regular file.
        result = self._execute(f"test -e {shlex.quote(target)}")
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError("ENOENT")
        resolved, size = super().inspect_file(target)
        if (_is_sensitive_path(PurePosixPath(resolved))
                or remote_path_is_denied(resolved, getattr(self._env, "_remote_home", None))):
            raise SshWorkspaceFsError("EACCES")
        return resolved, size

    def read_bytes(self, path, *, max_bytes, read_limit=None):
        target, _size = self.inspect_file(path)
        return super().read_bytes(target, max_bytes=max_bytes, read_limit=read_limit)

    def stream_file(self, path, chunk_size=64 * 1024, *, offset=0, length=None):
        target, size = self.inspect_file(path)
        remaining = max(0, size - offset)
        if length is not None:
            remaining = min(remaining, length)
        while remaining:
            count = min(chunk_size, remaining)
            block_size = 64 * 1024
            prefix = offset % block_size
            blocks = (prefix + count + block_size - 1) // block_size
            result = self._execute(
                f"set -o pipefail; dd if={shlex.quote(target)} bs={block_size} "
                f"skip={offset // block_size} count={blocks} "
                "2>/dev/null | base64"
            )
            if result.get("returncode") != 0:
                raise SshWorkspaceFsError("EIO", "Terminal file read failed")
            try:
                data = base64.b64decode("".join(result.get("output", "").splitlines()), validate=True)
                data = data[prefix:prefix + count]
            except (ValueError, binascii.Error) as exc:
                raise SshWorkspaceFsError("EIO", "Terminal returned invalid file data") from exc
            if len(data) != count:
                raise SshWorkspaceFsError("EIO", "Terminal file changed during read")
            yield data
            offset += len(data)
            remaining -= len(data)

    def write_text(self, *args, **kwargs):
        raise SshWorkspaceFsError("EROFS", "Terminal attachment access is read-only")


def get_terminal_workspace_fs(session):
    """Look up exactly this session in the owning profile scope; never create.

    Caller binds the profile's home and terminal policy. SSH and Docker retain
    their existing configured-reader and host mount-translation paths.
    """
    from tools import terminal_tool

    backend = terminal_tool._session_scope().env_type
    if backend in {"local", "docker", "ssh"}:
        return None
    sid = str((session or {}).get("id") or "")
    if not sid:
        raise SshWorkspaceFsError("ECONN", "A session is required for terminal attachment access")
    from tui_gateway.terminal_files import compute_session_fs, scoped_session_fs

    remote = compute_session_fs(sid, backend)
    if remote is not None:
        return remote
    return scoped_session_fs({"id": sid}, backend)
