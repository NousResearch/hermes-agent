"""Filesystem adapter for Desktop workspaces executed over SSH."""

from __future__ import annotations

import base64
import binascii
import json
import posixpath
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Iterator

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.ssh import SSHEnvironment


# This worker runs on the execution target, never against the agent host's filesystem.
# It inspects names/metadata only; the local live-SQLite registry is process-local and cannot
# establish whether another process on the SSH host has a database open.
_REMOTE_MUTATION_GUARD = r'''
import json, os, stat, sys


def fail(code):
    print('__HERMES_FS_ERROR__:' + code)
    sys.exit(1)


def sensitive(path):
    for part in path.split('/'):
        lowered = part.lower()
        if lowered in names or lowered in directories or lowered in {'.env', '.envrc'} or lowered.startswith('.env.'):
            return True
    return False


def check_policy(path):
    if sensitive(path):
        fail('EACCES')
    for root in roots:
        if path == root or root.startswith(path.rstrip('/') + '/'):
            fail('EINVAL')
    for prefix in protected:
        if path == prefix or path.startswith(prefix + '/'):
            fail('EACCES')


def check(path):
    check_policy(path)
    # Keep the final component: rename/unlink/replace act on the entry, not its referent.
    parent = os.path.dirname(path)
    current = parent
    while current != '/':
        if stat.S_ISLNK(os.lstat(current).st_mode):
            fail('EINVAL')
        current = os.path.dirname(current)
    entry = os.path.join(os.path.realpath(parent), os.path.basename(path))
    check_policy(entry)
    return entry


try:
    policy = json.loads(sys.argv[1])
    names, directories = set(policy['names']), set(policy['directories'])
    roots = {'/'}
    for root in policy['roots'] + [os.path.expanduser('~')]:
        if not root or not os.path.isabs(root):
            fail('EINVAL')
        os.stat(root)  # Require an inspectable root, including on older remote Python versions.
        roots.update((os.path.normpath(root), os.path.realpath(root)))
    protected = []
    for home in policy['homes'] + [os.path.expanduser('~')]:
        for location in {os.path.normpath(home), os.path.realpath(home)}:
            protected.extend(os.path.join(location, p) for p in policy['home_directories'])
    targets = policy['targets']
    entries = [check(target) for target in targets]
    source = targets[0]
    if policy['scan'] and stat.S_ISDIR(os.lstat(source).st_mode):
        # scandir/lstat raise on unreadable or vanished entries: never silently skip a subtree.
        pending = [source]
        while pending:
            parent = pending.pop()
            with os.scandir(parent) as children:
                for child in children:
                    check(child.path)
                    if len(targets) > 1:
                        relative = os.path.relpath(child.path, source)
                        # The destination subtree does not exist yet; inspect policy only.
                        check_policy(os.path.join(targets[1], relative))
                        check_policy(os.path.join(entries[1], relative))
                    if stat.S_ISDIR(child.stat(follow_symlinks=False).st_mode):
                        pending.append(child.path)
except FileNotFoundError:
    fail('ENOENT')
except PermissionError:
    fail('EACCES')
except (OSError, ValueError, RuntimeError):
    fail('EIO')
'''


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


# ``_CACHE_LOCK`` guards only the dicts and is never held across a dial or cleanup, so one
# unreachable host cannot stall file requests on every other SSH profile. A dial or cleanup holds
# its profile's lock plus that endpoint's lock: profiles on one user@host:port share a control
# socket, and one profile's cleanup would otherwise unlink the socket another profile's dial just
# created. At most one endpoint lock is held at a time, always inside the profile's lock.
_CACHE_LOCK = threading.Lock()
_BACKENDS: dict[str, tuple[_SshFsConfig, SshWorkspaceFs]] = {}
_DIAL_LOCKS: dict[str, threading.Lock] = {}
_ENDPOINT_LOCKS: dict[tuple[str, str, int], threading.Lock] = {}


def _lock_for(locks: dict, key) -> threading.Lock:
    with _CACHE_LOCK:
        return locks.setdefault(key, threading.Lock())


def _endpoint(config: _SshFsConfig) -> tuple[str, str, int]:
    """The key ``SSHEnvironment`` derives its control socket from."""
    return config.user, config.host, config.port


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
        """Return a regular file's resolved remote path and size without reading it."""
        target = self._normalize(path)
        command = f"""
p={shlex.quote(target)}
target=$(realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null) || {{ printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; }}
if [ ! -f "$target" ]; then printf '__HERMES_FS_ERROR__:ENOTREG\\n'; exit 45; fi
size=$(stat -c %s "$target" 2>/dev/null || stat -f %z "$target" 2>/dev/null) || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
printf '__HERMES_FS_TARGET__:%s\\n' "$(printf '%s' "$target" | base64 | tr -d '\\r\\n')"
printf '__HERMES_FS_SIZE__:%s\\n' "$size"
"""
        result = self._execute(command)
        output = result.get("output", "")
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(output))
        target_line = next((line for line in output.splitlines() if line.startswith("__HERMES_FS_TARGET__:")), "")
        size_line = next((line for line in output.splitlines() if line.startswith("__HERMES_FS_SIZE__:")), "")
        if not target_line or not size_line:
            raise SshWorkspaceFsError("EIO", "SSH filesystem did not return file metadata")
        try:
            resolved = base64.b64decode(target_line.split(":", 1)[1], validate=True).decode("utf-8", errors="strict")
            size = int(size_line.split(":", 1)[1])
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise SshWorkspaceFsError("EIO", "SSH filesystem returned invalid file metadata") from exc
        # Callers vet this path, then read or stream it through _normalize; a name it rewrites
        # (``.env `` loses its trailing space) would be vetted as one file and served as another.
        if self._normalize(resolved) != resolved:
            raise SshWorkspaceFsError("EINVAL", "Invalid path")
        return resolved, size

    def stream_file(
        self, path: str, chunk_size: int = 64 * 1024, *, offset: int = 0, length: int | None = None,
    ) -> Iterator[bytes]:
        """Stream an already-authorized remote file without buffering it in memory. ``offset`` and
        ``length`` read one byte range of it instead (a media element's ``Range`` request)."""
        target = self._normalize(path)
        source = f"< {shlex.quote(target)}"
        if offset or length is not None:
            script = f"tail -c +{int(offset) + 1} {source}"
            if length is not None:
                script += f" | head -c {int(length)}"
        else:
            script = f"exec cat {source}"
        command = self._env._build_ssh_command()
        command.extend(["bash", "-c", shlex.quote(script)])
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(chunk_size):
                yield chunk
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()

    def _mutation_guard(self, *targets: str, scan: bool = False) -> str:
        # Reuse the browser's canonical name policy without resolving remote paths on this host.
        # Import late: the router imports this adapter, and its policy can grow independently.
        from agent.file_safety import HOME_CREDENTIAL_DIRS, _READ_DENIED_DIRS
        from hermes_cli.web_routers.files import (
            _SENSITIVE_MANAGED_DIR_NAMES, _SENSITIVE_MANAGED_FILE_BASENAMES,
        )

        policy = json.dumps({
            "names": sorted(_SENSITIVE_MANAGED_FILE_BASENAMES),
            "directories": sorted(_SENSITIVE_MANAGED_DIR_NAMES | {name for name, _, _ in _READ_DENIED_DIRS}),
            "home_directories": list(HOME_CREDENTIAL_DIRS),
            "roots": [self.cwd, self._env._remote_home],
            "homes": [self._env._remote_home],
            "targets": list(targets),
            "scan": scan,
        })
        return (
            "command -v python3 >/dev/null 2>&1 || { printf '__HERMES_FS_ERROR__:EUNSUPPORTED\\n'; exit 1; }\n"
            f"python3 -c {shlex.quote(_REMOTE_MUTATION_GUARD)} {shlex.quote(policy)} || exit $?\n"
        )

    def create(self, path: str, *, directory: bool) -> str:
        target = self._mutation_path(path)
        parent = posixpath.dirname(target) or "/"
        worker = "import os, sys; fd = os.open(sys.argv[1], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666); os.close(fd)"
        command = self._mutation_guard(target) + self._safe_parent_function() + f"""
parent_dir={shlex.quote(parent)}
target={shlex.quote(target)}
if [ ! -d "$parent_dir" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if ! hermes_fs_safe_parent "$parent_dir"; then printf '__HERMES_FS_ERROR__:EINVAL\\n'; exit 45; fi
""" + ("""
if mkdir -- "$target" 2>/dev/null; then exit 0; fi
if [ -e "$target" ] || [ -L "$target" ]; then printf '__HERMES_FS_ERROR__:EEXIST\\n'; else printf '__HERMES_FS_ERROR__:EACCES\\n'; fi
exit 46
""" if directory else f"""
if python3 -c {shlex.quote(worker)} "$target" 2>/dev/null; then exit 0; fi
if [ -e "$target" ] || [ -L "$target" ]; then printf '__HERMES_FS_ERROR__:EEXIST\\n'; else printf '__HERMES_FS_ERROR__:EACCES\\n'; fi
exit 46
""")
        result = self._execute(command)
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(result.get("output", "")))
        return target

    def _mutation_path(self, path: str) -> str:
        raw = str(path or "").strip()
        if not raw or raw.lower().startswith("file:") or "\0" in raw:
            raise SshWorkspaceFsError("EINVAL", "Invalid path")
        if any(part in (".", "..") for part in raw.split("/")):
            raise SshWorkspaceFsError("EINVAL", "Invalid path")
        return "/" + self._normalize(raw).lstrip("/")

    @staticmethod
    def _safe_parent_function() -> str:
        return '''
hermes_fs_safe_parent() {
  local p=$1
  while :; do
    [ ! -L "$p" ] || return 1
    [ "$p" = / ] && break
    p=${p%/*}; [ -n "$p" ] || p=/
  done
}
'''

    def rename(self, path: str, name: str) -> str:
        source = self._mutation_path(path)
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\0" in name:
            raise SshWorkspaceFsError("EINVAL", "Invalid name")
        destination = posixpath.join(posixpath.dirname(source), name)
        if source == destination:
            # Ensure a no-op isn't reported as success for a missing source.
            command = f"p={shlex.quote(source)}; [ -e \"$p\" ] || [ -L \"$p\" ] || {{ printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; }}"
        else:
            worker = "import ctypes, errno, os, sys; src, dst = sys.argv[1:3]; libc = ctypes.CDLL(None, use_errno=True); fn = getattr(libc, 'renameat2', None); rc = fn(-100, ctypes.c_char_p(os.fsencode(src)), -100, ctypes.c_char_p(os.fsencode(dst)), 1) if fn else -1; e = ctypes.get_errno() if rc else 0; print('__HERMES_FS_ERROR__:' + ('EUNSUPPORTED' if not fn else 'EEXIST' if e == 17 else 'ENOENT' if e == 2 else 'EACCES' if e in (errno.EPERM, errno.EACCES) else 'EUNSUPPORTED' if e in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP) else 'EIO') if rc else ''); sys.exit(1 if rc else 0)"
            command = self._safe_parent_function() + f"""
src={shlex.quote(source)}
dst={shlex.quote(destination)}
parent={shlex.quote(posixpath.dirname(source) or '/')}
if [ ! -e "$src" ] && [ ! -L "$src" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if ! hermes_fs_safe_parent "$parent"; then printf '__HERMES_FS_ERROR__:EINVAL\\n'; exit 45; fi
python3 -c {shlex.quote(worker)} "$src" "$dst" || {{
  rc=$?
  [ "$rc" -eq 0 ] || {{ [ -n "$(command -v python3 2>/dev/null)" ] || printf '__HERMES_FS_ERROR__:EUNSUPPORTED\\n'; exit "$rc"; }}
}}
"""
        result = self._execute(self._mutation_guard(source, destination, scan=True) + command)
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(result.get("output", "")))
        return destination

    def delete(self, path: str, *, recursive: bool) -> str:
        target = self._mutation_path(path)
        if target == "/" or target == self.cwd or self.cwd.startswith(target.rstrip("/") + "/"):
            raise SshWorkspaceFsError("EINVAL", "Cannot delete a workspace root or its ancestor")
        parent = posixpath.dirname(target) or "/"
        command = self._mutation_guard(target, scan=True) + self._safe_parent_function() + f"""
target={shlex.quote(target)}
parent={shlex.quote(parent)}
if [ ! -e "$target" ] && [ ! -L "$target" ]; then printf '__HERMES_FS_ERROR__:ENOENT\\n'; exit 44; fi
if ! hermes_fs_safe_parent "$parent"; then printf '__HERMES_FS_ERROR__:EINVAL\\n'; exit 45; fi
if [ -L "$target" ] || [ ! -d "$target" ]; then
  rm -- "$target" || {{ printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }}
  exit 0
fi
""" + ("""
rm -rf -- "$target" || { printf '__HERMES_FS_ERROR__:EACCES\\n'; exit 46; }
""" if recursive else """
rmdir -- "$target" 2>/dev/null || { printf '__HERMES_FS_ERROR__:ENOTEMPTY\\n'; exit 47; }
""")
        result = self._execute(command)
        if result.get("returncode") != 0:
            raise SshWorkspaceFsError(self._error_code(result.get("output", "")))
        return target

    def write_text(self, path: str, content: str, *, max_bytes: int) -> tuple[str, int]:
        target = self._mutation_path(path)
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > max_bytes:
            raise SshWorkspaceFsError("EFBIG")
        # os.replace acts on the final entry even for a link to a directory; plain mv can
        # instead move the temporary file INTO that directory. Never write through the link.
        # Atomic replacement preserves regular-file mode, not ownership or ACLs. New files
        # and replaced links use 0666 masked by the remote process umask.
        worker = r'''
import os, stat, sys, tempfile
path = sys.argv[1]
tmp = None
try:
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing is not None and not (stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode)):
        print('__HERMES_FS_ERROR__:ENOTREG')
        sys.exit(1)
    if existing is not None and stat.S_ISREG(existing.st_mode):
        mode = stat.S_IMODE(existing.st_mode)
    else:
        mask = os.umask(0)
        os.umask(mask)
        mode = 0o666 & ~mask
    fd, tmp = tempfile.mkstemp(prefix='.hermes-tmp-', dir=os.path.dirname(path))
    with os.fdopen(fd, 'wb') as output:
        output.write(sys.stdin.buffer.read())
        output.flush()
        os.fchmod(output.fileno(), mode)
    os.replace(tmp, path)
except OSError as exc:
    print('__HERMES_FS_ERROR__:' + ('ENOENT' if isinstance(exc, FileNotFoundError) else 'EACCES'))
    sys.exit(1)
finally:
    if tmp is not None and os.path.lexists(tmp):
        os.unlink(tmp)
'''
        command = self._mutation_guard(target) + f"python3 -c {shlex.quote(worker)} {shlex.quote(target)}"
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

    with _lock_for(_DIAL_LOCKS, profile_key):
        with _CACHE_LOCK:
            cached = _BACKENDS.get(profile_key)
            if cached and cached[0] == config:
                return cached[1]
            _BACKENDS.pop(profile_key, None)
        # Close the replaced connection before dialing: the control socket is keyed on
        # user@host:port only, so a dial with new credentials would otherwise ride the old master.
        if cached:
            with _lock_for(_ENDPOINT_LOCKS, _endpoint(cached[0])):
                cached[1]._env.cleanup()
        try:
            with _lock_for(_ENDPOINT_LOCKS, _endpoint(config)):
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
        with _CACHE_LOCK:
            _BACKENDS[profile_key] = (config, backend)
        return backend
