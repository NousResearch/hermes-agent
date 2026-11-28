"""Usable-git resolution that refuses macOS's uninstalled ``/usr/bin/git`` CLT stub.

On macOS without Xcode Command Line Tools, ``/usr/bin/git`` exists and is executable, but
spawning it pops Apple's modal "The git command requires the command line developer tools"
dialog. Custom/Homebrew/Nix/PM git binaries live elsewhere and remain usable without CLT.

Deciding that runs ``/usr/bin/xcode-select -p``, the one spawn ``hermes_platform/AGENTS.md``
allows a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sys
import threading
import time
from typing import Callable, Literal, Mapping

from hermes_platform.resolver.core import ABSENT, LookupContext, locate_command

GitStatus = Literal["usable", "missing", "clt_stub"]

APPLE_GIT_STUB = "/usr/bin/git"
_XCODE_SELECT_PATH = "/usr/bin/xcode-select"
_XCODE_SELECT_TIMEOUT_S = 2.0
_NEGATIVE_CACHE_TTL_S = 5.0


@dataclass(frozen=True)
class GitResolution:
    executable: str | None
    status: GitStatus
    raw_path: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "usable" and self.executable is not None


class _CltGitCache:
    """Thread-safe cache for macOS ``xcode-select -p`` usability checks: permanent for
    positives, short-TTL for negatives so installing CLT while running heals promptly."""

    def __init__(self, *, negative_ttl_s: float = _NEGATIVE_CACHE_TTL_S) -> None:
        self._lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._negative_ttl_s = negative_ttl_s
        self._usable: bool | None = None
        self._expires_at: float | None = None

    def clear(self) -> None:
        with self._lock:
            self._usable = None
            self._expires_at = None

    def check(
        self,
        probe: Callable[[], bool],
        *,
        cached: bool = True,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> bool:
        if cached:
            hit = self._read_cached(now_fn())
            if hit is not None:
                return hit
        with self._probe_lock:
            if cached:
                hit = self._read_cached(now_fn())
                if hit is not None:
                    return hit
            usable = bool(probe())
            with self._lock:
                self._usable = usable
                self._expires_at = None if usable else now_fn() + self._negative_ttl_s
            return usable

    def _read_cached(self, now: float) -> bool | None:
        with self._lock:
            if self._usable is None:
                return None
            if self._expires_at is None or now < self._expires_at:
                return self._usable
            self._usable = None
            self._expires_at = None
            return None


_CLT_CACHE = _CltGitCache()


def clear_git_usability_cache() -> None:
    """Clear the process-wide macOS CLT git usability cache."""
    _CLT_CACHE.clear()


def _env_lookup(env: Mapping[str, str], name: str, *, case_insensitive: bool) -> tuple[bool, str | None]:
    if name in env:
        return True, env[name]
    if case_insensitive:
        upper = name.upper()
        for key, value in env.items():
            if key.upper() == upper:
                return True, value
    return False, None


def _lookup_context_for_env(env: Mapping[str, str] | None) -> LookupContext:
    if env is None:
        return LookupContext()
    case_insensitive = os.name == "nt"
    has_path, path_val = _env_lookup(env, "PATH", case_insensitive=case_insensitive)
    _, pathext_val = _env_lookup(env, "PATHEXT", case_insensitive=case_insensitive)
    if has_path:
        return LookupContext(path="" if path_val is None else path_val, pathext=pathext_val)
    # Match subprocess semantics: Windows CreateProcess searches ambient PATH; POSIX execvp falls back to os.defpath.
    return LookupContext(path=ABSENT if os.name == "nt" else os.defpath, pathext=pathext_val)


def _locate_git(env: Mapping[str, str] | None, cwd: str | os.PathLike[str] | None) -> str | None:
    ctx = _lookup_context_for_env(env)
    if cwd is None or os.name == "nt":
        res = locate_command("git", ctx)
        return res.command[0] if res.found else None
    path = ctx.effective_path()
    path = os.environ.get("PATH", os.defpath) if path is None else path
    if not path:
        return None
    # POSIX subprocess tries each PATH entry after chdir(cwd), so a relative entry names a directory under
    # cwd (an absolute one replaces it in the join). Test each entry on its own: a PATH string rebuilt from
    # cwd-joined entries would split a cwd that contains os.pathsep.
    for entry in path.split(os.pathsep):
        candidate = os.path.join(os.fspath(cwd), entry, "git")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _effective_developer_dir(env: Mapping[str, str] | None) -> str:
    source = os.environ if env is None else env
    return (source.get("DEVELOPER_DIR") or "").strip()


def _has_developer_git(dev_dir: str) -> bool:
    if not dev_dir:
        return False
    # A leftover empty developer dir after a macOS upgrade is not usable (mirrors scripts/install.sh).
    # xcode-select(1): DEVELOPER_DIR may also name the Xcode app, whose shims use Contents/Developer.
    for root in (dev_dir, os.path.join(dev_dir, "Contents", "Developer")):
        candidate = os.path.join(root, "usr", "bin", "git")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True
    return False


def _probe_xcode_select_developer_dir(
    env: Mapping[str, str] | None = None,
    *,
    xcode_select: str = _XCODE_SELECT_PATH,
    timeout: float = _XCODE_SELECT_TIMEOUT_S,
) -> str:
    try:
        proc = subprocess.run(
            [xcode_select, "-p"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _resolve_argv0_for_cwd(raw: str, cwd: str | os.PathLike[str] | None) -> str:
    if cwd is None or os.name == "nt" or os.path.isabs(raw):
        return raw
    return os.path.join(os.fspath(cwd), raw)


def is_git_command(
    argv0: str | os.PathLike[str] | None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> bool:
    """True when *argv0* names ``git`` (bare or path-qualified, or a macOS symlink to ``/usr/bin/git``)."""
    if not argv0:
        return False
    raw = os.fspath(argv0).strip()
    if not raw:
        return False
    base = os.path.basename(raw)
    if os.name == "nt":
        return base.lower() in {"git", "git.exe"}
    if base == "git":
        return True
    if sys.platform == "darwin" and (os.sep in raw or (os.altsep and os.altsep in raw)):
        return os.path.realpath(_resolve_argv0_for_cwd(raw, cwd)) == os.path.realpath(APPLE_GIT_STUB)
    return False


def is_usable_git_executable(
    executable: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None = None,
    *,
    cached: bool = True,
    platform: str | None = None,
    apple_stub_path: str = APPLE_GIT_STUB,
    developer_dir_probe: Callable[[Mapping[str, str] | None], str] | None = None,
    cache: _CltGitCache | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> bool:
    """Return True unless *executable* is macOS's uninstalled ``/usr/bin/git`` CLT stub."""
    if not executable:
        return False
    eff_platform = sys.platform if platform is None else platform
    if eff_platform != "darwin":
        return True
    if os.path.realpath(os.fspath(executable)) != os.path.realpath(apple_stub_path):
        return True
    dev_dir = _effective_developer_dir(env)
    if dev_dir:
        return _has_developer_git(dev_dir)
    probe = developer_dir_probe or _probe_xcode_select_developer_dir
    active_cache = _CLT_CACHE if cache is None else cache
    return active_cache.check(lambda: _has_developer_git(probe(env)), cached=cached, now_fn=now_fn)


def inspect_git(
    env: Mapping[str, str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
    cached: bool = True,
    platform: str | None = None,
    apple_stub_path: str = APPLE_GIT_STUB,
    developer_dir_probe: Callable[[Mapping[str, str] | None], str] | None = None,
    cache: _CltGitCache | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> GitResolution:
    """Resolve ``git`` from *env*'s effective PATH and report whether it is usable or a macOS CLT stub."""
    hit = _locate_git(env, cwd)
    if hit is None:
        return GitResolution(None, "missing", None)
    if is_usable_git_executable(
        hit,
        env,
        cached=cached,
        platform=platform,
        apple_stub_path=apple_stub_path,
        developer_dir_probe=developer_dir_probe,
        cache=cache,
        now_fn=now_fn,
    ):
        return GitResolution(hit, "usable", hit)
    return GitResolution(None, "clt_stub", hit)


def resolve_usable_git(
    env: Mapping[str, str] | None = None,
    *,
    cached: bool = True,
    platform: str | None = None,
    apple_stub_path: str = APPLE_GIT_STUB,
    developer_dir_probe: Callable[[Mapping[str, str] | None], str] | None = None,
    cache: _CltGitCache | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> str | None:
    """Resolved path to a usable ``git`` on *env*'s effective PATH, or ``None`` if missing/CLT stub."""
    return inspect_git(
        env,
        cached=cached,
        platform=platform,
        apple_stub_path=apple_stub_path,
        developer_dir_probe=developer_dir_probe,
        cache=cache,
        now_fn=now_fn,
    ).executable


def is_usable_git_command(
    argv0: str | os.PathLike[str] | None,
    env: Mapping[str, str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
    cached: bool = True,
) -> bool:
    """True when *argv0* (bare ``git`` or an explicit executable path) can run without hitting the macOS CLT stub.

    Only the stub is refused. A bare git this lookup cannot find is left to the launch to report:
    CreateProcess searches the parent's PATH and execvp follows a relative entry from the child's
    cwd, so "not found from here" is not proof the launch fails."""
    if not argv0:
        return False
    raw = os.fspath(argv0).strip()
    if not raw:
        return False
    if os.sep in raw or (os.altsep and os.altsep in raw):
        return is_usable_git_executable(_resolve_argv0_for_cwd(raw, cwd), env, cached=cached)
    return inspect_git(env, cwd=cwd, cached=cached).status != "clt_stub"
