"""Guard Python filesystem calls in tests, not arbitrary native/subprocess I/O."""
from __future__ import annotations

import builtins
from functools import lru_cache, wraps
import io
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import threading

_INTERPRETER_PREFIXES = tuple({
    Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
} | {
    # A PM-activated developer shell runs sys.prefix's python against a dependency generation
    # whose site-packages sits under the (real) Hermes home; third-party imports from it are the
    # interpreter's installation, not Hermes state.
    Path(p).resolve() for p in sys.path if p and Path(p).name in ("site-packages", "dist-packages")
} | {
    # The default install checks the repo out INSIDE the home (install.sh:
    # INSTALL_DIR=$HERMES_HOME/hermes-agent). Reading test data, sources for tracebacks, or the
    # checkout's own .venv is not Hermes state; without this every run from a default install
    # trips on its first traceback.
    Path(__file__).resolve().parent.parent,
})
# The same prefixes as plain strings for the check() fast path. PurePath comparison folds case on
# Windows; ``os.path.normcase`` (identity on POSIX) reproduces that for string compares. Prefixes
# resolve once at import, as before: they are fixed for the process lifetime.
_normcase = os.path.normcase
_INTERPRETER_PREFIX_STRS = tuple(_normcase(os.fspath(p)) for p in _INTERPRETER_PREFIXES)

# Install-layout markers beside a tree: a sealed payload ships ``manifest.json`` (repo/venv/store)
# in the directory ABOVE the tree (scripts/build/agent.py), an install records ``install-stamp.json``
# beside it, and PM's payload_venv()/store_root() probe both to find the environment and store for
# the tree they serve. The default install checks the repo out INSIDE the home, so those probes land
# in a guarded root even though they ask about the installation, not user state. Reads are allowed
# there — the manifest only counts when it names this checkout as its repo — and writes stay refused.
_LAYOUT_MARKERS = frozenset({"manifest.json", "install-stamp.json"})
_LAYOUT_DIRS = frozenset(
    _normcase(os.fspath(directory))
    for directory in (Path(__file__).resolve().parent.parent, *Path(__file__).resolve().parent.parent.parents)
)

# A launch with a temporary HERMES_HOME still BORROWS the owning root's dependency store
# (pm/environments.py: owning_home_root) instead of reinstalling: ``installs/<key>/facts.json``,
# the PM tool store, the managed node runtime and the package caches are what that borrow reads.
# Probing them is the product's design, not a test reaching into user state, and a checkout under
# a home (install.sh) makes the probes unavoidable there. Reads are allowed; writes, deletions and
# reads of user state (config, auth, memories, sessions, skills, cron, cache) stay refused.
_INSTALL_LAYOUT_NAMES = frozenset({
    "installs", "tools", "node", "node_modules", "uv-cache", "npm-cache", "plugins",
    "bin", "bootstrap", "pm-runtime",
})


def _within_install_layout(path: str, roots) -> bool:
    """*path* (normalized, absolute) is inside a guarded root's runtime install layout."""
    for root in roots:
        if not _within(path, root):
            continue
        rest = path[len(root):].lstrip(os.sep)
        if rest and _normcase(rest.split(os.sep, 1)[0]) in _INSTALL_LAYOUT_NAMES:
            return True
    return False


def _is_pkgutil_probe(path: str, roots) -> bool:
    """``pkgutil.extend_path`` stats ``<sys.path entry>/<name>.pkg`` for every namespace package
    it imports. Plugin discovery puts a real Hermes home on sys.path, so importing e.g. ``discord``
    probes ``<home>/discord.pkg``: import machinery, not Hermes state."""
    return path.endswith(".pkg") and os.path.dirname(path) in roots


def _within(path: str, prefix: str) -> bool:
    """``Path(path).is_relative_to(prefix)`` for two normalized, case-folded absolute strings."""
    if path == prefix:
        return True
    return path.startswith(prefix) if prefix == os.sep else path.startswith(prefix + os.sep)


def _contains(path: str, prefix: str) -> bool:
    """``Path(prefix).is_relative_to(path)``: *path* is *prefix* or one of its ancestors."""
    if path == prefix:
        return True
    return prefix.startswith(path) if path == os.sep else prefix.startswith(path + os.sep)


class HomeIOGuard:
    def __init__(self, roots, installed_apps=lambda: ()):
        self.roots = roots
        # The machine's installed desktop app (``packaged_gui_app_paths()`` before any test ran):
        # deleting, replacing or writing into it is refused. Reads stay allowed: several update
        # paths only probe it.
        self.installed_apps = installed_apps
        self.checking = threading.local()
        self.directories: dict[int, Path] = {}

    def check(self, value, *, dir_fd=None, metadata=False, destructive=False):
        if value is None or isinstance(value, int) or getattr(self.checking, "active", False):
            return
        self.checking.active = True
        try:
            candidate = os.fsdecode(value)
            if candidate.startswith("~"):
                # A test may have patched Path.expanduser to fail; the guard must not
                # turn that into its own crash — the unexpanded path is checked instead.
                try:
                    candidate = os.fspath(Path(candidate).expanduser())
                except Exception:
                    pass
            if dir_fd is not None and not os.path.isabs(candidate):
                parent = self.directories.get(dir_fd)
                if parent is None:
                    raise AssertionError("TEST BUG: untracked dir_fd in guarded filesystem I/O")
                candidate = os.path.join(os.fspath(parent), candidate)
            absolute = _normcase(os.path.abspath(candidate))
            # /proc/<pid>/fd/N is descriptor inspection (deleted-WAL holder scans stat the magic
            # link to compare inode identity); resolving it names whatever file that fd holds,
            # which is not I/O against the home.
            if metadata and (absolute == "/proc" or absolute.startswith("/proc" + os.sep)):
                return
            resolved = self._refuse_installed_app_change(value, absolute) if destructive else None
            roots = tuple(_normcase(os.fspath(r)) for r in self.roots())
            # Resolving the root itself (get_default_hermes_root's relative_to
            # probe) reads no state; only its contents are guarded.
            if metadata and absolute in roots:
                return
            if not destructive and os.path.dirname(absolute) in _LAYOUT_DIRS \
                    and os.path.basename(absolute) in _LAYOUT_MARKERS:
                return
            # The owning root's runtime install layout (borrowing launch) and the ``.pkg`` probes
            # namespace imports make beside it: installation state, not user state.
            if not destructive and _within_install_layout(absolute, roots):
                return
            if not destructive and metadata and _is_pkgutil_probe(absolute, roots):
                return
            # ``shutil.which`` stats/accesses ``<PATH entry>/<name>``. A developer shell puts
            # PM's tool store (~/.hermes/tools/...) on PATH; probing an executable there is
            # command lookup, not reading Hermes state. CI has no such entries.
            if metadata:
                path = os.environ.get("PATH", "")
                cwd = os.getcwd() if self._relative_path_entries(path) else None
                if os.path.dirname(absolute) in self._path_entries(path, cwd):
                    return
            # The interpreter's own installation (a PM-managed python under ~/.hermes/tools):
            # stdlib source reads (linecache, traceback) are not Hermes state either, nor is
            # realpath() walking up through its ancestors.
            for prefix in _INTERPRETER_PREFIX_STRS:
                if _within(absolute, prefix) or (metadata and _contains(absolute, prefix)):
                    return
            # Check the lexical path first: resolving must not probe a protected
            # tree merely to decide that the original path was forbidden.
            for root in roots:
                if _within(absolute, root):
                    self.refuse(value)
            if resolved is None:
                resolved = _normcase(os.path.realpath(absolute))
            if metadata and resolved in roots:
                return
            # A fixture symlink to the running interpreter resolves into its installation.
            for prefix in _INTERPRETER_PREFIX_STRS:
                if _within(resolved, prefix):
                    return
            # A fixture symlink to the managed node runtime (~/.local/bin/node, <tmp>/store/node/
            # bin/node) resolves into the owning root's install layout — same borrowing-launch
            # reads as above, so they are exempt lexically and by resolution.
            if not destructive and _within_install_layout(resolved, roots):
                return
            if not destructive and metadata and _is_pkgutil_probe(resolved, roots):
                return
            for root in roots:
                if _within(resolved, root):
                    self.refuse(value)
        finally:
            self.checking.active = False

    def _refuse_installed_app_change(self, value, absolute):
        """Fail on a change to an installed app (already normalized, literal and resolved);
        returns the resolved path so check() does not resolve it again, or None."""
        apps = self.installed_apps()
        if not apps:
            return None
        resolved = _normcase(os.path.realpath(absolute))
        for app in apps:
            if _within(absolute, app) or _within(resolved, app):
                import pytest
                # pytest.fail, not AssertionError: removal helpers catch Exception and would log
                # the refusal as a warning while the test passed.
                pytest.fail(f"TEST BUG: changing the REAL installed Hermes desktop app: {value}\n"
                            "Stub hermes_cli.gui_uninstall.packaged_gui_app_paths to a temporary path.")
        return resolved

    @staticmethod
    def refuse(value):
        raise AssertionError(
            f"TEST BUG: file I/O against the REAL hermes home: {value}\n"
            "Use the isolated HERMES_HOME or a temporary fixture instead."
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def _relative_path_entries(path: str) -> bool:
        return any(entry and not os.path.isabs(entry) for entry in path.split(os.pathsep))

    @staticmethod
    @lru_cache(maxsize=8)
    def _path_entries(path: str, cwd: str | None):
        # Relative PATH entries change meaning after chdir; absolute ones need no cwd.
        return frozenset(
            _normcase(os.path.normpath(os.path.join(cwd or "", entry)))
            for entry in path.split(os.pathsep) if entry
        )

    def install(self, monkeypatch):
        def wrap(module, name, parameters, *, metadata=False, destructive=False):
            original = getattr(module, name)

            @wraps(original)
            def guarded(*args, **kwargs):
                for index, (parameter, descriptor) in enumerate(parameters):
                    value = args[index] if index < len(args) else kwargs.get(parameter)
                    self.check(value, dir_fd=kwargs.get(descriptor) if descriptor else None, metadata=metadata,
                               destructive=destructive(args, kwargs) if callable(destructive) else destructive)
                return original(*args, **kwargs)

            monkeypatch.setattr(module, name, guarded)

        def open_writes(args, kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            return any(flag in mode for flag in "wax+")

        for module in (builtins, io):
            wrap(module, "open", (("file", None),), destructive=open_writes)
        for name in ("mkdir", "makedirs", "unlink", "remove", "rmdir", "chmod", "utime"):
            wrap(os, name, (("path" if name == "mkdir" else "name", "dir_fd"),), destructive=True)
        for name in ("stat", "lstat", "readlink", "access"):
            wrap(os, name, (("path", "dir_fd"),), metadata=True)
        for name in ("listdir", "scandir"):
            wrap(os, name, (("path", None),))
        for name in ("rename", "replace"):
            wrap(os, name, (("src", "src_dir_fd"), ("dst", "dst_dir_fd")), destructive=True)
        wrap(shutil, "rmtree", (("path", "dir_fd"),), destructive=True)
        wrap(sqlite3, "connect", (("database", None),))

        original_open, original_close = os.open, os.close

        @wraps(original_open)
        def guarded_open(path, flags, *args, **kwargs):
            self.check(path, dir_fd=kwargs.get("dir_fd"), destructive=bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            fd = original_open(path, flags, *args, **kwargs)
            candidate = Path(os.fsdecode(path))
            if kwargs.get("dir_fd") is not None and not candidate.is_absolute():
                candidate = self.directories[kwargs["dir_fd"]] / candidate
            self.directories[fd] = candidate.absolute()
            return fd

        @wraps(original_close)
        def guarded_close(fd):
            # Forget the old owner before close releases the number for reuse
            # by another thread's open; afterwards we could erase its mapping.
            self.directories.pop(fd, None)
            return original_close(fd)

        monkeypatch.setattr(os, "open", guarded_open)
        monkeypatch.setattr(os, "close", guarded_close)
