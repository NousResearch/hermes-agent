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


class HomeIOGuard:
    def __init__(self, roots):
        self.roots = roots
        self.checking = threading.local()
        self.directories: dict[int, Path] = {}

    def check(self, value, *, dir_fd=None, metadata=False):
        if value is None or isinstance(value, int) or getattr(self.checking, "active", False):
            return
        self.checking.active = True
        try:
            candidate = Path(os.fsdecode(value))
            if candidate.parts and candidate.parts[0].startswith("~"):
                # A test may have patched Path.expanduser to fail; the guard must not
                # turn that into its own crash — the unexpanded path is checked instead.
                try:
                    candidate = candidate.expanduser()
                except Exception:
                    pass
            if dir_fd is not None and not candidate.is_absolute():
                parent = self.directories.get(dir_fd)
                if parent is None:
                    raise AssertionError("TEST BUG: untracked dir_fd in guarded filesystem I/O")
                candidate = parent / candidate
            absolute = Path(os.path.abspath(candidate))
            # /proc/<pid>/fd/N is descriptor inspection (deleted-WAL holder scans stat the magic
            # link to compare inode identity); resolving it names whatever file that fd holds,
            # which is not I/O against the home.
            if metadata and absolute.is_relative_to("/proc"):
                return
            roots = self.roots()
            # Resolving the root itself (get_default_hermes_root's relative_to
            # probe) reads no state; only its contents are guarded.
            if metadata and absolute in roots:
                return
            # ``shutil.which`` stats/accesses ``<PATH entry>/<name>``. A developer shell puts
            # PM's tool store (~/.hermes/tools/...) on PATH; probing an executable there is
            # command lookup, not reading Hermes state. CI has no such entries.
            if metadata:
                path = os.environ.get("PATH", "")
                cwd = os.getcwd() if self._relative_path_entries(path) else None
                if absolute.parent in self._path_entries(path, cwd):
                    return
            # The interpreter's own installation (a PM-managed python under ~/.hermes/tools):
            # stdlib source reads (linecache, traceback) are not Hermes state either, nor is
            # realpath() walking up through its ancestors.
            if any(absolute.is_relative_to(prefix) or (metadata and prefix.is_relative_to(absolute))
                   for prefix in _INTERPRETER_PREFIXES):
                return
            # Check the lexical path first: resolving must not probe a protected
            # tree merely to decide that the original path was forbidden.
            if any(absolute.is_relative_to(root) for root in roots):
                self.refuse(value)
            resolved = absolute.resolve()
            if metadata and resolved in roots:
                return
            # A fixture symlink to the running interpreter resolves into its installation.
            if any(resolved.is_relative_to(prefix) for prefix in _INTERPRETER_PREFIXES):
                return
            if any(resolved.is_relative_to(root) for root in roots):
                self.refuse(value)
        finally:
            self.checking.active = False

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
            Path(os.path.normpath(os.path.join(cwd or "", entry)))
            for entry in path.split(os.pathsep) if entry
        )

    def install(self, monkeypatch):
        def wrap(module, name, parameters, *, metadata=False):
            original = getattr(module, name)

            @wraps(original)
            def guarded(*args, **kwargs):
                for index, (parameter, descriptor) in enumerate(parameters):
                    value = args[index] if index < len(args) else kwargs.get(parameter)
                    self.check(value, dir_fd=kwargs.get(descriptor) if descriptor else None, metadata=metadata)
                return original(*args, **kwargs)

            monkeypatch.setattr(module, name, guarded)

        for module in (builtins, io):
            wrap(module, "open", (("file", None),))
        for name in ("mkdir", "unlink", "remove", "rmdir", "chmod", "utime"):
            wrap(os, name, (("path", "dir_fd"),))
        for name in ("stat", "lstat", "readlink", "access"):
            wrap(os, name, (("path", "dir_fd"),), metadata=True)
        for name in ("makedirs", "listdir", "scandir"):
            wrap(os, name, (("name" if name == "makedirs" else "path", None),))
        for name in ("rename", "replace"):
            wrap(os, name, (("src", "src_dir_fd"), ("dst", "dst_dir_fd")))
        wrap(shutil, "rmtree", (("path", "dir_fd"),))
        wrap(sqlite3, "connect", (("database", None),))

        original_open, original_close = os.open, os.close

        @wraps(original_open)
        def guarded_open(path, flags, *args, **kwargs):
            self.check(path, dir_fd=kwargs.get("dir_fd"))
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
