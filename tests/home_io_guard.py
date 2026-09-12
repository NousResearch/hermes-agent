"""Guard Python filesystem calls in tests, not arbitrary native/subprocess I/O."""
from __future__ import annotations

import builtins
from functools import wraps
import io
import os
from pathlib import Path
import shutil
import sqlite3
import threading


class HomeIOGuard:
    def __init__(self, roots):
        self.roots = roots
        self.checking = threading.local()
        self.directories: dict[int, Path] = {}

    def check(self, value, *, dir_fd=None):
        if value is None or isinstance(value, int) or getattr(self.checking, "active", False):
            return
        self.checking.active = True
        try:
            candidate = Path(os.fsdecode(value)).expanduser()
            if dir_fd is not None and not candidate.is_absolute():
                parent = self.directories.get(dir_fd)
                if parent is None:
                    raise AssertionError("TEST BUG: untracked dir_fd in guarded filesystem I/O")
                candidate = parent / candidate
            absolute = Path(os.path.abspath(candidate))
            roots = self.roots()
            # Check the lexical path first: resolving must not probe a protected
            # tree merely to decide that the original path was forbidden.
            if any(absolute.is_relative_to(root) for root in roots):
                self.refuse(value)
            resolved = absolute.resolve()
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

    def install(self, monkeypatch):
        def wrap(module, name, parameters):
            original = getattr(module, name)

            @wraps(original)
            def guarded(*args, **kwargs):
                for index, (parameter, descriptor) in enumerate(parameters):
                    value = args[index] if index < len(args) else kwargs.get(parameter)
                    self.check(value, dir_fd=kwargs.get(descriptor) if descriptor else None)
                return original(*args, **kwargs)

            monkeypatch.setattr(module, name, guarded)

        for module in (builtins, io):
            wrap(module, "open", (("file", None),))
        for name in ("mkdir", "stat", "lstat", "unlink", "remove", "rmdir", "chmod", "utime", "readlink", "access"):
            wrap(os, name, (("path", "dir_fd"),))
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
