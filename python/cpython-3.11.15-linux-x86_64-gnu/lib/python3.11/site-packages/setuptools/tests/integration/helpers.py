"""Reusable functions and classes for different types of integration tests.

For example ``Archive`` can be used to check the contents of distribution built
with setuptools, and ``run`` will always try to be as verbose as possible to
facilitate debugging.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile, ZipInfo


def run(cmd, env=None):
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **(env or {})},
        # ^-- allow overwriting instead of discarding the current env
    )

    out = r.stdout + "\n" + r.stderr
    # pytest omits stdout/err by default, if the test fails they help debugging
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Command: {cmd}\nreturn code: {r.returncode}\n\n{out}")

    if r.returncode == 0:
        return out
    raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)


class Archive:
    """Compatibility layer for ZipFile/Info and TarFile/Info"""

    def __init__(self, filename) -> None:
        self._filename = filename
        if filename.endswith("tar.gz"):
            self._obj: tarfile.TarFile | ZipFile = tarfile.open(filename, "r:gz")
        elif filename.endswith("zip"):
            self._obj = ZipFile(filename)
        else:
            raise ValueError(f"{filename} doesn't seem to be a zip or tar.gz")

    def __iter__(self) -> Iterator[ZipInfo] | Iterator[tarfile.TarInfo]:
        if hasattr(self._obj, "infolist"):
            return iter(self._obj.infolist())
        return iter(self._obj)

    def get_name(self, zip_or_tar_info):
        if hasattr(zip_or_tar_info, "filename"):
            return zip_or_tar_info.filename
        return zip_or_tar_info.name

    def get_content(self, zip_or_tar_info):
        if hasattr(self._obj, "extractfile"):
            content = self._obj.extractfile(zip_or_tar_info)
            if content is None:
                msg = f"Invalid {zip_or_tar_info.name} in {self._filename}"
                raise ValueError(msg)
            return str(content.read(), "utf-8")
        return str(self._obj.read(zip_or_tar_info), "utf-8")


def get_sdist_members(sdist_path):
    with tarfile.open(sdist_path, "r:gz") as tar:
        files = [Path(f) for f in tar.getnames()]
    # remove root folder
    relative_files = ("/".join(f.parts[1:]) for f in files)
    return {f for f in relative_files if f}


def get_wheel_members(wheel_path):
    with ZipFile(wheel_path) as zipfile:
        return set(zipfile.namelist())
