from __future__ import annotations

from collections import defaultdict
import importlib.util
import os
from pathlib import Path
import tempfile

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.sdist import sdist as _sdist


REPO_ROOT = Path(__file__).parent.resolve()
_BUILD_INFO_SPEC = importlib.util.spec_from_file_location(
    "_hermes_build_info",
    REPO_ROOT / "hermes_cli" / "build_info.py",
)
if _BUILD_INFO_SPEC is None or _BUILD_INFO_SPEC.loader is None:
    raise RuntimeError("Unable to load Hermes build metadata helpers")
_BUILD_INFO = importlib.util.module_from_spec(_BUILD_INFO_SPEC)
_BUILD_INFO_SPEC.loader.exec_module(_BUILD_INFO)


def _source_tree_is_writable() -> bool:
    probe = REPO_ROOT / ".setuptools-write-probe"
    try:
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("")
        probe.unlink()
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def _temporary_build_dir(kind: str) -> str:
    return tempfile.mkdtemp(prefix=f"hermes-agent-{kind}-")


def _would_write_under_source(path_value: str | None) -> bool:
    if path_value is None:
        return True
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


class ReadOnlySourceBuild(_build):
    def finalize_options(self) -> None:
        if (
            not _source_tree_is_writable()
            and _would_write_under_source(self.build_base)
        ):
            self.build_base = _temporary_build_dir("build")
        super().finalize_options()


class ReadOnlySourceEggInfo(_egg_info):
    def finalize_options(self) -> None:
        if (
            not _source_tree_is_writable()
            and _would_write_under_source(self.egg_base)
        ):
            self.egg_base = _temporary_build_dir("egg-info")
        super().finalize_options()


def _artifact_source_revision() -> str | None:
    return _BUILD_INFO.resolve_build_source_revision(
        REPO_ROOT,
        os.environ.get("HERMES_GIT_SHA"),
    )


class BuildPythonWithMetadata(_build_py):
    def run(self) -> None:
        source_revision = _artifact_source_revision()
        super().run()
        _BUILD_INFO.write_build_metadata(self.build_lib, source_revision)


class SourceDistributionWithMetadata(_sdist):
    def make_release_tree(self, base_dir: str, files: list[str]) -> None:
        source_revision = _artifact_source_revision()
        super().make_release_tree(base_dir, files)
        _BUILD_INFO.write_build_metadata(base_dir, source_revision)


def _data_file_tree(root_name: str) -> list[tuple[str, list[str]]]:
    root = REPO_ROOT / root_name
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(REPO_ROOT)
        grouped[str(rel_path.parent)].append(str(rel_path))
    return sorted(grouped.items())


setup(
    cmdclass={
        "build": ReadOnlySourceBuild,
        "build_py": BuildPythonWithMetadata,
        "egg_info": ReadOnlySourceEggInfo,
        "sdist": SourceDistributionWithMetadata,
    },
    data_files=[
        *_data_file_tree("skills"),
        *_data_file_tree("optional-skills"),
    ]
)
