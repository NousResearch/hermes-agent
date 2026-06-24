from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.egg_info import egg_info as _egg_info


REPO_ROOT = Path(__file__).parent.resolve()


# Directory names that never carry shippable bundled-data content: VCS,
# virtualenv / dependency trees, byte-code and test caches, and the skills
# ``index-cache`` (a runtime download cache, also filtered by
# nix/hermes-agent.nix). Mirrors agent.skill_utils.EXCLUDED_SKILL_DIRS — kept
# as a literal so setup.py stays import-free at build time — plus
# ``index-cache``. NOTE: skill *support* dirs (references/templates/assets/
# scripts) are deliberately NOT excluded; they are real skill payload that
# tools/skills_sync.py copies into ~/.hermes/skills/ on seed.
_EXCLUDED_DATA_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "index-cache",
    }
)
_EXCLUDED_DATA_SUFFIXES = frozenset({".pyc", ".pyo"})
_EXCLUDED_DATA_NAMES = frozenset({".DS_Store", "Thumbs.db"})


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


def _is_shippable(rel_path: Path) -> bool:
    """True when *rel_path* (relative to REPO_ROOT) is real bundled-data content."""
    if _EXCLUDED_DATA_DIRS.intersection(rel_path.parts):
        return False
    if rel_path.suffix.lower() in _EXCLUDED_DATA_SUFFIXES:
        return False
    if rel_path.name in _EXCLUDED_DATA_NAMES:
        return False
    return True


def _data_file_tree(root_name: str) -> list[tuple[str, list[str]]]:
    """Map every file under *root_name* to a setuptools ``data_files`` entry.

    Returns ``(target_dir, [source_files])`` tuples with the on-disk directory
    layout preserved one-directory-per-target. setuptools ``data_files``
    FLATTENS every glob match into its single target dir, so a nested tree must
    be enumerated per directory or the structure collapses — skills are seeded
    by category/skill path (tools/skills_sync.py), so a flattened
    ``skills/*`` glob would corrupt the install. Paths are emitted as POSIX
    (forward slash) so the wheel is correct even when built on Windows, where
    ``str(Path(...))`` would otherwise bake in backslash targets.
    """
    root = REPO_ROOT / root_name
    if not root.is_dir():
        return []
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(REPO_ROOT)
        if not _is_shippable(rel_path):
            continue
        grouped[rel_path.parent.as_posix()].append(rel_path.as_posix())
    return sorted(grouped.items())


def bundled_data_files() -> list[tuple[str, list[str]]]:
    """All bundled non-package data shipped in the wheel's data scheme.

    setuptools resolves ``data_files`` from EITHER pyproject.toml's
    ``[tool.setuptools.data-files]`` table OR setup.py — never a merge of both.
    Whichever the pyproject table declares wins and silently drops setup.py's
    list (this clobbered the skills shipping for two releases). The skills and
    optional-skills trees are too deep and change too often to enumerate
    statically in TOML, so ALL bundled-data dirs are generated here in one
    place. Do NOT reintroduce ``[tool.setuptools.data-files]`` in pyproject.toml
    or it will shadow this and the wheel will ship zero skills again.
    """
    return [
        *_data_file_tree("skills"),
        *_data_file_tree("optional-skills"),
        *_data_file_tree("locales"),
        *_data_file_tree("optional-mcps"),
    ]


setup(
    cmdclass={
        "build": ReadOnlySourceBuild,
        "egg_info": ReadOnlySourceEggInfo,
    },
    data_files=bundled_data_files(),
)
