"""scripts/ci/catalog_core_imports.py: which pinned catalog plugins a core change breaks.

Real git repositories in tmp_path stand in for a catalog plugin and for two core commits; the index row
comes from the same fetch-and-parse path the scheduled build runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ci"))

import catalog_core_imports as cci


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8",
                          env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                               "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(cwd),
                               "PATH": "/usr/bin:/bin"}).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None]) -> str:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


class _Entry:
    def __init__(self, repo: Path, sha: str) -> None:
        self.name, self.repo, self.sha, self.subdir = "demo", str(repo), sha, ""


# The plugin imports one core name plainly, one through a try whose fallback is the OLD name, and one
# public name; hermes_state_wal is a real Hermes top-level, so the parser counts it as core.
PLUGIN = """\
from agent.memory_provider import MemoryProvider
from tools.registry import registry_helper
try:
    from sqlite_support import apply_wal_with_fallback
except ImportError:
    from hermes_state_wal import apply_wal_with_fallback
"""


@pytest.fixture
def index(tmp_path: Path) -> dict:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    _git(plugin, "init", "-q")
    sha = _commit(plugin, {"__init__.py": PLUGIN, "plugin.yaml": "name: demo\n"})
    work = tmp_path / "work"
    work.mkdir()
    return {"demo": cci.index_entry(_Entry(plugin, sha), work)}


@pytest.fixture
def core(tmp_path: Path):
    repo = tmp_path / "core"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, {
        "agent/memory_provider.py": "class MemoryProvider: ...\n",
        "tools/registry.py": "def registry_helper(): ...\n",
        "hermes_state_wal.py": "def apply_wal_with_fallback(conn): ...\n",
        "sqlite_support.py": "def apply_wal_with_fallback(conn): ...\n",
        "hermes_agent/compat/moves.py": "MOVES: dict[str, str] = {}\n",
    })
    return repo, base


def _check(repo: Path, index: dict, files: dict) -> list[tuple[str, str]]:
    """The breaks a commit of ``files`` on top of the current HEAD causes."""
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, files)
    return [(b.site, b.target) for b in cci.breaks(index, cci.Tree(repo, base), cci.Tree(repo, head))]


def test_removing_a_name_a_pinned_plugin_imports_is_named_and_a_kept_spelling_is_not(index, core):
    repo, base = core
    removed = _check(repo, index, {"tools/registry.py": "def other(): ...\n"})
    assert removed == [("__init__.py:2", "tools.registry.registry_helper")]
    assert "Open a PR for an official seam" in cci.report(
        cci.breaks(index, cci.Tree(repo, base), cci.Tree(repo, "HEAD")))
    _commit(repo, {"tools/registry.py": "def registry_helper(): ...\n"})

    # The removed module only ever stood in for sqlite_support behind the plugin's own try: nothing breaks.
    assert _check(repo, index, {"hermes_state_wal.py": None}) == []
    # A compat move keeps a moved module importable under its old name.
    assert _check(repo, index, {
        "tools/registry.py": None, "tools/registry_core.py": "def registry_helper(): ...\n",
        "hermes_agent/compat/moves.py": 'MOVES: dict[str, str] = {"tools.registry": "tools.registry_core"}\n'}) == []
    # Removing the module the try reaches first, with its fallback already gone, breaks it: both spellings
    # are named, since either one coming back would fix the plugin.
    assert _check(repo, index, {"sqlite_support.py": None}) == [
        ("__init__.py:4", "sqlite_support.apply_wal_with_fallback"),
        ("__init__.py:6", "hermes_state_wal.apply_wal_with_fallback")]


def test_the_label_lets_an_intended_break_through(index, core, monkeypatch, capsys):
    repo, base = core
    head = _commit(repo, {"agent/memory_provider.py": "class Other: ...\n"})
    monkeypatch.setattr(cci, "ROOT", repo)
    monkeypatch.setattr(cci, "read_index", lambda: index)
    assert cci.check(base, head, []) == 1
    assert "demo __init__.py:1 imports agent.memory_provider.MemoryProvider" in capsys.readouterr().out
    assert cci.check(base, head, [cci.OVERRIDE_LABEL]) == 0
