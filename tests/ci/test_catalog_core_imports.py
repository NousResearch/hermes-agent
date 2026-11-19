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


# One core name imported plainly, and one pair of names behind a try whose fallback is the OLD module; the
# public MemoryProvider import is a plain one too. hermes_state_wal is a real Hermes top-level, so the
# parser counts it as core.
PLUGIN = """\
from agent.memory_provider import MemoryProvider
from tools.registry import registry_helper
try:
    from sqlite_support import apply_wal_with_fallback, wal_mode
except ImportError:
    from hermes_state_wal import apply_wal_with_fallback, wal_mode
def data_dir():
    try:
        from plugins.plugin_storage import plugin_data_dir
    except Exception:
        try:
            from hermes_constants import get_hermes_home
        except Exception:
            get_hermes_home = None
"""


@pytest.fixture
def index(tmp_path: Path) -> dict:
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    _git(plugin, "init", "-q")
    sha = _commit(plugin, {"__init__.py": PLUGIN, "plugin.yaml": "name: demo\n"})
    work = tmp_path / "work"
    work.mkdir()
    # Through the on-disk format, as the PR check reads it.
    cci.write_index({"demo": cci.index_entry(_Entry(plugin, sha), work)}, tmp_path / "index")
    return cci.read_index(tmp_path / "index")


WAL = "def apply_wal_with_fallback(conn): ...\nwal_mode = 'wal'\n"


@pytest.fixture
def core(tmp_path: Path):
    repo = tmp_path / "core"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, {
        "agent/memory_provider.py": "class MemoryProvider: ...\n",
        "tools/registry.py": "def registry_helper(): ...\n",
        "hermes_state_wal.py": WAL,
        "sqlite_support.py": WAL,
        "hermes_agent/compat/moves.py": "MOVES: dict[str, str] = {}\n",
        "plugins/plugin_storage.py": "def plugin_data_dir(name): ...\n",
        "hermes_constants.py": "def get_hermes_home(): ...\n",
    })
    return repo, base


def _check(repo: Path, index: dict, files: dict) -> list[tuple[str, str]]:
    """The breaks a commit of ``files`` on top of the current HEAD causes."""
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, files)
    return [(f"{b.site.rel}:{b.site.line}", b.target)
            for b in cci.breaks(index, cci.Tree(repo, base), cci.Tree(repo, head))]


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
    # A try is judged branch by branch: with one name gone from BOTH spellings, neither import line works,
    # even though the other name still resolves in each.
    assert _check(repo, index, {"sqlite_support.py": "def apply_wal_with_fallback(conn): ...\n",
                                "hermes_state_wal.py": "def apply_wal_with_fallback(conn): ...\n"}) == [
        ("__init__.py:4", "sqlite_support.wal_mode")]
    # Reading globals() binds no names: removing one from such a module is still a break.
    _commit(repo, {"tools/registry.py": "def registry_helper(): ...\n", "tools/registry_core.py": None,
                   "hermes_agent/compat/moves.py": "MOVES: dict[str, str] = {}\n"})
    assert _check(repo, index, {"tools/registry.py": "def read():\n    return globals().get('x')\n"}) == [
        ("__init__.py:2", "tools.registry.registry_helper")]


def test_a_fallback_nested_in_the_handler_keeps_the_plugin_working(index, core):
    assert _check(core[0], index, {"plugins/plugin_storage.py": None}) == []
    # With the nested fallback gone too, the plugin runs on neither: the surviving branch is named.
    assert _check(core[0], index, {"hermes_constants.py": None}) == [("__init__.py:12", "hermes_constants.get_hermes_home")]


def test_the_label_lets_an_intended_break_through(index, core, monkeypatch, capsys):
    repo, base = core
    head = _commit(repo, {"agent/memory_provider.py": "class Other: ...\n"})
    monkeypatch.setattr(cci, "ROOT", repo)
    monkeypatch.setattr(cci, "read_index", lambda: index)
    assert cci.check(base, head, []) == 1
    assert "demo __init__.py:1 imports agent.memory_provider.MemoryProvider" in capsys.readouterr().out
    assert cci.check(base, head, [cci.OVERRIDE_LABEL]) == 0


def test_a_failed_read_or_a_moved_subdir_is_read_again(index):
    row = next(iter(index.values()))
    entry = _Entry(Path("."), row.sha)
    assert row.is_current(entry)
    entry.subdir = "media"
    assert not row.is_current(entry)
    assert not cci.Row(row.sha, "", row.imports, error="Could not connect to server").is_current(_Entry(Path("."), row.sha))
