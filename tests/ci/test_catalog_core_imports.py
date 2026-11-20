"""scripts/ci/catalog_core_imports.py: which pinned catalog plugins a core change breaks.

Real git repositories in tmp_path stand in for a catalog plugin and for two core commits; the index row
comes from the same fetch-and-parse path the scheduled build runs.
"""

from __future__ import annotations

import json
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
    repo, _base = core
    removed = _check(repo, index, {"tools/registry.py": "def other(): ...\n"})
    assert removed == [("__init__.py:2", "tools.registry.registry_helper")]
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


MOVED = 'MOVES: dict[str, str] = {"tools.old": "tools.new"}\n'
TWO_LEVELS = ("try:\n    from tools.a import x\n    try:\n        from tools.b import y\n    except ImportError:\n"
              "        from tools.c import y\nexcept ImportError:\n    x = None\n")
OPTIONAL_INNER = ("try:\n    from tools.a import x\n    try:\n        from tools.b import y\n    except ImportError:\n"
                  "        y = None\nexcept ImportError:\n    x = None\n")
ELSE_AND_TYPING = ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from tools.registry import only_typing\n"
                   "try:\n    import json\nexcept ImportError:\n    json = None\nelse:\n    from tools.registry import helper\n")


@pytest.mark.parametrize("plugin, base, change, expected", [
    # Dropping a compat key, with no other file touched, unmaps the old module name.
    ("from tools.old import helper\n", {"tools/new.py": "helper = 1\n", "hermes_agent/compat/moves.py": MOVED},
     {"hermes_agent/compat/moves.py": "MOVES: dict[str, str] = {}\n"}, [("__init__.py:1", "tools.old.helper")]),
    # ``from package import submodule`` is the submodule's file.
    ("from tools import registry\n", {"tools/__init__.py": "", "tools/registry.py": ""}, {"tools/registry.py": None},
     [("__init__.py:1", "tools.registry")]),
    # A name that moves behind a star import still imports.
    ("from tools.registry import helper\n", {"tools/registry.py": "helper = 1\n"},
     {"tools/registry.py": "from tools._impl import *\n", "tools/_impl.py": "helper = 1\n"}, []),
    # A body whose nested try fails as a whole does not run, whatever its own imports do.
    (TWO_LEVELS, {"tools/a.py": "x = 1\n", "tools/b.py": "y = 1\n", "tools/c.py": "y = 1\n"},
     {"tools/b.py": "", "tools/c.py": ""}, [("__init__.py:4", "tools.b.y"), ("__init__.py:6", "tools.c.y")]),
    # A try nested in a body that still runs loses its own feature: the outer import surviving hides nothing.
    (OPTIONAL_INNER, {"tools/a.py": "x = 1\n", "tools/b.py": "y = 1\n"}, {"tools/b.py": ""},
     [("__init__.py:4", "tools.b.y")]),
    # An ``else:`` block is checked like any plain import; a ``TYPE_CHECKING`` import never loads.
    (ELSE_AND_TYPING, {"tools/registry.py": "helper = only_typing = 1\n"}, {"tools/registry.py": ""},
     [("__init__.py:9", "tools.registry.helper")]),
])
def test_what_counts_as_a_lost_import(tmp_path, plugin, base, change, expected):
    source, core = tmp_path / "plugin", tmp_path / "core"
    for repo in (source, core, tmp_path / "work"):
        repo.mkdir()
    _git(source, "init", "-q")
    _git(core, "init", "-q")
    index = {"demo": cci.index_entry(_Entry(source, _commit(source, {"__init__.py": plugin})), tmp_path / "work")}
    _commit(core, {"hermes_agent/compat/moves.py": "MOVES: dict[str, str] = {}\n", **base})
    assert _check(core, index, change) == expected


def test_a_symlink_in_the_plugin_repository_is_not_followed(tmp_path):
    source = tmp_path / "plugin"
    for repo in (source, tmp_path / "work"):
        repo.mkdir()
    (tmp_path / "outside.py").write_text("from tools.registry import helper\n", encoding="utf-8")
    _git(source, "init", "-q")
    (source / "link.py").symlink_to(tmp_path / "outside.py")
    row = cci.index_entry(_Entry(source, _commit(source, {"__init__.py": ""})), tmp_path / "work")
    assert not row.error and row.imports == cci.CoreImports((), ())


def test_a_fallback_nested_in_the_handler_keeps_the_plugin_working(index, core):
    assert _check(core[0], index, {"plugins/plugin_storage.py": None}) == []
    # With the nested fallback gone too, the plugin runs on neither: the surviving branch is named.
    assert _check(core[0], index, {"hermes_constants.py": None}) == [("__init__.py:12", "hermes_constants.get_hermes_home")]


# A PEP 562 __getattr__ like core's facades: it serves a lookup table through a helper and caches what it serves.
LAZY = """\
_LAZY = {"OLD_NAME": "tools.registry_core"}


def unrelated(args):
    return getattr(args, "registry_helper", None)


def _legacy(name):
    return name == "LEGACY_NAME"


def _former(name):
    if _legacy(name) or name in _LAZY:
        return object()
    raise AttributeError(name)


def __getattr__(name):
    globals()[name] = value = _former(name)
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
"""


def test_a_module_getattr_answers_only_the_names_it_spells(index, core):
    repo, _base = core
    _commit(repo, {"tools/registry.py": "def registry_helper(): ...\n" + LAZY})
    assert _check(repo, index, {"tools/registry.py": LAZY}) == [("__init__.py:2", "tools.registry.registry_helper")]
    for served_by in ('"OLD_NAME"', '"LEGACY_NAME"'):  # its lookup table, a helper of its helper
        _commit(repo, {"tools/registry.py": "def registry_helper(): ...\n" + LAZY})
        assert _check(repo, index, {"tools/registry.py": LAZY.replace(served_by, '"registry_helper"')}) == []
    # Never served on the base, so dropping the hook breaks nothing that worked there.
    _commit(repo, {"tools/registry.py": LAZY})
    assert _check(repo, index, {"tools/registry.py": "def other(): ...\n"}) == []


def test_a_break_is_a_warning_naming_the_core_name_and_the_plugins_catalog_record(
        index, core, tmp_path, monkeypatch, capsys):
    repo, _base = core
    # Judged against the BASE's published API: this tree publishes MemoryProvider and nothing else.
    api = json.dumps({"groups": [{"entries": ["agent.memory_provider:MemoryProvider"]}]})
    base = _commit(repo, {"hermes_cli/data/plugin_api.json": api})
    head = _commit(repo, {"agent/memory_provider.py": "class Other: ...\n",
                          "hermes_state_wal.py": None, "sqlite_support.py": None})
    sha = next(iter(index.values())).sha
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "demo.yaml").write_text(
        f"name: demo\nrepo: https://github.com/acme/demo-plugin\nsha: {sha}\nsubdir: plugin\nversion: '1.4.0'\n"
        "description: d\nmaintainer: acme\ntier: community\ncategory: memory\n", encoding="utf-8")
    for name, value in {"ROOT": repo, "CATALOG": catalog, "read_index": lambda: index}.items():
        monkeypatch.setattr(cci, name, value)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    assert cci.check(base, head) == 0  # a warning, never a failure
    out = capsys.readouterr().out
    # The core name, whether a plugin may rely on it, and who imports it.
    assert "`agent.memory_provider.MemoryProvider` (public API): 1 plugin: demo" in out
    assert "`hermes_state_wal.wal_mode` (private): 1 plugin: demo" in out
    # The plugin's catalog record: repo, subdir and pin, catalog file, and a link to the import line.
    assert f"repo: https://github.com/acme/demo-plugin (subdir `plugin`) @ `{sha[:12]}`" in out
    assert "plugin-catalog/demo.yaml" in out and "maintainer `acme`" in out and "1.4.0" in out
    assert f"https://github.com/acme/demo-plugin/blob/{sha}/plugin/__init__.py#L1" in out
    assert "::warning title=Breaks catalog plugin demo::" in out
    assert "hermes_state_wal.wal_mode" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    status = (tmp_path / "runner" / "review-status" / "catalog-core-imports" / "review-status.json").read_text(
        encoding="utf-8")
    (entry,) = json.loads(status.removeprefix("review_status="))
    assert entry["results"][0]["kind"] == "warning"
    assert "acme/demo-plugin" in entry["results"][0]["detail"]


def test_a_failed_read_or_a_moved_subdir_is_read_again(index):
    row = next(iter(index.values()))
    entry = _Entry(Path("."), row.sha)
    assert row.is_current(entry)
    entry.subdir = "media"
    assert not row.is_current(entry)
    assert not cci.Row(row.sha, "", row.imports, error="Could not connect to server").is_current(_Entry(Path("."), row.sha))


@pytest.mark.parametrize(("module", "breaks"), [
    # A read binds nothing, so the removed name is a break.
    ("def read(name):\n    return globals()[name]\n", True),
    # A write through globals() can bind any name, so it stands in for the removed one.
    ("def export(namespace):\n    namespace['registry_helper'] = export\n\n\nexport(globals())\n", False),
    ("def bind():\n    globals().update(registry_helper=bind)\n\n\nbind()\n", False),
])
def test_only_a_globals_write_can_stand_in_for_a_removed_name(index, core, module, breaks):
    assert _check(core[0], index, {"tools/registry.py": module}) == (
        [("__init__.py:2", "tools.registry.registry_helper")] if breaks else [])


def test_an_import_already_broken_on_the_base_is_not_this_changes_break(index, core):
    _commit(core[0], {"tools/registry.py": "def other(): ...\n"})
    assert _check(core[0], index, {"tools/registry.py": "def other(): ...\ndef another(): ...\n"}) == []
