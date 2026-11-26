"""scripts/boundary_lint.py: CI fails when a core/module boundary crossing occurs more often than the
committed baseline records (One Core P2); a reduction sticks once the baseline is rewritten with
``--update``."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "boundary_lint.py"


def _load():
    spec = importlib.util.spec_from_file_location("boundary_lint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_boundary_violations_do_not_rise_above_the_baseline():
    lint = _load()
    assert not (rises := lint.regressions(lint.tally(lint.collect()), lint.read_baseline())), (
        "New core/module boundary imports:\n  " + "\n  ".join(rises) + "\n" + lint.FIX_HINT
    )


def test_both_directions_are_counted_and_public_imports_are_not(tmp_path):
    lint = _load()
    _write(tmp_path / "plugins" / "plugin_utils.py")
    _write(tmp_path / "plugins" / "platforms" / "foo" / "plugin.yaml", "name: foo\n")
    _write(tmp_path / "plugins" / "platforms" / "foo" / "adapter.py",
           "from gateway.run import GatewayRunner\n"
           "from agent.memory_provider import MemoryProvider\n"
           "from plugins.platforms.foo import helpers\n")
    _write(tmp_path / "tools" / "sender.py",
           "from plugins.platforms.foo.adapter import FooAdapter\n"
           "from plugins import plugin_utils\n")
    assert lint.module_to_core(tmp_path) == ["plugins/platforms/foo/adapter.py:1 gateway.run.GatewayRunner"]
    assert lint.core_to_module(tmp_path) == ["tools/sender.py:1 plugins.platforms.foo.adapter"]


def test_dashboard_modules_relative_imports_and_type_checking(tmp_path):
    lint = _load()
    _write(tmp_path / "plugins" / "board" / "dashboard" / "manifest.json", "{}\n")
    _write(tmp_path / "plugins" / "board" / "dashboard" / "plugin_api.py", "from gateway.run import GatewayRunner\n")
    _write(tmp_path / "plugins" / "memory" / "honcho" / "plugin.yaml", "name: honcho\n")
    _write(tmp_path / "plugins" / "memory" / "honcho" / "client.py")
    _write(tmp_path / "plugins" / "memory" / "loader.py", "from .honcho import client\n")
    _write(tmp_path / "plugins" / "memory" / "sub" / "deep.py", "from ..honcho import client\n")
    _write(tmp_path / "tools" / "typed.py",
           "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from plugins.memory.honcho import client\n")
    assert lint.module_to_core(tmp_path) == ["plugins/board/dashboard/plugin_api.py:1 gateway.run.GatewayRunner"]
    assert lint.core_to_module(tmp_path) == ["plugins/memory/loader.py:1 plugins.memory.honcho",
                                             "plugins/memory/sub/deep.py:1 plugins.memory.honcho"]


def test_grouped_imports_count_each_module(tmp_path):
    lint = _load()
    for name in ("exa", "firecrawl"):
        _write(tmp_path / "plugins" / "web" / name / "plugin.yaml", f"name: {name}\n")
        _write(tmp_path / "plugins" / "web" / name / "provider.py")
    _write(tmp_path / "tools" / "web.py",
           "import plugins.web.exa, plugins.web.firecrawl\n"
           "from plugins.web import exa, firecrawl\n"
           "from plugins.web.exa.provider import A, B\n")
    assert lint.core_to_module(tmp_path) == [
        "tools/web.py:1 plugins.web.exa", "tools/web.py:1 plugins.web.firecrawl",
        "tools/web.py:2 plugins.web.exa", "tools/web.py:2 plugins.web.firecrawl",
        "tools/web.py:3 plugins.web.exa.provider",
    ]


def test_only_a_rise_is_a_regression_and_it_names_the_import():
    lint = _load()
    baseline = {"module_to_core": {"plugins/a/x.py gateway.run.A": 2}, "core_to_module": {"tools/t.py plugins.a": 1}}
    lower = {"module_to_core": {"plugins/a/x.py gateway.run.A": 1}, "core_to_module": {"tools/t.py plugins.a": 1}}
    assert lint.regressions(lower, baseline) == []
    # A swap: one import removed, another added. The total is unchanged, the new crossing still fails.
    swap = {"module_to_core": {"plugins/a/x.py gateway.run.A": 1, "plugins/b/y.py gateway.run._B": 1},
            "core_to_module": {"tools/t.py plugins.a": 1}}
    assert lint.regressions(swap, baseline) == ["module_to_core: plugins/b/y.py gateway.run._B (1, baseline 0)"]


def test_tally_keys_by_importing_file_and_counts_repeats():
    lint = _load()
    found = {"module_to_core": ["plugins/a/x.py:3 gateway.run.A", "plugins/a/x.py:90 gateway.run.A"],
             "core_to_module": ["tools/t.py:1 plugins.a"]}
    assert lint.tally(found) == {"module_to_core": {"plugins/a/x.py gateway.run.A": 2},
                                 "core_to_module": {"tools/t.py plugins.a": 1}}


def _merge(*baselines):
    """What git makes of baselines whose new keys sit on different lines: the union of their entries."""
    return {d: {k: v for b in baselines for k, v in b[d].items()} for d in ("module_to_core", "core_to_module")}


def test_two_branches_adding_one_target_from_different_files_merge_green():
    lint = _load()

    def tally(*sites):
        return lint.tally({"module_to_core": [], "core_to_module": ["tools/base.py:1 plugins.a", *sites]})

    x, y = "tools/x.py:4 plugins.a", "tools/y.py:9 plugins.a"
    assert lint.regressions(tally(x, y), _merge(tally(x), tally(y))) == []
    # Known limit: both branches adding the target in the SAME file record the same key at 1, so the
    # merge keeps 1 while main has 2 (CI on the merged head catches it).
    later = "tools/x.py:30 plugins.a"
    assert lint.regressions(tally(x, later), _merge(tally(x), tally(later))) == [
        "core_to_module: tools/x.py plugins.a (2, baseline 1)"]
