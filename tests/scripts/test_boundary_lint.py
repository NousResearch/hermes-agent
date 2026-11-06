"""scripts/boundary_lint.py: CI fails when a per-direction core/module boundary count rises above the
committed baseline (One Core P2); a reduction sticks once the baseline is lowered with ``--update``."""

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
    found = lint.collect()
    counts = {d: len(found[d]) for d in lint.DIRECTIONS}
    assert not lint.regressions(counts, lint.read_baseline()), (
        "New core/module boundary imports. Run `python scripts/boundary_lint.py --list` to see them; "
        "a module imports only the published plugin API, and core reaches modules through discovery."
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


def test_only_a_rise_is_a_regression():
    lint = _load()
    baseline = {"module_to_core": 5, "core_to_module": 2}
    assert lint.regressions({"module_to_core": 4, "core_to_module": 2}, baseline) == []
    assert lint.regressions({"module_to_core": 5, "core_to_module": 3}, baseline) == [
        "core_to_module: 3 violations, baseline 2"
    ]
