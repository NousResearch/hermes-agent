"""scripts/boundary_lint.py: the core/module import boundary count can only go down (One Core P2)."""

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


def test_only_a_rise_is_a_regression():
    lint = _load()
    baseline = {"module_to_core": 5, "core_to_module": 2}
    assert lint.regressions({"module_to_core": 4, "core_to_module": 2}, baseline) == []
    assert lint.regressions({"module_to_core": 5, "core_to_module": 3}, baseline) == [
        "core_to_module: 3 violations, baseline 2"
    ]
