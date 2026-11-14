"""Retired updater shims hand off only from a historical updater's own call stack (#122656)."""
import ast
import importlib
import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _retired_shims():
    """Every function in the shipped tree that can start the takeover (a bare stop_for_relaunch
    or the shared gate), found by AST so a new shim cannot skip these contracts."""
    found = []
    for path in sorted(p for d in ("hermes_cli", "tools", "gateway") for p in (ROOT / d).rglob("*.py")):
        if path.name == "_old_updater.py":
            continue
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id in {"stop_for_relaunch", "stop_historical_updater"}
                   and not call.keywords for call in ast.walk(fn)):
                found.append((module, fn.name))
    # hermes_cli.main's lazy frozen export, resolved through main.__getattr__.
    found.append(("hermes_cli.main", "_write_update_incomplete_marker"))
    assert len(found) > 40, found  # the scan itself must not silently find nothing
    return found


def _call(module, name):
    fn = getattr(importlib.import_module(module), name)
    params = inspect.signature(fn).parameters.values()
    args = [None for p in params if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kwargs = {p.name: None for p in params if p.default is p.empty and p.kind is p.KEYWORD_ONLY}
    return fn(*args, **kwargs)


@pytest.fixture
def takeover(monkeypatch):
    from hermes_cli import _old_updater

    child = Mock(return_value=(0, {}))  # never spawn the real _update_takeover.py
    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(_old_updater, "_result", None)
    return child


@pytest.mark.parametrize("module,name", _retired_shims(), ids="{0[0]}.{0[1]}".format)
@pytest.mark.parametrize("frame", ["live", "current-updater"])
def test_current_code_never_starts_the_takeover(module, name, frame, takeover):
    # A live serve/gateway/plugin, or the CURRENT `hermes update` (sentinel local), reaching a
    # retired shim must get a catchable error, not a full update finish that restarts it.
    sentinel = "    _hermes_current_updater_frame = True\n" if frame == "current-updater" else ""
    namespace = {"__name__": "hermes_cli.update_cmd", "_call": _call}
    exec(f"def _cmd_update_impl(m, n):\n{sentinel}    _call(m, n)\n" if sentinel else
         "def serve(m, n):\n    _call(m, n)\n", namespace)
    entry = namespace["_cmd_update_impl" if sentinel else "serve"]
    with pytest.raises(Exception, match="retired") as refused:
        entry(module, name)
    assert not isinstance(refused.value, TypeError)
    takeover.assert_not_called()


@pytest.mark.parametrize("module,name", _retired_shims(), ids="{0[0]}.{0[1]}".format)
def test_historical_updater_frame_still_hands_off(module, name, takeover):
    # A historical on-disk `_cmd_update_impl` (no sentinel local) keeps handing off exactly as before.
    namespace = {"__name__": "hermes_cli.update_cmd", "_call": _call}
    exec("def _cmd_update_impl(m, n):\n    pre_update_version = '1.0'\n    _helper(m, n)\n"
         "def _helper(m, n):\n    _call(m, n)\n", namespace)
    with pytest.raises(SystemExit) as stopped:
        namespace["_cmd_update_impl"](module, name)
    assert stopped.value.code == 0
    takeover.assert_called_once()
