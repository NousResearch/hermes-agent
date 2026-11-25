"""Retired updater shims hand off only from a historical updater's own call stack (#122656)."""
import importlib
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

# Every gated shim, as data. scripts/ci/check_retired_shims_manifest.py (a repo-guards step) fails
# when a function that can start the takeover is missing here; `--write` regenerates it.
SHIMS = [tuple(item) for item in json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/retired_updater_shims.json").read_text(encoding="utf-8-sig"))]


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


@pytest.mark.parametrize("module,name", SHIMS, ids="{0[0]}.{0[1]}".format)
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


@pytest.mark.parametrize("module,name", SHIMS, ids="{0[0]}.{0[1]}".format)
def test_historical_updater_frame_still_hands_off(module, name, takeover):
    # A historical on-disk `_cmd_update_impl` (no sentinel local) keeps handing off exactly as before.
    namespace = {"__name__": "hermes_cli.update_cmd", "_call": _call}
    exec("def _cmd_update_impl(m, n):\n    pre_update_version = '1.0'\n    _helper(m, n)\n"
         "def _helper(m, n):\n    _call(m, n)\n", namespace)
    with pytest.raises(SystemExit) as stopped:
        namespace["_cmd_update_impl"](module, name)
    assert stopped.value.code == 0
    takeover.assert_called_once()


def test_module_mode_historical_updater_still_hands_off(takeover):
    # `python -m hermes_cli.main update` from a release before update_cmd.py existed: its
    # cmd_update runs as `__main__`, compiled from this checkout's hermes_cli/main.py.
    from hermes_cli import _old_updater

    main_py = str(Path(_old_updater.__file__).resolve().with_name("main.py"))
    namespace = {"__name__": "__main__", "_call": _call}
    exec(compile("def cmd_update(m, n):\n    pre_update_version = '1.0'\n    _call(m, n)\n",
                 main_py, "exec"), namespace)
    with pytest.raises(SystemExit) as stopped:
        namespace["cmd_update"]("tools.lazy_deps", "install_specs")
    assert stopped.value.code == 0
    takeover.assert_called_once()


def test_a_main_script_elsewhere_is_live_code(takeover):
    # Any other `__main__` cmd_update (a user script, a tool) is not the updater.
    namespace = {"__name__": "__main__", "_call": _call}
    exec(compile("def cmd_update(m, n):\n    _call(m, n)\n", "/elsewhere/main.py", "exec"), namespace)
    with pytest.raises(Exception, match="retired"):
        namespace["cmd_update"]("tools.lazy_deps", "install_specs")
    takeover.assert_not_called()


def test_current_module_mode_updater_is_live_code(takeover):
    # The current tree's own cmd_update run via `python -m` carries the sentinel local.
    from hermes_cli import _old_updater

    main_py = str(Path(_old_updater.__file__).resolve().with_name("main.py"))
    namespace = {"__name__": "__main__", "_call": _call}
    exec(compile("def cmd_update(m, n):\n    _hermes_current_updater_frame = True\n    _call(m, n)\n",
                 main_py, "exec"), namespace)
    with pytest.raises(Exception, match="retired"):
        namespace["cmd_update"]("tools.lazy_deps", "install_specs")
    takeover.assert_not_called()
