"""Caller attribution and logging survive the startup-guard extraction."""
import asyncio
import importlib.util
import logging
from pathlib import Path
from traceback import FrameSummary
from types import SimpleNamespace

import pytest

from hermes_cli._startup_fast import project_root_str


@pytest.mark.parametrize("alias_import", [False, True], ids=["canonical", "symlink"])
def test_loop_guard_attributes_storage_calls_to_external_caller(tmp_path, alias_import):
    root = Path(project_root_str())
    if alias_import:
        alias = tmp_path / "checkout"
        try:
            alias.symlink_to(root, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Directory symlinks unavailable: {exc}")
        root = alias
    spec = importlib.util.spec_from_file_location(
        "loop_guard_attribution_probe", root / "hermes_state_loop_guard.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callers = []
    for function in ("first_caller", "second_caller"):
        caller = FrameSummary(str(root / "gateway/session_authority.py"), 100, function)
        stack = [
            caller,
            FrameSummary(str(root / "hermes_state.py"), 300, "_read_ctx"),
            FrameSummary(str(root / "hermes_state_loop_guard.py"), 90, "wrapped"),
        ]
        assert module._caller(stack) is caller
        callers.append(module._caller(stack).name)
    assert callers == ["first_caller", "second_caller"]
    # An unrelated file sharing a storage-module basename remains attributable.
    foreign = FrameSummary(str(tmp_path / "foreign/hermes_state.py"), 1, "foreign_call")
    assert module._caller([foreign]) is foreign


def test_startup_watchdog_failure_retains_gateway_logger(monkeypatch, caplog):
    from gateway import shutdown_watchdog
    from gateway.run_startup_liveness import start_loop_liveness_guards
    import hermes_state_loop_guard

    monkeypatch.setattr(hermes_state_loop_guard, "install_from_config", lambda cfg: None)

    def fail_start(*args, **kwargs):
        raise RuntimeError("synthetic watchdog startup failure")

    monkeypatch.setattr(shutdown_watchdog, "start_loop_liveness_watchdog", fail_start)
    owner = SimpleNamespace(config=None, _loop_floor_timer_handle=object())
    loop = asyncio.new_event_loop()
    try:
        with caplog.at_level(logging.DEBUG):
            start_loop_liveness_guards(owner, loop)
    finally:
        loop.close()
    records = [r for r in caplog.records if r.message == "Failed to start gateway loop liveness watchdog"]
    assert len(records) == 1
    assert records[0].name == "gateway.run"
