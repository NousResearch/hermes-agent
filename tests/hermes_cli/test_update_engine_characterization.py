"""Characterization tests for the native update seam.

These tests freeze the small request/result/event model and the delegation
boundary that will sit in front of the legacy ``hermes update`` behavior.
"""

from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_update_engine_dataclasses_are_pure():
    from hermes_cli.update_engine import UpdateEvent, UpdateRequest, UpdateResult

    request = UpdateRequest(args=SimpleNamespace(branch="main"), gateway_mode=True)
    result = UpdateResult(exit_code=0, message="done")
    event = UpdateEvent(kind="fetch", message="Fetching updates")

    assert is_dataclass(UpdateRequest)
    assert is_dataclass(UpdateResult)
    assert is_dataclass(UpdateEvent)
    assert request.args.branch == "main"
    assert request.gateway_mode is True
    assert result.exit_code == 0
    assert result.message == "done"
    assert event.kind == "fetch"
    assert event.message == "Fetching updates"


def test_native_update_engine_delegates_to_injected_legacy_runner():
    from hermes_cli.update_engine import NativeUpdateEngine, UpdateRequest, UpdateResult

    calls = []

    def legacy_runner(request):
        calls.append(request)
        return UpdateResult(exit_code=17, message="legacy path")

    engine = NativeUpdateEngine(legacy_runner=legacy_runner)
    request = UpdateRequest(args=SimpleNamespace(force=True), gateway_mode=False)

    result = engine.run(request)

    assert result == UpdateResult(exit_code=17, message="legacy path")
    assert calls == [request]


def test_cmd_update_routes_through_native_update_seam(mock_args=None):
    from hermes_cli import main as hm
    from hermes_cli.update_engine import UpdateRequest, UpdateResult

    engine_calls = []
    ctor_calls = []

    class FakeEngine:
        def __init__(self, legacy_runner):
            ctor_calls.append(legacy_runner)
            self.legacy_runner = legacy_runner

        def run(self, request):
            engine_calls.append(request)
            return UpdateResult(exit_code=0, message="ok")

    if mock_args is None:
        mock_args = SimpleNamespace(check=False, gateway=False, yes=False)
    else:
        mock_args.check = False
        mock_args.gateway = False
        mock_args.yes = False

    with patch.object(hm, "NativeUpdateEngine", FakeEngine), \
        patch.object(hm, "_install_hangup_protection", return_value=object()) as hangup_mock, \
        patch.object(hm, "_finalize_update_output") as finalize_mock, \
        patch("hermes_cli.config.detect_install_method", return_value="git"), \
        patch("hermes_cli.config.is_unsupported_install_method", return_value=False), \
        patch("hermes_cli.config.is_managed", return_value=False), \
        patch("hermes_cli.config.managed_error") as managed_error_mock, \
        patch("hermes_cli.main._cmd_update_check") as check_mock:
        hm.cmd_update(mock_args)

    assert engine_calls == [UpdateRequest(args=mock_args, gateway_mode=False)]
    assert ctor_calls == [hm._run_legacy_native_update]
    assert hangup_mock.call_count == 1
    finalize_mock.assert_called_once()
    managed_error_mock.assert_not_called()
    check_mock.assert_not_called()


def test_cmd_update_preserves_system_exit_and_finalizes_output():
    from hermes_cli import main as hm

    class ExitingEngine:
        def __init__(self, legacy_runner):
            assert legacy_runner is hm._run_legacy_native_update

        def run(self, request):
            raise SystemExit(7)

    args = SimpleNamespace(check=False, gateway=False, yes=False)
    io_state = object()
    with patch.object(hm, "NativeUpdateEngine", ExitingEngine), \
        patch.object(hm, "_install_hangup_protection", return_value=io_state), \
        patch.object(hm, "_finalize_update_output") as finalize_mock, \
        patch("hermes_cli.config.detect_install_method", return_value="git"), \
        patch("hermes_cli.config.is_unsupported_install_method", return_value=False), \
        patch("hermes_cli.config.is_managed", return_value=False), \
        pytest.raises(SystemExit) as exc_info:
        hm.cmd_update(args)

    assert exc_info.value.code == 7
    finalize_mock.assert_called_once_with(io_state)


def test_cmd_update_preserves_generic_exception_and_finalizes_output():
    from hermes_cli import main as hm

    failure = RuntimeError("boom")

    class FailingEngine:
        def __init__(self, legacy_runner):
            assert legacy_runner is hm._run_legacy_native_update

        def run(self, request):
            raise failure

    args = SimpleNamespace(check=False, gateway=False, yes=False)
    io_state = object()
    with patch.object(hm, "NativeUpdateEngine", FailingEngine), \
        patch.object(hm, "_install_hangup_protection", return_value=io_state), \
        patch.object(hm, "_finalize_update_output") as finalize_mock, \
        patch("hermes_cli.config.detect_install_method", return_value="git"), \
        patch("hermes_cli.config.is_unsupported_install_method", return_value=False), \
        patch("hermes_cli.config.is_managed", return_value=False), \
        pytest.raises(RuntimeError) as exc_info:
        hm.cmd_update(args)

    assert exc_info.value is failure
    finalize_mock.assert_called_once_with(io_state)
