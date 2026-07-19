"""Characterization tests for the existing native update flow.

These tests freeze the small request/result model, legacy delegation boundary,
repository lock, and native-Windows compatibility behavior.
"""

from dataclasses import is_dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_main_import_survives_without_posix_lock_primitives():
    import subprocess
    import sys

    script = (
        "import builtins, os\n"
        "real_import = builtins.__import__\n"
        "def guarded_import(name, *args, **kwargs):\n"
        "    if name == 'fcntl':\n"
        "        raise ModuleNotFoundError('simulated non-POSIX runtime')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded_import\n"
        "if hasattr(os, 'getuid'):\n"
        "    delattr(os, 'getuid')\n"
        "import hermes_cli.main\n"
        "from hermes_cli.update_lock import acquire_shared_update_lock, UpdateLockError\n"
        "try:\n"
        "    with acquire_shared_update_lock('/unsupported-platform', timeout_seconds=0.0):\n"
        "        raise AssertionError('lock must not fail open')\n"
        "except UpdateLockError as exc:\n"
        "    assert exc.code.value == 'acquisition_failed'\n"
        "    assert exc.boundary.value == 'pre_mutation'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_cmd_update_preserves_legacy_path_without_posix_lock_backend():
    import subprocess
    import sys

    script = r'''
import builtins
from types import SimpleNamespace
from unittest.mock import patch

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("simulated native Windows runtime")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

from hermes_cli import main as hm
calls = []
args = SimpleNamespace(check=False, gateway=False, yes=False)
with patch("hermes_cli.config.detect_install_method", return_value="git"), \
     patch("hermes_cli.config.is_unsupported_install_method", return_value=False), \
     patch("hermes_cli.config.is_managed", return_value=False), \
     patch.object(hm, "_install_hangup_protection", return_value=object()), \
     patch.object(hm, "_finalize_update_output"), \
     patch.object(hm, "_cmd_update_impl", side_effect=lambda *a, **kw: calls.append((a, kw))):
    hm.cmd_update(args)
assert len(calls) == 1, calls
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_update_engine_dataclasses_are_pure():
    from hermes_cli.update_engine import UpdateRequest, UpdateResult

    request = UpdateRequest(args=SimpleNamespace(branch="main"), gateway_mode=True)
    result = UpdateResult(exit_code=0, message="done")

    assert is_dataclass(UpdateRequest)
    assert is_dataclass(UpdateResult)
    assert request.args.branch == "main"
    assert request.gateway_mode is True
    assert result.exit_code == 0
    assert result.message == "done"


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
        def __init__(self, legacy_runner, update_lock_identity=None):
            ctor_calls.append((legacy_runner, update_lock_identity))
            self.legacy_runner = legacy_runner
            self.update_lock_identity = update_lock_identity

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
    assert ctor_calls == [(hm._run_legacy_native_update, str(hm.PROJECT_ROOT))]
    assert hangup_mock.call_count == 1
    finalize_mock.assert_called_once()
    managed_error_mock.assert_not_called()
    check_mock.assert_not_called()


def test_cmd_update_preserves_system_exit_and_finalizes_output():
    from hermes_cli import main as hm

    class ExitingEngine:
        def __init__(self, legacy_runner, update_lock_identity=None):
            assert legacy_runner is hm._run_legacy_native_update
            assert update_lock_identity == str(hm.PROJECT_ROOT)

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
        def __init__(self, legacy_runner, update_lock_identity=None):
            assert legacy_runner is hm._run_legacy_native_update
            assert update_lock_identity == str(hm.PROJECT_ROOT)

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


def test_legacy_run_contends_on_shared_update_lock_when_locked(tmp_path):
    import subprocess
    import sys
    import time

    from hermes_cli.update_engine import NativeUpdateEngine, UpdateRequest, UpdateResult
    from hermes_cli.update_lock import UpdateLockError, shared_update_lock_identity

    repo = tmp_path / "repo"
    repo.mkdir()
    identity = shared_update_lock_identity(repo)

    script = (
        "import sys, time\n"
        "from hermes_cli.update_lock import acquire_shared_update_lock\n"
        "with acquire_shared_update_lock(sys.argv[1], timeout_seconds=1.0):\n"
        "    time.sleep(1.0)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", script, identity],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        time.sleep(0.2)
        calls = []

        def legacy_runner(request):
            calls.append(request)
            return UpdateResult(exit_code=0)

        engine = NativeUpdateEngine(
            legacy_runner=legacy_runner,
            update_lock_identity=identity,
            update_lock_timeout_seconds=0.1,
        )
        with pytest.raises(UpdateLockError):
            engine.run(UpdateRequest(args={"branch": "main"}, gateway_mode=False))
        assert calls == []
    finally:
        holder.wait(timeout=5)


def test_legacy_release_failure_is_post_mutation_uncertain(tmp_path, monkeypatch):
    from hermes_cli import update_lock
    from hermes_cli.update_engine import (
        NativeUpdateEngine,
        UpdateMutationBoundary,
        UpdateRequest,
        UpdateResult,
    )
    from hermes_cli.update_lock import UpdateLockError, UpdateLockErrorCode

    repo = tmp_path / "repo"
    repo.mkdir()
    real_close = update_lock._close_lock_fd

    def close_then_report_failure(fd):
        real_close(fd)
        raise OSError("simulated close uncertainty")

    monkeypatch.setattr(update_lock, "_close_lock_fd", close_then_report_failure)
    calls = []

    def legacy_runner(request):
        calls.append(request)
        return UpdateResult(exit_code=0)

    engine = NativeUpdateEngine(
        legacy_runner=legacy_runner,
        update_lock_identity=str(repo),
    )
    request = UpdateRequest(args={"yes": True}, gateway_mode=False)
    with pytest.raises(UpdateLockError) as exc_info:
        engine.run(request)

    assert calls == [request]
    assert exc_info.value.code is UpdateLockErrorCode.RELEASE_FAILED
    assert exc_info.value.boundary is UpdateMutationBoundary.POST_MUTATION_UNCERTAIN
