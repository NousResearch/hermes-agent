"""Fresh takeover imports and recovery ownership after a historical checkout swap.

Keep the former purge regressions at their fork path, using the current
acknowledged child boundary rather than the retired Windows shim protocol.
"""
from __future__ import annotations

import contextvars
import copy
import dataclasses
import json
import sys
import types

import pytest

import hermes_constants
import hermes_logging
from hermes_cli import _old_updater, update_cmd, update_cmd_windows, update_finish, update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan


@pytest.fixture
def handoff_env(tmp_path, monkeypatch):
    """No discovery or installed CLI: every output and home is temporary."""
    home, checkout = tmp_path / "home", tmp_path / "checkout"
    home.mkdir()
    checkout.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", checkout)
    monkeypatch.setattr(update_receipt, "_current", contextvars.ContextVar("test_receipt", default=None))
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {})
    monkeypatch.setattr(update_receipt, "_receipt_dir", lambda: home / "logs" / "update_receipts")
    monkeypatch.setattr(_old_updater, "_result", None)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    home_token = hermes_constants.set_hermes_home_override(home)
    try:
        yield home, checkout
    finally:
        hermes_constants.reset_hermes_home_override(home_token)


def _request(checkout, token):
    receipt = update_receipt._current.get()
    return dict(
        root=str(checkout), argv=[], assume_yes=True, gateway_mode=False,
        desktop=True, windows_resume=token, restart_update=False,
        pre_update_snapshot_id="snapshot-default", pre_update_version="old",
        plan=dataclasses.asdict(UpdatePlan(
            expected_sha="a" * 40,
            runtimes=[RuntimeRecord(kind="gateway", profile="work", pid=41)])),
        receipt=copy.deepcopy(receipt.data) if receipt else None,
        update_id=update_receipt.current_correlation_id(),
    )


@pytest.mark.real_post_swap_handoff
def test_handoff_imports_fresh_root_and_package_without_mutating_parent(handoff_env, monkeypatch, capfd):
    """Real isolated child imports only a synthetic checkout and stdlib."""
    home, checkout = handoff_env
    package = checkout / "hermes_cli"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli_output.py").write_text("def new_package_symbol(): return 'fresh package'\n", encoding="utf-8")
    (checkout / "utils.py").write_text("def new_root_symbol(): return 'fresh root'\n", encoding="utf-8")
    (package / "_update_takeover.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "sys.path.insert(0, request['root'])\n"
        "from utils import new_root_symbol\nfrom hermes_cli.cli_output import new_package_symbol\n"
        "print(json.dumps({'symbols': [new_root_symbol(), new_package_symbol()], 'payload': request}))\n"
        "Path(sys.argv[2]).write_text(json.dumps({'resume_handled': True, 'receipt_handled': True}))\n",
        encoding="utf-8")
    monkeypatch.setattr(_old_updater, "__file__", str(package / "_old_updater.py"))

    import utils as parent_utils
    from hermes_cli import cli_output as parent_cli_output

    stale = {}
    for original in (parent_utils, parent_cli_output):
        module = types.ModuleType(original.__name__)
        module.__dict__.update(vars(original))
        stale[original.__name__] = module
        monkeypatch.setitem(sys.modules, original.__name__, module)
    listener = hermes_logging._queue_listener
    token = {"resume_needed": True, "profiles": {"work": 41}}
    update_receipt.begin_update_receipt()
    update_receipt.record_step("git_pull", True, "checkout replaced")
    request = _request(checkout, token)
    monkeypatch.setattr(_old_updater, "_historical_context", lambda: (request, [token], update_receipt._current))

    with pytest.raises(SystemExit) as stopped:
        _old_updater.stop_for_relaunch()
    assert stopped.value.code == 0
    child = next(json.loads(line) for line in capfd.readouterr().out.splitlines() if line.startswith('{"symbols"'))
    assert child["symbols"] == ["fresh root", "fresh package"]
    assert child["payload"]["windows_resume"]["resume_needed"] is True
    assert token["resume_needed"] is False
    assert child["payload"]["receipt"]["steps"][0]["name"] == "git_pull"
    assert child["payload"]["pre_update_snapshot_id"] == "snapshot-default"
    restored = update_finish._restore_plan(child["payload"]["plan"])
    assert isinstance(restored.runtimes[0], RuntimeRecord)
    assert restored.runtimes[0].profile == "work"
    assert update_receipt.finalize_pending_update_receipt(0) is None
    for name, module in stale.items():
        assert sys.modules[name] is module
    assert stale["utils"].atomic_json_write is parent_utils.atomic_json_write
    assert not hasattr(stale["utils"], "new_root_symbol")
    assert not hasattr(stale["hermes_cli.cli_output"], "new_package_symbol")
    assert sys.modules["hermes_logging"] is hermes_logging
    assert hermes_logging._queue_listener is listener
    assert sys.modules["hermes_constants"] is hermes_constants
    assert hermes_constants.get_hermes_home_override() == str(home)


@pytest.mark.platforms('windows')
@pytest.mark.real_post_swap_handoff
@pytest.mark.parametrize("spawn_ok", [True, False], ids=["acknowledged", "spawn-refused"])
def test_windows_handoff_transfers_or_retains_recovery_ownership(handoff_env, monkeypatch, spawn_ok):
    """Only the child's acknowledgement settles parent recovery and receipt."""
    home, checkout = handoff_env
    token = {"resume_needed": True, "profiles": {"work": 41}}
    update_receipt.begin_update_receipt()
    update_receipt.record_step("git_pull", True, "checkout replaced")
    request = _request(checkout, token)
    monkeypatch.setattr(_old_updater, "_historical_context", lambda: (request, [token], update_receipt._current))
    calls, recovered = [], []

    def child(payload):
        calls.append(copy.deepcopy(payload))
        if not spawn_ok:
            raise OSError("synthetic spawn refusal")
        with update_receipt.update_receipt_scope():
            update_receipt.begin_update_receipt(previous=payload["receipt"], correlation_id=payload["update_id"])
            update_receipt.finalize_pending_update_receipt(0)
        return 0, {"resume_handled": True, "receipt_handled": True}

    def recover(resume):
        recovered.append(resume)
        resume["resume_needed"] = False

    monkeypatch.setattr(_old_updater, "_run_child", child)
    monkeypatch.setattr(update_cmd._m(), "_resume_windows_gateways_after_update", recover)
    with pytest.raises(SystemExit) as stopped:
        _old_updater.stop_for_relaunch()
    assert stopped.value.code == (0 if spawn_ok else 1)
    assert calls[0]["windows_resume"]["resume_needed"] is True
    assert token["resume_needed"] is (not spawn_ok)
    if token["resume_needed"]:
        update_cmd_windows._resume_windows_update_runtimes(token)
    assert token["resume_needed"] is False
    assert recovered == ([] if spawn_ok else [token])
    parent_receipt = update_receipt.finalize_pending_update_receipt(stopped.value.code)
    assert (parent_receipt is None) is spawn_ok
    receipts = list((home / "logs" / "update_receipts").glob("update_*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert [step["name"] for step in receipt["steps"]] == ["git_pull"]
    assert receipt["outcome"] == ("success" if spawn_ok else "failed")
    assert update_receipt.finalize_pending_update_receipt(stopped.value.code) is None
    with pytest.raises(SystemExit) as repeated:
        _old_updater.stop_for_relaunch()
    assert repeated.value.code == stopped.value.code
    assert len(calls) == 1


@pytest.mark.parametrize("failure_stage", ["products", "tail"])
def test_post_swap_refusal_resumes_gateways_before_hard_exit(handoff_env, monkeypatch, failure_stage):
    """Fresh completion settles recovery even when building or finishing refuses."""
    from hermes_cli import source_build

    home, checkout = handoff_env
    token = {"resume_needed": True, "profiles": {"work": 41}}
    update_receipt.begin_update_receipt()
    context, result = home / "request.json", home / "result.json"
    context.write_text(json.dumps(_request(checkout, token)), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "hermes_bootstrap", types.ModuleType("hermes_bootstrap"))
    resumed, stages = [], []

    def products(*args, **kwargs):
        stages.append("products")
        if failure_stage == "products":
            raise SystemExit(2)

    def tail(**kwargs):
        stages.append("tail")
        raise SystemExit(2)

    def resume(current):
        resumed.append(copy.deepcopy(current))
        current["resume_needed"] = False

    monkeypatch.setattr(source_build, "build_update_products", products)
    monkeypatch.setattr(update_finish, "finish_update", tail)
    monkeypatch.setattr(update_cmd_windows, "_resume_windows_gateways_after_update", resume)
    with update_receipt.update_receipt_scope():
        assert update_finish.main(context, result) == 2
    assert stages == (["products"] if failure_stage == "products" else ["products", "tail"])
    assert resumed == [{"resume_needed": True, "profiles": {"work": 41}}]
    assert json.loads(result.read_text(encoding="utf-8")) == {"resume_handled": True, "receipt_handled": True}
    receipts = list((home / "logs" / "update_receipts").glob("update_*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["outcome"] == "refused"
