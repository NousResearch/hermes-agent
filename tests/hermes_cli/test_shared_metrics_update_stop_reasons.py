"""Every `hermes update` exit before the apply step names its closed reason (hermes.update.run
``failure_class``), in-process and through the parked copy alike; exits that fire before the
receipt opens still count once; a run that committed and owes work never reads ``failed``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import update_receipt
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_update as update_metrics


@pytest.fixture
def rows(tmp_path, monkeypatch):
    captured: list[tuple[str, dict]] = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly",
                        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}})
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: True)
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: captured.append((mark, data)))
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {"sha": "a" * 40, "commit_date": None})
    with update_receipt.update_receipt_scope():
        yield SimpleNamespace(runs=lambda: [d for m, d in captured if m == contract.UPDATE_RUN_MARK],
                              home=tmp_path / "home")


def _run(stop: str | None, *, applied: bool = False, code: int = 1, user_action: bool = False) -> dict:
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    if user_action:
        update_receipt.record_user_action("local_changes", "stash ref /home/alice/private")
    if stop:
        update_receipt.record_stop_reason(stop)
    if applied:
        update_receipt.record_stage("apply", "success", mode="git")
    path = update_receipt.finalize_pending_update_receipt(code, f"sys.exit({code})")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("stop", "kwargs", "expected"), [
    *[(name, {}, (name, "failed")) for name in sorted(contract.UPDATE_STOP_CLASSES - {"lock_held", "managed_install"})],
    ("not_a_class: /home/alice", {}, ("aborted_before_apply", "failed")),  # only closed tokens pass
    (None, {}, ("aborted_before_apply", "failed")),  # the fallback stays for an exit with no reason
    ("fetch_failed", {"applied": True}, ("deps_failed", "failed")),  # a reason never outlives the apply
    (None, {"applied": True, "code": 1, "user_action": True}, ("local_changes_parked", "partial")),
])
def test_a_pre_apply_exit_reads_its_own_class_in_process_and_parked(rows, stop, kwargs, expected):
    """Invariant: the class is the exit that fired (recorded at it), the parked copy a pre-pull
    interpreter writes classifies the same run identically and keeps no free text, and a committed
    run that only owes the user's parked changes is ``partial``, never ``failed``."""
    receipt = _run(stop, **kwargs)
    (run,) = rows.runs()
    assert (run["failure_class"], run["outcome"]) == expected
    assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, run)
    parked = update_receipt._metric_receipt(receipt)
    assert "/home/alice" not in json.dumps(parked)
    assert update_metrics.update_receipt_fields(json.loads(json.dumps(parked)))[0] == run


def test_exits_before_the_receipt_opens_count_once_and_leave_the_holders_receipt_alone(rows, monkeypatch):
    """Invariant: the update-lock refusal and the Git-operation refusal fire before this run's
    receipt opens; each still yields exactly one row with its class, and neither writes a receipt
    (``latest.json`` belongs to the update holding the lock)."""
    from hermes_cli import main, update_cmd, update_lock, update_owning_install

    update_receipt.begin_update_receipt()  # the holder's open run
    holders = (rows.home / "logs/update_receipts/latest.json").read_bytes()
    update_receipt._current.set(None)

    class Held:
        holder = None

        def __init__(self, **_kw):
            pass

        def acquire(self):
            return False

    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr(update_lock, "UpdateLock", Held)
    monkeypatch.setattr(update_lock, "describe_holder", lambda holder: "another update is running")
    with pytest.raises(SystemExit) as refused:
        main.cmd_update(SimpleNamespace(gateway=False))
    monkeypatch.setattr(update_cmd, "git_operation_in_progress", lambda root: "rebase")
    with pytest.raises(SystemExit) as in_progress:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert (refused.value.code, in_progress.value.code) == (2, 1)
    assert [(r["failure_class"], r["outcome"], r["failed_stage"]) for r in rows.runs()] == [
        ("lock_held", "refused", "none"), ("git_in_progress", "failed", "other")]
    assert all(contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, r) for r in rows.runs())
    assert (rows.home / "logs/update_receipts/latest.json").read_bytes() == holders
