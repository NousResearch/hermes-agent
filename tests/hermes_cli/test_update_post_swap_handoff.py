"""Current and shipped post-swap callers preserve the interpreter boundary.

Real subprocess correlation and cleanup are covered by test_update_completion_process.
These tests keep the frozen legacy surface and current transport wiring distinct.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from hermes_cli import update_cmd, update_completion, update_handoff, update_receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan


@pytest.fixture(autouse=True)
def _isolated_handoff_runtime(tmp_path, monkeypatch):
    home, root = tmp_path / "home", tmp_path / "checkout"
    home.mkdir()
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {})
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_unrestored_autostash_notice", lambda: None)
    monkeypatch.setattr(update_handoff, "post_swap_python", lambda: Path(sys.executable))
    token = update_receipt._current.set(None)
    yield
    update_receipt._current.reset(token)


def _request():
    opts = SimpleNamespace(assume_yes=True, pre_update_version="0.21.3")
    return update_cmd._source_completion_request(opts, None, "s1", None, False, False)


def test_parent_uses_handoff_module_loaded_before_checkout_changes(monkeypatch):
    request = _request()
    seen = []

    def captured(payload):
        seen.append(deepcopy(payload))
        receipt = {**payload["receipt"], "outcome": "success", "finished_at": "2026-10-02T00:00:00Z"}
        return {"exit_code": 0, "receipt": receipt, "windows_resume": None}

    monkeypatch.setattr(update_cmd, "run_completion", captured)
    pulled = ModuleType("hermes_cli.update_completion")
    pulled.run_completion = lambda *_: pytest.fail("imported pulled transport into old process")
    monkeypatch.setitem(sys.modules, "hermes_cli.update_completion", pulled)
    update_cmd._complete_source_update(request)
    assert len(seen) == 1 and seen[0]["source"] == request["source"]
    assert seen[0]["snapshot_id"] == "s1"
    assert update_receipt._current.get() is None


def test_parent_reexecs_tail_on_pulled_tree_and_relays_exit_code(monkeypatch, tmp_path):
    from hermes_cli import _old_updater

    seen = []
    payload = {"receipt": {"started_at": "before-swap"},
               "windows_gateway_resume": {"resume_needed": True},
               "had_desktop_app_before_update": True, "plan": {"profiles": ["work"]}}
    monkeypatch.setattr(_old_updater, "_run_child", lambda request: (seen.append(deepcopy(request)) or 7, True))
    assert update_handoff.continue_update_in_fresh_interpreter(payload, argv_tail=["--yes"]) == 7
    assert seen[0]["desktop"] and seen[0]["assume_yes"]
    assert seen[0]["windows_resume"] == payload["windows_gateway_resume"]
    assert payload["windows_gateway_resume"]["resume_needed"]
    assert seen[0]["receipt"]["started_at"] == "before-swap"
    path = tmp_path / "post_swap.json"
    assert update_handoff.post_swap_command(path, ["--yes"])[1:] == [
        "-m", "hermes_cli.main", "update", "--yes", "--post-swap", str(path)]


@pytest.mark.parametrize("upstream_moves", [False, True])
def test_upstream_sync_finishes_before_final_tree_handoff(monkeypatch, upstream_moves):
    final_sha = ("c" if upstream_moves else "b") * 40
    request = _request()
    plan = update_cmd._CheckoutPlan(in_place_update=False, auto_stash_ref=None,
        parked_branch_switched=False, upstream_checked=True, commit_count=1,
        prompt_for_restore=False, switch_block_reason=None)
    handed = []
    monkeypatch.setattr(update_cmd, "_verify_head_after_pull", lambda *a, **kw: final_sha)
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *a: final_sha)
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda data: handed.append(deepcopy(data)))
    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed",
                        lambda *a, **kw: pytest.fail("upstream swap after final checkout selection"))
    update_cmd._apply_pulled_update(["git"], "main", "a" * 40, plan,
                                  _windows_gateway_resume=None, completion_request=request)
    assert len(handed) == 1 and handed[0]["expected_sha"] == final_sha


def test_post_swap_tail_never_changes_checkout_again(monkeypatch):
    events = []
    request = _request()
    request["no_gateway_restart"] = True
    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed",
                        lambda *a, **kw: pytest.fail("tree swap after interpreter handoff"))
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update", lambda *_: None)
    monkeypatch.setattr("hermes_cli.source_completion.complete_source_checkout",
                        lambda *a, **kw: events.append(("build", kw["pre_update_snapshot_id"])) or True)
    monkeypatch.setattr("hermes_cli.venv_sync.clear_completion", lambda *_: events.append("clear"))
    update_completion._complete_selected(request)
    assert events == [("build", "s1"), "clear"]


def test_parent_records_failure_when_child_cannot_start(monkeypatch, capsys):
    from hermes_cli import _old_updater

    monkeypatch.setattr(_old_updater, "_run_child", lambda *_: (_ for _ in ()).throw(OSError("no python")))
    assert update_handoff.continue_update_in_fresh_interpreter({"receipt": {}}, argv_tail=["--yes"]) is None
    output = capsys.readouterr().out
    assert "Could not start" in output and "--post-swap" in output
    assert list((Path(update_cmd.get_hermes_home()) / "logs/update_receipts").glob("post_swap_*.json"))


def test_child_resumes_receipt_and_runs_git_tail_from_payload(monkeypatch, tmp_path):
    from hermes_cli import _old_updater

    receipt = {"started_at": "2026-09-16T15:57:48+00:00", "steps": [{"name": "pre_update_backup", "ok": True}]}
    plan = UpdatePlan(runtimes=[RuntimeRecord(kind="gateway", profile="work", pid=9)]).to_dict()
    payload = {"receipt": receipt, "plan": plan, "sibling_snapshots": {"work": "snap-w"},
               "had_desktop_app_before_update": True, "windows_gateway_resume": None}
    handoff = tmp_path / "post_swap.json"
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    seen = []
    monkeypatch.setattr(_old_updater, "_run_child", lambda request: (seen.append(request) or 0, True))
    assert update_handoff._continue_legacy_post_swap(handoff, argv_tail=["--yes"]) == 0
    assert seen[0]["receipt"] == receipt and seen[0]["plan"] == plan
    assert seen[0]["sibling_snapshots"] == {"work": "snap-w"}
    assert seen[0]["desktop"] and not handoff.exists()
