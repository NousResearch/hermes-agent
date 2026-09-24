"""``hermes -z`` exit-code contract: the outcome is judged from the turn result, not from
whether any text was printed (#111770 — an incomplete or failed run that left an explanation on
stdout used to exit 0, so scripts treated a half-done job as success)."""

from unittest import mock

import hermes_cli.oneshot as oneshot


def _run(monkeypatch, tmp_path, response, result):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    usage = tmp_path / "usage.json"
    with mock.patch.object(oneshot, "_run_agent", return_value=(response, dict(result))):
        code = oneshot.run_oneshot("q", usage_file=str(usage))
    return code, usage


def test_incomplete_run_with_text_exits_nonzero_and_reports_why(monkeypatch, tmp_path):
    import json

    partial = {"final_response": "Got as far as step 2.", "completed": False, "partial": True,
               "turn_exit_reason": "iteration_limit"}
    code, usage = _run(monkeypatch, tmp_path, "Got as far as step 2.", partial)
    assert code == 2
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["completed"] is False and report["partial"] is True
    assert report["turn_exit_reason"] == "iteration_limit"

    interrupted = {"final_response": "Stopping.", "completed": False, "interrupted": True}
    code, usage = _run(monkeypatch, tmp_path, "Stopping.", interrupted)
    assert code == 130
    assert json.loads(usage.read_text(encoding="utf-8"))["interrupted"] is True

    failed = {"final_response": "Provider returned 401.", "completed": False, "failed": True}
    code, _ = _run(monkeypatch, tmp_path, "Provider returned 401.", failed)
    assert code == 2

    # ``error`` is a failure carrier too (the codex app-server runtime reports a failed turn as
    # ``result["error"]`` text): it must not exit 0 even when the flag fields are absent.
    errored = {"final_response": "", "error": "provider unavailable"}
    code, usage = _run(monkeypatch, tmp_path, "", errored)
    assert code == 2  # a failure, not a completed-but-empty turn (1)
    assert json.loads(usage.read_text(encoding="utf-8"))["failed"] is True

    # The sharpest case: error text riding next to a non-empty reply used to exit 0.
    errored_with_text = {"final_response": "Provider returned 401.", "error": "provider 401"}
    code, _ = _run(monkeypatch, tmp_path, "Provider returned 401.", errored_with_text)
    assert code == 2


def test_completed_run_still_exits_zero(monkeypatch, tmp_path):
    code, _ = _run(monkeypatch, tmp_path, "Paris.", {"final_response": "Paris.", "completed": True, "failed": False})
    assert code == 0
