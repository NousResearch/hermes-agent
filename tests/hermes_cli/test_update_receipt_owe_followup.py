"""A post-commit follow-up lands on the run exactly once, whether its receipt is still open or closed."""
from __future__ import annotations

import json

from hermes_cli import update_receipt


def _steps(path):
    return [f["step"] for f in json.loads(path.read_text(encoding="utf-8")).get("followups") or []]


def test_follow_up_on_an_open_run_is_recorded_once_not_amended_again(_isolate_hermes_home):
    update_receipt.begin_update_receipt()
    run = update_receipt._current.get().data
    archive = update_receipt._run_file(update_receipt._receipt_dir(), run)

    update_receipt.owe_followup(run["update_id"], "channel_adoption", "remote refused",
                                retry="the next `hermes update` adopts it again")

    # The durable running record (what an interrupted run leaves behind) owes it once.
    assert _steps(archive) == ["channel_adoption"]
    update_receipt.finalize_update_receipt("success")
    assert _steps(archive) == ["channel_adoption"]


def test_follow_up_after_the_run_closed_amends_its_terminal_receipt(_isolate_hermes_home):
    update_receipt.begin_update_receipt()
    run = dict(update_receipt._current.get().data)
    update_receipt.finalize_update_receipt("success")

    update_receipt.owe_followup(run["update_id"], "windows_resume", "Windows gateway recovery failed: boom")

    assert _steps(update_receipt._run_file(update_receipt._receipt_dir(), run)) == ["windows_resume"]
    assert [f["step"] for f in update_receipt.read_latest_receipt()["followups"]] == ["windows_resume"]
    assert update_receipt.read_latest_receipt()["outcome"] == "success"
