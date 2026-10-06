"""A run's own archive is read through one lookup: an unreadable record is a missing record."""
from __future__ import annotations

import pytest

from hermes_cli import update_completion, update_receipt


def _lost_run(tmp_path):
    update_receipt.begin_update_receipt()
    run = dict(update_receipt._current.get().data)
    request = {"source": str(tmp_path / "checkout"), "receipt": run, "desktop": False, "gateway_mode": False}
    return run, update_receipt._run_file(update_receipt._receipt_dir(), run), request


@pytest.mark.parametrize("content", ["null", "[]", "{torn"])
def test_lost_completion_over_a_malformed_run_archive_still_closes_the_run(_isolate_hermes_home, tmp_path, content):
    # The code is committed: an archive that is valid JSON but not a record (hand edit, foreign
    # writer) must settle exactly like a missing one, never raise out of the post-commit path.
    run, archive, request = _lost_run(tmp_path)
    archive.write_text(content, encoding="utf-8")

    result = update_completion.settle_lost_completion(request, "the completion process exited -9")

    assert result["exit_code"] == 0
    assert result["receipt"]["update_id"] == run["update_id"]
    assert result["receipt"]["outcome"] == "success"
    assert [row["step"] for row in result["receipt"]["followups"]] == ["completion"]
