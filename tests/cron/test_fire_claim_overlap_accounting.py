from __future__ import annotations


def test_overlap_loser_discards_unstarted_placeholder(monkeypatch):
    import cron.scheduler as scheduler

    calls = []
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scheduler,
        "discard_unstarted_execution",
        lambda execution_id: calls.append(("discard", execution_id)) or True,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda *args, **kwargs: calls.append(("finish", args, kwargs)),
    )

    assert scheduler._process_due_job(
        {"id": "job-1", "execution_id": "exec-loser"},
        adapters=None,
        loop=None,
        verbose=False,
    ) is True
    assert calls == [("discard", "exec-loser")]


def test_overlap_loser_fails_truthfully_if_placeholder_cannot_be_discarded(monkeypatch):
    import cron.scheduler as scheduler

    calls = []
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda *args, **kwargs: False)
    monkeypatch.setattr(scheduler, "discard_unstarted_execution", lambda _execution_id: False)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: calls.append((execution_id, kwargs)),
    )

    assert scheduler._process_due_job(
        {"id": "job-1", "execution_id": "exec-mismatch"},
        adapters=None,
        loop=None,
        verbose=False,
    ) is True
    assert calls == [
        (
            "exec-mismatch",
            {
                "success": False,
                "error": "Fire claim lost; execution was not started.",
            },
        )
    ]
