"""Characterization + unit tests for the `run_one_job` shared helper (Phase 4A).

`tick`'s per-job body (`_process_job`) is the execute → save → deliver → mark
sequence that fires ONE due job. Phase 4A extracts it into a module-level
`run_one_job(job, *, adapters=None, loop=None, verbose=False)` so the external
Chronos provider's `fire_due` can reuse the IDENTICAL body — no duplicated
correctness.

The first test characterizes the sequence as driven through `tick()` (proving
the extraction didn't change `tick`'s behavior); the rest unit-test the
extracted helper directly.
"""
import asyncio

import pytest

import cron.scheduler as s


def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, silent_marker_in=None):
    """Patch the job pipeline primitives and record the call order."""
    calls = []

    def fake_run_job(job, **_kwargs):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "advance_next_run", lambda jid: True)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_run_one_job_silent_skips_delivery(monkeypatch):
    """A [SILENT] final response saves output + marks the run but does NOT
    deliver."""
    calls = _patch_pipeline(monkeypatch, silent_marker_in="[SILENT]")

    s.run_one_job({"id": "j3", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "run_job" in kinds and "save" in kinds and "mark" in kinds
    assert "deliver" not in kinds


def test_run_one_job_report_that_mentions_silent_marker_still_delivers(monkeypatch):
    """Only an exact [SILENT] response suppresses delivery.

    A real status report may explain the [SILENT] policy or mention the marker
    while still containing user-visible information. That report must deliver.
    """
    calls = _patch_pipeline(
        monkeypatch,
        final="Part 55 report: delivery policy mentions [SILENT], but this is real output.",
    )

    s.run_one_job({"id": "j3b", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "deliver" in kinds
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j3b", True)


def test_run_one_job_empty_response_is_soft_failure(monkeypatch):
    """An empty final response marks the run as NOT ok (issue #8585)."""
    calls = _patch_pipeline(monkeypatch, final="   ")

    s.run_one_job({"id": "j4", "name": "t"})

    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j4", False)


def test_run_one_job_failed_job_delivers_error(monkeypatch):
    """A failed job still delivers (the error notice) and marks not-ok."""
    calls = _patch_pipeline(monkeypatch, success=False, final="", error="boom")

    s.run_one_job({"id": "j5", "name": "t"})

    kinds = [c[0] for c in calls]
    assert "deliver" in kinds  # failures always deliver
    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark == ("mark", "j5", False)


def test_run_one_job_exception_marks_failure(monkeypatch):
    """If run_job raises, the helper marks the run failed and returns False
    rather than propagating."""
    def boom(job, **_kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(s, "run_job", boom)
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: marks.append((jid, ok)),
    )

    ok = s.run_one_job({"id": "j6", "name": "t"})

    assert ok is False
    assert marks == [("j6", False)]


@pytest.mark.asyncio
async def test_run_one_job_background_mode_launches_gateway_background(monkeypatch):
    """Opt-in cron background mode schedules the live gateway /background path.

    The cron run saves/delivers only a launch acknowledgement; the real result
    is owned by GatewayRunner._run_background_task.
    """
    monkeypatch.setattr(s, "_build_job_prompt", lambda job, prerun_script=None: "assembled prompt")

    saved = []
    delivered = []
    marks = []
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: saved.append((jid, out)) or f"/tmp/{jid}.md")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, adapters=None, loop=None: delivered.append(content) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: marks.append((jid, ok, err, delivery_error)),
    )

    launched = []

    class FakeRunner:
        def __init__(self):
            self._background_tasks = set()

        async def _run_background_task(self, prompt, source, task_id):
            launched.append((prompt, source, task_id))

    runner = FakeRunner()
    s.register_gateway_background_runner(runner)
    try:
        ok = s.run_one_job(
            {
                "id": "bg1",
                "name": "background job",
                "background": True,
                "deliver": "origin",
                "origin": {
                    "platform": "discord",
                    "chat_id": "1518605229694914680",
                    "thread_id": "thread-1",
                },
            },
            loop=asyncio.get_running_loop(),
        )
        for _ in range(20):
            if launched:
                break
            await asyncio.sleep(0.01)
    finally:
        s.unregister_gateway_background_runner(runner)

    assert ok is True
    assert launched
    prompt, source, task_id = launched[0]
    assert prompt == "assembled prompt"
    assert source.platform.value == "discord"
    assert source.chat_id == "1518605229694914680"
    assert source.thread_id == "thread-1"
    assert task_id.startswith("bgcron_bg1_")
    assert saved and "**Mode:** gateway-background" in saved[0][1]
    assert delivered and "Cron background task started" in delivered[0]
    assert marks == [("bg1", True, None, None)]


def test_run_one_job_background_mode_requires_live_gateway_runner(monkeypatch):
    """A background cron job must not silently fall back to the old cron agent."""
    monkeypatch.setattr(s, "_build_job_prompt", lambda job, prerun_script=None: "assembled prompt")
    s.unregister_gateway_background_runner()
    calls = []
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: calls.append(("save", out)) or f"/tmp/{jid}.md")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, adapters=None, loop=None: calls.append(("deliver", content)) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None: calls.append(("mark", ok, err)),
    )

    ok = s.run_one_job(
        {
            "id": "bg2",
            "name": "background job",
            "background": True,
            "deliver": "origin",
            "origin": {"platform": "discord", "chat_id": "1518605229694914680"},
        }
    )

    assert ok is True
    assert calls[0][0] == "save"
    assert "no live GatewayRunner is registered" in calls[0][1]
    assert calls[-1][0] == "mark"
    assert calls[-1][1] is False
