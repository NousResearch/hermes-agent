"""Behaviour contract for the cron scheduler's own degenerate-output guard.

The guard has two jobs and they pull in opposite directions:

  * a real runaway loop must never reach a chat (the 2026-11-02 incident), and
  * an HONEST repetitive report must never be destroyed or reported as a failed run.

The second half is the one this file exists for. A ``no_agent`` script that prints
the same WARN line on every sample produces a log whose body *is* one repeated
block, so ``collapse_degenerate`` — which keeps only the text outside the
degenerate span — finds nothing to keep and used to collapse the whole report to
zero characters. The run was then booked ``success=False`` with "degenerate model
output with no readable body": the operator lost an honest report AND received a
failure alert for a script that exited 0.

Contract pinned here: when the degenerate span covers the entire response, the
delivered copy is a bounded, clearly marked excerpt of the original — never empty,
never a failure.
"""

from __future__ import annotations

import pytest

from cron import jobs, scheduler
from cron.scheduler import _RunDelivery


# A real shape: a watchdog that samples every few seconds and prints one line per
# sample. 100 copies is above MIN_PERIODIC_COPIES (48), so the detector fires.
_HONEST_LOG = "\n".join(["WARN repeated alert"] * 100)

# The incident shape: one phrase to exhaustion, no readable preamble or suffix.
_PURE_LOOP = "NOW FIN. " * 4000


@pytest.fixture
def delivered(monkeypatch, tmp_path):
    """Run one completed fire and capture the composed delivery text."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    captured: dict = {}

    def fake_deliver_result(job, content, adapters=None, loop=None, **kwargs):
        captured["content"] = content

    monkeypatch.setattr(scheduler, "_deliver_result", fake_deliver_result)
    return captured


def _fire(job: dict, final_response: str, output: str, monkeypatch) -> None:
    monkeypatch.setattr(scheduler, "run_job", lambda j, **kw: (True, output, final_response, None))
    scheduler.run_one_job(dict(job))


def _job(tmp_path, **extra) -> dict:
    created = jobs.create_job(prompt="p", schedule="every 1h", deliver="local")
    return {**created, **extra}


class TestHonestRepetitiveLogIsPreserved:
    def test_whole_response_repetitive_is_not_collapsed_to_nothing(self, delivered, tmp_path, monkeypatch):
        job = _job(tmp_path, no_agent=True, script="w.sh")

        _fire(job, _HONEST_LOG, _HONEST_LOG, monkeypatch)

        content = delivered["content"]
        assert content.strip(), "an honest repetitive log must not be collapsed to nothing"
        assert "WARN repeated alert" in content

    def test_whole_response_repetitive_is_not_a_failed_run(self, delivered, tmp_path, monkeypatch):
        job = _job(tmp_path, no_agent=True, script="w.sh")

        _fire(job, _HONEST_LOG, _HONEST_LOG, monkeypatch)

        stored = jobs.get_job(job["id"])
        assert stored["last_status"] == "ok", (
            f"an honest repetitive log was booked as {stored['last_status']!r} "
            f"({stored.get('last_error')!r})"
        )
        assert "no readable body" not in str(stored.get("last_error") or "")

    def test_preserved_excerpt_is_bounded_and_marked(self, delivered, tmp_path, monkeypatch):
        """The excerpt must stay one readable message and say that it was shortened."""
        job = _job(tmp_path, no_agent=True, script="w.sh")

        _fire(job, _HONEST_LOG, _HONEST_LOG, monkeypatch)

        content = delivered["content"]
        assert len(content) <= 4000, "the excerpt must fit one platform message"
        assert "退化" in content or "truncated" in content.lower()

    def test_pure_loop_with_no_readable_body_still_delivers_something(self, delivered, tmp_path, monkeypatch):
        """A genuine loop is bounded and marked, not silently dropped and not verbatim."""
        job = _job(tmp_path, no_agent=True, script="w.sh")

        _fire(job, _PURE_LOOP, _PURE_LOOP, monkeypatch)

        content = delivered["content"]
        assert content.strip()
        assert len(content) <= 4000
        assert len(content) < len(_PURE_LOOP)

    def test_a_readable_report_around_a_loop_is_still_collapsed(self, delivered, tmp_path, monkeypatch):
        """The original guarantee stays: text outside the loop survives, the loop does not."""
        job = _job(tmp_path, no_agent=True, script="w.sh")
        body = (
            "Patrol summary written before the model started looping.\n"
            + _PURE_LOOP
            + "\nFinal state: board A has 2 open cards."
        )

        _fire(job, body, body, monkeypatch)

        content = delivered["content"]
        assert "Patrol summary" in content
        assert "Final state" in content
        assert "NOW FIN" not in content


class TestRunDeliveryDefaults:
    def test_guard_is_not_reached_for_failed_runs(self, delivered, tmp_path, monkeypatch):
        """A failure notice is our own copy, not model output: it is delivered as composed.

        ``no_agent=True`` pins the delivery route: with it the composed notice goes through
        ``_deliver_result`` (which the ``delivered`` fixture captures), while an agent-shaped
        job carrying an execution id routes through ``cron.delivery_queue`` and never reaches
        the monkeypatched seam.  The shape is harness plumbing, not the assertion — what is
        pinned here is that a failed run's notice is NOT caught by the degeneracy guard.
        """
        job = _job(tmp_path, no_agent=True, script="w.sh")
        monkeypatch.setattr(
            scheduler, "run_job",
            lambda j, **kw: (False, "doc", _HONEST_LOG, "script exited 3"))

        _fire(job, _HONEST_LOG, _HONEST_LOG, monkeypatch)

        assert delivered["content"].strip()
