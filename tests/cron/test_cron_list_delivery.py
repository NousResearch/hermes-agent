"""`cron list` surfaces failure_deliver and the model pin.

Kazue's incident-rehearsal rule (#NS-788 follow-up): an operator must be able
to re-verify a job's failure routing from `hermes cron list` without digging
into jobs.json by hand. The list formatter (`_job_rows`) now shows a
``Fail deliver`` row whenever `failure_deliver` is set, and a ``Model`` row
when the job pins a model.

This test exercises `_job_rows` directly — the row tuple list is the schema
the room relies on.
"""

from hermes_cli.cron import _format_model_pin, _job_rows


def _base_job(**extra):
    """Minimal viable job dict that won't crash _job_rows."""
    job = {
        "id": "test-job",
        "name": "test-job",
        "schedule": {"kind": "cron", "expr": "0 8 * * *"},
        "next_run_at": "2026-10-08T08:00:00+00:00",
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "signal:6d208f55-26ca-4e10-a931-9fe1591247ea",
        "last_status": "ok",
        "last_run_at": "2026-10-07T08:00:00+00:00",
    }
    job.update(extra)
    return job


class TestFailureDeliverRow:
    def test_failure_deliver_shown_when_set(self):
        job = _base_job(failure_deliver="signal:group:Alerts")
        rows = dict(_job_rows(job))
        assert rows["Fail deliver"] == "signal:group:Alerts"

    def test_failure_deliver_hidden_when_unset(self):
        """Unset = falls back to Deliver at fire time, so nothing extra renders."""
        rows = _job_rows(_base_job())
        assert "Fail deliver" not in dict(_job_rows(_base_job()))

    def test_failure_deliver_local_shown_explicitly(self):
        """failure_deliver: local (structural silence) is worth seeing on the list."""
        job = _base_job(failure_deliver="local")
        assert dict(_job_rows(job))["Fail deliver"] == "local"

    def test_failure_deliver_placed_right_after_deliver(self):
        """Row ordering matters for scan-ability: Fail deliver immediately follows Deliver."""
        job = _base_job(failure_deliver="signal:group:Alerts")
        labels = [label for label, _ in _job_rows(job)]
        assert labels.index("Fail deliver") == labels.index("Deliver") + 1

    def test_failure_deliver_list_form_joined_like_deliver(self):
        """A hand-edited list value renders comma-joined, same grammar as Deliver."""
        job = _base_job(failure_deliver=["slack:ALERTS", "telegram:123:17"])
        assert dict(_job_rows(job))["Fail deliver"] == "slack:ALERTS, telegram:123:17"


class TestModelPinRow:
    def test_model_pin_shown_with_provider(self):
        job = _base_job(model="claude-4-sonnet", provider="anthropic")
        assert _format_model_pin(job) == "anthropic/claude-4-sonnet"
        assert dict(_job_rows(job))["Model"] == "anthropic/claude-4-sonnet"

    def test_model_pin_shown_without_provider(self):
        job = _base_job(model="gpt-4.1")
        assert _format_model_pin(job) == "gpt-4.1"

    def test_model_pin_hidden_when_unpinned(self):
        """No model field = no Model row (falls back to profile default at fire time)."""
        rows = dict(_job_rows(_base_job()))
        assert "Model" not in rows

    def test_model_pin_alone_without_provider(self):
        assert _format_model_pin({"model": "glm-5.3-flash"}) == "glm-5.3-flash"
        assert _format_model_pin({"model": "", "provider": "openrouter"}) == ""
        assert _format_model_pin({}) == ""
