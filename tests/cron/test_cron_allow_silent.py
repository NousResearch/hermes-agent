"""Per-job ``allow_silent`` policy (#53230).

A recurring briefing that promises an all-clear must not silently skip a run
because the model decided there was "nothing new". ``allow_silent=False`` on a
job:

1. replaces the ``[SILENT]`` suppression guidance in the prompt with an
   always-report instruction, and
2. makes the delivery path send a short all-clear when the model emits a
   silence marker anyway, instead of suppressing the delivery or leaking the
   literal marker text to the user.

Default (``True``, including jobs stored before the field existed) keeps the
historical behaviour byte-for-byte. Internal script-job silences (``no_agent``
empty stdout, ``wakeAgent=false``) are scheduler signals rather than model
decisions and stay silent either way.
"""

import pytest

import cron.scheduler as s
from cron.jobs import create_job, update_job
from cron.scheduler import (
    CRON_ALL_CLEAR_MESSAGE,
    SILENT_MARKER,
    _build_job_prompt,
    _is_cron_silence_response,
)


class TestPromptInjection:
    """The flag selects which delivery guidance the prompt carries."""

    def test_default_job_injects_silent_guidance(self):
        job = create_job(prompt="Daily report", schedule="0 9 * * *")
        assert SILENT_MARKER in _build_job_prompt(job)

    def test_allow_silent_true_injects_guidance(self):
        job = create_job(prompt="Daily report", schedule="0 9 * * *", allow_silent=True)
        assert SILENT_MARKER in _build_job_prompt(job)

    def test_allow_silent_false_omits_silent_guidance(self):
        job = create_job(
            prompt="Daily report — always send an all-clear",
            schedule="0 9 * * *",
            allow_silent=False,
        )
        prompt = _build_job_prompt(job)
        assert SILENT_MARKER not in prompt
        # The delivery contract itself is not what is being turned off.
        assert "scheduled cron job" in prompt
        assert "DELIVERY:" in prompt
        assert "ALWAYS REPORT" in prompt

    def test_legacy_job_without_key_defaults_to_silent_guidance(self):
        job = create_job(prompt="Legacy job", schedule="0 9 * * *")
        job.pop("allow_silent", None)
        assert SILENT_MARKER in _build_job_prompt(job)

    def test_failure_and_recursion_markers_survive_the_switch(self):
        for allow_silent in (True, False):
            job = create_job(prompt="Test", schedule="0 9 * * *", allow_silent=allow_silent)
            prompt = _build_job_prompt(job)
            assert "[CRON_FAILURE]" in prompt
            assert "RECURSION:" in prompt


class TestJobField:
    """The flag round-trips through the job store."""

    def test_field_defaults_to_true(self):
        assert create_job(prompt="Test", schedule="0 9 * * *").get("allow_silent") is True

    def test_field_persists_false(self):
        job = create_job(prompt="Test", schedule="0 9 * * *", allow_silent=False)
        assert job.get("allow_silent") is False
        assert isinstance(job.get("allow_silent"), bool)

    def test_update_flips_the_field(self):
        job = create_job(prompt="Test", schedule="0 9 * * *")
        updated = update_job(job["id"], {"allow_silent": False})
        assert updated is not None
        assert updated.get("allow_silent") is False
        assert SILENT_MARKER not in _build_job_prompt(updated)


class TestSilenceClassification:
    """``_is_cron_silence_response`` is a pure string check — the flag gates
    whether the scheduler *acts* on it, not whether it is detectable."""

    @pytest.mark.parametrize("marker", ["[SILENT]", "[SILENT]\n", "SILENT", "NO_REPLY", "NO REPLY"])
    def test_detects_markers(self, marker):
        assert _is_cron_silence_response(marker)

    def test_does_not_detect_normal_text(self):
        assert not _is_cron_silence_response("All systems normal")
        assert not _is_cron_silence_response("Daily report: nothing changed.")


@pytest.fixture
def run_env(monkeypatch):
    """Drive the real ``run_one_job`` delivery decision, captured at _deliver_result."""
    delivered = []

    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-t"})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda *_a, **_kw: {})
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "mark_job_run", lambda *_a, **_kw: True)
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)
    monkeypatch.setattr(s, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(s, "load_config", lambda: {})
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda job, content, **_kw: delivered.append(content) or None,
    )
    return delivered


def _succeeding_run_job(final):
    def _fake(job, **_kw):
        return (True, "raw output", final, None)

    return _fake


@pytest.mark.parametrize(
    ("no_agent", "allow_silent", "response", "expected"),
    [
        (False, True, "[SILENT]", None),
        (False, None, "[SILENT]", None),  # legacy job: key absent
        (False, False, "[SILENT]", CRON_ALL_CLEAR_MESSAGE),
        (False, False, "NO_REPLY", CRON_ALL_CLEAR_MESSAGE),
        (False, False, "NO REPLY", CRON_ALL_CLEAR_MESSAGE),
        (False, False, "Daily report: normal", "Daily report: normal"),
        (True, False, "[SILENT]", None),  # internal script silence, not a model decision
    ],
)
def test_delivery_policy(monkeypatch, run_env, no_agent, allow_silent, response, expected):
    job = {"id": "silence-policy", "name": "policy-test", "deliver": "local",
           "no_agent": no_agent}
    if allow_silent is not None:
        job["allow_silent"] = allow_silent
    monkeypatch.setattr(s, "run_job", _succeeding_run_job(response))

    assert s.run_one_job(job) is True

    if expected is None:
        assert run_env == []
    else:
        assert run_env == [expected]
