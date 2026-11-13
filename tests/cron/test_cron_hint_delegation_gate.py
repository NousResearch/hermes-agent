"""The cron FAILURE clause is emitted only for jobs that can delegate.

Regression: ``cron/scheduler_prompt.py`` put "FAILURE: If a delegated child fails ... put
[CRON_FAILURE] on the first line" into the cron hint for EVERY job. For a job whose toolset list
excludes ``delegation`` that advice is unreachable — and a job whose subject matter IS failing
cron runs reads it as "declare this run failed": such a run produced its full report and then
stamped ``[CRON_FAILURE]`` on the first line of the response, which
``cron/scheduler.py::_cron_failure_marker_error`` records as a real failure (``failure_streak``
1, ``last_error`` = the report text).

The fix splits the sentence out of ``_CRON_HINT`` into ``_CRON_FAILURE_CLAUSE`` and emits it only
when ``enabled_toolsets`` is None (no per-job restriction → platform default → delegation
reachable) or explicitly contains ``delegation``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def prompt_mod(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so prompt assembly touches no real cron store."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from cron import scheduler_prompt

    return scheduler_prompt


# A job whose toolset excludes delegation, with a subject matter about failing jobs.
DIGEST_JOB = {
    "id": "digest-job",
    "name": "cron diagnosis digest",
    "prompt": "Report the failing cron jobs and open a card for each one.",
    "enabled_toolsets": ["terminal", "file", "kanban"],
}


class TestFailureClauseGate:

    def test_omitted_for_job_without_delegation_toolset(self, prompt_mod):
        prompt = prompt_mod._build_job_prompt(dict(DIGEST_JOB))
        assert prompt is not None
        assert "CRON_FAILURE" not in prompt
        assert "FAILURE:" not in prompt

    def test_marker_constant_never_reaches_a_non_delegating_prompt(self, prompt_mod):
        """The token the scheduler keys on must not appear anywhere in the prompt."""
        from cron.scheduler import CRON_FAILURE_MARKER

        prompt = prompt_mod._build_job_prompt(dict(DIGEST_JOB))
        assert CRON_FAILURE_MARKER not in prompt

    def test_present_when_delegation_is_explicitly_enabled(self, prompt_mod):
        job = dict(DIGEST_JOB, enabled_toolsets=["delegation", "terminal"])
        prompt = prompt_mod._build_job_prompt(job)
        assert "[CRON_FAILURE] on the first line" in prompt

    def test_present_when_toolsets_are_unrestricted(self, prompt_mod):
        job = {k: v for k, v in DIGEST_JOB.items() if k != "enabled_toolsets"}
        prompt = prompt_mod._build_job_prompt(job)
        assert "[CRON_FAILURE] on the first line" in prompt

    def test_omitted_for_empty_toolset_list(self, prompt_mod):
        job = dict(DIGEST_JOB, enabled_toolsets=[])
        assert prompt_mod._cron_hint(job) == (
            prompt_mod._CRON_HINT_HEAD + prompt_mod._CRON_HINT_TAIL
        )

    def test_clause_is_child_scoped_and_run_scoped(self, prompt_mod):
        """The tightened wording: a child THIS job delegated in THIS run."""
        assert (
            "ONLY if a child YOU delegated in THIS run failed"
            in prompt_mod._CRON_FAILURE_CLAUSE
        )
        assert "If a delegated child fails" not in prompt_mod._CRON_FAILURE_CLAUSE


class TestHintStability:

    def test_delegating_jobs_get_the_byte_identical_hint(self, prompt_mod):
        """No prompt-cache churn: the hint a delegating job sees is unchanged."""
        job = dict(DIGEST_JOB, enabled_toolsets=["delegation"])
        assert prompt_mod._cron_hint(job) == prompt_mod._CRON_HINT
        assert prompt_mod._CRON_HINT == (
            prompt_mod._CRON_HINT_HEAD
            + prompt_mod._CRON_FAILURE_CLAUSE
            + prompt_mod._CRON_HINT_TAIL
        )

    def test_gated_hint_is_the_full_hint_minus_exactly_the_clause(self, prompt_mod):
        gated = prompt_mod._cron_hint(dict(DIGEST_JOB))
        assert prompt_mod._CRON_HINT.replace(prompt_mod._CRON_FAILURE_CLAUSE, "") == gated

    def test_non_delegating_hint_keeps_delivery_and_recursion_guidance(self, prompt_mod):
        """Only the FAILURE sentence is dropped — SILENT/DELIVERY/RECURSION stay."""
        gated = prompt_mod._cron_hint(dict(DIGEST_JOB))
        assert "[SILENT]" in gated
        assert "DELIVERY: Your final response will be automatically delivered" in gated
        assert "RECURSION: This is a run of an EXISTING scheduled job" in gated
