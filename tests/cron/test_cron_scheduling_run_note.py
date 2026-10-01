"""The run note and the ``cronjob`` toolset must tell the same story.

``cron.allow_agent_scheduling`` decides whether a cron-spawned agent gets the ``cronjob`` toolset
at all (``_resolve_cron_disabled_toolsets``). With the gate on, a chained job — one whose prompt
says "book your next run from the calendar" — has the tool it needs, but the run note's RECURSION
sentence is a blanket ban on reacting to recurring language, and a model that reads it that way
refuses its own next step. The chain then stops with no error and no next fire (#130896).

Contract:

  - gate off: the prompt is byte-exact what it was before this change, note included.
  - gate on: the note still forbids reacting to recurring language, and adds one line saying the
    task's own "book the next run" step is the task, not recurring language.
  - the line appears inside the note (before its closing bracket), so it cannot be mistaken for
    something the job prompt said, and the job prompt itself is untouched.
"""

import pytest

from cron.scheduler import _agent_scheduling_allowed, _resolve_cron_disabled_toolsets
from cron.scheduler_prompt import _build_job_prompt

RECURSION_SENTENCE = (
    "treat phrasing like \"each Monday\" or \"every day at 9\" as context for this run, "
    "not as a request to schedule another job."
)
SELF_SCHEDULING = "SELF-SCHEDULING"
SELF_SCHEDULING_SENTENCE = (
    "SELF-SCHEDULING: if the task below tells you to book this job's next run, do it — "
    "that is the task itself, not recurring language."
)
NOTE_HEAD = "[IMPORTANT: You are running as a scheduled cron job."
NOTE_TAIL = "]\n\n"


def _job(prompt: str = "Summarise yesterday's tickets.") -> dict:
    return {"id": "job-1", "name": "digest", "prompt": prompt}


class TestGateOff:
    def test_the_prompt_has_no_self_scheduling_line(self):
        assert SELF_SCHEDULING not in _build_job_prompt(_job())

    def test_the_recursion_ban_is_intact(self):
        assert RECURSION_SENTENCE in _build_job_prompt(_job())

    def test_the_call_defaults_to_the_gate_off_prompt(self):
        """A caller that forgets the flag gets today's behavior, not a surprise instruction."""
        assert _build_job_prompt(_job()) == _build_job_prompt(
            _job(), allow_agent_scheduling=False)


class TestGateOn:
    def test_the_prompt_carries_the_self_scheduling_line(self):
        assert SELF_SCHEDULING in _build_job_prompt(_job(), allow_agent_scheduling=True)

    def test_the_recursion_ban_survives(self):
        assert RECURSION_SENTENCE in _build_job_prompt(_job(), allow_agent_scheduling=True)

    def test_one_sentence_is_inserted_between_the_ban_and_the_note_tail(self):
        note = _build_job_prompt(_job(), allow_agent_scheduling=True)
        assert note.startswith(NOTE_HEAD)
        head, tail = note.split(RECURSION_SENTENCE, 1)
        assert head.endswith(RECURSION_SENTENCE) is False  # the split consumed the sentence
        # Between the ban and the note's closing bracket sits the added sentence and nothing else.
        assert tail == " " + SELF_SCHEDULING_SENTENCE + NOTE_TAIL + _job()["prompt"]

    def test_the_job_prompt_is_untouched(self):
        """Only the note grows: a chained job's own prompt must survive verbatim, tail included."""
        job = _job("Book this job's next run for tomorrow 9am, every day.")
        assert _build_job_prompt(
            job, allow_agent_scheduling=True).endswith(NOTE_TAIL + job["prompt"])
        assert _build_job_prompt(
            job, allow_agent_scheduling=False).endswith(NOTE_TAIL + job["prompt"])

    def test_dropping_the_added_sentence_restores_the_gate_off_prompt(self):
        """One sentence is the whole difference: removing it reproduces today's prompt byte for byte."""
        job = _job()
        on = _build_job_prompt(job, allow_agent_scheduling=True)
        off = _build_job_prompt(job, allow_agent_scheduling=False)
        assert on.replace(" " + SELF_SCHEDULING_SENTENCE, "", 1) == off


class TestNoteAndToolsetAgree:
    """The note may only ever promise the toolset this run actually receives."""

    CASES = (
        {"cron": {"allow_agent_scheduling": True}},
        {"cron": {"allow_agent_scheduling": False}},
        {"cron": {}},
        {},
        {"cron": {"allow_agent_scheduling": "yes"}},
        {"cron": {"allow_agent_scheduling": None}},
    )

    @pytest.mark.parametrize("cfg", CASES)
    def test_the_gate_agrees_with_the_toolset_resolution(self, cfg):
        assert _agent_scheduling_allowed(cfg) == (
            "cronjob" not in _resolve_cron_disabled_toolsets(cfg))

    @pytest.mark.parametrize("cfg", CASES)
    def test_the_note_reflects_the_same_gate(self, cfg):
        prompt = _build_job_prompt(_job(), allow_agent_scheduling=_agent_scheduling_allowed(cfg))
        assert (SELF_SCHEDULING in prompt) is _agent_scheduling_allowed(cfg)


class TestUserDenialStillWins:
    """``agent.disabled_toolsets`` outranks the gate. The note explains how to read the task; it is
    not what grants a toolset, so a user who denies ``cronjob`` keeps it denied either way."""

    def test_a_user_level_cronjob_denial_still_withholds_the_tool(self):
        cfg = {"cron": {"allow_agent_scheduling": True},
               "agent": {"disabled_toolsets": ["cronjob"]}}
        assert _agent_scheduling_allowed(cfg) is True
        assert "cronjob" in _resolve_cron_disabled_toolsets(cfg)