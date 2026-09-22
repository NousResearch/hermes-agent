"""A skill that appears or disappears mid-conversation reaches the model without a prompt rebuild.

The persisted ``<available_skills>`` index is reused byte-for-byte for the prefix cache, so the
delta is delivered on the per-turn note channel (cloudflare/cloudflare-os#267 port).
"""
from unittest.mock import MagicMock, patch

from agent.skills_index_delta import (
    _ADDED_HEAD, _REMOVED_HEAD, _SKILLS_INDEX_NOTE_PREFIX, index_entries, stage_skills_index_note,
)

_STORED = (
    "BODY\n\n## Skills\nintro\n\n<available_skills>\n"
    "  devops: ops tools\n"
    "    - docker-ops: Run and debug containers.\n"
    "    - k8s-ops: Kubernetes rollouts.\n"
    "  creative [names only]: ascii-art, manim-video\n"
    "</available_skills>\n\nOnly proceed without loading a skill if none fit.\n\nPlatform: cli"
)
_CURRENT = _STORED.replace(
    "    - k8s-ops: Kubernetes rollouts.\n",
    "    - k8s-ops: Kubernetes rollouts.\n    - terraform-ops: Plan and apply infra changes.\n",
).replace("ascii-art, manim-video", "manim-video")


def _agent():
    agent = MagicMock()
    agent.provider = "openrouter"
    agent.api_mode = "chat_completions"
    agent._skills_index_note = ""
    return agent


def _stage(agent, history, current=_CURRENT):
    with patch("agent.system_prompt._skills_prompt", return_value=current):
        return stage_skills_index_note(agent, _STORED, history)


def test_index_entries_reads_described_and_names_only_skills():
    assert set(index_entries(_STORED)) == {"docker-ops", "k8s-ops", "ascii-art", "manim-video"}
    assert index_entries(_STORED)["docker-ops"] == "    - docker-ops: Run and debug containers."
    assert index_entries("no block here") == {}


def test_delta_is_announced_once_with_its_description_and_restaged_only_when_it_changes():
    agent = _agent()
    assert _stage(agent, [{"role": "user", "content": "hi"}]) is True
    note = agent._skills_index_note
    assert note.startswith(_SKILLS_INDEX_NOTE_PREFIX)
    assert "    - terraform-ops: Plan and apply infra changes." in note  # the routing signal, not just a name
    assert f"{_REMOVED_HEAD} ascii-art]" in note
    assert "docker-ops" not in note.split(_ADDED_HEAD, 1)[1]  # unchanged skills are not re-listed

    # Gateway shape: a fresh agent next turn, the transcript already carries this exact delta.
    carried = [{"role": "user", "content": "hi", "api_content": f"hi\n\n{note}"}, {"role": "assistant", "content": "ok"}]
    again = _agent()
    assert _stage(again, carried) is False
    assert again._skills_index_note == ""

    # The delta grows (another skill lands): the cumulative note is re-staged.
    grown = _CURRENT.replace("</available_skills>", "    - ansible-ops: Configuration runs.\n</available_skills>")
    third = _agent()
    assert _stage(third, carried, current=grown) is True
    assert "ansible-ops" in third._skills_index_note and "terraform-ops" in third._skills_index_note

    # Undone (index accurate again): the stale note is retired, then nothing more is staged.
    fourth = _agent()
    assert _stage(fourth, carried, current=_STORED) is True
    assert "accurate again" in fourth._skills_index_note
    retired = carried + [{"role": "user", "content": "x", "api_content": f"x\n\n{fourth._skills_index_note}"}]
    fifth = _agent()
    assert _stage(fifth, retired, current=_STORED) is False

    # A prompt without an index has no baseline; an unchanged index stages nothing.
    assert stage_skills_index_note(_agent(), "no skills block", []) is False
    assert _stage(_agent(), [], current=_STORED) is False
