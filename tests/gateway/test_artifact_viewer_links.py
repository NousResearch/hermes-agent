"""A chat reply carries the viewer link of every hosted artifact its turn made, changed or showed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.artifact_links import append_viewer_links, turn_viewer_links
from gateway.config import Platform
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_results import execution_result
from gateway.turn_context import TurnContext
from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _RecordingAgent

VIEWER = "https://portal.example.com/a/01hx"
OTHER = "https://portal.example.com/a/02jk"
TICKET = "https://01hx.frame.example.com/__av?t=secret-ticket"


def _result(body: dict) -> str:
    return json.dumps({"result": json.dumps(body)})


def _call(call_id: str, tool: str, content: str) -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": f"mcp__hermes_artifacts__{tool}", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


CREATED = _call("c1", "artifact_create", _result(
    {"id": "01hx", "version": 1, "viewer_url": VIEWER, "content_url": TICKET}))
UPDATED = _call("c2", "artifact_update", _result(
    {"id": "01hx", "version": 2, "viewer_url": VIEWER, "content_url": TICKET}))
STALE = _call("c3", "artifact_update", json.dumps({"error": "stale_base: head is 3"}))
SHOWN = _call("c4", "artifact_show", _result(
    {"id": "02jk", "version": 4, "title": "Deck", "viewer_url": OTHER, "content_url": TICKET}))


def test_missing_links_are_appended_once_each_in_order():
    turn = [{"role": "user", "content": "make it"}, *CREATED, *STALE, *UPDATED, *SHOWN]
    reply = append_viewer_links("Done.", turn, 0)
    assert reply == f"Done.\n\nArtifact: {VIEWER}\nArtifact \u201cDeck\u201d: {OTHER}"
    assert "secret-ticket" not in reply


def test_a_link_the_model_already_gave_is_not_repeated():
    reply = f"Here it is: {VIEWER}"
    assert append_viewer_links(reply, [*CREATED], 0) == reply


def test_earlier_turns_never_ride_this_reply():
    history = [{"role": "user", "content": "first"}, *CREATED, {"role": "assistant", "content": "ok"}]
    turn = [*history, {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "bye"}]
    assert append_viewer_links("bye", turn, len(history)) == "bye"
    # A list shorter than the turn's offset (compressed mid-run) cannot be split safely: nothing is added.
    assert turn_viewer_links([*CREATED], 5) == []


@pytest.mark.parametrize("messages", [None, "x", [{"role": "tool", "content": _result({"viewer_url": VIEWER})}]])
def test_unattributable_rows_add_nothing(messages):
    assert append_viewer_links("hi", messages, 0) == "hi"


class _ArtifactAgent(_RecordingAgent):
    final = "I made the page."

    def run_conversation(self, message, conversation_history=None, **kwargs):
        history = list(conversation_history or [])
        turn = [{"role": "user", "content": message}, *CREATED, {"role": "assistant", "content": self.final}]
        return {"final_response": self.final, "messages": history + turn}


def _run_turn(platform):
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id="sid", user_id="u"),
        message="make me a page", history=[], session_id="sid", session_key="***", user_config={},
        AIAgent=_ArtifactAgent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    token = execution_result.set({})
    try:
        turn = TurnRunner(_gateway_runner(), ctx)
        turn._approval_owner = (SimpleNamespace(sessions={"sid": SimpleNamespace(tool_progress_mode=None)},
                                                profile_id="p", adopt_agent=lambda *a: None), "sid", 1)
        return turn.run_sync()
    finally:
        execution_result.reset(token)


@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK, Platform.DISCORD])
def test_a_chat_turn_delivers_the_viewer_link(platform):
    final = _run_turn(platform)["final_response"]
    assert final.startswith(_ArtifactAgent.final) and final.endswith(f"Artifact: {VIEWER}")
    assert "secret-ticket" not in final


def test_a_local_viewer_turn_leaves_the_reply_alone():
    # Local viewers (CLI, Ink TUI, Desktop) show the link on the tool call itself.
    assert _run_turn(Platform.LOCAL)["final_response"] == _ArtifactAgent.final
