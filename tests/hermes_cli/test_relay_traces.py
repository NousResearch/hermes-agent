"""Execution traces recorded from Hermes's real NeMo Relay events (``relay_traces``)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from agent import relay_runtime
from hermes_cli import lifecycle, plugins
from hermes_cli.observability import relay_traces
from hermes_cli.plugins import PluginManager


@pytest.fixture
def relay_home(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly", lambda: {})
    manager = PluginManager()
    manager._discovered = True  # a runtime with zero plugins: only first-party observers
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    relay_traces._reset_for_tests()
    relay_runtime._reset_for_tests()
    yield home
    relay_traces._reset_for_tests()
    relay_runtime._reset_for_tests()


def _flush() -> None:
    for relay in list(relay_traces.RECORDER._registered_relays.values()):
        relay.subscribers.flush()


def _turn(session_id: str, *, parent: str = "") -> tuple[Any, Any]:
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id=session_id, platform="cli",
        parent_session_id=parent,
    )
    return lease, relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id=f"{session_id}-turn", task_id="task")


def _spans(events: list[dict]) -> dict[str, dict]:
    return {e["uuid"]: e for e in events if e.get("kind") == "scope" and e.get("scope_category") == "start"}


def test_unmanaged_turn_records_relay_llm_and_tool_spans_under_its_turn(relay_home):
    """Without a managed-execution consumer Relay emits no LLM/tool events of its own; the
    recorder fills them from the observer hooks as Relay spans parented under the turn scope,
    and a delegated child's run lands in its own file yet reads back as part of the parent."""
    live: list[tuple[str, str]] = []
    relay_traces.add_listener(lambda _profile, root, session, _event: live.append((root, session)))
    _lease, turn = _turn("parent")
    host = relay_runtime.get_runtime(create=False)
    assert host is not None and not host.managed_execution_enabled()

    started = time.time()
    lifecycle.invoke_hook("pre_llm_call", session_id="parent", user_message="list the files")
    lifecycle.invoke_hook("pre_api_request", session_id="parent", api_request_id="r1", provider="openrouter",
                          api_mode="chat_completions", model="m1", started_at=started,
                          request_messages=[{"role": "user", "content": "list the files"}])
    relay_traces.note_stream_delta("parent", "r1", "reasoning", "let me look")
    relay_traces.note_stream_delta("parent", "r1", "text", "Listing")
    lifecycle.invoke_hook("post_api_request", session_id="parent", api_request_id="r1", ended_at=started + 1,
                          usage={"input_tokens": 12, "output_tokens": 3}, finish_reason="tool_calls",
                          response={"assistant_message": {"content": "", "tool_calls": []}})
    lifecycle.invoke_hook("pre_tool_call", session_id="parent", tool_name="delegate_task",
                          args={"goal": "read it"}, tool_call_id="call-1")
    with relay_runtime.spawning_tool_call("call-1"):  # what delegate_task binds around its children
        child_lease, child_turn = _turn("child", parent="parent")
    lifecycle.invoke_hook("pre_tool_call", session_id="child", tool_name="read_file", args={"path": "a"},
                          tool_call_id="call-2")
    lifecycle.invoke_hook("post_tool_call", session_id="child", tool_name="read_file", tool_call_id="call-2",
                          result='{"content": "abc"}', status="ok")
    relay_runtime.SESSION_COORDINATOR.end_turn(child_turn, outcome="success")
    relay_runtime.SESSION_COORDINATOR.release_conversation(child_lease)
    lifecycle.invoke_hook("post_tool_call", session_id="parent", tool_name="delegate_task",
                          tool_call_id="call-1", result="{}", status="ok")
    relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
    _flush()

    events, truncated = relay_traces.read_session_events(relay_home, "parent")
    spans = _spans(events)
    by_name = {start["name"]: start for start in spans.values()}
    parent_turn = next(s for s in spans.values()
                       if s["name"] == "hermes.turn" and spans[s["parent_uuid"]]["metadata"]["hermes.session_id"] == "parent")

    assert not truncated
    assert by_name["openai.chat_completions"]["category"] == "llm"
    assert by_name["openai.chat_completions"]["parent_uuid"] == parent_turn["uuid"]
    assert by_name["delegate_task"]["parent_uuid"] == parent_turn["uuid"]
    assert spans[by_name["read_file"]["parent_uuid"]]["name"] == "hermes.turn"  # the child's own turn
    child_scope = next(s for s in spans.values() if (s.get("metadata") or {}).get("hermes.session_id") == "child")
    assert child_scope["metadata"][relay_runtime.SPAWNED_BY_TOOL_CALL_KEY] == "call-1"
    assert by_name["delegate_task"]["category_profile"]["tool_call_id"] == "call-1"
    # Relay marks can't hang on an LLM call handle: they sit on the call's scope and name the call.
    call = {"api_request_id": "r1", "llm_uuid": by_name["openai.chat_completions"]["uuid"]}
    stream = [(e["name"], e.get("data")) for e in events
              if e.get("kind") == "mark" and (e.get("data") or {}).get("api_request_id") == "r1"]
    assert stream == [
        (relay_traces.STREAM_FIRST_TOKEN_MARK, {"kind": "reasoning", **call}),
        (relay_traces.STREAM_MARK, {"kind": "reasoning", "text": "let me look", **call}),
        (relay_traces.STREAM_MARK, {"kind": "text", "text": "Listing", **call}),
    ]
    ends = {e["uuid"]: e for e in events if e.get("scope_category") == "end"}
    llm_end = ends[by_name["openai.chat_completions"]["uuid"]]
    assert llm_end["category_profile"]["annotated_response"]["usage"]["prompt_tokens"] == 12
    assert any(e["name"] == relay_traces.TURN_INPUT_MARK and e["data"]["preview"] == "list the files" for e in events)
    # The child's own file holds only its subtree; the parent read folds it in.
    child_only, _ = relay_traces.read_session_events(relay_home, "child")
    assert {e["name"] for e in _spans(child_only).values()} == {"hermes.session", "hermes.turn", "read_file"}
    assert ("parent", "child") in live and all(root == "parent" for root, _session in live)


def test_managed_execution_consumer_suppresses_hook_spans(relay_home):
    """When a Relay consumer routes calls through the managed pipeline, Relay emits the LLM and
    tool events itself; hook-driven spans would draw every call twice."""
    _lease, turn = _turn("managed")
    host = relay_runtime.get_runtime(create=False)
    assert host is not None
    host.retain_managed_execution("test.consumer")
    lifecycle.invoke_hook("pre_api_request", session_id="managed", api_request_id="r1", provider="p",
                          model="m", started_at=time.time(), request_messages=[])
    lifecycle.invoke_hook("post_api_request", session_id="managed", api_request_id="r1", ended_at=time.time())
    relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
    _flush()

    events, _ = relay_traces.read_session_events(relay_home, "managed")
    assert {e["category"] for e in _spans(events).values()} == {"agent", "function"}
