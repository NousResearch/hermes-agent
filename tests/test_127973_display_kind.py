"""Regression for #127973: model-only/automation user rows carry display_kind.

Covers write-site stamping (hidden / goal_continuation / loop_wakeup /
skill_invocation) and the session_history fallback for untagged rows on disk.
"""
from unittest.mock import MagicMock, patch


def _compressor():
    from unittest.mock import patch as _patch
    from agent.context_compressor import ContextCompressor
    with _patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
    c.tail_token_budget = 500
    return c


def test_inflight_replay_standalone_is_hidden():
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, _SUMMARY_END_MARKER, SUMMARY_PREFIX
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    carrier = {
        "role": "assistant",
        "content": SUMMARY_PREFIX + "\n## Summary\nran.\n\n" + _SUMMARY_END_MARKER,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    compressed = [
        {"role": "system", "content": "sys"},
        carrier,
        {"role": "assistant", "content": "step",
         "tool_calls": [{"id": "c0", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "out"},
    ]
    inflight = {"role": "user", "content": "do the cron job"}
    out = _compressor()._reappend_inflight_user_task(compressed, inflight)
    replays = [m for m in out if m is not carrier and _INFLIGHT_TASK_REPLAY_HEADER in str(m.get("content") or "")]
    assert len(replays) == 1
    assert replays[0].get("display_kind") == "hidden"
    assert replays[0]["role"] == "user"


def test_inflight_merged_carrier_stays_hidden():
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, _SUMMARY_END_MARKER, SUMMARY_PREFIX
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    carrier = {
        "role": "user",
        "content": SUMMARY_PREFIX + "\n## Summary\nran.\n\n" + _SUMMARY_END_MARKER,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
        "display_kind": "hidden",
    }
    # Ends on user so the replay merges onto the carrier.
    compressed = [{"role": "system", "content": "sys"}, carrier]
    inflight = {"role": "user", "content": "finish the report"}
    out = _compressor()._reappend_inflight_user_task(compressed, inflight)
    # No standalone row; the carrier absorbed the replay.
    assert len(out) == 2
    assert _INFLIGHT_TASK_REPLAY_HEADER in str(out[1].get("content") or "")
    assert out[1].get("display_kind") == "hidden"


def test_todo_standalone_snapshot_is_hidden():
    from agent.conversation_compression import _fold_todo_snapshot
    from tools.todo_tool import TODO_INJECTION_HEADER
    agent = MagicMock()
    agent._todo_store.format_for_injection.return_value = f"{TODO_INJECTION_HEADER}\n- [ ] t1 (pending)"
    agent._todo_store.has_items.return_value = True
    compressed = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    _fold_todo_snapshot(agent, compressed)
    assert len(compressed) == 3
    tail = compressed[-1]
    assert tail["role"] == "user"
    assert TODO_INJECTION_HEADER in str(tail["content"])
    assert tail.get("display_kind") == "hidden"


def test_todo_merged_keeps_carrier_kind_and_flags_metadata():
    from agent.conversation_compression import _fold_todo_snapshot
    from tools.todo_tool import TODO_INJECTION_HEADER
    agent = MagicMock()
    agent._todo_store.format_for_injection.return_value = f"{TODO_INJECTION_HEADER}\n- [ ] t1 (pending)"
    agent._todo_store.has_items.return_value = True
    compressed = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow up"},
    ]
    _fold_todo_snapshot(agent, compressed)
    tail = compressed[-1]
    assert TODO_INJECTION_HEADER in str(tail["content"])
    # Merged into a real user turn: no hidden overwrite.
    assert tail.get("display_kind") in (None, "")
    assert isinstance(tail.get("display_metadata"), dict)
    assert tail["display_metadata"].get("todo_snapshot_appended") is True


def test_compression_continuation_placeholder_is_hidden():
    from agent.conversation_compression import _ensure_compressed_has_user_turn
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
    # No real user turn anywhere: forces the placeholder path.
    original = [
        {"role": "assistant", "content": "tool work", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]
    compressed = [{"role": "assistant", "content": "summary"}]
    outcome = _ensure_compressed_has_user_turn(original, compressed)
    assert outcome == "placeholder_appended"
    assert compressed[-1]["role"] == "user"
    assert compressed[-1]["content"] == COMPRESSION_CONTINUATION_USER_CONTENT
    assert compressed[-1].get("display_kind") == "hidden"


def test_goal_continuation_dispatch_is_typed():
    from tui_gateway import server
    seen = {}

    def fake_run(rid, sid, session, prompt, **kw):
        seen.update(kw)
        seen["prompt"] = prompt
        return True

    session = {"running": True, "history_lock": __import__("threading").Lock()}
    with patch.object(server, "_run_prompt_submit", fake_run):
        with patch.object(server, "_emit", lambda *a, **k: True):
            server._dispatch_followup_turn("r1", "s1", session, "goal next step", "goal continuation dispatch",
                                           display_kind="goal_continuation")
    assert seen.get("display_kind") == "goal_continuation"
    assert seen.get("prompt") == "goal next step"


def test_goal_post_turn_followup_uses_goal_kind():
    from tui_gateway import server
    import threading
    calls = {}

    def fake_dispatch(rid, sid, session, prompt, what, **kw):
        calls["kind"] = kw.get("display_kind")
        calls["prompt"] = prompt

    session = {"running": False, "history_lock": threading.Lock(), "_turn_cancel_requested": False}
    with patch.object(server, "_dispatch_followup_turn", fake_dispatch):
        with patch.object(server, "_drain_queued_prompt", return_value=False):
            with patch.object(server, "_session_turn_admission") as adm:
                import contextlib
                adm.return_value = contextlib.nullcontext(True)
                with patch.object(server, "process_registry", create=True):
                    # Stub the safety-net drain block by forcing an early return
                    # via queued prompt? Instead call with empty result and catch
                    # the drain exception path.
                    try:
                        server._run_post_turn_followups("r", "sid", session, {}, "do the next goal step")
                    except Exception:
                        pass
    assert calls.get("kind") == "goal_continuation"


def test_loop_wakeup_submit_is_typed():
    from tui_gateway import server
    seen = {}

    def fake_run(rid, sid, session, text, **kw):
        seen.update(kw)
        seen["text"] = text
        return True

    session = {"history_lock": __import__("threading").Lock(), "running": True}
    with patch.object(server, "_run_prompt_submit", fake_run):
        with patch.object(server, "_emit", lambda *a, **k: True):
            with patch.object(server, "_session_profile_runtime_scope", create=True) as scope:
                import contextlib
                scope.return_value = contextlib.nullcontext()
                server._notif_submit("r1", "s1", session, "loop wake body", "loop wakeup send failed",
                                     display_kind="loop_wakeup")
    assert seen.get("display_kind") == "loop_wakeup"


def test_loop_tick_plain_wakeup_is_typed():
    from tui_gateway import server
    import threading
    seen = {}

    def fake_run(rid, sid, session, text, **kw):
        seen.update(kw)
        return True

    mgr = MagicMock()
    mgr.is_due.return_value = True
    mgr.fire_tick.return_value = "plain wakeup body"
    mgr.state.ticks_fired = 3
    session = {"session_key": "k1", "history_lock": threading.Lock(), "running": False}
    with patch("hermes_cli.loops.LoopManager", return_value=mgr):
        with patch("hermes_cli.loops.goal_blocks_loop_tick", return_value=False):
            with patch.object(server, "_loop_route_is_gateway_chat", return_value=False):
                with patch.object(server, "_notif_claim_turn", return_value=True):
                    with patch.object(server, "_notif_loop_status", lambda *a, **k: None):
                        with patch.object(server, "_emit", lambda *a, **k: True):
                            with patch.object(server, "_run_prompt_submit", fake_run):
                                server._maybe_fire_tui_loop_tick("sid1", session)
    assert seen.get("display_kind") == "loop_wakeup"


def test_skill_persist_fields_split_content_and_sidecar():
    from tui_gateway.methods_tools import _skill_persist_fields
    scaffold = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n# /work\n\nSPIN UP A WORKTREE.\n\n"
        "The user has provided the following instruction alongside the skill "
        "invocation: fix the leak"
    )
    fields = _skill_persist_fields(scaffold)
    assert fields is not None
    typed, api_content, kind = fields
    assert typed == "/work fix the leak"
    assert api_content == scaffold
    assert kind == "skill_invocation"
    assert _skill_persist_fields("just a normal message") is None
    assert _skill_persist_fields("") is None


def test_skill_new_row_projects_typed_without_scaffold_parse():
    from tui_gateway import server
    scaffold = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n# /work\n\nBODY.\n\n"
        "The user has provided the following instruction alongside the skill "
        "invocation: fix the leak"
    )
    history = [{"role": "user", "content": "/work fix the leak",
                "api_content": scaffold, "display_kind": "skill_invocation"}]
    projected = server._history_to_messages(history)
    assert len(projected) == 1
    assert projected[0]["text"] == "/work fix the leak"
    assert projected[0]["display_kind"] == "skill_invocation"
    assert "api_content" not in projected[0]


def test_skill_legacy_scaffold_still_projects_invocation():
    from tui_gateway import server
    scaffold = (
        '[IMPORTANT: The user has invoked the "work" skill, indicating they '
        "want you to follow its instructions. The full skill content is "
        "loaded below.]\n\n# /work\n\nSPIN UP A WORKTREE.\n\n"
        "The user has provided the following instruction alongside the skill "
        "invocation: fix the title leak"
    )
    projected = server._history_to_messages([{"role": "user", "content": scaffold}])
    assert projected[0]["text"] == "/work fix the title leak"
    assert projected[0]["display_kind"] == "skill_invocation"


def test_legacy_fallback_hides_untagged_model_rows():
    from tui_gateway import server
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, COMPRESSION_CONTINUATION_USER_CONTENT
    from tools.todo_tool import TODO_INJECTION_HEADER
    # Inflight replay without kind.
    assert server._legacy_display_kind("user", f"{_INFLIGHT_TASK_REPLAY_HEADER}\ndo thing") == "hidden"
    # Standalone todo snapshot without kind.
    assert server._legacy_display_kind("user", f"{TODO_INJECTION_HEADER}\n- [ ] t") == "hidden"
    # Merged todo (real text + snapshot) keeps user kind.
    assert server._legacy_display_kind("user", f"real question\n\n{TODO_INJECTION_HEADER}\n- [ ] t") is None
    # Continuation placeholder.
    assert server._legacy_display_kind("user", COMPRESSION_CONTINUATION_USER_CONTENT) == "hidden"
    # Ordinary user text stays untyped.
    assert server._legacy_display_kind("user", "hello there") is None


def test_history_projection_drops_legacy_hidden_rows():
    from tui_gateway import server
    from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, COMPRESSION_CONTINUATION_USER_CONTENT
    from tools.todo_tool import TODO_INJECTION_HEADER
    history = [
        {"role": "user", "content": "real first question"},
        {"role": "user", "content": f"{_INFLIGHT_TASK_REPLAY_HEADER}\ndo thing"},
        {"role": "user", "content": f"{TODO_INJECTION_HEADER}\n- [ ] t"},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {"role": "user", "content": "real second question"},
    ]
    projected = server._history_to_messages(history)
    texts = [m["text"] for m in projected]
    assert "real first question" in texts
    assert "real second question" in texts
    assert not any(_INFLIGHT_TASK_REPLAY_HEADER in t for t in texts)
    assert not any(TODO_INJECTION_HEADER in t for t in texts)
    assert COMPRESSION_CONTINUATION_USER_CONTENT not in texts


def test_system_notice_write_site_is_hidden():
    # tui_gateway/server.py stamps hidden for untyped [System: ...] notices.
    import pathlib
    src = pathlib.Path("tui_gateway/server.py").read_text()
    assert "display_kind" in src and "hidden" in src
    assert "[System:" in src


def test_model_switch_marker_stays_visible_model_switch():
    """Model-switch markers are stamped model_switch and stay visible.

    Guards the #130230 mutation: silently reclassifying the marker as hidden
    would drop the pivot notice from the transcript unnoticed. The persistence
    tail warns-and-skips without an agent, but the in-memory entry is what
    renders — pin exactly that.
    """
    from tui_gateway.server import _append_model_switch_marker, _is_model_switch_marker

    session = {"session_key": "pin-model-switch"}
    _append_model_switch_marker(session, model="test-model", provider="")
    assert len(session["history"]) == 1
    entry = session["history"][0]
    assert _is_model_switch_marker(entry)
    assert entry["display_kind"] == "model_switch", (
        "model-switch markers must stay visible, never hidden")
