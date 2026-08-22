"""Tests for opt-in log-reconstruction desync check.

Detects silent loss after known wire transforms. Default off = zero cost.
E2E cases build api_messages the way conversation_loop does.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Sequence

import pytest

from agent.log_reconstruction import (
    LogReconstructionDesyncError,
    check_log_reconstruction,
    compare_history_to_api_messages,
    extract_api_history,
    is_log_reconstruction_check_enabled,
    maybe_check_before_request,
    project_turn,
)


def simulate_send_path_api_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    system: str = "You are Hermes.",
    prefill_messages: Optional[Sequence[Mapping[str, Any]]] = None,
    current_turn_user_idx: Optional[int] = None,
    inject_user_suffix: str = "",
    apply_cache_control_on_last_user: bool = False,
    drop_codex_reasoning_items: bool = True,
) -> List[dict]:
    """Build an ``api_messages`` list the way the conversation loop does.

    Test-only helper (not used on the hot path). Mirrors send-path ordering:

    clone → api_content prefer → copy reasoning→reasoning_content then pop
    reasoning → system → prefill insert → drop thinking-only + merge users →
    strip → optional cache_control shape.

    Reasoning fidelity: production calls ``_copy_reasoning_content_for_api``
    before popping ``reasoning``, so reasoning-only assistants still carry
    ``reasoning_content`` into the drop pass. Without the copy step, a
    reasoning-only turn with only ``reasoning`` would survive simulate while
    ``project_live_through_wire_transforms`` drops it.
    """
    api: List[dict] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            continue
        api_msg = {
            k: v
            for k, v in msg.items()
            if k not in ("display_kind", "display_metadata", "_row_id")
        }
        api_msg = dict(api_msg)
        _api_content = api_msg.pop("api_content", None)
        if idx == current_turn_user_idx and msg.get("role") == "user":
            if isinstance(_api_content, str) and _api_content:
                api_msg["content"] = _api_content
            if inject_user_suffix:
                base = api_msg.get("content") or ""
                if isinstance(base, str):
                    api_msg["content"] = base + inject_user_suffix
        elif (
            isinstance(_api_content, str)
            and _api_content
            and msg.get("role") in ("user", "assistant")
        ):
            api_msg["content"] = _api_content

        # Mirror conversation_loop: copy reasoning for API, then drop storage key.
        # Minimal fidelity of apply_reasoning_content_policy for drop detection:
        # promote non-empty ``reasoning`` onto ``reasoning_content`` when absent.
        if "reasoning" in api_msg:
            reasoning = api_msg.get("reasoning")
            if (
                isinstance(reasoning, str)
                and reasoning.strip()
                and not (
                    isinstance(api_msg.get("reasoning_content"), str)
                    and api_msg.get("reasoning_content", "").strip()
                )
            ):
                api_msg["reasoning_content"] = reasoning
            api_msg.pop("reasoning", None)
        api_msg.pop("finish_reason", None)
        api.append(api_msg)

    if system:
        api = [{"role": "system", "content": system}] + api

    if prefill_messages:
        sys_offset = 1 if (api and api[0].get("role") == "system") else 0
        for i, pfm in enumerate(prefill_messages):
            api.insert(sys_offset + i, dict(pfm))

    from agent.agent_runtime_helpers import drop_thinking_only_and_merge_users

    api = drop_thinking_only_and_merge_users(
        api,
        drop_codex_reasoning_items=drop_codex_reasoning_items,
    )

    for am in api:
        if isinstance(am.get("content"), str):
            am["content"] = am["content"].strip()

    if apply_cache_control_on_last_user:
        for am in reversed(api):
            if am.get("role") == "user" and isinstance(am.get("content"), str):
                text = am["content"]
                am["content"] = [
                    {
                        "type": "text",
                        "text": text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                break

    return api


# ---------------------------------------------------------------------------
# Unit: projection + comparison basics
# ---------------------------------------------------------------------------


class TestProjectAndCompare:
    def test_matching_history_is_ok(self):
        live = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "next"},
        ]
        api = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "next"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=2)
        assert report.ok
        assert report.mismatches == []

    def test_system_and_prefill_are_skipped(self):
        live = [{"role": "user", "content": "hi"}]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prefill-user"},
            {"role": "assistant", "content": "prefill-asst"},
            {"role": "user", "content": "hi"},
        ]
        report = compare_history_to_api_messages(live, api, prefill_count=2)
        assert report.ok

    def test_silent_loss_of_historical_turn_is_desync(self):
        """Drop a middle turn that wire transforms would keep → fire."""
        live = [
            {"role": "user", "content": "turn-1"},
            {"role": "assistant", "content": "reply-1"},
            {"role": "user", "content": "turn-2"},
            {"role": "assistant", "content": "reply-2"},
            {"role": "user", "content": "turn-3"},
        ]
        # Lost reply-1 + turn-2 while keeping ends — not a contiguous suffix
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "turn-1"},
            {"role": "assistant", "content": "reply-2"},
            {"role": "user", "content": "turn-3"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=4)
        assert not report.ok
        assert any("drift" in m or "mismatch" in m for m in report.mismatches)

    def test_content_drift_on_historical_turn_is_desync(self):
        live = [
            {"role": "user", "content": "original"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now"},
        ]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "MUTATED"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=2)
        assert not report.ok
        assert any("content fingerprint drift" in m for m in report.mismatches)

    def test_current_user_injection_prefix_is_ok(self):
        live = [{"role": "user", "content": "hello"}]
        api = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "hello\n\n<memory-context>\nlikes tea\n</memory-context>",
            },
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert report.ok

    def test_current_user_short_string_in_trap_is_not_ok(self):
        """Loose substring match would pass; prefix-only must fail."""
        live = [{"role": "user", "content": "a"}]
        api = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "totally unrelated paragraph containing a letter",
            },
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert not report.ok

    def test_empty_current_user_does_not_pass_arbitrary_injection(self):
        """Wiped current-user body must not be masked by wire injection."""
        live = [{"role": "user", "content": ""}]
        api = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "\n\n<memory-context>\ninjected</memory-context>",
            },
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert not report.ok

    def test_empty_current_user_matches_empty_api(self):
        live = [{"role": "user", "content": ""}]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": ""},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert report.ok

    def test_empty_content_with_stamped_api_content_matches_stamp(self):
        live = [{"role": "user", "content": "", "api_content": "stamped-body"}]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "stamped-body"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert report.ok

    def test_historical_api_content_sidecar_is_authoritative(self):
        live = [
            {
                "role": "user",
                "content": "clean",
                "api_content": "clean\n\nINJECTED",
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "next"},
        ]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "clean\n\nINJECTED"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "next"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=2)
        assert report.ok

    def test_tool_call_integrity_name_args(self):
        live = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
        ]
        api = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"cmd":"rm"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
        ]
        report = compare_history_to_api_messages(live, api)
        assert not report.ok
        assert any("tool_calls integrity" in m for m in report.mismatches)

    def test_wire_only_tool_stub_insertion_still_matches_live(self):
        live = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "done"},
        ]
        api = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "tool_call_id": "orphan",
                "content": "[tool result missing]",
            },
            {"role": "assistant", "content": "done"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert report.ok

    def test_compression_summary_plus_tail_no_fp(self):
        """api = marked summary + last turn; live still full → ok (no FP)."""
        live = [
            {"role": "user", "content": "old-1"},
            {"role": "assistant", "content": "old-a"},
            {"role": "user", "content": "old-2"},
            {"role": "assistant", "content": "old-b"},
            {"role": "user", "content": "latest"},
        ]
        api = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Summary of prior conversation...",
                "_compressed_summary": True,
            },
            {"role": "user", "content": "latest"},
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=4)
        assert report.ok, report.mismatches


# ---------------------------------------------------------------------------
# E2E: real send-path ordering via simulate_send_path_api_messages
# ---------------------------------------------------------------------------


class TestSendPathE2E:
    def test_legit_transforms_return_ok(self):
        """thinking-only drop, strip, cache-control shape, user merge → ok."""
        live = [
            {"role": "user", "content": "first\n"},  # trailing newline → strip
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I ponder quietly",
            },  # thinking-only → dropped on wire
            {"role": "user", "content": "second"},  # merges with first after drop
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "third"},
        ]
        api = simulate_send_path_api_messages(
            live,
            current_turn_user_idx=4,
            inject_user_suffix="\n\n<memory>x</memory>",
            apply_cache_control_on_last_user=True,
            prefill_messages=[
                {"role": "user", "content": "prefill-u"},
                {"role": "assistant", "content": "prefill-a"},
            ],
        )
        agent = SimpleNamespace(
            log_reconstruction_check=True,
            log_reconstruction_check_raise=False,
            prefill_messages=[
                {"role": "user", "content": "prefill-u"},
                {"role": "assistant", "content": "prefill-a"},
            ],
            api_mode="chat_completions",
            model="test-model",
        )
        report = check_log_reconstruction(
            agent,
            messages=live,
            api_messages=api,
            api_kwargs={"model": "test-model", "messages": api},
            current_turn_user_idx=4,
            raise_on_desync=False,
        )
        assert report.ok, report.mismatches

    def test_reasoning_storage_key_only_is_dropped_like_prod(self):
        """``reasoning`` (storage) without ``reasoning_content`` still drops."""
        live = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "trajectory-only thoughts",
            },
            {"role": "user", "content": "follow-up"},
        ]
        api = simulate_send_path_api_messages(live, current_turn_user_idx=2)
        report = compare_history_to_api_messages(
            live, api, prefill_count=0, current_turn_user_idx=2
        )
        assert report.ok, report.mismatches
        hist = extract_api_history(api)
        assert len(hist) == 1
        assert hist[0]["role"] == "user"

    def test_genuine_silent_loss_fires(self):
        live = [
            {"role": "user", "content": "keep-me"},
            {"role": "assistant", "content": "visible-reply"},
            {"role": "user", "content": "now"},
        ]
        # Build legit api then drop a retained assistant turn
        api = simulate_send_path_api_messages(live, current_turn_user_idx=2)
        # Remove the assistant that wire transforms would keep
        api = [m for m in api if m.get("content") != "visible-reply"]
        agent = SimpleNamespace(
            log_reconstruction_check=True,
            log_reconstruction_check_raise=True,
            prefill_messages=[],
            api_mode="chat_completions",
        )
        with pytest.raises(LogReconstructionDesyncError) as excinfo:
            check_log_reconstruction(
                agent,
                messages=live,
                api_messages=api,
                current_turn_user_idx=2,
            )
        assert "log-reconstruction desync" in str(excinfo.value).lower()
        assert excinfo.value.diff

    def test_thinking_only_alone_no_fp(self):
        live = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "   ",
                "reasoning_content": "secret thoughts",
            },
            {"role": "user", "content": "follow-up"},
        ]
        api = simulate_send_path_api_messages(live, current_turn_user_idx=2)
        report = compare_history_to_api_messages(
            live, api, prefill_count=0, current_turn_user_idx=2
        )
        assert report.ok, report.mismatches
        # Wire should have merged users
        hist = extract_api_history(api)
        assert len(hist) == 1
        assert hist[0]["role"] == "user"
        assert "q" in hist[0]["content"] and "follow-up" in hist[0]["content"]


# ---------------------------------------------------------------------------
# Guard / raise behavior
# ---------------------------------------------------------------------------


class TestGuardAndRaise:
    def test_disabled_is_zero_cost_no_op(self):
        agent = SimpleNamespace(log_reconstruction_check=False)
        assert is_log_reconstruction_check_enabled(agent) is False
        maybe_check_before_request(
            agent,
            messages=[{"role": "user", "content": "a"}],
            api_messages=[{"role": "user", "content": "b"}],
        )
        report = check_log_reconstruction(
            agent,
            messages=[{"role": "user", "content": "a"}],
            api_messages=[],
        )
        assert report.ok

    def test_enabled_soft_by_default_no_raise(self):
        agent = SimpleNamespace(
            log_reconstruction_check=True,
            log_reconstruction_check_raise=False,
            prefill_messages=[],
            api_mode="chat_completions",
        )
        # Content drift on a retained historical turn (not compression-shaped)
        report = maybe_check_before_request(
            agent,
            messages=[
                {"role": "user", "content": "keep-me"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "now"},
            ],
            api_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "MUTATED"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "now"},
            ],
            current_turn_user_idx=2,
        )
        assert report is not None
        assert not report.ok

    def test_enabled_raise_flag_hard_fails(self):
        agent = SimpleNamespace(
            log_reconstruction_check=True,
            log_reconstruction_check_raise=True,
            prefill_messages=[],
            api_mode="chat_completions",
        )
        with pytest.raises(LogReconstructionDesyncError):
            maybe_check_before_request(
                agent,
                messages=[
                    {"role": "user", "content": "keep-me"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "now"},
                ],
                api_messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "MUTATED"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "now"},
                ],
                current_turn_user_idx=2,
            )

    def test_model_meta_never_raises(self):
        """Fallback changes model above the hook — must not raise."""
        agent = SimpleNamespace(
            log_reconstruction_check=True,
            log_reconstruction_check_raise=True,
            prefill_messages=[],
            api_mode="chat_completions",
            model="gpt-test",
        )
        # Aligned history, mismatched model — must NOT raise
        maybe_check_before_request(
            agent,
            messages=[{"role": "user", "content": "hi"}],
            api_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            api_kwargs={"model": "other-model", "messages": []},
            current_turn_user_idx=0,
        )


# ---------------------------------------------------------------------------
# Config default + real agent_init stamp via temp HERMES_HOME
# ---------------------------------------------------------------------------


class TestConfigDefault:
    def test_default_config_flags_are_false(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["agent"]["log_reconstruction_check"] is False
        assert DEFAULT_CONFIG["agent"]["log_reconstruction_check_raise"] is False

    def test_config_propagates_through_agent_init_stamp(self, tmp_path, monkeypatch):
        """Write config.yaml under temp HERMES_HOME and stamp via agent_init path."""
        home = tmp_path / "hermes_home"
        home.mkdir()
        cfg_path = home / "config.yaml"
        cfg_path.write_text(
            textwrap.dedent(
                """\
                agent:
                  log_reconstruction_check: true
                  log_reconstruction_check_raise: true
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        # Clear cached config if any
        import hermes_cli.config as config_mod

        if hasattr(config_mod, "_config_cache"):
            monkeypatch.setattr(config_mod, "_config_cache", None, raising=False)

        from hermes_cli.config import load_config

        loaded = load_config()
        agent_section = loaded.get("agent") or {}
        assert agent_section.get("log_reconstruction_check") is True
        assert agent_section.get("log_reconstruction_check_raise") is True

        # Stamp the same way agent_init.py does (the real wiring lines)
        agent = SimpleNamespace()
        _agent_section = agent_section if isinstance(agent_section, dict) else {}
        agent.log_reconstruction_check = bool(
            _agent_section.get("log_reconstruction_check", False)
        )
        agent.log_reconstruction_check_raise = bool(
            _agent_section.get("log_reconstruction_check_raise", False)
        )
        assert agent.log_reconstruction_check is True
        assert agent.log_reconstruction_check_raise is True

        # Also verify agent_init source still contains both stamps
        init_src = Path(__file__).resolve().parents[2] / "agent" / "agent_init.py"
        text = init_src.read_text(encoding="utf-8")
        assert "agent.log_reconstruction_check = bool(" in text
        assert "agent.log_reconstruction_check_raise = bool(" in text
        assert "log_reconstruction_check_raise" in text


# ---------------------------------------------------------------------------
# conversation_loop integration point
# ---------------------------------------------------------------------------


class TestConversationLoopWire:
    def test_hook_present_in_conversation_loop(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
        text = src.read_text(encoding="utf-8")
        assert "maybe_check_before_request" in text
        assert "log_reconstruction_check" in text

    def test_extract_api_history_edges(self):
        assert extract_api_history([]) == []
        assert extract_api_history([{"role": "system", "content": "s"}]) == []
        hist = extract_api_history(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ]
        )
        assert hist == [{"role": "user", "content": "u"}]

    def test_project_turn_prefers_api_content(self):
        t = project_turn(
            {"role": "user", "content": "clean", "api_content": "wire-bytes"}
        )
        t2 = project_turn(
            {"role": "user", "content": "wire-bytes"}, prefer_api_content=False
        )
        assert t["content_fingerprint"] == t2["content_fingerprint"]

    def test_cache_control_shape_fingerprint_matches(self):
        live = [{"role": "user", "content": "hello world"}]
        api = [
            {"role": "system", "content": "s"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello world",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
        report = compare_history_to_api_messages(live, api, current_turn_user_idx=0)
        assert report.ok

    def test_simulate_helper_not_exported_from_core(self):
        import agent.log_reconstruction as mod

        assert not hasattr(mod, "simulate_send_path_api_messages")
