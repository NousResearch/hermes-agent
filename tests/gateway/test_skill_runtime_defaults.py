"""Gateway-side tests for skill runtime defaults (Tasks 7 and 8).

Anchors:
  - tests/gateway/test_plan_command.py:1-100  (_make_runner / _make_event)
  - tests/gateway/test_session_race_guard.py:187-211 (merge fixtures)
  - tests/gateway/test_agent_cache.py:193-214 (reasoning refresh pattern)
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.skill_commands import scan_skill_commands
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.session import SessionEntry, SessionSource


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = {"enabled": True, "effort": "high"}
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._smart_model_routing = {}
    runner._service_tier = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


def _make_event(text="/plan", **kwargs):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m1",
        **kwargs,
    )


def _make_brainstorm_skill(skills_dir, body="Brainstorm loudly.", extra=""):
    skill_dir = skills_dir / "brainstorm"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: brainstorm
description: Brainstorm skill.
{extra}---

# Brainstorm

{body}
"""
    )


# ── Task 7: MessageEvent plumbing + merge rule ────────────────────────────


class TestMessageEventTurnRuntimeDefaultsField:
    def test_default_is_none(self):
        event = _make_event()
        assert event.turn_runtime_defaults is None

    def test_can_be_populated(self):
        event = _make_event(
            turn_runtime_defaults={"reasoning_effort": "low"}
        )
        assert event.turn_runtime_defaults == {"reasoning_effort": "low"}


class TestMergePendingMessageEventRuntimeDefaults:
    def test_existing_has_defaults_new_has_none_preserves(self):
        """Photo-burst case: first event carried the skill; follow-on photos
        must inherit defaults instead of silently losing them."""
        store = {}
        existing = _make_event(
            text="",
            message_type=MessageType.PHOTO,
            turn_runtime_defaults={"reasoning_effort": "low"},
        )
        existing.media_urls = ["a.jpg"]
        existing.media_types = ["image/jpeg"]
        store["sk"] = existing

        incoming = _make_event(text="", message_type=MessageType.PHOTO)
        incoming.media_urls = ["b.jpg"]
        incoming.media_types = ["image/jpeg"]
        merge_pending_message_event(store, "sk", incoming)

        assert store["sk"].turn_runtime_defaults == {"reasoning_effort": "low"}
        assert store["sk"].media_urls == ["a.jpg", "b.jpg"]

    def test_new_has_defaults_existing_has_none_copies_over(self):
        """Non-skill event first, then skill event → skill defaults attach to
        the combined photo burst."""
        store = {}
        existing = _make_event(text="", message_type=MessageType.PHOTO)
        existing.media_urls = ["a.jpg"]
        existing.media_types = ["image/jpeg"]
        store["sk"] = existing

        incoming = _make_event(
            text="",
            message_type=MessageType.PHOTO,
            turn_runtime_defaults={"reasoning_effort": "low"},
        )
        incoming.media_urls = ["b.jpg"]
        incoming.media_types = ["image/jpeg"]
        merge_pending_message_event(store, "sk", incoming)

        assert store["sk"].turn_runtime_defaults == {"reasoning_effort": "low"}

    def test_both_agree_merges_trivially(self):
        store = {}
        defaults = {"reasoning_effort": "low"}
        existing = _make_event(
            text="", message_type=MessageType.PHOTO, turn_runtime_defaults=defaults
        )
        existing.media_urls = ["a.jpg"]
        existing.media_types = ["image/jpeg"]
        store["sk"] = existing
        incoming = _make_event(
            text="", message_type=MessageType.PHOTO, turn_runtime_defaults=dict(defaults)
        )
        incoming.media_urls = ["b.jpg"]
        incoming.media_types = ["image/jpeg"]
        merge_pending_message_event(store, "sk", incoming)
        assert store["sk"].turn_runtime_defaults == defaults

    def test_text_burst_carries_runtime_defaults(self):
        """Skill event followed by text-burst text-only event (merge_text=True)
        must keep the skill's runtime defaults."""
        store = {}
        existing = _make_event(
            text="I want to",
            turn_runtime_defaults={"reasoning_effort": "low"},
        )
        store["sk"] = existing
        incoming = _make_event(text="brainstorm this")
        merge_pending_message_event(store, "sk", incoming, merge_text=True)
        assert store["sk"].turn_runtime_defaults == {"reasoning_effort": "low"}
        assert "brainstorm" in store["sk"].text

    def test_replace_path_with_latest_defaults_wins(self):
        """No merge criteria hit → straight replace at line 787.  When both
        events declare defaults and they differ, the latest wins (the incoming
        event becomes the new stored event)."""
        store = {}
        existing = _make_event(
            text="hi", turn_runtime_defaults={"reasoning_effort": "low"}
        )
        store["sk"] = existing
        incoming = _make_event(
            text="hello", turn_runtime_defaults={"reasoning_effort": "medium"}
        )
        merge_pending_message_event(store, "sk", incoming)
        # Replace path: incoming event stored, its defaults survive.
        assert store["sk"].turn_runtime_defaults == {"reasoning_effort": "medium"}


class TestGatewaySkillCommandSetsTurnRuntimeDefaults:
    @pytest.mark.asyncio
    async def test_plan_command_sets_runtime_defaults_on_event(
        self, monkeypatch, tmp_path
    ):
        import gateway.run as gateway_run

        runner = _make_runner()
        event = _make_event("/plan Add OAuth")

        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {"api_key": "***"},
        )
        monkeypatch.setattr(
            "agent.model_metadata.get_model_context_length",
            lambda *_a, **_k: 100_000,
        )
        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path),
            patch(
                "agent.skill_utils._runtime_defaults_flag_enabled",
                return_value=True,
            ),
        ):
            _make_brainstorm_skill(
                tmp_path,
                extra=(
                    "metadata:\n"
                    "  hermes:\n"
                    "    runtime_defaults:\n"
                    "      reasoning_effort: low\n"
                ),
            )
            # Overwrite the plan skill with a runtime_defaults variant so
            # /plan routes through build_skill_invocation_payload and sets
            # event.turn_runtime_defaults.
            plan_dir = tmp_path / "plan"
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "SKILL.md").write_text(
                "---\nname: plan\ndescription: Plan.\n"
                "metadata:\n  hermes:\n    runtime_defaults:\n      reasoning_effort: low\n"
                "---\n\nPlan body.\n"
            )
            scan_skill_commands()
            await runner._handle_message(event)

        kwargs = runner._run_agent.call_args.kwargs
        # skill_name is carried inside runtime_defaults so the merger can
        # emit a non-empty skill_name_sanitized in structured log events.
        assert kwargs.get("turn_runtime_defaults") == {
            "reasoning_effort": "low",
            "reasoning_config": {"enabled": True, "effort": "low"},
            "required": [],
            "skill_name": "plan",
        }


# ── Task 8: reasoning override applied on cached agent ───────────────────


class TestResolveTurnAgentConfigWithSkillDefaults:
    def test_skill_reasoning_applied_on_session_with_higher_default(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._smart_model_routing = {}
        runner._service_tier = None
        runner._reasoning_config = {"enabled": True, "effort": "high"}

        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value={
                "model": "m",
                "runtime": {},
                "label": None,
                "signature": ("sig",),
            },
        ):
            route = GatewayRunner._resolve_turn_agent_config(
                runner,
                "hi",
                "m",
                {"api_key": "k"},
                skill_runtime_defaults={
                    "reasoning_effort": "low",
                    "reasoning_config": {"enabled": True, "effort": "low"},
                    "required": [],
                },
            )
        assert route["skill_reasoning_config"] == {
            "enabled": True,
            "effort": "low",
        }
        assert route["hard_fail"] is None

    def test_skill_reasoning_clamped_when_above_session_default(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._smart_model_routing = {}
        runner._service_tier = None
        runner._reasoning_config = {"enabled": True, "effort": "low"}

        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value={
                "model": "m",
                "runtime": {},
                "label": None,
                "signature": ("sig",),
            },
        ):
            route = GatewayRunner._resolve_turn_agent_config(
                runner,
                "hi",
                "m",
                {"api_key": "k"},
                skill_runtime_defaults={
                    "reasoning_effort": "high",
                    "reasoning_config": {"enabled": True, "effort": "high"},
                    "required": [],
                },
            )
        # Clamp kicked in.
        assert route["skill_reasoning_config"] is None

    def test_required_reasoning_hard_fails_when_clamped(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._smart_model_routing = {}
        runner._service_tier = None
        runner._reasoning_config = {"enabled": True, "effort": "low"}

        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value={
                "model": "m",
                "runtime": {},
                "label": None,
                "signature": ("sig",),
            },
        ):
            route = GatewayRunner._resolve_turn_agent_config(
                runner,
                "hi",
                "m",
                {"api_key": "k"},
                skill_runtime_defaults={
                    "reasoning_effort": "high",
                    "reasoning_config": {"enabled": True, "effort": "high"},
                    "required": ["reasoning_effort"],
                },
            )
        assert route["hard_fail"] is not None
        assert route["hard_fail"].failing_field == "reasoning_effort"

    def test_no_skill_defaults_returns_normal_route_backward_compat(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._smart_model_routing = {}
        runner._service_tier = None
        runner._reasoning_config = None

        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value={
                "model": "m",
                "runtime": {},
                "label": None,
                "signature": ("sig",),
            },
        ):
            route = GatewayRunner._resolve_turn_agent_config(
                runner, "hi", "m", {"api_key": "k"}
            )
        assert route.get("hard_fail") is None
        assert route.get("skill_reasoning_config") is None
