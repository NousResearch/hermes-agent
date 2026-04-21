"""Tests for CLI skill runtime defaults plumbing.

Covers Task 5 (structured CLI queue) and Task 6 (per-turn agent mutation).
Anchors:
  - tests/cli/test_cli_plan_command.py:9-68 (CLI helpers)
  - tests/cli/test_fast_command.py:195-247  (route patching)
  - tests/gateway/test_agent_cache.py:193-214 (reasoning_config refresh)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.skill_commands import scan_skill_commands
from cli import HermesCLI, _normalize_pending_turn


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "sess-123"
    cli_obj._pending_input = MagicMock()
    cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
    cli_obj.service_tier = None
    return cli_obj


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


# ── Task 5 normalizer tests ──────────────────────────────────────────────


class TestNormalizePendingTurn:
    def test_plain_string(self):
        turn = _normalize_pending_turn("hello world")
        assert turn == {
            "text": "hello world",
            "images": [],
            "runtime_defaults": {},
            "skill_name": "",
        }

    def test_tuple_with_images(self):
        turn = _normalize_pending_turn(("caption", ["img1.png"]))
        assert turn["text"] == "caption"
        assert turn["images"] == ["img1.png"]
        assert turn["runtime_defaults"] == {}

    def test_structured_skill_payload(self):
        payload = {
            "message": "rendered skill body",
            "skill_name": "brainstorm",
            "runtime_defaults": {
                "reasoning_effort": "low",
                "reasoning_config": {"enabled": True, "effort": "low"},
                "required": [],
            },
        }
        turn = _normalize_pending_turn(payload)
        assert turn["text"] == "rendered skill body"
        assert turn["skill_name"] == "brainstorm"
        assert turn["runtime_defaults"]["reasoning_effort"] == "low"

    def test_already_normalized_shape(self):
        payload = {
            "text": "plain-again",
            "images": ["x.png"],
            "runtime_defaults": {},
            "skill_name": "",
        }
        turn = _normalize_pending_turn(payload)
        assert turn["text"] == "plain-again"
        assert turn["images"] == ["x.png"]

    def test_unknown_shape_coerced_with_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            turn = _normalize_pending_turn(12345)
        assert turn["text"] == "12345"
        assert turn["runtime_defaults"] == {}
        assert any(
            "Unexpected _pending_input shape" in rec.message
            for rec in caplog.records
        )

    def test_empty_dict_is_safe(self):
        turn = _normalize_pending_turn({})
        assert turn["text"] == ""
        assert turn["runtime_defaults"] == {}


# ── Task 5 skill enqueue shape tests ─────────────────────────────────────


class TestSkillSlashEnqueuesStructuredPayload:
    def test_skill_command_enqueues_dict_with_runtime_defaults(
        self, tmp_path, monkeypatch
    ):
        import cli as cli_mod

        cli_obj = _make_cli()
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
            scanned = scan_skill_commands()
            monkeypatch.setattr(cli_mod, "_skill_commands", scanned)
            cli_obj.process_command("/brainstorm Figure this out")

        cli_obj._pending_input.put.assert_called_once()
        queued = cli_obj._pending_input.put.call_args[0][0]
        assert isinstance(queued, dict)
        assert "brainstorm" in queued["message"]
        assert queued["skill_name"] == "brainstorm"
        assert queued["runtime_defaults"]["reasoning_effort"] == "low"

    def test_skill_without_runtime_defaults_enqueues_empty_defaults(
        self, tmp_path, monkeypatch
    ):
        import cli as cli_mod

        cli_obj = _make_cli()
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_brainstorm_skill(tmp_path)
            scanned = scan_skill_commands()
            monkeypatch.setattr(cli_mod, "_skill_commands", scanned)
            cli_obj.process_command("/brainstorm")

        queued = cli_obj._pending_input.put.call_args[0][0]
        assert isinstance(queued, dict)
        assert queued["runtime_defaults"] == {}


# ── Task 6 per-turn mutation tests ───────────────────────────────────────


class TestResolveTurnAgentConfigWithSkillDefaults:
    def _stub_cli(self):
        return SimpleNamespace(
            model="anthropic/claude-opus-4.6",
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            _credential_pool=None,
            _smart_model_routing={},
            service_tier=None,
            reasoning_config={"enabled": True, "effort": "high"},
        )

    def test_skill_reasoning_applied_when_below_session_default(self):
        stub = self._stub_cli()
        primary_route = {
            "model": "anthropic/claude-opus-4.6",
            "runtime": {},
            "label": None,
            "signature": ("sig",),
        }
        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value=primary_route,
        ):
            route = HermesCLI._resolve_turn_agent_config(
                stub,
                "hi",
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

    def test_skill_reasoning_clamped_when_higher_than_session(self):
        stub = self._stub_cli()
        stub.reasoning_config = {"enabled": True, "effort": "low"}
        primary_route = {
            "model": "m",
            "runtime": {},
            "label": None,
            "signature": ("sig",),
        }
        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value=primary_route,
        ):
            route = HermesCLI._resolve_turn_agent_config(
                stub,
                "hi",
                skill_runtime_defaults={
                    "reasoning_effort": "high",
                    "reasoning_config": {"enabled": True, "effort": "high"},
                    "required": [],
                },
            )
        # Clamp kicked in: no skill reasoning attached.
        assert route["skill_reasoning_config"] is None
        assert route["hard_fail"] is None

    def test_required_reasoning_hard_fails_when_clamped(self):
        stub = self._stub_cli()
        stub.reasoning_config = {"enabled": True, "effort": "low"}
        primary_route = {
            "model": "m",
            "runtime": {},
            "label": None,
            "signature": ("sig",),
        }
        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value=primary_route,
        ):
            route = HermesCLI._resolve_turn_agent_config(
                stub,
                "hi",
                skill_runtime_defaults={
                    "reasoning_effort": "high",
                    "reasoning_config": {"enabled": True, "effort": "high"},
                    "required": ["reasoning_effort"],
                },
            )
        assert route["hard_fail"] is not None
        assert route["hard_fail"].failing_field == "reasoning_effort"

    def test_no_skill_defaults_is_backward_compatible(self):
        stub = self._stub_cli()
        primary_route = {
            "model": "m",
            "runtime": {},
            "label": None,
            "signature": ("sig",),
        }
        with patch(
            "agent.smart_model_routing.resolve_turn_route",
            return_value=primary_route,
        ):
            route = HermesCLI._resolve_turn_agent_config(stub, "hi")
        assert route.get("hard_fail") is None
        assert route.get("skill_reasoning_config") is None


class TestApplyPerTurnAgentMutation:
    def test_skill_reasoning_applied_to_reused_agent_identity_preserved(self):
        """test_reasoning_only_skill_override_mutates_cached_cli_agent_identity_preserved"""
        agent = SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "medium"},
            service_tier=None,
            request_overrides=None,
        )
        stub = SimpleNamespace(
            agent=agent,
            reasoning_config={"enabled": True, "effort": "medium"},
            service_tier=None,
        )
        turn_route = {
            "skill_reasoning_config": {"enabled": True, "effort": "low"},
            "request_overrides": None,
        }
        # Call the classmethod indirectly via the function's descriptor.
        HermesCLI._apply_per_turn_agent_mutation(stub, turn_route)
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
        # Identity preserved — no rebuild.
        assert stub.agent is agent

    def test_next_plain_turn_reverts_to_session_default(self):
        """Plain turn's route has no skill_reasoning_config; mutation restores
        session default so the effect of the previous skill turn decays."""
        agent = SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "low"},
            service_tier=None,
            request_overrides=None,
        )
        stub = SimpleNamespace(
            agent=agent,
            reasoning_config={"enabled": True, "effort": "medium"},
            service_tier=None,
        )
        turn_route = {
            "skill_reasoning_config": None,
            "request_overrides": None,
        }
        HermesCLI._apply_per_turn_agent_mutation(stub, turn_route)
        assert agent.reasoning_config == {"enabled": True, "effort": "medium"}

    def test_mutation_noop_when_agent_is_none(self):
        stub = SimpleNamespace(
            agent=None,
            reasoning_config={"enabled": True, "effort": "medium"},
            service_tier=None,
        )
        # Must not raise even when there is no agent yet.
        HermesCLI._apply_per_turn_agent_mutation(
            stub, {"skill_reasoning_config": None, "request_overrides": None}
        )
