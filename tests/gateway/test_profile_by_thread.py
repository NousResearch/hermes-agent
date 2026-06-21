"""Tests for Telegram topic→profile routing on live gateway turns.

A Telegram thread/topic can be mapped to a named Hermes profile via
``telegram.profile_by_thread``.  When a live (non-Kanban) turn arrives in
that topic, the gateway resolves:

  * reasoning effort  — from the mapped profile's own ``config.yaml``
                        (single source of truth lives in the profile)
  * identity / role   — from the mapped profile's ``SOUL.md``, injected as an
                        authoritative role overlay into the ephemeral prompt

Memory, USER profile, sessions and skills stay shared from the default home —
this is deliberately a *behaviour* overlay (Option B), not a HERMES_HOME swap.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_event_source(
    platform=Platform.TELEGRAM,
    user_id="12345",
    chat_id="-100999",
    thread_id=None,
):
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        thread_id=thread_id,
    )


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _seed_profile(hermes_home, name, *, effort=None, soul=None):
    """Create profiles/<name>/ with optional config.yaml + SOUL.md."""
    pdir = hermes_home / "profiles" / name
    pdir.mkdir(parents=True, exist_ok=True)
    if effort is not None:
        (pdir / "config.yaml").write_text(
            f"agent:\n  reasoning_effort: {effort}\n", encoding="utf-8"
        )
    if soul is not None:
        (pdir / "SOUL.md").write_text(soul, encoding="utf-8")
    return pdir


def _write_root_config(hermes_home, body):
    (hermes_home / "config.yaml").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# _load_source_profile — thread → profile mapping
# ---------------------------------------------------------------------------


class TestLoadSourceProfile:
    def test_maps_chat_and_thread_key(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '-100999:2': engineer\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(chat_id="-100999", thread_id="2")
        assert runner._load_source_profile(source) == "engineer"

    def test_maps_bare_thread_key(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '107': task-manager\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(chat_id="-100999", thread_id="107")
        assert runner._load_source_profile(source) == "task-manager"

    def test_chat_thread_key_beats_bare_thread(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '-100999:2': engineer\n"
            "    '2': admin\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(chat_id="-100999", thread_id="2")
        assert runner._load_source_profile(source) == "engineer"

    def test_returns_none_when_unmapped(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(chat_id="-100999", thread_id="999")
        assert runner._load_source_profile(source) is None

    def test_returns_none_for_non_telegram(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(platform=Platform.LOCAL, chat_id="cli", thread_id="2")
        assert runner._load_source_profile(source) is None

    def test_accepts_profile_overrides_alias(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "telegram:\n"
            "  profile_overrides:\n"
            "    '2': engineer\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(chat_id="-100999", thread_id="2")
        assert runner._load_source_profile(source) == "engineer"


# ---------------------------------------------------------------------------
# reasoning effort resolves from the mapped profile's own config
# ---------------------------------------------------------------------------


class TestProfileReasoning:
    def test_effort_comes_from_profile_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "agent:\n"
            "  reasoning_effort: medium\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        _seed_profile(hermes_home, "engineer", effort="xhigh")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(thread_id="2")
        assert runner._resolve_session_reasoning_config(source=source) == {
            "enabled": True,
            "effort": "xhigh",
        }

    def test_unmapped_thread_uses_global(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "agent:\n"
            "  reasoning_effort: medium\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        _seed_profile(hermes_home, "engineer", effort="xhigh")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(thread_id="999")
        assert runner._resolve_session_reasoning_config(source=source) == {
            "enabled": True,
            "effort": "medium",
        }

    def test_session_override_beats_profile(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "agent:\n"
            "  reasoning_effort: medium\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        _seed_profile(hermes_home, "engineer", effort="xhigh")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(thread_id="2")
        session_key = runner._session_key_for_source(source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "low"}
        assert runner._resolve_session_reasoning_config(source=source) == {
            "enabled": True,
            "effort": "low",
        }

    def test_profile_without_effort_falls_through_to_global(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "agent:\n"
            "  reasoning_effort: high\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        # profile dir exists but no reasoning_effort set
        _seed_profile(hermes_home, "engineer", soul="role")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        source = _make_event_source(thread_id="2")
        assert runner._resolve_session_reasoning_config(source=source) == {
            "enabled": True,
            "effort": "high",
        }


# ---------------------------------------------------------------------------
# SOUL overlay loader
# ---------------------------------------------------------------------------


class TestProfileSoul:
    def test_loads_profile_soul(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _seed_profile(hermes_home, "engineer", soul="You are the Engineer worker.")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        assert runner._load_profile_soul("engineer") == "You are the Engineer worker."

    def test_missing_soul_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _seed_profile(hermes_home, "engineer", effort="xhigh")  # no SOUL.md
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        assert runner._load_profile_soul("engineer") is None


# ---------------------------------------------------------------------------
# End-to-end: live turn picks up profile effort + SOUL overlay
# ---------------------------------------------------------------------------


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {"final_response": "ok", "messages": [], "api_calls": 1}


class TestRunAgentProfileOverlay:
    def _patch_runtime(self, monkeypatch, hermes_home):
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *a, **k: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
        _CapturingAgent.last_init = None

    def test_live_turn_uses_profile_effort_and_soul(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "model:\n"
            "  default: test-model\n"
            "agent:\n"
            "  reasoning_effort: medium\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        _seed_profile(
            hermes_home,
            "engineer",
            effort="xhigh",
            soul="ENGINEER ROLE: maximum rigor.",
        )
        self._patch_runtime(monkeypatch, hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        source = _make_event_source(thread_id="2")

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:telegram:-100999:2",
            )
        )

        assert result["final_response"] == "ok"
        init = _CapturingAgent.last_init
        assert init is not None
        # profile effort applied
        assert init["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
        # profile SOUL injected into ephemeral overlay
        assert "ENGINEER ROLE: maximum rigor." in (init["ephemeral_system_prompt"] or "")

    def test_unmapped_thread_no_overlay(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        _write_root_config(
            hermes_home,
            "model:\n"
            "  default: test-model\n"
            "agent:\n"
            "  reasoning_effort: medium\n"
            "telegram:\n"
            "  profile_by_thread:\n"
            "    '2': engineer\n",
        )
        _seed_profile(
            hermes_home,
            "engineer",
            effort="xhigh",
            soul="ENGINEER ROLE: maximum rigor.",
        )
        self._patch_runtime(monkeypatch, hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        source = _make_event_source(thread_id="999")

        asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:telegram:-100999:999",
            )
        )

        init = _CapturingAgent.last_init
        assert init["reasoning_config"] == {"enabled": True, "effort": "medium"}
        assert "ENGINEER ROLE" not in (init["ephemeral_system_prompt"] or "")
