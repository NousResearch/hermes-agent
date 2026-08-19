"""GATEWAY_MESSAGING_GUIDANCE scoping in the assembled system prompt.

The guidance block (cannot self-restart the gateway; reply gating is
config-enforced, e.g. ``native_mention_only_channels``) must be appended
exactly once after the platform hint for gateway messaging sessions
(Slack, Telegram, ...), and never for local/one-shot surfaces (cli, tui,
cron, desktop, api_server, webui) or unknown/empty platforms.

These tests run against the real prompt builders (no mocks of the
resolver) because the gating lives at the assembly site: it keys off the
built-in *default* hint plus the platform key, so a config
``platform_hints`` override must not strip the guidance.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_builder import (
    GATEWAY_MESSAGING_GUIDANCE,
    NON_MESSAGING_PLATFORM_KEYS,
    PLATFORM_HINTS,
)
from agent.system_prompt import build_system_prompt_parts


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _make_agent(platform="", **overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _platform_hint_overrides={},
        model="",
        provider="",
        pass_session_id=False,
        session_id="",
    )
    base["platform"] = platform
    base.update(overrides)
    return SimpleNamespace(**base)


class TestMessagingPlatformsGetGuidance:
    """Gateway messaging sessions carry the operational-constraints block,
    positioned after their platform hint."""

    @pytest.mark.parametrize("platform", ["slack", "telegram"])
    def test_guidance_present_for_messaging_platform(self, platform):
        stable = _stable_prompt(_make_agent(platform=platform))
        assert GATEWAY_MESSAGING_GUIDANCE in stable

    @pytest.mark.parametrize("platform", ["slack", "telegram"])
    def test_guidance_follows_platform_hint(self, platform):
        stable = _stable_prompt(_make_agent(platform=platform))
        hint = PLATFORM_HINTS[platform]
        assert hint in stable
        assert stable.index(GATEWAY_MESSAGING_GUIDANCE) > stable.index(hint)

    @pytest.mark.parametrize("platform", ["slack", "telegram"])
    def test_guidance_appears_exactly_once(self, platform):
        stable = _stable_prompt(_make_agent(platform=platform))
        assert stable.count(GATEWAY_MESSAGING_GUIDANCE) == 1


class TestNonMessagingSurfacesNeverSeeGuidance:
    """cli/tui/cron/desktop have PLATFORM_HINTS entries but are local or
    one-shot surfaces — the gateway constraints do not apply and their
    prompt bytes must be unchanged by this feature."""

    @pytest.mark.parametrize("platform", ["cli", "tui", "cron", "desktop"])
    def test_guidance_absent_for_local_surface(self, platform):
        assert platform in NON_MESSAGING_PLATFORM_KEYS
        stable = _stable_prompt(_make_agent(platform=platform))
        assert GATEWAY_MESSAGING_GUIDANCE not in stable

    def test_guidance_absent_for_empty_platform(self):
        stable = _stable_prompt(_make_agent(platform=""))
        assert GATEWAY_MESSAGING_GUIDANCE not in stable

    def test_guidance_absent_for_unknown_platform(self):
        """A platform with no built-in hint and no plugin hint has no
        default hint, so the guidance gate must stay closed."""
        stable = _stable_prompt(_make_agent(platform="no_such_platform"))
        assert GATEWAY_MESSAGING_GUIDANCE not in stable


class TestByteStability:
    """The guidance is static text; its gate reads platform_key (fixed at
    agent construction) and _default_hint (re-resolved each build but
    deterministic for the life of the process) — rebuilding the prompt for
    the same agent must yield byte-identical text (prompt-cache contract)."""

    @pytest.mark.parametrize("platform", ["slack", "telegram", "cli"])
    def test_rebuild_is_byte_identical(self, platform):
        agent = _make_agent(platform=platform)
        first = _stable_prompt(agent)
        second = _stable_prompt(agent)
        assert first == second


class TestGuidanceSurvivesPlatformHintOverride:
    """The gate keys off the *default* hint, not the config-overridable
    effective hint — a ``platform_hints.slack`` replace-override swaps the
    hint text but must NOT strip the gateway constraints."""

    def test_replace_override_keeps_guidance(self):
        agent = _make_agent(
            platform="slack",
            _platform_hint_overrides={
                "slack": {"replace": "Custom replacement slack hint."}
            },
        )
        stable = _stable_prompt(agent)
        assert "Custom replacement slack hint." in stable
        assert PLATFORM_HINTS["slack"] not in stable
        assert GATEWAY_MESSAGING_GUIDANCE in stable

    def test_append_override_keeps_guidance(self):
        agent = _make_agent(
            platform="telegram",
            _platform_hint_overrides={"telegram": "Extra appended note."},
        )
        stable = _stable_prompt(agent)
        assert PLATFORM_HINTS["telegram"] in stable
        assert "Extra appended note." in stable
        assert GATEWAY_MESSAGING_GUIDANCE in stable

    def test_override_on_local_surface_does_not_add_guidance(self):
        """An override for a non-messaging surface changes only the hint —
        it must not smuggle the gateway constraints into a cli session."""
        agent = _make_agent(
            platform="cli",
            _platform_hint_overrides={"cli": {"replace": "Custom cli hint."}},
        )
        stable = _stable_prompt(agent)
        assert "Custom cli hint." in stable
        assert GATEWAY_MESSAGING_GUIDANCE not in stable
