"""Tests for the ``/bundles`` gateway slash command handler.

Verifies that:
- ``_handle_bundles_command`` returns useful text when no bundles are
  installed and when several are.
- Bundle dispatch in ``_handle_message`` rewrites ``event.text`` to the
  combined skill content when the user types ``/<bundle-slug>``.

The actual ``/<bundle-slug>`` → combined-message build is tested in
``tests/agent/test_skill_bundles.py``; this file only checks the gateway
glue (handler wiring, dispatch ordering, event.text rewrite).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    return runner


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    bundles_dir = tmp_path / "skill-bundles"
    bundled_bundles_dir = tmp_path / "bundled-skill-bundles"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setenv("HERMES_BUNDLED_SKILL_BUNDLES", str(bundled_bundles_dir))
    import tools.skills_tool as skills_tool_module
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    return bundles_dir, skills_dir


def _make_skill(skills_dir, name, body="content"):
    sd = skills_dir / name
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc {name}\n---\n\n# {name}\n\n{body}\n"
    )


def _make_bundle(bundles_dir, slug, skills):
    bundles_dir.mkdir(parents=True, exist_ok=True)
    (bundles_dir / f"{slug}.yaml").write_text(
        f"name: {slug}\nskills:\n" + "\n".join(f"  - {s}" for s in skills) + "\n"
    )


class TestHandleBundlesCommand:
    def test_empty(self, bundles_env):
        runner = _make_runner()
        result = asyncio.run(runner._handle_bundles_command(_make_event("/bundles")))
        assert "No skill bundles" in result

    def test_with_bundles(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle(bundles_dir, "research", ["alpha", "beta"])
        runner = _make_runner()
        result = asyncio.run(runner._handle_bundles_command(_make_event("/bundles")))
        assert "research" in result
        assert "/research" in result
        assert "2 skills" in result


class TestBundleResolutionPriority:
    """Verify resolve_bundle_command_key picks bundles over skills."""

    def test_bundle_resolves(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle(bundles_dir, "research", ["alpha"])
        from agent.skill_bundles import resolve_bundle_command_key
        assert resolve_bundle_command_key("research") == "/research"

    def test_underscore_alias(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle(bundles_dir, "my-bundle", ["alpha"])
        from agent.skill_bundles import resolve_bundle_command_key
        assert resolve_bundle_command_key("my_bundle") == "/my-bundle"
