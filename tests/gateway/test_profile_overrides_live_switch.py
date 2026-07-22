"""Tests for live per-source profile switching (``/profile set``).

Covers:
  * gateway.profile_overrides — the persistent switch table (set/clear/
    resolve/list, hierarchical thread > chat).
  * build_source() honouring a live override before profile_routes.
  * _handle_profile_command subcommands (set/status/clears, multiplex gate,
    missing-profile error).

Run under an isolated HERMES_HOME so the real overrides file is untouched.
The isolated home also gets a couple of fake profile dirs (product/
development) so profile_exists() behaves as it would on a real machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway import profile_overrides as po
from gateway.platforms.base import BasePlatformAdapter


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point the override store + profile dirs at a throwaway home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Create fake profile dirs so profile_exists() is realistic.
    for name in ("product", "development"):
        (tmp_path / "profiles" / name).mkdir(parents=True, exist_ok=True)
    p = tmp_path / po._OVERRIDES_FILE
    if p.exists():
        p.unlink()
    yield tmp_path
    if p.exists():
        p.unlink()


@pytest.fixture
def fake_adapter():
    """A BasePlatformAdapter with platform + gateway_runner; real build_source."""
    ad = MagicMock(spec=BasePlatformAdapter)
    ad.platform = MagicMock(value="telegram")
    ad.gateway_runner = None
    # Bind the *real* build_source so the override hook actually runs.
    ad.build_source = BasePlatformAdapter.build_source.__get__(ad)
    return ad


# --------------------------------------------------------------------------
# gateway.profile_overrides unit tests
# --------------------------------------------------------------------------

class TestOverrideStore:
    def test_set_and_resolve_chat(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        assert po.resolve_override("telegram", "chatA") == "product"

    def test_set_and_resolve_thread_wins_over_chat(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        po.set_override("telegram", "chatA", "development", thread_id="t1")
        assert po.resolve_override("telegram", "chatA", "t1") == "development"
        assert po.resolve_override("telegram", "chatA") == "product"

    def test_clear_chat(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        assert po.clear_override("telegram", "chatA") is True
        assert po.resolve_override("telegram", "chatA") is None
        assert po.clear_override("telegram", "chatA") is False

    def test_clear_thread_only(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        po.set_override("telegram", "chatA", "development", thread_id="t1")
        assert po.clear_override("telegram", "chatA", "t1") is True
        assert po.resolve_override("telegram", "chatA") == "product"

    def test_list_roundtrip(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        po.set_override("discord", "g1", "dev", thread_id="th")
        data = po.list_overrides()
        assert data["telegram:chatA"] == "product"
        assert data["discord:g1:th"] == "dev"

    def test_persists_to_disk(self, tmp_path):
        po.set_override("telegram", "chatA", "product")
        raw = (tmp_path / po._OVERRIDES_FILE).read_text()
        assert "product" in raw and "telegram:chatA" in raw

    def test_missing_file_returns_empty(self, tmp_path):
        assert po.resolve_override("telegram", "nope") is None
        assert po.list_overrides() == {}


# --------------------------------------------------------------------------
# build_source() honours the live override
# --------------------------------------------------------------------------

class TestBuildSourceOverride:
    def _make_source(self, adapter, chat_id="chatA", thread_id=None):
        return adapter.build_source(
            chat_id=chat_id,
            chat_type="dm",
            user_id="u1",
            thread_id=thread_id,
        )

    def test_override_wins_over_routes(self, fake_adapter, tmp_path):
        runner = MagicMock()
        runner._profile_name_for_source.return_value = "routed"
        fake_adapter.gateway_runner = runner
        po.set_override("telegram", "chatA", "product")

        src = self._make_source(fake_adapter)
        assert src.profile == "product"
        # routing engine must NOT have been consulted when override exists
        runner._profile_name_for_source.assert_not_called()

    def test_falls_back_to_routes_when_no_override(self, fake_adapter, tmp_path):
        runner = MagicMock()
        runner._profile_name_for_source.return_value = "routed"
        fake_adapter.gateway_runner = runner

        src = self._make_source(fake_adapter)
        assert src.profile == "routed"
        runner._profile_name_for_source.assert_called_once()

    def test_thread_override_scope(self, fake_adapter, tmp_path):
        runner = MagicMock()
        runner._profile_name_for_source.return_value = None
        fake_adapter.gateway_runner = runner
        po.set_override("telegram", "chatA", "development", thread_id="t9")

        chat_only = self._make_source(fake_adapter, thread_id=None)
        assert chat_only.profile is None  # no chat-level override
        threaded = self._make_source(fake_adapter, thread_id="t9")
        assert threaded.profile == "development"


# --------------------------------------------------------------------------
# _handle_profile_command subcommands
# --------------------------------------------------------------------------

class TestProfileCommand:
    def _make_event(self, args="", chat_id="chatA", thread_id=None):
        ev = MagicMock()
        ev.get_command_args.return_value = args
        src = MagicMock()
        src.platform = MagicMock(value="telegram")
        src.chat_id = chat_id
        src.thread_id = thread_id
        src.profile = None
        ev.source = src
        return ev

    def _runner(self, multiplex=True):
        runner = MagicMock()
        runner.config = MagicMock(multiplex_profiles=multiplex)
        from gateway.run import GatewayRunner

        runner._resolve_profile_home_for_source = (
            GatewayRunner._resolve_profile_home_for_source.__get__(runner)
        )
        return runner

    @pytest.mark.asyncio
    async def test_set_requires_multiplex(self):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        runner = self._runner(multiplex=False)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("set product"))
        assert "multiplex_profiles" in out

    @pytest.mark.asyncio
    async def test_set_unknown_profile(self):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("set ghost"))
        assert "does not exist" in out

    @pytest.mark.asyncio
    async def test_set_writes_override(self, tmp_path):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("set product"))
        assert "product" in out
        assert po.resolve_override("telegram", "chatA") == "product"

    @pytest.mark.asyncio
    async def test_set_validation_rejects_bad_name(self):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("set .."))
        assert "Invalid profile name" in out

    @pytest.mark.asyncio
    async def test_clear_removes_override(self, tmp_path):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        po.set_override("telegram", "chatA", "product")
        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("clear"))
        assert "cleared" in out.lower() or "no profile pin" in out.lower()
        assert po.resolve_override("telegram", "chatA") is None

    @pytest.mark.asyncio
    async def test_status_reports_pinned(self, tmp_path):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        po.set_override("telegram", "chatA", "product")
        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event(""))
        assert "product" in out

    @pytest.mark.asyncio
    async def test_list_shows_profiles(self, monkeypatch):
        from gateway.slash_commands import GatewaySlashCommandsMixin
        import hermes_cli.profiles as hp

        monkeypatch.setattr(
            hp, "list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="product")],
        )
        monkeypatch.setattr(hp, "get_active_profile_name", lambda: "default")

        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        out = await mixin._handle_profile_command(self._make_event("list"))
        assert "default" in out

    @pytest.mark.asyncio
    async def test_no_args_opens_picker_and_set(self, tmp_path, monkeypatch):
        """`/profile` (no args) opens the interactive picker; selecting a
        profile performs the live set (parity with /model)."""
        from gateway.slash_commands import GatewaySlashCommandsMixin
        import hermes_cli.profiles as hp

        monkeypatch.setattr(
            hp, "list_profiles",
            lambda: [SimpleNamespace(name="default"), SimpleNamespace(name="product")],
        )
        monkeypatch.setattr(hp, "get_active_profile_name", lambda: "default")

        runner = self._runner(multiplex=True)
        mixin = GatewaySlashCommandsMixin()
        mixin.config = runner.config
        mixin._session_key_for_source = lambda src: "k:chatA"
        captured = {}

        async def fake_send_picker(event, session_key, title, choices, on_choice_selected, metadata=None):
            captured["choices"] = choices
            captured["on_choice_selected"] = on_choice_selected
            captured["title"] = title
            return SimpleNamespace(success=True)

        mixin._try_send_choice_picker = fake_send_picker

        out = await mixin._handle_profile_command(self._make_event(""))
        assert out is None
        names = [c["value"] for c in captured["choices"]]
        assert names == ["default", "product"]
        assert any(c["is_current"] for c in captured["choices"] if c["value"] == "default")

        result = await captured["on_choice_selected"]("chatA", "product")
        assert "product" in result
        assert po.resolve_override("telegram", "chatA") == "product"
