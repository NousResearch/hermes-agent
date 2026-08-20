"""Seam-identity tests for the DiscordGateMixin extraction (adapter.py god-file slice R4-S1).

Verifies the mixin-first class line keeps every C1 gate member bound through
the MRO with zero test edits, plus aggressive behavior cases for the
admission/mention policy and the lazy adapter-helper resolver.
"""

import sys

import pytest

import plugins.platforms.discord.adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.discord_gate_mixin import DiscordGateMixin

GATE_VARS = [
    "DISCORD_ALLOWED_CHANNELS",
    "DISCORD_IGNORED_CHANNELS",
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ALLOWED_ROLES",
    "DISCORD_ALLOW_ALL_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "DISCORD_NO_THREAD_CHANNELS",
    "DISCORD_FREE_RESPONSE_CHANNELS",
    "DISCORD_ALLOW_BOTS",
]


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch):
    """Keep real host gate env vars from leaking into gate assertions."""
    for var in GATE_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in GATE_VARS:
        import os

        os.environ.pop(var, None)


C1_MEMBERS = [
    "_resolve_channel_skills",
    "_resolve_channel_prompt",
    "_discord_require_mention",
    "_discord_allow_any_attachment",
    "_discord_max_attachment_bytes",
    "_is_discord_voice_message_attachment",
    "_snapshot_gate_env",
    "_gate_env",
    "_gate_raw",
    "_gate_csv_set",
    "_get_allowed_channels",
    "_get_ignored_channels",
    "_get_no_thread_channels",
    "_get_allowed_users",
    "_get_allowed_roles",
    "_discord_allow_all_users",
    "_gateway_allow_all_users",
    "_get_allow_bots",
    "_discord_free_response_channels",
    "_raw_mentioned_user_ids",
    "_self_is_explicitly_mentioned",
    "_self_is_raw_mentioned",
    "_discord_bots_require_inline_mention",
    "_discord_channel_keys",
    "_discord_channel_keys_from_channel",
    "_discord_thread_require_mention",
    "_discord_history_backfill",
    "_discord_history_backfill_limit",
]


def _adapter(extra: dict | None = None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.config = type(
        "C", (), {"extra": dict(extra or {})}
    )()  # duck-typed config; gates only touch .extra
    adapter._gate_env_snapshot = None
    return adapter


class TestSeamIdentity:
    """The class line must resolve every C1 member to the mixin (MRO)."""

    def test_adapter_subclasses_mixin(self):
        assert issubclass(DiscordAdapter, DiscordGateMixin)

    def test_all_28_members_identity_bound(self):
        for name in C1_MEMBERS:
            assert getattr(DiscordAdapter, name) is getattr(DiscordGateMixin, name), name

    def test_mixin_has_no_module_level_adapter_import(self):
        # Circular-import guard: the mixin must not import adapter at module
        # level — only indented (in-method) lazy imports are permitted.
        import ast

        tree = ast.parse(open(adapter_mod.__file__.replace("adapter.py", "discord_gate_mixin.py")).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "plugins.platforms.discord.adapter" in (
                node.module or ""
            ):
                assert node.col_offset > 0, f"module-level adapter import at {node.lineno}"
        assert True


class TestGateAllowDeny:
    """Aggressive: per-profile allow/deny gates resolve through the mixin."""

    def test_allowed_channels_gate(self, monkeypatch):
        adapter = _adapter()
        adapter._gate_env_snapshot = {
            "DISCORD_ALLOWED_CHANNELS": "111,222",
            "DISCORD_IGNORED_CHANNELS": "",
            "DISCORD_NO_THREAD_CHANNELS": "",
        }
        assert adapter._get_allowed_channels() == {"111", "222"}
        assert adapter._get_ignored_channels() == set()

    def test_ignored_channels_deny(self, monkeypatch):
        adapter = _adapter({"ignored_channels": ["333", " 444 "]})
        adapter._gate_env_snapshot = {}
        assert adapter._get_ignored_channels() == {"333", "444"}

    def test_allow_all_users_flags(self):
        adapter = _adapter({"allow_all_users": "true"})
        adapter._gate_env_snapshot = {"GATEWAY_ALLOW_ALL_USERS": "1"}
        assert adapter._discord_allow_all_users() is True
        assert adapter._gateway_allow_all_users() is True


class TestMentionPolicy:
    """Aggressive: mention/admission policy helpers behave through the mixin."""

    def test_require_mention_default_true(self, monkeypatch):
        monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
        assert _adapter()._discord_require_mention() is True

    def test_require_mention_extra_false(self):
        adapter = _adapter({"require_mention": "false"})
        assert adapter._discord_require_mention() is False

    def test_self_is_raw_mentioned_requires_client(self):
        adapter = _adapter()
        adapter._client = None
        assert adapter._self_is_raw_mentioned(object()) is False


class TestSenderAuthorizationSeam:
    """Aggressive: base-class authorization check still reachable via MRO."""

    def test_is_sender_authorized_unknown_without_check(self):
        adapter = _adapter()
        adapter._authorization_check = None
        # Resolves to BasePlatformAdapter via MRO — not shadowed by the mixin.
        assert adapter._is_sender_authorized("12345") is None


class TestLazyHelperResolver:
    """The mixin must resolve adapter-module helpers lazily, at call time."""

    def test_snapshot_gate_env_uses_live_adapter_helper(self, monkeypatch):
        calls = []

        def fake_scoped(name):
            calls.append(name)
            return "777"

        monkeypatch.setattr(adapter_mod, "_scoped_gate_env", fake_scoped)
        adapter = _adapter()
        adapter._snapshot_gate_env()
        assert calls == list(adapter_mod._GATE_ENV_KEYS)
        assert adapter._gate_env_snapshot
        assert all(v == "777" for v in adapter._gate_env_snapshot.values())

    def test_gate_env_falls_back_to_live_helper(self, monkeypatch):
        monkeypatch.setattr(adapter_mod, "_scoped_gate_env", lambda name, default="": "live")
        adapter = _adapter()
        # Key absent from snapshot -> fallback to the live adapter helper.
        adapter._gate_env_snapshot = {"SOME_OTHER_KEY": ""}
        assert adapter._gate_env("SOME_KEY") == "live"

    def test_clean_discord_id_resolved_from_adapter(self, monkeypatch):
        monkeypatch.setattr(adapter_mod, "_clean_discord_id", lambda s: s.strip().lstrip("@"))
        adapter = _adapter({"allow_from": "@user123, @user456"})
        assert adapter._get_allowed_users() == {"user123", "user456"}


class TestImportabilityWithoutDiscord:
    """The mixin module must import with the discord library blocked."""

    def test_mixin_imports_without_discord(self, monkeypatch):
        for modname in list(sys.modules):
            if modname == "discord" or modname.startswith("discord."):
                monkeypatch.setitem(sys.modules, modname, None)
        import importlib

        module = importlib.import_module("plugins.platforms.discord.discord_gate_mixin")
        assert module.DiscordGateMixin is DiscordGateMixin
