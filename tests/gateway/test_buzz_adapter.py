"""Tests for the Buzz platform adapter plugin."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/buzz/adapter.py under a unique module name
# (plugin_adapter_buzz) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same xdist worker.
_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter
hex_to_npub = _buzz_mod.hex_to_npub
npub_to_hex = _buzz_mod.npub_to_hex
_normalize_user_ref = _buzz_mod._normalize_user_ref
_cli_error_message = _buzz_mod._cli_error_message
_resolve_private_key = _buzz_mod._resolve_private_key
_resolve_auth_tag = _buzz_mod._resolve_auth_tag
check_requirements = _buzz_mod.check_requirements
validate_config = _buzz_mod.validate_config
register = _buzz_mod.register
_env_enablement = _buzz_mod._env_enablement
_standalone_send = _buzz_mod._standalone_send

# Real key pair (Chip's public identity — public information, not a secret)
SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
OTHER_PUBKEY = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
# Real DM conversation as materialized by a hosted relay: `dms list` returns
# [] for it (#68871) while `channels list` shows it as name "DM", empty
# description, indistinguishable from a channel except via message p-tags.
DM_CHANNEL = "6468cc16-a114-4f23-8b8c-02c1655cbf6b"

_ENV_VARS = (
    "BUZZ_RELAY_URL",
    "BUZZ_PRIVATE_KEY",
    "BUZZ_CHANNELS",
    "BUZZ_HOME_CHANNEL",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "BUZZ_POLL_INTERVAL",
    "BUZZ_AUTH_TAG",
    "BUZZ_CLI_PATH",
    "BUZZ_CREDENTIALS_FILE",
    "BUZZ_AUTH_TAG",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Keep tests hermetic: no ambient Buzz env vars or real credentials."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _event(event_id, pubkey=OTHER_PUBKEY, content="hello", created_at=1000, kind=9):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": [["h", CHANNEL]],
    }


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = SELF_NPUB
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    return adapter


class _ScriptedCli:
    """Fake ``_run_cli`` that routes on the buzz subcommand and records calls."""

    def __init__(self):
        self.responses = {}  # (group, cmd) -> list of (code, stdout, stderr)
        self.calls = []

    def script(self, group, cmd, payload, code=0, stderr=""):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        self.responses.setdefault((group, cmd), []).append((code, stdout, stderr))

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        queue = self.responses.get((args[0], args[1]), [])
        if len(queue) > 1:
            return queue.pop(0)
        if queue:
            return queue[0]
        return 0, "[]", ""


# ── bech32 / identity helpers ─────────────────────────────────────────────


class TestBech32Helpers:

    def test_hex_to_npub_known_pair(self):
        assert hex_to_npub(SELF_PUBKEY) == SELF_NPUB

    def test_npub_to_hex_known_pair(self):
        assert npub_to_hex(SELF_NPUB) == SELF_PUBKEY


# ── Adapter init / config precedence ──────────────────────────────────────


class TestBuzzAdapterInit:


    def test_init_from_config_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://cfg.relay",
                "channels": ["ccc"],
                "poll_interval": 2,
                "home_channel": "ccc",
            },
        )
        adapter = BuzzAdapter(cfg)
        assert adapter.relay_url == "https://cfg.relay"
        assert adapter.channels == ["ccc"]
        assert adapter.poll_interval == 2.0
        assert adapter.home_channel == "ccc"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://env.relay")
        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://cfg.relay"}))
        assert adapter.relay_url == "https://env.relay"


# ── Multiplex secondary-profile scope (#98738) ─────────────────────────────


@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""

    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("BUZZ_RELAY_URL", "https://default.relay")
    monkeypatch.setenv("BUZZ_CHANNELS", "chan-a,chan-b,chan-c")
    monkeypatch.setenv("BUZZ_HOME_CHANNEL", "chan-a")
    monkeypatch.setenv("BUZZ_POLL_INTERVAL", "9")
    monkeypatch.setenv("BUZZ_CLI_PATH", "/default/bin/buzz")
    monkeypatch.setenv("BUZZ_TRANSPORT", "poll")
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", "default-user-npub")
    monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", "/default/creds.json")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1default")
    monkeypatch.setenv("BUZZ_AUTH_TAG", '["auth","default-profile-tag","","x"]')


class TestMultiplexProfileScope:

    def test_secondary_extra_wins_over_default_profile_env(
        self, multiplex_scope, default_profile_env, tmp_path
    ):
        """The secondary profile's PlatformConfig is authoritative (#98738)."""
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        multiplex_scope()
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://profile.relay",
                "channels": ["pchan"],
                "home_channel": "pchan",
                "poll_interval": 2,
                "cli_path": str(cli),
                "transport": "websocket",
                "allowed_users": [SELF_NPUB],
            },
        )
        adapter = BuzzAdapter(cfg)
        assert adapter.relay_url == "https://profile.relay"
        assert adapter.channels == ["pchan"]
        assert adapter.home_channel == "pchan"
        assert adapter.poll_interval == 2.0
        assert adapter.cli_path == str(cli)
        assert adapter.transport == "websocket"
        assert adapter._allowed_pubkeys == {SELF_PUBKEY}

    def test_secondary_missing_keys_fail_closed(
        self, multiplex_scope, default_profile_env
    ):
        """Keys absent from the profile's config must NOT borrow the default
        profile's bridged env values — that would connect this adapter to the
        default profile's relay and watch its channels."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter.relay_url == ""
        assert adapter.channels == []
        assert adapter.home_channel == ""
        assert adapter.poll_interval == _buzz_mod._DEFAULT_POLL_INTERVAL
        assert adapter.transport == "auto"
        assert adapter._allowed_pubkeys == set()

    def test_secondary_credentials_file_not_borrowed(
        self, multiplex_scope, default_profile_env, tmp_path, monkeypatch
    ):
        """BUZZ_CREDENTIALS_FILE in env points at the DEFAULT profile's key
        file; the scoped adapter must not read the default identity's key."""
        default_creds = tmp_path / "default-creds.json"
        default_creds.write_text(
            json.dumps({"nsec": "nsec1default-identity"}), encoding="utf-8"
        )
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(default_creds))
        multiplex_scope()
        # Scope has no key: the profile is unconfigured and must fail closed
        # to "" rather than resolving the default profile's credentials.
        assert _buzz_mod._resolve_private_key({}) == ""

    def test_default_profile_unscoped_keeps_env_precedence(
        self, monkeypatch, default_profile_env
    ):
        """Multiplex ON but no scope (the DEFAULT profile constructs
        unscoped): env is its own bridge output and still wins."""
        from agent.secret_scope import set_multiplex_active
        from gateway.config import PlatformConfig

        set_multiplex_active(True)
        try:
            adapter = BuzzAdapter(
                PlatformConfig(enabled=True, extra={"relay_url": "https://cfg.relay"})
            )
        finally:
            set_multiplex_active(False)
        assert adapter.relay_url == "https://default.relay"

    def test_check_requirements_scoped_reads_profile_config(
        self, multiplex_scope, default_profile_env, tmp_path
    ):
        """The gate must consult the profile's own config.yaml + secret scope,
        not the default profile's env values."""
        import yaml
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"nsec": "nsec1profile"}), encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "gateway": {
                        "platforms": {
                            "buzz": {
                                "enabled": True,
                                "extra": {
                                    "relay_url": "https://profile.relay",
                                    "credentials_file": str(creds),
                                },
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        multiplex_scope()
        token = set_hermes_home_override(str(tmp_path))
        try:
            # The default profile's env relay+key must NOT pass the gate on
            # their own for a profile without a buzz config...
            assert check_requirements() is True  # profile config passes
        finally:
            reset_hermes_home_override(token)

        # A profile whose config.yaml has no buzz entry fails closed even
        # though the default profile's env values are present.
        empty_home = tmp_path / "empty-profile"
        empty_home.mkdir()
        multiplex_scope()
        token = set_hermes_home_override(str(empty_home))
        try:
            assert check_requirements() is False
        finally:
            reset_hermes_home_override(token)

    def test_env_enablement_scoped_returns_none(self, multiplex_scope, default_profile_env):
        """Scoped env enablement must not fabricate Buzz for a profile from
        the default profile's env values."""
        multiplex_scope()
        assert _env_enablement() is None

    def test_apply_yaml_config_scoped_skips_env_bridge(
        self, multiplex_scope, default_profile_env, monkeypatch
    ):
        """A secondary profile's YAML values must not be pinned into the
        process env for every other profile (first-writer-wins)."""
        for var in ("BUZZ_RELAY_URL", "BUZZ_HOME_CHANNEL", "BUZZ_CHANNELS"):
            monkeypatch.delenv(var, raising=False)
        multiplex_scope()
        _buzz_mod._apply_yaml_config(
            {},
            {"extra": {"relay_url": "https://profile.relay", "home_channel": "pchan"}},
        )
        import os as _os

        assert "BUZZ_RELAY_URL" not in _os.environ
        assert "BUZZ_HOME_CHANNEL" not in _os.environ
        assert "BUZZ_CHANNELS" not in _os.environ

    def test_standalone_send_scoped_uses_profile_extra(
        self, multiplex_scope, default_profile_env, monkeypatch, tmp_path
    ):
        multiplex_scope()
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        calls = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=None):
            calls["relay"] = relay_url
            return 0, '{"event_id": "e1"}', ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        monkeypatch.setattr(
            _buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1profile"
        )
        result = asyncio.run(
            _standalone_send(
                PlatformConfig(
                    enabled=True,
                    extra={"relay_url": "https://profile.relay", "cli_path": str(cli)},
                ),
                "chan-x",
                "hello",
            )
        )
        assert result.get("success") is True
        assert calls["relay"] == "https://profile.relay"

    def test_secondary_partial_extra_fills_missing_keys_from_defaults(
        self, multiplex_scope, default_profile_env
    ):
        """Partial config: configured keys win, unconfigured keys fall back to
        their own defaults — never to the default profile's env values."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        adapter = BuzzAdapter(
            PlatformConfig(enabled=True, extra={"relay_url": "https://profile.relay"})
        )
        assert adapter.relay_url == "https://profile.relay"
        assert adapter.channels == []
        assert adapter.home_channel == ""
        assert adapter.poll_interval == _buzz_mod._DEFAULT_POLL_INTERVAL
        assert adapter.transport == "auto"
        assert adapter._allowed_pubkeys == set()

    def test_ws_auth_tag_not_borrowed_from_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """BUZZ_AUTH_TAG is per-identity NIP-OA owner attestation: a scoped
        secondary profile without one must not sign its NIP-42 auth event
        with the default profile's tag from os.environ (#98738)."""
        import asyncio as _asyncio

        multiplex_scope()
        adapter = BuzzAdapter.__new__(BuzzAdapter)
        adapter._private_key = "00" * 31 + "03"
        adapter._websocket_url = lambda: "wss://relay.example"

        class _FakeWS:
            def __init__(self):
                self.sent = []

            async def recv(self):
                if self.sent:
                    return json.dumps(["OK", self.sent[0][1]["id"], True, "ok"])
                return json.dumps(["AUTH", "challenge-1"])

            async def send(self, raw):
                self.sent.append(json.loads(raw))

        ws = _FakeWS()
        _asyncio.run(adapter._authenticate_websocket(ws))
        tags = [t for t in ws.sent[0][1]["tags"] if t and t[0] == "auth"]
        assert tags == []

    def test_ws_auth_tag_scoped_profile_uses_its_own_tag(
        self, multiplex_scope
    ):
        """Positive control: a tag present in the profile's own secret scope
        IS attached to the NIP-42 auth event."""
        import asyncio as _asyncio

        profile_tag = json.dumps(["auth", "p" * 64, "", "q" * 128])
        multiplex_scope({"BUZZ_AUTH_TAG": profile_tag})
        adapter = BuzzAdapter.__new__(BuzzAdapter)
        adapter._private_key = "00" * 31 + "03"
        adapter._websocket_url = lambda: "wss://relay.example"

        class _FakeWS:
            def __init__(self):
                self.sent = []

            async def recv(self):
                if self.sent:
                    return json.dumps(["OK", self.sent[0][1]["id"], True, "ok"])
                return json.dumps(["AUTH", "challenge-1"])

            async def send(self, raw):
                self.sent.append(json.loads(raw))

        ws = _FakeWS()
        _asyncio.run(adapter._authenticate_websocket(ws))
        tags = [t for t in ws.sent[0][1]["tags"] if t and t[0] == "auth"]
        assert tags == [json.loads(profile_tag)]

    def test_ws_auth_tag_unscoped_default_profile_keeps_env(
        self, default_profile_env
    ):
        """The default profile constructs unscoped even under multiplex, so
        its env-provided auth tag still applies (legacy behavior kept)."""
        import asyncio as _asyncio

        from agent.secret_scope import set_multiplex_active

        set_multiplex_active(True)
        try:
            adapter = BuzzAdapter.__new__(BuzzAdapter)
            adapter._private_key = "00" * 31 + "03"
            adapter._websocket_url = lambda: "wss://relay.example"

            class _FakeWS:
                def __init__(self):
                    self.sent = []

                async def recv(self):
                    if self.sent:
                        return json.dumps(["OK", self.sent[0][1]["id"], True, "ok"])
                    return json.dumps(["AUTH", "challenge-1"])

                async def send(self, raw):
                    self.sent.append(json.loads(raw))

            ws = _FakeWS()
            _asyncio.run(adapter._authenticate_websocket(ws))
        finally:
            set_multiplex_active(False)
        tags = [t for t in ws.sent[0][1]["tags"] if t and t[0] == "auth"]
        assert tags == [["auth", "default-profile-tag", "", "x"]]

    def test_validate_config_scoped_extra_is_authoritative(
        self, multiplex_scope, default_profile_env, tmp_path
    ):
        """Scoped validation reads the profile's extra, not the default
        profile's env relay/key: unconfigured fails closed, configured
        passes via its own credentials file."""
        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"nsec": "nsec1profile"}), encoding="utf-8")
        from gateway.config import PlatformConfig

        multiplex_scope()
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False
        assert (
            validate_config(
                PlatformConfig(
                    enabled=True,
                    extra={
                        "relay_url": "https://profile.relay",
                        "credentials_file": str(creds),
                    },
                )
            )
            is True
        )

    def test_validate_config_unscoped_keeps_env_precedence(
        self, default_profile_env
    ):
        """Single-profile/unscoped: env relay + env key still validate even
        with an empty extra mapping."""
        from gateway.config import PlatformConfig

        assert validate_config(PlatformConfig(enabled=True, extra={})) is True

    def test_standalone_send_scoped_target_falls_back_to_profile_home(
        self, multiplex_scope, default_profile_env, monkeypatch, tmp_path
    ):
        """With no explicit chat_id, the scoped standalone send targets the
        profile's own home_channel — never the default profile's env one."""
        multiplex_scope()
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")
        calls = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=None):
            calls["args"] = args
            return 0, '{"event_id": "e1"}', ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        monkeypatch.setattr(
            _buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1profile"
        )
        result = asyncio.run(
            _standalone_send(
                PlatformConfig(
                    enabled=True,
                    extra={
                        "relay_url": "https://profile.relay",
                        "cli_path": str(cli),
                        "home_channel": "pchan",
                    },
                ),
                "",
                "hello",
            )
        )
        assert result.get("success") is True
        assert calls["args"][calls["args"].index("--channel") + 1] == "pchan"

    def test_standalone_send_scoped_without_target_fails_closed(
        self, multiplex_scope, default_profile_env, monkeypatch, tmp_path
    ):
        """No chat_id and no profile home_channel: the error is returned —
        the default profile's env BUZZ_HOME_CHANNEL must not be borrowed."""
        multiplex_scope()
        from gateway.config import PlatformConfig

        cli = tmp_path / "buzz"
        cli.write_text("#!/bin/sh\n", encoding="utf-8")

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=None):
            raise AssertionError("CLI must not run without a resolved target")

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        monkeypatch.setattr(
            _buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1profile"
        )
        result = asyncio.run(
            _standalone_send(
                PlatformConfig(
                    enabled=True,
                    extra={"relay_url": "https://profile.relay", "cli_path": str(cli)},
                ),
                "",
                "hello",
            )
        )
        assert result == {
            "error": "Buzz standalone send: no target channel (set BUZZ_HOME_CHANNEL)"
        }


# ── CLI error contract ────────────────────────────────────────────────────


class TestCliErrorContract:

    def test_parses_json_error(self):
        msg = _cli_error_message('{"error":"relay_error","message":"boom","retryable":false}', 2)
        assert "relay_error" in msg and "boom" in msg and "exit 2" in msg


# ── Seeding / high-water mark / de-dupe ───────────────────────────────────


class TestPollingDedupe:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_seed_sets_high_water_mark_without_dispatch(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event("e1", content="@Chip old history", created_at=100),
            _event("e2", content="@Chip newer history", created_at=200),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        state = adapter._channel_state[CHANNEL]
        assert state["last_ts"] == 200
        assert set(state["seen"]) == {"e1", "e2"}
        # Seeding must never replay history into the agent
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_new_event_dispatched_once(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [_event("e1", content="@Chip hi", created_at=100)])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        # Poll 1: seeded event + a genuinely new mention
        cli.responses.clear()
        cli.script("messages", "get", [
            _event("e1", content="@Chip hi", created_at=100),
            _event("e2", content="hey @Chip, ping", created_at=150),
        ])
        await adapter._poll_channel(CHANNEL)
        assert [d["message_id"] for d in adapter._dispatched] == ["e2"]
        assert adapter._dispatched[0]["text"] == "hey @Chip, ping"
        assert adapter._channel_state[CHANNEL]["last_ts"] == 150

        # Poll 2: identical response — the seen-id set must de-dupe
        await adapter._poll_channel(CHANNEL)
        assert len(adapter._dispatched) == 1


# ── Mention gating / DMs / authorization ──────────────────────────────────


class TestMentionGating:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)

    @pytest.mark.asyncio
    async def test_unaddressed_channel_message_ignored(self, adapter):
        await self._poll_with(adapter, _event("e1", content="just chatting", created_at=10))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_name_mention_dispatched(self, adapter):
        await self._poll_with(adapter, _event("e1", content="hey @Chip can you help?", created_at=10))
        assert len(adapter._dispatched) == 1


    @pytest.mark.asyncio
    async def test_allowlist_blocks_unauthorized(self, adapter):
        adapter._allowed_pubkeys = {"b" * 64}
        await self._poll_with(adapter, _event("e1", content="@Chip hello", created_at=10))
        assert adapter._dispatched == []


# ── DM classification via p-tags (issue #68871) ──────────────────────────
#
# `buzz dms list` returns [] on some hosted relays, so DM conversations leak
# in via `channels list` and get seeded chat_type="group".  The adapter must
# reclassify them from the Nostr tags of real traffic: DM messages are
# p-tagged to our own pubkey WITHOUT the text mentioning us, while channel
# messages only ever p-tag us when the text visibly @mentions us.


def _tagged_event(event_id, channel, *, content, pubkey=OTHER_PUBKEY,
                  created_at=1000, kind=9, p=None, reply_to=None):
    """Event with the tag shapes observed on a live relay (h/p/e tags)."""
    tags = [["h", channel]]
    if reply_to:
        tags.append(["e", reply_to, "", "reply"])
    if p:
        tags.append(["p", p])
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
    }


class TestDmClassification:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        # Metadata exactly as `channels list` returns it on the hosted relay.
        a._channel_meta = {
            DM_CHANNEL: {"channel_id": DM_CHANNEL, "name": "DM", "description": ""},
            CHANNEL: {
                "channel_id": CHANNEL,
                "name": "general",
                "description": "General conversation and community updates.",
            },
        }
        a._channel_names = {DM_CHANNEL: "DM", CHANNEL: "general"}
        # Both leaked in as group — the bug under test.
        a._channel_state[DM_CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, channel, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(channel)

    @pytest.mark.asyncio
    async def test_unmentioned_ptagged_dm_latches_and_dispatches(self, adapter):
        """The reported bug: a DM without an @mention must dispatch."""
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e1", DM_CHANNEL, content="here's a test message", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[DM_CHANNEL]["chat_type"] == "dm"
        assert [d["message_id"] for d in adapter._dispatched] == ["e1"]
        assert adapter._dispatched[0]["chat_type"] == "dm"


    @pytest.mark.asyncio
    async def test_general_reply_ptagging_self_stays_channel(self, adapter):
        """A #general reply to us p-tags our pubkey (observed live) — that
        must NOT reclassify the channel; mention gating still applies."""
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="@chip what's up?",
                          p=SELF_PUBKEY, reply_to="root-event"),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        # It carried a mention, so it dispatches — but as a group message.
        assert [d["chat_type"] for d in adapter._dispatched] == ["group"]

        # And once the mention is absent, the channel gate drops the message
        # even though the earlier reply p-tagged us.
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e2", CHANNEL, content="thanks everyone", created_at=1001),
        )
        assert len(adapter._dispatched) == 1


    @pytest.mark.asyncio
    async def test_channel_like_metadata_blocks_latch_even_without_mention(self, adapter):
        """Second guard on its own: even a p-tagged, un-mentioned message
        cannot reclassify a conversation whose metadata says real channel."""
        adapter._channel_meta[CHANNEL]["description"] = ""
        adapter._channel_meta[CHANNEL]["name"] = "announcements"
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="fyi everyone", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        assert adapter._dispatched == []


    @pytest.mark.asyncio
    async def test_dm_shaped_channel_discovered_when_dms_list_empty(self):
        """Fallback discovery: with `dms list` broken (returns []), a
        DM-shaped `channels list` entry gets watched; real channels not
        already watched are left alone."""
        a = _make_adapter()
        cli = _ScriptedCli()
        cli.script("dms", "list", [])
        cli.script("channels", "list", [
            {"channel_id": DM_CHANNEL, "name": "DM", "description": "", "created_at": 1},
            {"channel_id": CHANNEL, "name": "general",
             "description": "General conversation and community updates.", "created_at": 2},
        ])
        a._run_cli = cli
        await a._discover_dms(seed=False)
        # Watched as group; the p-tag latch flips it on the first real DM.
        assert a._channel_state[DM_CHANNEL]["chat_type"] == "group"
        assert a._may_reclassify_as_dm(DM_CHANNEL) is True
        assert CHANNEL not in a._channel_state
        assert a._may_reclassify_as_dm(CHANNEL) is False


# ── Sending ───────────────────────────────────────────────────────────────


class TestBuzzAdapterSend:

    @pytest.mark.asyncio
    async def test_send_success_via_stdin(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt123", "message": ""})
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "hello **markdown**")
        assert result.success is True
        assert result.message_id == "evt123"

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "send"]
        assert args[args.index("--channel") + 1] == CHANNEL
        # Content travels via stdin (--content -), never argv
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "hello **markdown**"
        # Our own event id is marked seen for echo suppression
        assert "evt123" in adapter._channel_state[CHANNEL]["seen"]


    @pytest.mark.asyncio
    async def test_send_image_local_file_uses_file_flag(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt126", "message": ""})
        adapter._run_cli = cli
        result = await adapter.send_image(CHANNEL, str(img), caption="screenshot")
        assert result.success is True
        args, _stdin = cli.calls[0]
        assert args[args.index("--file") + 1] == str(img)


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestBuzzAdapterLifecycle:


    @pytest.mark.asyncio
    async def test_disconnect_releases_scoped_lock(self, monkeypatch):
        """The identity lock taken in connect() must be released on disconnect."""
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        adapter = _make_adapter()
        adapter._lock_key = "wss://relay.example:" + SELF_PUBKEY
        await adapter.disconnect()
        assert released == [("buzz", "wss://relay.example:" + SELF_PUBKEY)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """A second profile using the same relay+pubkey must fail fast."""
        import gateway.status as gateway_status

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: False
        )
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        cli = _ScriptedCli()
        cli.script(
            "users", "get",
            [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}],
        )
        adapter._run_cli = cli
        assert await adapter.connect() is False
        assert adapter._lock_key is None


# ── Credentials / requirements ────────────────────────────────────────────


class TestCredentialResolution:

    def test_env_key_wins(self, monkeypatch):
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1fromenv")
        assert _resolve_private_key() == "nsec1fromenv"

    def test_credentials_file_fallback(self, monkeypatch, tmp_path):
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "npub": "npub1x"}), encoding="utf-8")
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        assert _resolve_private_key() == "nsec1fromfile"

    def test_owner_auth_tag_from_credentials_file(self, monkeypatch, tmp_path):
        tag = ["auth", "b" * 64, "", "c" * 128]
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "auth_tag": tag}), encoding="utf-8")
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        assert json.loads(_resolve_auth_tag()) == tag

    def test_invalid_owner_auth_tag_fails_closed(self, monkeypatch, tmp_path):
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "auth_tag": ["bad"]}), encoding="utf-8")
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        with pytest.raises(ValueError, match="auth tag"):
            _resolve_auth_tag()

    def test_scoped_credentials_and_tag_do_not_borrow_ambient_profile(self, monkeypatch, tmp_path):
        from agent import secret_scope as ss

        ambient_tag = ["auth", "a" * 64, "", "d" * 128]
        scoped_tag = ["auth", "b" * 64, "", "c" * 128]
        ambient = tmp_path / "ambient.json"
        scoped = tmp_path / "scoped.json"
        ambient.write_text(json.dumps({"nsec": "nsec1ambient", "auth_tag": ambient_tag}))
        scoped.write_text(json.dumps({"nsec": "nsec1scoped", "auth_tag": scoped_tag}))
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(ambient))
        monkeypatch.setenv("BUZZ_AUTH_TAG", json.dumps(ambient_tag))
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"BUZZ_CREDENTIALS_FILE": str(scoped)})
        try:
            assert _resolve_private_key() == "nsec1scoped"
            assert json.loads(_resolve_auth_tag()) == scoped_tag
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

    def test_owner_auth_tag_uses_credentials_autodiscovery(self, monkeypatch, tmp_path):
        tag = ["auth", "b" * 64, "", "c" * 128]
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "auth_tag": tag}), encoding="utf-8")
        monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path)
        assert _resolve_private_key() == "nsec1fromfile"
        assert json.loads(_resolve_auth_tag()) == tag

    def test_empty_multiplex_scope_never_autodiscovers_ambient_credentials(self, monkeypatch, tmp_path):
        from agent import secret_scope as ss

        tag = ["auth", "a" * 64, "", "d" * 128]
        (tmp_path / "general_credentials.json").write_text(
            json.dumps({"nsec": "nsec1ambient", "auth_tag": tag}), encoding="utf-8"
        )
        monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path)
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert _resolve_private_key() == ""
            assert _resolve_auth_tag() == ""
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)


# ── Env enablement / registration / standalone send ──────────────────────


class TestEnvEnablement:

    def test_returns_none_when_unconfigured(self):
        assert _env_enablement() is None


class TestBuzzPluginRegistration:

    def test_register_platform_contract(self):
        from gateway.platform_registry import platform_registry

        platform_registry.unregister("buzz")
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "buzz"
        assert kwargs["cron_deliver_env_var"] == "BUZZ_HOME_CHANNEL"
        assert kwargs["allowed_users_env"] == "BUZZ_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "BUZZ_ALLOW_ALL_USERS"
        assert callable(kwargs["standalone_sender_fn"])
        assert callable(kwargs["env_enablement_fn"])
        assert set(kwargs["required_env"]) == {"BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"}


class TestStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_success(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))

        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=30.0):
            captured.update(cli_path=cli_path, args=args, relay_url=relay_url, auth_tag=auth_tag, input_text=input_text)
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), CHANNEL, "cron says hi")
        assert result == {"success": True, "message_id": "evt-cron"}
        assert captured["args"][:2] == ["messages", "send"]
        assert captured["input_text"] == "cron says hi"
        # The private key must never be part of argv
        assert all("nsec1x" not in str(a) for a in captured["args"])

    @pytest.mark.asyncio
    async def test_standalone_send_injects_owner_auth_tag_from_credentials_file(
        self, monkeypatch, tmp_path
    ):
        """Cron/standalone path must load NIP-OA auth_tag from credentials JSON.

        Main-line regression: when only BUZZ_PRIVATE_KEY is ambient and the
        credentials file holds auth_tag, omitting injection causes relay 403
        membership failures on owner-gated relays.
        """
        from gateway.config import PlatformConfig

        tag = ["auth", "b" * 64, "", "c" * 128]
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(
            json.dumps({"nsec": "nsec1fromfile", "auth_tag": tag}),
            encoding="utf-8",
        )
        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)

        captured = {}

        async def fake_exec(
            cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=30.0
        ):
            captured.update(
                private_key=private_key,
                auth_tag=auth_tag,
                args=args,
                input_text=input_text,
            )
            return 0, json.dumps({"accepted": True, "event_id": "evt-auth", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}), CHANNEL, "cron needs owner auth"
        )
        assert result == {"success": True, "message_id": "evt-auth"}
        assert captured["private_key"] == "nsec1fromfile"
        assert json.loads(captured["auth_tag"]) == tag
        # Secrets stay out of argv (auth_tag is env-injected by _exec_buzz).
        joined_args = " ".join(str(a) for a in captured["args"])
        assert "nsec1fromfile" not in joined_args
        assert tag[1] not in joined_args
        assert tag[3] not in joined_args

    @pytest.mark.asyncio
    async def test_standalone_send_key_only_does_not_invent_auth_tag(
        self, monkeypatch, tmp_path
    ):
        """Direct private key without credentials_file must not invent an auth tag."""
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
        monkeypatch.delenv("BUZZ_CREDENTIALS_FILE", raising=False)
        # Ambient credentials dir must not be borrowed when a direct key is set.
        ambient = tmp_path / "ambient_credentials.json"
        ambient.write_text(
            json.dumps({"nsec": "nsec1ambient", "auth_tag": ["auth", "a" * 64, "", "d" * 128]}),
            encoding="utf-8",
        )
        monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path)

        captured = {}

        async def fake_exec(
            cli_path, args, *, relay_url, private_key, auth_tag="", input_text=None, timeout=30.0
        ):
            captured.update(private_key=private_key, auth_tag=auth_tag)
            return 0, json.dumps({"accepted": True, "event_id": "evt-key", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}), CHANNEL, "key only"
        )
        assert result == {"success": True, "message_id": "evt-key"}
        assert captured["private_key"] == "nsec1x"
        assert captured["auth_tag"] == ""


# ── Editing and deleting (streaming) ──────────────────────────────────


class TestBuzzAdapterEdit:

    @pytest.mark.asyncio
    async def test_edit_targets_the_original_event_and_uses_stdin(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1", "message": ""})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "partial answer")
        assert result.success is True

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "edit"]
        assert args[args.index("--event") + 1] == "orig1"
        # Content travels via stdin (--content -), never argv, same as send
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "partial answer"

    @pytest.mark.asyncio
    async def test_edit_returns_the_original_id_not_the_cli_event_id(self):
        """The stream consumer re-edits ONE message id for the whole stream.

        buzz-cli reports a fresh event id for each edit; returning that would
        make the second edit address a message that was never sent.
        """
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert result.message_id == "orig1"

    @pytest.mark.asyncio
    async def test_edit_marks_its_own_event_seen(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        await adapter.edit_message(CHANNEL, "orig1", "text")
        assert "edit1" in adapter._channel_state[CHANNEL]["seen"]

    @pytest.mark.asyncio
    async def test_edit_accepts_finalize_without_changing_behaviour(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "text", finalize=True)
        assert result.success is True
        assert len(cli.calls) == 1

    @pytest.mark.asyncio
    async def test_edit_without_a_message_id_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "", "text")
        assert result.success is False
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_edit_with_empty_content_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "")
        assert result.success is False
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_edit_relay_error_is_retryable_but_bad_input_is_not(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "edit", "", code=2, stderr="relay unreachable")
        adapter._run_cli = cli
        relay_failure = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert relay_failure.success is False
        assert relay_failure.retryable is True

        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "edit", "", code=1, stderr="bad input")
        adapter._run_cli = cli
        input_failure = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert input_failure.success is False
        assert input_failure.retryable is False

    @pytest.mark.asyncio
    async def test_delete_targets_the_event(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "delete", {"accepted": True, "event_id": "del1"})
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "orig1") is True
        assert cli.calls[0][0] == ["messages", "delete", "--event", "orig1"]

    @pytest.mark.asyncio
    async def test_delete_failure_returns_false(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "delete", "", code=2, stderr="relay unreachable")
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "orig1") is False

    @pytest.mark.asyncio
    async def test_delete_without_a_message_id_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "") is False
        assert cli.calls == []
