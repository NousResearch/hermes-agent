"""Multi-agent Buzz wiring: per-agent credentials + identity-based routing.

Covers the three layers added on top of the Buzz adapter (#71610) and the
single-gateway-multi-agent registry:

1. Credential resolution — ``extra["private_key_env"]`` NAMES the env var
   holding one agent's key; resolution is env/secret-scope first, then the
   agent's own ``.env``, and fail-closed (never the shared
   ``BUZZ_PRIVATE_KEY``).
2. Binding construction — ``agents.<id>.buzz.nsec_env`` blocks become
   per-agent ``PlatformConfig``s (gateway/agent_platforms.py).
3. Routing — the connection an event arrives on determines the owning
   agent: mention gating uses each agent's OWN identity, DM classification
   uses each agent's OWN p-tag, and replies resolve back to the owning
   agent's adapter (fail-closed when it is down).

All tests here run against a mock relay (scripted ``_run_cli``); the live
end-to-end pass lives in tests/integration/buzz/test_live_relay_multi_agent.py.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter
hex_to_npub = _buzz_mod.hex_to_npub
_resolve_private_key = _buzz_mod._resolve_private_key

from agent.profile import AgentProfile
from gateway.agent_platforms import (
    AgentPlatformBinding,
    build_agent_platform_bindings,
    resolve_binding_secret,
    secret_fingerprint,
)
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource


AGENT_A_PUBKEY = "b" * 64
AGENT_B_PUBKEY = "c" * 64
POSTER_PUBKEY = "d" * 64
CHANNEL = "11111111-2222-3333-4444-555555555555"
DM_CHANNEL = "66666666-7777-8888-9999-aaaaaaaaaaaa"

_ENV_VARS = (
    "BUZZ_RELAY_URL",
    "BUZZ_PRIVATE_KEY",
    "BUZZ_CHANNELS",
    "BUZZ_HOME_CHANNEL",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "BUZZ_POLL_INTERVAL",
    "BUZZ_CLI_PATH",
    "BUZZ_CREDENTIALS_FILE",
    "BUZZ_REQUIRE_MENTION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


# ── 1. Per-agent credential resolution ────────────────────────────────────


class TestPerAgentKeyResolution:

    def test_resolves_named_env_var_from_process_env(self, monkeypatch):
        monkeypatch.setenv("CHIP_TEST_NSEC", "nsec1-chip-value")
        assert (
            _resolve_private_key({"private_key_env": "CHIP_TEST_NSEC"})
            == "nsec1-chip-value"
        )

    def test_falls_back_to_agent_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('CHIP_TEST_NSEC="nsec1-from-file"\n')
        extra = {"private_key_env": "CHIP_TEST_NSEC", "env_file": str(env_file)}
        assert _resolve_private_key(extra) == "nsec1-from-file"

    def test_process_env_wins_over_env_file(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("CHIP_TEST_NSEC=nsec1-from-file\n")
        monkeypatch.setenv("CHIP_TEST_NSEC", "nsec1-from-env")
        extra = {"private_key_env": "CHIP_TEST_NSEC", "env_file": str(env_file)}
        assert _resolve_private_key(extra) == "nsec1-from-env"

    def test_fail_closed_never_borrows_shared_key(self, monkeypatch, tmp_path):
        """An unresolvable per-agent name must NOT fall back to
        BUZZ_PRIVATE_KEY — that would connect this agent as another member."""
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1-shared")
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=whatever\n")
        extra = {"private_key_env": "MISSING_TEST_NSEC", "env_file": str(env_file)}
        assert _resolve_private_key(extra) == ""

    def test_legacy_resolution_unchanged(self, monkeypatch):
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1-shared")
        assert _resolve_private_key({}) == "nsec1-shared"
        assert _resolve_private_key(None) == "nsec1-shared"


# ── 2. Agent-scoped adapter settings ──────────────────────────────────────


def _adapter(extra=None):
    cfg = PlatformConfig(
        enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})}
    )
    return BuzzAdapter(cfg)


class TestAgentScopedSettings:

    def test_agent_block_wins_over_global_env(self, monkeypatch):
        """One process hosts N connections: a global BUZZ_* var can only be a
        default, never an override of a specific agent's block."""
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://global.relay")
        monkeypatch.setenv("BUZZ_CHANNELS", "global-channel")
        monkeypatch.setenv("BUZZ_REQUIRE_MENTION", "true")
        a = _adapter(
            {
                "agent_id": "chip",
                "relay_url": "https://agent.relay",
                "channels": [CHANNEL],
                "require_mention": False,
            }
        )
        assert a.relay_url == "https://agent.relay"
        assert a.channels == [CHANNEL]
        assert a.require_mention is False

    def test_env_still_wins_without_agent_id(self, monkeypatch):
        """Legacy single-connection precedence (env > extra) is untouched."""
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://global.relay")
        monkeypatch.setenv("BUZZ_REQUIRE_MENTION", "false")
        a = _adapter({"relay_url": "https://agent.relay", "require_mention": True})
        assert a.relay_url == "https://global.relay"
        assert a.require_mention is False

    def test_global_env_remains_default_for_unset_keys(self, monkeypatch):
        monkeypatch.setenv("BUZZ_CHANNELS", f"{CHANNEL},{DM_CHANNEL}")
        a = _adapter({"agent_id": "chip"})
        assert a.channels == [CHANNEL, DM_CHANNEL]

    def test_name_carries_agent_id(self):
        assert _adapter({"agent_id": "chip"}).name == "Buzz:chip"
        assert _adapter().name == "Buzz"


# ── 3. Binding construction from the AgentProfile registry ────────────────


def _registry(tmp_path):
    return {
        "main": AgentProfile(),
        "chip": AgentProfile(
            id="chip",
            home_dir=tmp_path / "chip",
            config_overrides={
                "buzz": {"nsec_env": "CHIP_TEST_NSEC", "require_mention": False}
            },
        ),
        "scout": AgentProfile(
            id="scout",
            home_dir=tmp_path / "scout",
            config_overrides={"buzz": {"nsec_env": "SCOUT_TEST_NSEC"}},
        ),
        "nobuzz": AgentProfile(id="nobuzz", config_overrides={"model": "x"}),
    }


def _gateway_config(extra=None):
    return SimpleNamespace(
        platforms={
            Platform("buzz"): PlatformConfig(
                enabled=True,
                extra={"relay_url": "https://shared.relay", "poll_interval": 2,
                       **(extra or {})},
            )
        }
    )


class TestBindingBuilder:

    def test_builds_one_binding_per_agent_with_credential(self, tmp_path):
        bindings, problems = build_agent_platform_bindings(
            _registry(tmp_path), _gateway_config()
        )
        assert problems == []
        assert [(b.agent_id, b.platform_value) for b in bindings] == [
            ("chip", "buzz"),
            ("scout", "buzz"),
        ]

    def test_extra_merges_shared_defaults_with_agent_overrides(self, tmp_path):
        bindings, _ = build_agent_platform_bindings(
            _registry(tmp_path), _gateway_config()
        )
        chip = bindings[0].config.extra
        assert chip["relay_url"] == "https://shared.relay"  # inherited
        assert chip["poll_interval"] == 2  # inherited
        assert chip["require_mention"] is False  # agent override
        assert chip["private_key_env"] == "CHIP_TEST_NSEC"
        assert chip["agent_id"] == "chip"
        assert chip["env_file"].endswith("chip/.env")
        assert "nsec_env" not in chip  # renamed to the adapter-facing key
        assert bindings[0].config.enabled is True

    def test_works_without_shared_platform_block(self, tmp_path):
        bindings, problems = build_agent_platform_bindings(
            _registry(tmp_path), SimpleNamespace(platforms={})
        )
        assert problems == []
        assert len(bindings) == 2
        assert bindings[0].config.extra["private_key_env"] == "CHIP_TEST_NSEC"

    def test_missing_nsec_env_is_reported_and_skipped(self, tmp_path):
        registry = {
            "chip": AgentProfile(id="chip", config_overrides={"buzz": {}}),
        }
        bindings, problems = build_agent_platform_bindings(registry, _gateway_config())
        assert bindings == []
        assert len(problems) == 1 and "nsec_env" in problems[0]

    def test_non_mapping_block_is_reported(self, tmp_path):
        registry = {
            "chip": AgentProfile(id="chip", config_overrides={"buzz": "yes"}),
        }
        bindings, problems = build_agent_platform_bindings(registry, _gateway_config())
        assert bindings == []
        assert len(problems) == 1 and "expected a mapping" in problems[0]

    def test_duplicate_nsec_env_refused(self, tmp_path):
        registry = {
            "chip": AgentProfile(
                id="chip", config_overrides={"buzz": {"nsec_env": "SAME_NSEC"}}
            ),
            "scout": AgentProfile(
                id="scout", config_overrides={"buzz": {"nsec_env": "SAME_NSEC"}}
            ),
        }
        bindings, problems = build_agent_platform_bindings(registry, _gateway_config())
        assert [b.agent_id for b in bindings] == ["chip"]  # first (sorted) wins
        assert len(problems) == 1 and "already claimed" in problems[0]

    def test_resolve_binding_secret_orders_and_fails_closed(
        self, monkeypatch, tmp_path
    ):
        env_file = tmp_path / ".env"
        env_file.write_text("SCOUT_TEST_NSEC=nsec1-scout-file\n")
        binding = AgentPlatformBinding(
            agent_id="scout",
            platform_value="buzz",
            secret_env="SCOUT_TEST_NSEC",
            env_file=env_file,
            config=PlatformConfig(enabled=True),
        )
        assert resolve_binding_secret(binding) == "nsec1-scout-file"
        monkeypatch.setenv("SCOUT_TEST_NSEC", "nsec1-scout-env")
        assert resolve_binding_secret(binding) == "nsec1-scout-env"
        missing = AgentPlatformBinding(
            agent_id="x",
            platform_value="buzz",
            secret_env="NOPE_TEST_NSEC",
            env_file=None,
            config=PlatformConfig(enabled=True),
        )
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1-shared")
        assert resolve_binding_secret(missing) == ""

    def test_secret_fingerprint_is_salted_and_stable(self):
        fp = secret_fingerprint("nsec1-anything")
        assert fp == secret_fingerprint("nsec1-anything")
        assert fp != secret_fingerprint("nsec1-other")
        assert fp is not None and "nsec1" not in fp and len(fp) == 16
        assert secret_fingerprint("") is None
        assert secret_fingerprint(None) is None  # type: ignore[arg-type]


# ── 4. Identity routing on a mock relay ───────────────────────────────────


def _mock_agent_adapter(agent_id, pubkey, display_name, require_mention=True):
    """A per-agent adapter as gateway/agent_platforms.py would build it,
    with connection identity pre-seeded (as connect() would learn it)."""
    a = _adapter(
        {
            "agent_id": agent_id,
            "channels": [CHANNEL],
            "require_mention": require_mention,
        }
    )
    a._self_pubkey = pubkey
    a._self_npub = hex_to_npub(pubkey) or ""
    a._display_name = display_name
    a._private_key = "nsec1test"
    a.set_routing_context(routes=[], default_agent=agent_id)
    a._message_handler = AsyncMock()
    a._events = []

    async def _capture_handle_message(event):
        # The slice of BasePlatformAdapter.handle_message under test:
        # agent-id stamping. Session machinery is out of scope here.
        a._attach_agent_id(event)
        a._events.append(event)

    a.handle_message = _capture_handle_message
    a.send_reaction = AsyncMock(return_value=True)
    a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
    return a


def _channel_event(event_id, content, pubkey=POSTER_PUBKEY, tags=None):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": 1000,
        "kind": 9,
        "tags": tags if tags is not None else [["h", CHANNEL]],
    }


async def _poll(adapter, channel, events):
    async def _run_cli(args, *, input_text=None):
        if args[:2] == ["messages", "get"]:
            return 0, json.dumps(events), ""
        if args[:2] == ["users", "get"]:
            return 0, json.dumps([{"pubkey": POSTER_PUBKEY, "display_name": "Poster"}]), ""
        return 0, "[]", ""

    adapter._run_cli = _run_cli
    await adapter._poll_channel(channel)


class TestIdentityRouting:

    @pytest.mark.asyncio
    async def test_channel_mention_routes_to_owning_agent_only(self):
        """'@AgentA hello' in a shared channel: agent A's connection
        dispatches (stamped with A's agent_id), agent B's stays silent."""
        a = _mock_agent_adapter("agent_a", AGENT_A_PUBKEY, "AgentA")
        b = _mock_agent_adapter("agent_b", AGENT_B_PUBKEY, "AgentB")
        events = [_channel_event("e1", "@AgentA hello")]
        await _poll(a, CHANNEL, events)
        await _poll(b, CHANNEL, events)

        assert len(a._events) == 1
        assert a._events[0].text == "hello"  # leading mention stripped
        assert a._events[0].source.agent_id == "agent_a"
        assert b._events == []

    @pytest.mark.asyncio
    async def test_unaddressed_channel_message_reaches_neither(self):
        a = _mock_agent_adapter("agent_a", AGENT_A_PUBKEY, "AgentA")
        b = _mock_agent_adapter("agent_b", AGENT_B_PUBKEY, "AgentB")
        events = [_channel_event("e2", "just chatting amongst humans")]
        await _poll(a, CHANNEL, events)
        await _poll(b, CHANNEL, events)
        assert a._events == [] and b._events == []

    @pytest.mark.asyncio
    async def test_require_mention_false_is_per_agent(self):
        a = _mock_agent_adapter("agent_a", AGENT_A_PUBKEY, "AgentA")
        b = _mock_agent_adapter(
            "agent_b", AGENT_B_PUBKEY, "AgentB", require_mention=False
        )
        events = [_channel_event("e3", "morning everyone")]
        await _poll(a, CHANNEL, events)
        await _poll(b, CHANNEL, events)
        assert a._events == []
        assert len(b._events) == 1
        assert b._events[0].source.agent_id == "agent_b"

    @pytest.mark.asyncio
    async def test_dm_routes_by_recipient_p_tag(self):
        """The same relay-materialized DM conversation is visible to both
        connections, but the message p-tags agent B: only B's connection
        latches it as a DM and dispatches (the #2897 workaround, applied
        per identity)."""
        a = _mock_agent_adapter("agent_a", AGENT_A_PUBKEY, "AgentA")
        b = _mock_agent_adapter("agent_b", AGENT_B_PUBKEY, "AgentB")
        for adapter in (a, b):
            adapter._channel_state[DM_CHANNEL] = {
                "chat_type": "group",
                "last_ts": 0,
                "seen": {},
            }
            adapter._channel_meta[DM_CHANNEL] = {
                "channel_id": DM_CHANNEL,
                "name": "DM",
                "description": "",
            }
        dm = _channel_event(
            "e4",
            "psst, this is private",
            tags=[["h", DM_CHANNEL], ["p", AGENT_B_PUBKEY]],
        )
        await _poll(a, DM_CHANNEL, [dm])
        await _poll(b, DM_CHANNEL, [dm])

        assert a._events == []
        assert a._channel_state[DM_CHANNEL]["chat_type"] == "group"
        assert len(b._events) == 1
        assert b._events[0].source.agent_id == "agent_b"
        assert b._events[0].source.chat_type == "dm"
        assert b._channel_state[DM_CHANNEL]["chat_type"] == "dm"


# ── 5. Outbound adapter resolution (replies leave as the right member) ────


class _FakeRunner(GatewayAuthorizationMixin):
    def __init__(self):
        self.adapters = {}
        self._profile_adapters = {}
        self._agent_adapters = {}
        self._agent_bindings = {}


def _source(agent_id=None):
    return SessionSource(
        platform=Platform("buzz"),
        chat_id=CHANNEL,
        chat_type="group",
        user_id=POSTER_PUBKEY,
        agent_id=agent_id,
    )


class TestAdapterForSource:

    def test_agent_owned_adapter_wins(self):
        runner = _FakeRunner()
        shared = object()
        chip_adapter = object()
        runner.adapters[Platform("buzz")] = shared
        runner._agent_adapters["chip"] = {Platform("buzz"): chip_adapter}
        runner._agent_bindings[("chip", Platform("buzz"))] = object()
        assert runner._adapter_for_source(_source("chip")) is chip_adapter

    def test_declared_binding_fails_closed_when_connection_down(self):
        """A dead per-agent connection must NOT fall back to the shared
        adapter — the reply would leave as the wrong workspace member."""
        runner = _FakeRunner()
        runner.adapters[Platform("buzz")] = object()
        runner._agent_bindings[("chip", Platform("buzz"))] = object()
        assert runner._adapter_for_source(_source("chip")) is None

    def test_unbound_agent_falls_back_to_shared_adapter(self):
        runner = _FakeRunner()
        shared = object()
        runner.adapters[Platform("buzz")] = shared
        assert runner._adapter_for_source(_source("main")) is shared
        assert runner._adapter_for_source(_source(None)) is shared

    def test_transport_ref_recognizes_agent_adapters(self):
        import weakref

        runner = _FakeRunner()
        adapter = _mock_agent_adapter("chip", AGENT_A_PUBKEY, "Chip")
        runner._agent_adapters["chip"] = {Platform("buzz"): adapter}
        source = _source("chip")
        source._transport_adapter_ref = weakref.ref(adapter)
        assert runner._registered_transport_adapter(source) is adapter


# ── 6. Runner-side adapter configuration ──────────────────────────────────


class TestConfigureAgentAdapter:

    def test_connection_identity_is_the_routing_default(self):
        from gateway.run import GatewayRunner

        adapter = _adapter({"agent_id": "chip"})
        fake = SimpleNamespace(
            _handle_message=AsyncMock(),
            session_store=object(),
            _handle_active_session_busy_message=AsyncMock(),
            _handle_reaction_event=AsyncMock(),
            _recover_telegram_topic_thread_id=lambda *a, **k: None,
            _make_adapter_auth_check=lambda p: (lambda *a, **k: True),
            _make_agent_fatal_error_handler=lambda aid, plat: AsyncMock(),
            _busy_text_mode="normal",
        )
        GatewayRunner._configure_agent_adapter(
            fake, adapter, "chip", Platform("buzz")
        )
        # Connection identity IS the routing decision: no gateway routes
        # table, default agent = the owning agent.
        assert adapter._gateway_routes == []
        assert adapter._default_agent_id == "chip"
        assert adapter._message_handler is fake._handle_message


# ── 7. Registry check_fn bypass for proven per-agent instances ────────────


class TestRegistryCheckFnBypass:
    """A platform configured ONLY through per-agent bindings has no shared
    BUZZ_RELAY_URL / BUZZ_PRIVATE_KEY in the process env, so the plugin's
    no-arg ``check_fn`` gate would refuse every per-agent adapter despite
    each binding having proven its credential by name. The per-agent
    startup path passes ``skip_check_fn=True``; ``validate_config`` (which
    receives the instance config) still gates."""

    def test_skip_check_fn_bypasses_env_gate_but_keeps_validate_config(
        self, tmp_path, monkeypatch
    ):
        from gateway.platform_registry import platform_registry, PlatformEntry
        from gateway.config import PlatformConfig

        entry = PlatformEntry(
            name="buzz",
            label="Buzz",
            adapter_factory=lambda cfg: _buzz_mod.BuzzAdapter(cfg),
            check_fn=_buzz_mod.check_requirements,
            validate_config=_buzz_mod.validate_config,
            source="test",
        )
        platform_registry.register(entry)
        try:
            env_file = tmp_path / ".env"
            env_file.write_text("BUZZ_NSEC_CHIP=nsec1chipkey\n", encoding="utf-8")

            # No shared env config: the no-arg check_fn gate refuses.
            cfg = PlatformConfig(
                enabled=True,
                extra={
                    "relay_url": "https://relay.example",
                    "private_key_env": "BUZZ_NSEC_CHIP",
                    "env_file": str(env_file),
                    "agent_id": "chip",
                },
            )
            assert platform_registry.create_adapter("buzz", cfg) is None

            # Bypassing check_fn creates the adapter; validate_config ran
            # and passed because the instance config resolves its own
            # credential.
            adapter = platform_registry.create_adapter(
                "buzz", cfg, skip_check_fn=True
            )
            assert adapter is not None
            assert adapter.agent_id == "chip"

            # validate_config still fails closed for an unconfigured
            # instance.
            bad = PlatformConfig(enabled=True, extra={"agent_id": "chip"})
            assert (
                platform_registry.create_adapter("buzz", bad, skip_check_fn=True)
                is None
            )
        finally:
            platform_registry.unregister("buzz")

    def test_runner_per_agent_create_uses_skip_check_fn(self):
        import inspect
        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._start_agent_platform_adapters)
        assert "skip_check_fn=True" in src
        src = inspect.getsource(GatewayRunner._run_agent_adapter_reconnect)
        assert "skip_check_fn=True" in src
