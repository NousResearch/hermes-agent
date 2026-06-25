"""Phase 0 foundations for multi-profile gateway multiplexing.

Covers the three Phase 0 deliverables:
  1. ``gateway.multiplex_profiles`` config flag (default False, round-trips).
  2. ``hermes_cli.profiles.profiles_to_serve`` enumeration.
  3. Profile-stamped ``build_session_key`` that is BYTE-IDENTICAL when the
     flag is off (the orphan-every-session guard) and namespace-segmented when
     on, without disturbing the positional key layout downstream parsers rely
     on.
"""
import pytest
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key


def _src(**kw) -> SessionSource:
    kw.setdefault("platform", Platform.TELEGRAM)
    kw.setdefault("chat_id", "99")
    kw.setdefault("chat_type", "dm")
    return SessionSource(**kw)


class TestSessionKeyByteIdenticalWhenOff:
    """The non-negotiable guard: with no profile (or 'default'), every key is
    byte-for-byte what it was before Phase 0. A diff here orphans every
    existing session on upgrade."""

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_dm_with_chat_id(self, profile):
        s = _src(chat_id="99", chat_type="dm")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:99"

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_dm_with_thread(self, profile):
        s = _src(chat_id="99", chat_type="dm", thread_id="t1")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:99:t1"

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_dm_without_chat_id_falls_back_to_user(self, profile):
        s = _src(chat_id="", chat_type="dm", user_id="jordan")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:jordan"

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_group_per_user(self, profile):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, profile=profile)
            == "agent:main:discord:group:g1:alice"
        )

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_group_shared_when_disabled(self, profile):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, group_sessions_per_user=False, profile=profile)
            == "agent:main:discord:group:g1"
        )


class TestSessionKeyNamespacedWhenOn:
    """A named profile occupies the namespace slot, isolating its sessions."""

    def test_named_profile_dm(self):
        s = _src(chat_id="99", chat_type="dm")
        assert build_session_key(s, profile="coder") == "agent:coder:telegram:dm:99"

    def test_named_profile_group_per_user(self):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, profile="coder")
            == "agent:coder:discord:group:g1:alice"
        )

    def test_two_profiles_same_chat_do_not_collide(self):
        s = _src(chat_id="99", chat_type="dm")
        a = build_session_key(s, profile="default")
        b = build_session_key(s, profile="coder")
        c = build_session_key(s, profile="writer")
        assert a != b != c and a != c

    def test_positional_layout_preserved_for_parsers(self):
        """Downstream parsers split on ':' and read parts[2]=platform,
        parts[3]=chat_type, parts[4]=chat_id (see qqbot adapter
        _parse_gateway_session_key). The profile must occupy parts[1] only."""
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        parts = build_session_key(s, profile="coder").split(":")
        assert parts[0] == "agent"
        assert parts[1] == "coder"  # namespace slot (was always 'main')
        assert parts[2] == "discord"  # platform — unchanged offset
        assert parts[3] == "group"  # chat_type — unchanged offset
        assert parts[4] == "g1"  # chat_id — unchanged offset

    def test_default_namespace_layout_matches_named(self):
        """Default and named keys differ ONLY in parts[1]."""
        s = _src(platform=Platform.SLACK, chat_id="c1", chat_type="channel", user_id="u1")
        d = build_session_key(s, profile="default").split(":")
        n = build_session_key(s, profile="coder").split(":")
        assert d[0] == n[0] == "agent"
        assert d[1] == "main" and n[1] == "coder"
        assert d[2:] == n[2:]  # everything after the namespace is identical


class TestMultiplexConfigFlag:
    """gateway.multiplex_profiles defaults off and round-trips."""

    def test_default_is_false(self):
        assert GatewayConfig().multiplex_profiles is False

    def test_to_dict_includes_flag(self):
        assert GatewayConfig().to_dict()["multiplex_profiles"] is False

    def test_from_dict_top_level(self):
        cfg = GatewayConfig.from_dict({"multiplex_profiles": True})
        assert cfg.multiplex_profiles is True

    def test_from_dict_nested_gateway(self):
        cfg = GatewayConfig.from_dict({"gateway": {"multiplex_profiles": True}})
        assert cfg.multiplex_profiles is True

    def test_from_dict_coerces_truthy_string(self):
        cfg = GatewayConfig.from_dict({"multiplex_profiles": "true"})
        assert cfg.multiplex_profiles is True

    def test_roundtrip(self):
        cfg = GatewayConfig.from_dict(GatewayConfig(multiplex_profiles=True).to_dict())
        assert cfg.multiplex_profiles is True


class TestSessionStoreProfileResolution:
    """SessionStore._generate_session_key honors the flag: legacy namespace
    when off, active-profile namespace when on."""

    def _store(self, tmp_path, **cfg_kw):
        config = GatewayConfig(**cfg_kw)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_flag_off_uses_legacy_namespace(self, tmp_path):
        store = self._store(tmp_path)  # multiplex_profiles defaults False
        s = _src(chat_id="99", chat_type="dm")
        assert store._generate_session_key(s) == "agent:main:telegram:dm:99"
        assert store._generate_session_key(s) == build_session_key(s)

    def test_flag_off_resolve_profile_is_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store._resolve_profile_for_key() is None

    def test_flag_on_uses_active_profile_namespace(self, tmp_path):
        store = self._store(tmp_path, multiplex_profiles=True)
        s = _src(chat_id="99", chat_type="dm")
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"):
            assert store._generate_session_key(s) == "agent:coder:telegram:dm:99"

    def test_flag_on_default_profile_stays_legacy(self, tmp_path):
        store = self._store(tmp_path, multiplex_profiles=True)
        s = _src(chat_id="99", chat_type="dm")
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert store._generate_session_key(s) == "agent:main:telegram:dm:99"


class TestAgentIdDMKeyIsolation:
    """DM with agent_id set and no chat_id must route to a per-user key,
    never the bare shared sink "<agent_prefix>:<platform>:dm".

    Regression guard for the MGA routing path: source.agent_id takes the
    highest-priority namespace slot, and the DM fallback chain must reach
    user_id before falling through to the bare sink — otherwise every DM
    from every user that lacks a chat_id collapses into one shared session
    and cross-user history bleeds across routing boundaries.
    """

    def test_mga_dm_no_chat_id_uses_user_id(self):
        """agent_id="myagent", chat_id=None → key is agent:myagent:<platform>:dm:<user_id>.

        Why: validates that the MGA agent_id prefix + DM fallback to user_id
        works end-to-end, so a DM arriving without a chat_id doesn't sink into
        the shared "agent:myagent:telegram:dm" bucket and pollute other users.
        What: builds key from a source with agent_id set, chat_id empty, user_id present.
        Test: assert key matches the expected per-user pattern, not the bare :dm sink.
        """
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="",          # no chat_id — exercises the DM fallback path
            chat_type="dm",
            user_id="user42",
            agent_id="myagent",  # MGA routing — takes priority over profile
        )
        key = build_session_key(source)
        assert key == "agent:myagent:telegram:dm:user42", (
            f"Expected agent:myagent:telegram:dm:user42 but got {key!r}. "
            "The DM fallback must reach user_id, not collapse to the bare :dm sink."
        )

    def test_mga_dm_no_chat_id_no_user_id_uses_bare_sink(self):
        """With agent_id set but no chat_id AND no user_id, key is the bare per-agent sink.

        Why: documents the intended (unavoidable) fallback when zero identifiers
        are available — at least the MGA agent_id prefix keeps each agent's
        bare sink isolated from other agents.
        What: key ends with :dm and contains the agent_id prefix.
        Test: assert key is "agent:myagent:telegram:dm".
        """
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="",
            chat_type="dm",
            user_id=None,
            agent_id="myagent",
        )
        key = build_session_key(source)
        assert key == "agent:myagent:telegram:dm", (
            f"Expected agent:myagent:telegram:dm but got {key!r}."
        )

    def test_mga_prefix_takes_priority_over_profile(self):
        """agent_id overrides profile in the namespace slot — they must not collide.

        Why: documents the three-tier priority (agent_id > profile > 'main').
        A source with both agent_id and profile set must use the agent_id prefix
        so the session store's MGA routing is authoritative.
        What: build_session_key with agent_id="X" and profile="X" → "agent:X:…",
        but the same key would arise from profile="X" without agent_id — the
        call sites that pass agent_id are the MGA routing path.
        Test: assert key prefix is "agent:myagent:" not "agent:worker:".
        """
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="99",
            chat_type="dm",
            agent_id="myagent",
        )
        # profile kwarg is "worker" but agent_id="myagent" takes priority
        key = build_session_key(source, profile="worker")
        assert key == "agent:myagent:telegram:dm:99", (
            f"agent_id must override profile in key namespace, got {key!r}"
        )


