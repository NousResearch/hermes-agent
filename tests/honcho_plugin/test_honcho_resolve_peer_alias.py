"""Tests for _resolve_peer_id alias resolution in HonchoSessionManager."""

from types import SimpleNamespace

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _make_manager(aliases=None):
    """Create a minimal HonchoSessionManager with optional user_peer_aliases."""
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
        user_peer_aliases=aliases or {},
    )
    return HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)


def _make_session():
    return HonchoSession(
        key="test",
        user_peer_id="chris",
        assistant_peer_id="hermes",
        honcho_session_id="test-honcho",
    )


def test_resolve_peer_id_returns_user_peer_for_none():
    mgr = _make_manager()
    session = _make_session()
    assert mgr._resolve_peer_id(session, None) == "chris"


def test_resolve_peer_id_returns_user_peer_for_empty():
    mgr = _make_manager()
    session = _make_session()
    assert mgr._resolve_peer_id(session, "  ") == "chris"


def test_resolve_peer_id_maps_ai_keyword():
    mgr = _make_manager()
    session = _make_session()
    assert mgr._resolve_peer_id(session, "ai") == "hermes"


def test_resolve_peer_id_alias_resolves_configured_mapping():
    """Issue #40874: per-call peer args should consult user_peer_aliases."""
    aliases = {"@user:example.org": "_user_aliased"}
    mgr = _make_manager(aliases=aliases)
    session = _make_session()
    assert mgr._resolve_peer_id(session, "@user:example.org") == "_user_aliased"


def test_resolve_peer_id_alias_with_whitespace_strips_and_resolves():
    aliases = {"@user:example.org": "  _user_aliased  "}
    mgr = _make_manager(aliases=aliases)
    session = _make_session()
    assert mgr._resolve_peer_id(session, "@user:example.org") == "_user_aliased"


def test_resolve_peer_id_no_alias_falls_through_to_sanitize():
    """When candidate is not in aliases, normal sanitize path applies."""
    mgr = _make_manager(aliases={})
    session = _make_session()
    # "unknown_peer" is not "user" or "ai", so it gets sanitized and returned
    assert mgr._resolve_peer_id(session, "unknown_peer") == "unknown_peer"


def test_resolve_peer_id_alias_empty_value_falls_through():
    """Alias with empty string value should not match, falls through to sanitize."""
    aliases = {"@user:example.org": ""}
    mgr = _make_manager(aliases=aliases)
    session = _make_session()
    # Empty alias value → falls through to sanitize
    result = mgr._resolve_peer_id(session, "@user:example.org")
    assert result == "-user-example-org"  # sanitized


def test_resolve_peer_id_alias_none_config_uses_empty_dict():
    """When config has no user_peer_aliases attribute, falls through gracefully."""
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=600,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
        # No user_peer_aliases attribute
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = _make_session()
    # Should not crash, should fall through to sanitize
    assert mgr._resolve_peer_id(session, "some_peer") == "some_peer"
