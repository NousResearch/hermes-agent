"""Shared-credential chat/topic routes for multiplexed gateways."""
from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from gateway.run import _resolve_shared_credential_profile_route


def _src(**kw) -> SessionSource:
    kw.setdefault("platform", Platform.TELEGRAM)
    kw.setdefault("chat_id", "-100111")
    kw.setdefault("chat_type", "group")
    return SessionSource(**kw)


def _resolve(config, source):
    with patch("hermes_cli.profiles.profile_exists", return_value=True):
        return _resolve_shared_credential_profile_route(config, source)


def test_mapping_routes_chat_to_profile():
    profile = _resolve(
        {"telegram": {"profile_routes": {"-100111": "research"}}},
        _src(),
    )
    assert profile == "research"


def test_mapping_prefers_topic_specific_route():
    profile = _resolve(
        {
            "telegram": {
                "profile_routes": {
                    "-100111": "general",
                    "-100111:42": "support",
                }
            }
        },
        _src(thread_id="42"),
    )
    assert profile == "support"


def test_list_routes_can_scope_threads():
    profile = _resolve(
        {
            "telegram": {
                "profile_routes": [
                    {"chats": ["-100111"], "threads": ["42"], "profile": "support"},
                    {"chats": ["-100111"], "profile": "fallback"},
                ]
            }
        },
        _src(thread_id="42"),
    )
    assert profile == "support"


def test_aliases_are_accepted():
    profile = _resolve(
        {"telegram": {"chat_profiles": {"-100111": "writer"}}},
        _src(),
    )
    assert profile == "writer"


def test_missing_profile_is_ignored():
    with patch("hermes_cli.profiles.profile_exists", return_value=False):
        profile = _resolve_shared_credential_profile_route(
            {"telegram": {"profile_routes": {"-100111": "ghost"}}},
            _src(),
        )
    assert profile is None


def test_routed_source_uses_namespaced_session_key():
    source = _src(profile="research")
    assert build_session_key(source, profile=source.profile) == "agent:research:telegram:group:-100111"
