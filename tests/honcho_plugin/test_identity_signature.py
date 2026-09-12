"""``HonchoMemoryProvider.identity_signature()``: the identity-mapping values the gateway folds
into its agent-cache key, read from honcho.json without touching the network."""

import json

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


@pytest.fixture
def honcho_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "honcho.json"

    def _write(**values):
        path.write_text(json.dumps({"apiKey": "k", **values}), encoding="utf-8")
        return path

    return _write


def test_signature_uses_neutral_keys(honcho_json):
    honcho_json(workspace="team", peerName="eri", aiPeer="hermes", pinUserPeer=True, runtimePeerPrefix="tg_",
                userPeerAliases={"222": "bob", "111": "alice"}, sessionPeerPrefix=True, a2aSessions=False)

    sig = HonchoMemoryProvider().identity_signature()

    expected = {
        "workspace": "team",
        "user_identity": "eri",
        "agent_identity": "hermes",
        "pin_user_identity": True,
        "runtime_identity_prefix": "tg_",
        "user_identity_aliases": [("111", "alice"), ("222", "bob")],
        "session_prefixing": [True],
        "a2a_sessions": False,
    }
    assert {key: sig[key] for key in expected} == expected
    assert not any(k.startswith("honcho") for k in sig)


def test_signature_tracks_edits_to_the_file(honcho_json):
    provider = HonchoMemoryProvider()
    honcho_json(peerName="eri", pinUserPeer=True)
    assert provider.identity_signature()["pin_user_identity"] is True

    honcho_json(peerName="eri", pinUserPeer=False)
    assert provider.identity_signature()["pin_user_identity"] is False


def test_signature_never_touches_the_network(honcho_json, network_attempts):
    honcho_json(peerName="eri")
    HonchoMemoryProvider().identity_signature()
    assert network_attempts == []


@pytest.mark.parametrize("setting", [
    {'workspace': 'other'},
    {'apiKey': 'secret'},
    {'environment': 'local'},
    {'baseUrl': 'https://honcho.example/v3'},
    {'timeout': 12},
    {'peerName': 'alice'},
    {'aiPeer': 'assistant'},
    {'pinUserPeer': True},
    {'runtimePeerPrefix': 'tg_'},
    {'userPeerAliases': {'1': 'alice'}},
    {'enabled': True},
    {'saveMessages': False},
    {'contextTokens': 1000},
    {'writeFrequency': 'turn'},
    {'dialecticReasoningLevel': 'medium'},
    {'dialecticDynamic': False},
    {'dialecticMaxChars': 700},
    {'dialecticDepth': 2},
    {'dialecticDepth': 2, 'dialecticDepthLevels': ['low', 'high']},
    {'reasoningHeuristic': False},
    {'reasoningLevelCap': 'max'},
    {'observation': {'user': {'observeMe': False}}},
    {'observation': {'user': {'observeOthers': False}}},
    {'observation': {'ai': {'observeMe': False}}},
    {'observation': {'ai': {'observeOthers': False}}},
    {'observationMode': 'directional'},
    {'messageMaxChars': 20000},
    {'dialecticMaxInputChars': 8000},
    {'recallMode': 'tools'},
    {'initOnSessionStart': True},
    {'injectionFrequency': 'first-turn'},
    {'contextCadence': 2},
    {'dialecticCadence': 3},
    {'queryRewrite': True},
    {'firstTurnBaseWait': 0.5},
    {'firstTurnDialecticWait': 0.75},
    {'sessionStrategy': 'per-session'},
    {'sessionPeerPrefix': True},
    {'sessions': {'/project': 'stable'}},
    {"a2aSessions": False},
])
def test_every_effective_setting_changes_signature(honcho_json, setting):
    provider = HonchoMemoryProvider()
    honcho_json()
    before = provider.identity_signature()
    honcho_json(**setting)
    assert provider.identity_signature() != before


def test_active_host_change_without_file_edit_changes_signature(honcho_json, monkeypatch):
    from plugins.memory.honcho import client

    honcho_json(hosts={name: {"workspace": "same", "aiPeer": "same"} for name in ("hermes_alpha", "hermes_beta")})
    provider = HonchoMemoryProvider()
    monkeypatch.setattr(client, "resolve_active_host", lambda: "hermes_alpha")
    before = provider.identity_signature()
    monkeypatch.setattr(client, "resolve_active_host", lambda: "hermes_beta")
    assert provider.identity_signature() != before


@pytest.mark.parametrize("setting", [
    {"base_url": "https://cfg.example"}, {"timeout": 10}, {"request_timeout": 11},
])
def test_yaml_transport_fallback_changes_signature(honcho_json, monkeypatch, setting):
    from hermes_cli import config

    honcho_json()
    provider = HonchoMemoryProvider()
    monkeypatch.setattr(config, "load_config_readonly", lambda: {})
    before = provider.identity_signature()
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"honcho": setting})
    assert provider.identity_signature() != before


def test_signature_ignores_unknown_settings_and_token_refresh(honcho_json):
    provider = HonchoMemoryProvider()

    def configure(unrelated, access, refresh):
        honcho_json(unrelated=unrelated, hosts={"hermes": {
            "apiKey": access, "oauth": {"refreshToken": refresh},
        }})

    configure(1, "access-a", "refresh-a")
    before = provider.identity_signature()
    configure(2, "access-b", "refresh-a")
    assert provider.identity_signature() == before
    configure(2, "access-c", "refresh-b")
    after = provider.identity_signature()
    assert after != before
    assert "access-c" not in json.dumps(after)
    assert "refresh-b" not in json.dumps(after)
