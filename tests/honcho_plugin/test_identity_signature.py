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
        path.write_text(json.dumps({"apiKey": "k", **values}))
        return path

    return _write


def test_signature_uses_neutral_keys(honcho_json):
    honcho_json(workspace="team", peerName="eri", aiPeer="hermes", pinUserPeer=True, runtimePeerPrefix="tg_",
                userPeerAliases={"222": "bob", "111": "alice"}, sessionPeerPrefix=True, a2aSessions=False)

    sig = HonchoMemoryProvider().identity_signature()

    assert sig == {
        "workspace": "team",
        "user_identity": "eri",
        "agent_identity": "hermes",
        "pin_user_identity": True,
        "runtime_identity_prefix": "tg_",
        "user_identity_aliases": [("111", "alice"), ("222", "bob")],
        "session_prefixing": [True],
        "a2a_sessions": False,
    }
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
