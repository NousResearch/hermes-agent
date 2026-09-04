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
    honcho_json(peerName="eri", aiPeer="hermes", pinUserPeer=True, runtimePeerPrefix="tg_",
                userPeerAliases={"222": "bob", "111": "alice"}, sessionPeerPrefix=True)

    sig = HonchoMemoryProvider().identity_signature()

    assert sig == {
        "user_identity": "eri",
        "agent_identity": "hermes",
        "pin_user_identity": True,
        "runtime_identity_prefix": "tg_",
        "user_identity_aliases": [("111", "alice"), ("222", "bob")],
        "session_prefixing": [True],
    }
    assert not any(k.startswith("honcho") for k in sig)


def test_signature_defaults_without_identity_keys(honcho_json):
    honcho_json()

    sig = HonchoMemoryProvider().identity_signature()

    assert sig["user_identity"] is None
    assert sig["pin_user_identity"] is False
    assert sig["runtime_identity_prefix"] == ""
    assert sig["user_identity_aliases"] == []
    assert sig["session_prefixing"] == [False]


def test_signature_tracks_edits_to_the_file(honcho_json):
    provider = HonchoMemoryProvider()
    honcho_json(peerName="eri", pinUserPeer=True)
    assert provider.identity_signature()["pin_user_identity"] is True

    honcho_json(peerName="eri", pinUserPeer=False)
    assert provider.identity_signature()["pin_user_identity"] is False


def test_signature_is_memoized_on_an_unchanged_file(honcho_json, monkeypatch):
    honcho_json(peerName="eri")
    provider = HonchoMemoryProvider()
    first = provider.identity_signature()

    from plugins.memory.honcho import client as client_module
    monkeypatch.setattr(client_module.HonchoClientConfig, "from_global_config",
                        classmethod(lambda cls, **kw: pytest.fail("config re-read on an unchanged file")))
    assert provider.identity_signature() == first


def test_signature_never_touches_the_network(honcho_json, network_attempts):
    honcho_json(peerName="eri")
    HonchoMemoryProvider().identity_signature()
    assert network_attempts == []


def test_signature_is_empty_when_config_cannot_be_read(honcho_json, monkeypatch):
    honcho_json(peerName="eri")
    from plugins.memory.honcho import client as client_module

    def _boom(cls, **kw):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(client_module.HonchoClientConfig, "from_global_config", classmethod(_boom))
    assert HonchoMemoryProvider().identity_signature() == {}
