"""MC-7915 regression: `honcho status` must not print OK when peer data is
unreachable (e.g. Invalid JWT).

Before the guard, `_show_peer_cards` swallowed connection/auth errors and the
status wrapper (`cmd_status`) printed a misleading OK. The guard makes
`_show_peer_cards` return a reachability bool the wrapper honours.

Two layers of coverage:
  * unit — `_show_peer_cards` returns False on failure, True on success;
  * integration — the real `cmd_status` prints DEGRADED (never a bare OK)
    when the peer store is unreachable, and OK when it is reachable.
"""

from types import SimpleNamespace


def _fake_mgr_factory(behaviour):
    class _FakeMgr:
        def __init__(self, *a, **k):
            pass

        def get_or_create(self, key):
            return behaviour()

        def get_peer_card(self, key, peer="user"):
            return []

        def get_ai_representation(self, key):
            return {"representation": ""}

    return _FakeMgr


def _install_fake_mgr(monkeypatch, behaviour):
    import plugins.memory.honcho.session as sess_mod

    monkeypatch.setattr(sess_mod, "HonchoSessionManager", _fake_mgr_factory(behaviour))


def _boom():
    raise Exception("Invalid JWT")


# --------------------------------------------------------------------------- #
# unit: _show_peer_cards reachability verdict
# --------------------------------------------------------------------------- #
class TestPeerDataReachabilityGuard:
    def test_invalid_jwt_returns_false(self, monkeypatch, capsys):
        import plugins.memory.honcho.cli as honcho_cli

        _install_fake_mgr(monkeypatch, _boom)
        hcfg = SimpleNamespace(resolve_session_name=lambda: "workspace")
        ok = honcho_cli._show_peer_cards(hcfg, client=object())
        out = capsys.readouterr().out
        assert ok is False
        assert "Peer data unavailable" in out
        assert "Invalid JWT" in out

    def test_reachable_returns_true(self, monkeypatch):
        import plugins.memory.honcho.cli as honcho_cli

        _install_fake_mgr(monkeypatch, lambda: None)
        hcfg = SimpleNamespace(resolve_session_name=lambda: "workspace")
        assert honcho_cli._show_peer_cards(hcfg, client=object()) is True


# --------------------------------------------------------------------------- #
# integration: drive the real cmd_status wrapper and read its printed verdict
# --------------------------------------------------------------------------- #
def _dummy_hcfg():
    """A fully-populated hcfg so cmd_status's print block runs unmodified.
    raw={} → empty host block → OAuthCredential is None → API-key auth path."""
    return SimpleNamespace(
        host="hermes_iris",
        enabled=True,
        api_key="eyJ.fake.key",
        workspace_id="pka-shared",
        ai_peer="iris",
        peer_name="elmar",
        resolve_session_name=lambda: "workspace",
        session_strategy="per-directory",
        recall_mode="hybrid",
        context_tokens=4000,
        base_url="http://127.0.0.1:8000",
        raw={},
        dialectic_cadence=4,
        reasoning_level_cap="high",
        reasoning_heuristic=True,
        dialectic_reasoning_level="minimal",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=True,
        write_frequency="async",
    )


def _drive_cmd_status(monkeypatch, mgr_behaviour):
    import plugins.memory.honcho.cli as honcho_cli
    import plugins.memory.honcho.client as client_mod
    from pathlib import Path

    monkeypatch.setattr(
        honcho_cli, "_read_config", lambda: {"enabled": True, "apiKey": "x"}
    )
    monkeypatch.setattr(
        honcho_cli, "_config_path", lambda: Path("/tmp/mc7915-fake-honcho.json")
    )
    monkeypatch.setattr(
        honcho_cli, "_local_config_path", lambda: Path("/tmp/mc7915-fake-honcho.json")
    )
    monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: "iris")
    monkeypatch.setattr(
        client_mod.HonchoClientConfig,
        "from_global_config",
        classmethod(lambda cls, host=None: _dummy_hcfg()),
    )
    monkeypatch.setattr(client_mod, "get_honcho_client", lambda hcfg: object())
    _install_fake_mgr(monkeypatch, mgr_behaviour)

    honcho_cli.cmd_status(SimpleNamespace(all=False))


class TestCmdStatusWrapperVerdict:
    def test_status_degraded_not_ok_when_unreachable(self, monkeypatch, capsys):
        _drive_cmd_status(monkeypatch, _boom)
        out = capsys.readouterr().out
        assert "Invalid JWT" in out
        assert "DEGRADED (peer data unavailable)" in out
        # the whole point: no bare OK line survives an unreachable peer store
        assert not any(line.strip() == "OK" for line in out.splitlines())

    def test_status_ok_when_reachable(self, monkeypatch, capsys):
        _drive_cmd_status(monkeypatch, lambda: None)
        out = capsys.readouterr().out
        assert any(line.strip() == "OK" for line in out.splitlines())
        assert "DEGRADED" not in out
