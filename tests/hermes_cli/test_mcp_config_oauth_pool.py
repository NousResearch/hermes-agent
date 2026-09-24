"""Config-mutation ↔ OAuth-credential transaction boundary (``hermes_cli.mcp_config``).

Two invariants the manager keeps for LIVE providers must also hold when config.yaml is edited
through ``_save_mcp_server`` / ``_remove_mcp_server`` / ``_replace_mcp_servers``:

* one profile leaving a root-exported shared pool must not revoke the grant for root and its
  siblings, while the root changing or removing its own export must;
* nothing on disk is destroyed (tokens, client info, tombstone) until ``save_config()`` has
  committed, so a failed write leaves the old entry authoritative WITH its credentials.

Every test exercises the real mutation helpers against a temp root + named profiles laid out the
way ``get_hermes_home()`` sees them (``<root>/profiles/<name>``), switching the active home the way
the gateway does (``set_hermes_home_override``).
"""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

SERVER = {"url": "https://mcp.example.com/mcp", "auth": "oauth"}
CHANGED = {"url": "https://mcp.example.com/other", "auth": "oauth"}


def _write_config(home: Path, servers: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")


def _servers(home: Path) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("mcp_servers") or {}


def _seed_tokens(home: Path, name: str, access_token: str) -> Path:
    token_dir = home / "mcp-tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    (token_dir / f"{name}.client.json").write_text(json.dumps({"client_id": "cid"}), encoding="utf-8")
    path = token_dir / f"{name}.json"
    path.write_text(json.dumps({"access_token": access_token, "token_type": "Bearer", "expires_in": 3600}), encoding="utf-8")
    return path


@pytest.fixture
def estate(tmp_path, monkeypatch):
    """Root exporting ``team`` + profiles A and B that carry an identical entry (both share the root
    grant). Yields ``(root, a, b, activate)`` where ``activate(home)`` re-points every home-derived
    reader (env, override, config path) at *home*, the way a profile-scoped turn does."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth_manager import reset_manager_for_tests

    root = tmp_path / ".hermes"
    a, b = root / "profiles" / "a", root / "profiles" / "b"
    _write_config(root, {"team": {**SERVER, "oauth": {"share_with_profiles": True}}})
    _write_config(a, {"team": dict(SERVER)})
    _write_config(b, {"team": dict(SERVER)})
    _seed_tokens(root, "team", "ROOT_GRANT")
    reset_manager_for_tests()

    state = {"token": None}

    def activate(home: Path):
        if state["token"] is not None:
            reset_hermes_home_override(state["token"])
        state["token"] = set_hermes_home_override(home)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
        monkeypatch.setattr("hermes_cli.mcp_config.get_hermes_home", lambda: home)

    yield root, a, b, activate
    if state["token"] is not None:
        reset_hermes_home_override(state["token"])
    reset_manager_for_tests()


def _grant(root: Path) -> Path:
    return root / "mcp-tokens" / "team.json"


def _build_all(root: Path, a: Path, b: Path, activate):
    """Cache a live provider for root, A and B — all three on the root pool."""
    from tools.mcp_oauth_manager import get_manager

    providers = {}
    for home in (root, a, b):
        activate(home)
        providers[home] = get_manager().get_or_build_provider("team", SERVER["url"], None)
        assert providers[home] is not None
        assert asyncio.run(providers[home].context.storage.get_tokens()).access_token == "ROOT_GRANT"
    return providers


# ---------------------------------------------------------------------------
# Blocker 1: participant detach vs shared-pool revocation
# ---------------------------------------------------------------------------

def test_profile_identity_change_via_save_detaches_without_revoking_root_grant(estate):
    from hermes_cli.mcp_config import _save_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)
    manager = get_manager()

    activate(a)
    assert _save_mcp_server("team", dict(CHANGED)) is True

    assert _grant(root).exists(), "A detaching must not delete the root grant"
    assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT"
    assert _servers(a)["team"] == CHANGED
    # A's provider is gone (it was built on the root pool); root's and B's survive untouched.
    assert manager._key("team", a) not in manager._entries
    assert manager._key("team", root) in manager._entries
    assert manager._key("team", b) in manager._entries
    # A now resolves a LOCAL pool for the new identity, root/B still the shared one.
    assert HermesTokenStorage("team", hermes_home=a).pool_path == str((a / "mcp-tokens" / "team.json").resolve())
    assert HermesTokenStorage("team", hermes_home=b).pool_path == str(_grant(root).resolve())
    assert not (a / "mcp-tokens" / "team.json").exists()


def test_profile_identity_change_via_replace_detaches_without_revoking_root_grant(estate):
    from hermes_cli.mcp_config import _replace_mcp_servers
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)

    activate(a)
    assert _replace_mcp_servers({"team": dict(CHANGED)}) == (True, [])

    assert _grant(root).exists()
    assert _servers(a)["team"] == CHANGED
    manager = get_manager()
    assert manager._key("team", a) not in manager._entries
    assert manager._key("team", root) in manager._entries
    assert manager._key("team", b) in manager._entries


def test_profile_removing_its_entry_detaches_without_revoking_or_tombstoning_root_grant(estate):
    from hermes_cli.mcp_config import _remove_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)

    activate(a)
    assert _remove_mcp_server("team") is True

    assert "team" not in _servers(a)
    assert _grant(root).exists(), "a participant's removal is a detach, not a revocation"
    assert not HermesTokenStorage("team", hermes_home=root).rebuild_blocked()
    assert not HermesTokenStorage("team", hermes_home=b).rebuild_blocked()
    manager = get_manager()
    assert manager._key("team", a) not in manager._entries
    assert manager._key("team", b) in manager._entries


def test_root_identity_change_revokes_the_shared_grant_for_every_participant(estate):
    from hermes_cli.mcp_config import _save_mcp_server
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    providers = _build_all(root, a, b, activate)

    activate(root)
    assert _save_mcp_server("team", {**CHANGED, "oauth": {"share_with_profiles": True}}) is True

    assert not _grant(root).exists(), "the owner changing the exported identity revokes the pool"
    manager = get_manager()
    assert not any(k[1] == "team" for k in manager._entries), "root, A and B were all on that pool"
    for provider in providers.values():
        assert provider.context.current_tokens is None


def test_root_removal_revokes_and_tombstones_the_shared_grant(estate):
    from hermes_cli.mcp_config import _remove_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)

    activate(root)
    assert _remove_mcp_server("team") is True

    assert not _grant(root).exists()
    assert HermesTokenStorage("team", hermes_home=root).rebuild_blocked()
    assert not any(k[1] == "team" for k in get_manager()._entries)
    # B's config still names the server but the export is gone: B resolves its own (empty) pool.
    activate(b)
    assert HermesTokenStorage("team", hermes_home=b).pool_path == str((b / "mcp-tokens" / "team.json").resolve())


def test_toggling_the_export_flag_alone_revokes_nothing(estate):
    """``share_with_profiles`` decides WHERE the grant lives, not what it is for."""
    from hermes_cli.mcp_config import _save_mcp_server

    root, a, b, activate = estate
    activate(root)
    assert _save_mcp_server("team", dict(SERVER)) is True  # export withdrawn, identity unchanged
    assert _grant(root).exists()
    assert _save_mcp_server("team", {**SERVER, "oauth": {"share_with_profiles": True}}) is True
    assert _grant(root).exists()


# ---------------------------------------------------------------------------
# Blocker 2: credentials survive a failed save_config()
# ---------------------------------------------------------------------------

def _fail_save(monkeypatch):
    def boom(*_a, **_k):
        raise OSError(30, "Read-only file system")
    monkeypatch.setattr("hermes_cli.mcp_config.save_config", boom)


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A single home with a local OAuth grant + a cached provider."""
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    home = tmp_path / ".hermes"
    _write_config(home, {"srv": dict(SERVER)})
    grant = _seed_tokens(home, "srv", "LOCAL")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
    monkeypatch.setattr("hermes_cli.mcp_config.get_hermes_home", lambda: home)
    reset_manager_for_tests()
    assert get_manager().get_or_build_provider("srv", SERVER["url"], None) is not None
    yield home, grant
    reset_manager_for_tests()


def _assert_untouched(home: Path, grant: Path) -> None:
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    assert _servers(home)["srv"] == SERVER, "old config must remain authoritative"
    assert json.loads(grant.read_text())["access_token"] == "LOCAL", "old token must remain usable"
    assert (home / "mcp-tokens" / "srv.client.json").exists()
    assert not HermesTokenStorage("srv", hermes_home=home).rebuild_blocked(), "no tombstone from a failed mutation"
    assert get_manager()._key("srv", home) in get_manager()._entries, "cached provider still serves the old config"


def test_failed_save_on_remove_keeps_config_credentials_and_leaves_no_tombstone(local, monkeypatch):
    from hermes_cli.mcp_config import _remove_mcp_server

    home, grant = local
    _fail_save(monkeypatch)
    with pytest.raises(OSError):
        _remove_mcp_server("srv")
    _assert_untouched(home, grant)


def test_failed_save_on_identity_change_keeps_config_and_credentials(local, monkeypatch):
    from hermes_cli.mcp_config import _save_mcp_server

    home, grant = local
    _fail_save(monkeypatch)
    with pytest.raises(OSError):
        _save_mcp_server("srv", dict(CHANGED))
    _assert_untouched(home, grant)


def test_failed_save_on_replace_keeps_config_and_credentials(local, monkeypatch):
    from hermes_cli.mcp_config import _replace_mcp_servers

    home, grant = local
    _fail_save(monkeypatch)
    with pytest.raises(OSError):
        _replace_mcp_servers({})
    _assert_untouched(home, grant)


def test_failed_save_on_shared_pool_keeps_root_grant_for_every_participant(estate, monkeypatch):
    """The shared-pool variant: root's failed removal leaves the grant root, A and B all use."""
    from hermes_cli.mcp_config import _remove_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)
    activate(root)
    _fail_save(monkeypatch)
    with pytest.raises(OSError):
        _remove_mcp_server("team")
    assert _servers(root)["team"]["oauth"] == {"share_with_profiles": True}
    assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT"
    assert not HermesTokenStorage("team", hermes_home=root).rebuild_blocked()
    assert all(get_manager()._key("team", h) in get_manager()._entries for h in (root, a, b))


def test_successful_change_never_resurrects_over_a_concurrent_authorization(local):
    """Post-commit revocation deletes the OLD pool; there is no restore step that could clobber a
    grant another process wrote in the meantime. Mirror the reviewer's concurrency concern by
    landing a fresh token between the config write and the credential commit."""
    from hermes_cli import mcp_config
    from hermes_cli.mcp_config import _save_mcp_server

    home, grant = local
    real_save = mcp_config.save_config

    def save_then_peer_authorizes(config, **kw):
        real_save(config, **kw)
        grant.write_text(json.dumps({"access_token": "PEER", "token_type": "Bearer"}), encoding="utf-8")

    mcp_config.save_config = save_then_peer_authorizes
    try:
        assert _save_mcp_server("srv", dict(CHANGED)) is True
    finally:
        mcp_config.save_config = real_save
    # The old identity's pool IS this home's pool, so it is revoked — including the peer's write,
    # which was minted for the old endpoint. Nothing is written back.
    assert not grant.exists()
    assert not (home / "mcp-tokens" / "srv.client.json").exists()


# ---------------------------------------------------------------------------
# Tombstone lifecycle through the config surface
# ---------------------------------------------------------------------------

def test_readding_a_removed_server_lifts_the_rebuild_tombstone(local, monkeypatch):
    from hermes_cli.mcp_config import _remove_mcp_server, _save_mcp_server
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import get_manager

    home, grant = local
    assert _remove_mcp_server("srv") is True
    assert HermesTokenStorage("srv", hermes_home=home).rebuild_blocked()
    # Interactive stdin so the only thing standing between us and a provider is the tombstone.
    from unittest.mock import MagicMock
    stdin = MagicMock(); stdin.isatty.return_value = True
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", stdin)
    assert get_manager().get_or_build_provider("srv", SERVER["url"], None) is None, "blocked while tombstoned"

    assert _save_mcp_server("srv", dict(SERVER)) is True
    assert not HermesTokenStorage("srv", hermes_home=home).rebuild_blocked()
    assert get_manager().get_or_build_provider("srv", SERVER["url"], None) is not None


# ---------------------------------------------------------------------------
# Re-authorization on a shared pool: ``hermes mcp login`` from a participant
# ---------------------------------------------------------------------------

def _args(name: str):
    import argparse
    return argparse.Namespace(name=name, flow=None)


def test_participant_login_failure_restores_the_shared_grant_for_everyone(estate, monkeypatch, capsys):
    """B runs ``hermes mcp login team`` and the flow fails: root's grant must come back — a failed
    participant login must not log root and A out."""
    from hermes_cli.mcp_config import cmd_mcp_login
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    _build_all(root, a, b, activate)
    activate(b)

    def failed_probe(name, cfg, connect_timeout=30):
        assert not _grant(root).exists(), "the flow starts from a cleared pool"
        raise RuntimeError("authorization failed")

    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", failed_probe)
    cmd_mcp_login(_args("team"))

    assert "Authentication failed" in capsys.readouterr().out
    assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT"
    assert (root / "mcp-tokens" / "team.client.json").exists()
    manager = get_manager()
    assert manager._key("team", root) in manager._entries
    assert manager._key("team", a) in manager._entries


def test_participant_login_without_a_token_restores_the_shared_grant(estate, monkeypatch, capsys):
    """The Google-Drive shape: the probe lists tools but no token lands. Same restore."""
    from hermes_cli.mcp_config import cmd_mcp_login

    root, a, b, activate = estate
    _build_all(root, a, b, activate)
    activate(b)
    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server",
                        lambda name, cfg, connect_timeout=30: [("public_tool", "no auth needed")])
    cmd_mcp_login(_args("team"))

    assert "no OAuth token was obtained" in capsys.readouterr().out
    assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT"


def test_participant_login_success_rotates_the_shared_grant_for_everyone(estate, monkeypatch, capsys):
    """B re-authorizes successfully: root and A keep their providers and pick up the new grant from
    disk on their next request (the disk watch), instead of being fenced."""
    from hermes_cli.mcp_config import cmd_mcp_login
    from tools.mcp_oauth_manager import get_manager

    root, a, b, activate = estate
    providers = _build_all(root, a, b, activate)
    activate(b)

    def successful_probe(name, cfg, connect_timeout=30):
        _grant(root).write_text(json.dumps({"access_token": "NEW", "token_type": "Bearer", "expires_in": 3600}))
        return [("tool", "")]

    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", successful_probe)
    cmd_mcp_login(_args("team"))

    assert "Authenticated" in capsys.readouterr().out
    assert json.loads(_grant(root).read_text())["access_token"] == "NEW"
    manager = get_manager()
    for home in (root, a):
        assert manager._key("team", home) in manager._entries
        assert not providers[home]._hermes_detached
        assert asyncio.run(manager.invalidate_if_disk_changed("team", hermes_home=home)) is True
        assert asyncio.run(providers[home].context.storage.get_tokens()).access_token == "NEW"


# ---------------------------------------------------------------------------
# Authorizing an entry that is not saved yet (dashboard / connector card flows)
# ---------------------------------------------------------------------------

def test_authorizing_a_changed_endpoint_never_touches_the_shared_pool(estate):
    """A shares the root grant; the dashboard authorizes an EDIT of A's entry to another endpoint
    before saving it. The saved config still matches root, but the pool for the entry being
    authorized is A's own — the root grant must be neither cleared nor overwritten."""
    from tools.mcp_oauth import HermesTokenStorage

    root, a, b, activate = estate
    activate(a)
    assert HermesTokenStorage("team").pool_path == str(_grant(root).resolve()), "precondition: A shares"
    storage = HermesTokenStorage("team", requested=CHANGED)
    assert storage.pool_path == str((a / "mcp-tokens" / "team.json").resolve())


def test_authorizing_a_changed_oauth_client_never_touches_the_shared_pool(estate):
    from tools.mcp_oauth import HermesTokenStorage

    root, a, b, activate = estate
    activate(a)
    edited = {**SERVER, "oauth": {"client_id": "someone-elses-client"}}
    assert HermesTokenStorage("team", requested=edited).pool_path == str((a / "mcp-tokens" / "team.json").resolve())


def test_authorizing_a_changed_transport_never_touches_the_shared_pool(estate):
    """Transport is part of the shared identity: a pending transport edit resolves to A's own pool
    even though the saved entry (still ``http``) matches root. The resolver must read the whole
    pending identity, not overlay url/oauth on the saved entry."""
    from tools.mcp_oauth import HermesTokenStorage

    root, a, b, activate = estate
    activate(a)
    assert HermesTokenStorage("team").pool_path == str(_grant(root).resolve()), "precondition: A shares"
    edited = {**SERVER, "transport": "sse"}
    assert HermesTokenStorage("team", requested=edited).pool_path == str((a / "mcp-tokens" / "team.json").resolve())


def test_dashboard_authorizing_a_transport_edit_lands_locally_and_saves_it(estate, monkeypatch):
    """End to end through the dashboard worker: a sharing profile authorizes a transport change
    before saving it. The grant lands in A's own pool, root and sibling state stay byte-identical,
    and the save keeps the grant it just minted."""
    import hermes_cli.mcp_config as mcp_config
    from hermes_cli.web_server_mcp import _run_dashboard_mcp_oauth
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    root, a, b, activate = estate
    activate(a)
    before = {p: p.read_bytes() for p in (root / "mcp-tokens").iterdir()}
    edited = {**SERVER, "transport": "sse"}

    def probe(name, cfg, **_kwargs):
        # The authorization: the provider writes the new grant to the pool resolved for *cfg*.
        _seed_tokens(a, name, "MINTED_FOR_SSE")
        return [("tool", "desc")]

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    flow = DashboardOAuthFlow(flow_id="f-transport", server_name="team", profile="a",
                              hermes_home=str(a), redirect_uri="http://127.0.0.1/cb")
    _run_dashboard_mcp_oauth(flow, dict(edited))

    assert flow.status != "error", getattr(flow, "error", None)
    assert {p: p.read_bytes() for p in (root / "mcp-tokens").iterdir()} == before, "root pool untouched"
    assert json.loads(_grant(a).read_text())["access_token"] == "MINTED_FOR_SSE"
    assert _servers(a)["team"].get("transport") == "sse"
    assert _servers(b)["team"] == SERVER, "sibling config untouched"


def test_saving_an_authorized_edit_keeps_the_grant_it_just_minted(estate, monkeypatch):
    """A local-pool profile authorizes an endpoint change, then the flow saves it: the grant the
    authorization wrote must survive the identity-change revocation that the save would
    otherwise apply to that same pool."""
    from hermes_cli.mcp_config import _save_mcp_server
    from tools.mcp_oauth import HermesTokenStorage

    root, a, b, activate = estate
    _write_config(b, {"team": dict(SERVER)})
    _write_config(root, {"team": dict(SERVER)})  # no export: B's pool is its own
    activate(b)
    storage = HermesTokenStorage("team", requested=CHANGED)
    minted = _seed_tokens(b, "team", "MINTED_FOR_CHANGED")
    assert storage.pool_path == str(minted.resolve())

    assert _save_mcp_server("team", dict(CHANGED), authorized_pool=storage.pool_path) is True
    assert json.loads(minted.read_text())["access_token"] == "MINTED_FOR_CHANGED"


def test_saving_an_unauthorized_edit_still_revokes_the_old_identity(estate):
    """The inverse: without an authorization for the new identity, the old grant goes."""
    from hermes_cli.mcp_config import _save_mcp_server

    root, a, b, activate = estate
    _write_config(b, {"team": dict(SERVER)})
    _write_config(root, {"team": dict(SERVER)})
    activate(b)
    old = _seed_tokens(b, "team", "FOR_OLD_ENDPOINT")
    assert _save_mcp_server("team", dict(CHANGED)) is True
    assert not old.exists()
