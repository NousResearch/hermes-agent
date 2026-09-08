"""Tests for ``hermes honcho peers map``: account discovery, the resolution preview, and alias writes."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

import plugins.memory.honcho.cli as honcho_cli
from plugins.memory.honcho.cli import _preview_peer_resolution, _seen_gateway_accounts


def _make_state_db(path, rows):
    """Create a minimal sessions table with only the columns the query reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE sessions (
               id TEXT PRIMARY KEY, source TEXT, user_id TEXT, session_key TEXT,
               display_name TEXT, origin_json TEXT, started_at REAL
           )"""
    )
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _origin(user_name=None, is_bot=False, user_id_alt=None):
    return json.dumps({"user_name": user_name, "is_bot": is_bot, "user_id_alt": user_id_alt})


def _row(user_id, platform="telegram", name="eri", started=100.0):
    return (f"s{user_id}", platform, user_id, "k", None, _origin(name), started)


def _cfg(root=None, **host):
    return {"apiKey": "***", **(root or {}), "hosts": {"hermes": host}}


class TestSeenGatewayAccounts:
    def test_groups_orders_and_filters_rows(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "telegram", "111", None, "Eri", _origin("eri"), 100.0),  # predates session_key
            ("s2", "telegram", "111", "", "Eri", _origin("eri"), 200.0),
            ("s3", "discord", "222", "k2", "Tek DM", None, 900.0),
            ("s4", "discord", "333", "k3", "Bot", _origin("webhook", is_bot=True), 950.0),
            ("s5", "cli", None, None, None, None, 999.0),
        ])
        assert _seen_gateway_accounts(db) == [
            {"platform": "discord", "user_id": "222", "user_id_alt": "", "label": "Tek DM", "sessions": 1, "profiles": []},
            {"platform": "telegram", "user_id": "111", "user_id_alt": "", "label": "eri", "sessions": 2, "profiles": []},
        ]

    def test_missing_db_or_table_returns_empty(self, tmp_path):
        assert _seen_gateway_accounts(tmp_path / "absent.db") == []
        sqlite3.connect(tmp_path / "empty.db").close()
        assert _seen_gateway_accounts(tmp_path / "empty.db") == []


@pytest.mark.parametrize("user_id, kwargs, expected", [
    ("111", dict(pin=True, aliases={"111": "alice"}, prefix="tg_", peer_name="eri"), "eri (pinned)"),
    ("111", dict(pin=True, aliases={}, prefix="", peer_name=""), "111"),
    ("111", dict(pin=False, aliases={"111": "alice"}, prefix="tg_", peer_name="eri"), "alice"),
    ("111", dict(pin=False, aliases={}, prefix="tg_", peer_name="eri"), "tg_111 (prefixed)"),
    ("@you:matrix.org", dict(pin=False, aliases={}, prefix="", peer_name=""), "-you-matrix-org"),
    ("device-777", dict(pin=False, aliases={"uuid-abc": "eri"}, prefix="", peer_name="", user_id_alt="uuid-abc"), "eri"),
    ("111", dict(pin=False, aliases={"111": "alice", "uuid-abc": "bob"}, prefix="", peer_name="", user_id_alt="uuid-abc"),
     "alice"),
], ids=["pin-wins", "pin-without-peer-name", "alias", "prefix", "raw-sanitized", "alt-id-alias", "primary-alias-over-alt"])
def test_preview_peer_resolution(user_id, kwargs, expected):
    assert _preview_peer_resolution(user_id, **kwargs) == expected


def _run_map(monkeypatch, tmp_path, *, answers, cfg, db_rows=(), ws_peers=None, workspaces=None):
    """Drive cmd_peers_map with scripted answers; returns what _write_config received."""
    db = tmp_path / "state.db"
    if db_rows:
        _make_state_db(db, list(db_rows))
    written = {}
    answer_iter = iter(answers)
    # One profile per host block: "hermes" is default, "hermes.<name>" is profile <name>.
    profiles = [("default" if k == "hermes" else k.removeprefix("hermes."), k, v) for k, v in cfg["hosts"].items()]
    # API seams are offline unless ws_peers is given, then a sentinel client stands in.
    client = object() if ws_peers is not None else None
    for name, value in {
        "_read_config": lambda: cfg,
        "_host_key": lambda: "hermes",
        "_active_profile_name": lambda: "default",
        "_state_db_path": lambda: db,
        "_local_config_path": lambda: tmp_path / "honcho.json",
        "_write_config": lambda c, path=None: written.update({"cfg": c}),
        "_all_profile_host_configs": lambda: profiles,
        "_peers_map_client": lambda workspace=None: (client, SimpleNamespace(workspace_id="hermes") if client else None),
        "_api_workspace_peers": lambda c: list(ws_peers) if c is not None and ws_peers is not None else None,
        "_api_workspaces": lambda c: list(workspaces) if workspaces is not None else None,
        "_api_peer_detail": lambda c, pid: f"(card of {pid})",
        "_prompt": lambda label, default=None, secret=False: next(answer_iter, default or ""),
    }.items():
        monkeypatch.setattr(honcho_cli, name, value)
    honcho_cli.cmd_peers_map(SimpleNamespace())
    return written


class TestCmdPeersMap:
    @pytest.mark.parametrize("cfg, answers, kwargs", [
        (_cfg(peerName="eri"), ["1", "eri", ""], dict(db_rows=[_row("111")])),
        (_cfg(), ["111", "eri", ""], {}),
        (_cfg(peerName="eri"), ["1", "p1", ""], dict(db_rows=[_row("111")], ws_peers=["eri", "hermes"])),
        (_cfg(pinUserPeer=True, peerName="eri"), ["y", "111", "eri", ""], {}),
    ], ids=["account-number", "typed-runtime-id", "picked-from-peers-table", "pinned-but-accepted"])
    def test_writes_alias_to_host_block(self, monkeypatch, tmp_path, cfg, answers, kwargs):
        written = _run_map(monkeypatch, tmp_path, answers=answers, cfg=cfg, **kwargs)
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"111": "eri"}

    @pytest.mark.parametrize("aliases, expected_host", [
        ({"111": "eri", "222": "tek"}, {"userPeerAliases": {"222": "tek"}}),
        ({"111": "eri"}, {"userPeerAliases": {}}),
    ], ids=["one-of-two", "last-alias-keeps-empty-map"])
    def test_dash_clears_alias(self, monkeypatch, tmp_path, aliases, expected_host):
        written = _run_map(monkeypatch, tmp_path, answers=["111", "-", ""], cfg=_cfg(userPeerAliases=aliases))
        assert written["cfg"]["hosts"]["hermes"] == expected_host

    def test_cleared_host_map_does_not_resurrect_root_aliases(self, monkeypatch, tmp_path, capsys):
        """A host block without the key inherits root, so clearing must leave an empty map behind."""
        from plugins.memory.honcho.client import HonchoClientConfig

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _cfg(root={"userPeerAliases": {"111": "root-person"}}, userPeerAliases={"111": "host-person"})
        written = _run_map(monkeypatch, tmp_path, answers=["111", "-", ""], cfg=cfg)
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {}
        assert "root aliases no longer apply to [hermes]" in capsys.readouterr().out

        path = tmp_path / "honcho.json"
        path.write_text(json.dumps(written["cfg"]))
        assert HonchoClientConfig.from_global_config(host="hermes", config_path=path).user_peer_aliases == {}

    @pytest.mark.parametrize("cfg, answers, kwargs", [
        (_cfg(), [""], {}),
        (_cfg(userPeerAliases={"111": "eri"}), ["111", "eri", ""], {}),
        (_cfg(pinUserPeer=True, peerName="eri"), ["n"], {}),
        (_cfg(peerName="eri"), ["w", "2", "n", ""], dict(ws_peers=["eri"], workspaces=["hermes", "cosmania-dex"])),
    ], ids=["nothing-entered", "kept-current-value", "pinned-declined", "workspace-switch-declined"])
    def test_nothing_changed_writes_nothing(self, monkeypatch, tmp_path, cfg, answers, kwargs):
        assert _run_map(monkeypatch, tmp_path, answers=answers, cfg=cfg, **kwargs) == {}

    @pytest.mark.parametrize("cfg, answers, kwargs, expected", [
        (_cfg(peerName="eri"), ["1", "fresh-name", ""], dict(db_rows=[_row("111", name="bob")], ws_peers=["eri", "111"]),
         ["'fresh-name' is a new peer", "peer '111' keeps its existing history"]),
        (_cfg(peerName="eri"), [""], dict(db_rows=[_row("111"), _row("222", "discord", "b", 50.0)], ws_peers=["eri", "111"]),
         ["111 ✓", "222 ○ new"]),
        (_cfg(), ["7654321", "eri", ""], {}, ["peers unavailable"]),
        (_cfg(peerName="eri"), ["p2", ""], dict(ws_peers=["eri", "meow"]), ["(card of meow)"]),
        (_cfg(peerName="eri"), [""], dict(ws_peers=[]), ["No peers here yet", "Wrong workspace?"]),
        (_cfg(peerName="eri"), [""], dict(ws_peers=["stranger1", "stranger2"]),
         ["None of these match your configured identity"]),
        ({"apiKey": "***", "userPeerAliases": {},
          "hosts": {"hermes": {"workspace": "hermes"}, "hermes.dreamer": {"workspace": "dreamland"}}},
         ["222", "tek", "", "all"], {}, ["also apply in workspace 'dreamland'"]),
        ({"apiKey": "***", "hosts": {"hermes": {"peerName": "eri", "userPeerAliases": {"111": "eri"}},
                                     "hermes.dreamer": {"peerName": "eri", "userPeerAliases": {"111": "bob"}}}},
         [""], dict(db_rows=[_row("111", name="x")]), ["≠ dreamer→bob"]),
    ], ids=["new-peer-and-history-consequences", "exists-markers", "offline-typed-targets", "inspect-peer-card",
            "empty-workspace-hint", "unrecognized-workspace-hint", "cross-workspace-root-write-warns",
            "sibling-divergence-marked"])
    def test_view_and_consequence_messages(self, monkeypatch, tmp_path, capsys, cfg, answers, kwargs, expected):
        _run_map(monkeypatch, tmp_path, answers=answers, cfg=cfg, **kwargs)
        out = capsys.readouterr().out
        for text in expected:
            assert text in out

    def test_peers_dispatches_map_action(self, monkeypatch):
        called = {}
        monkeypatch.setattr(honcho_cli, "cmd_peers_map", lambda a: called.update({"map": True}))
        honcho_cli.cmd_peers(SimpleNamespace(peers_action="map"))
        assert called == {"map": True}


class TestWorkspaceSwitch:
    def test_browse_and_confirmed_switch_writes_workspace(self, monkeypatch, tmp_path):
        written = _run_map(
            monkeypatch, tmp_path, answers=["w", "2", "y", ""],
            cfg=_cfg(peerName="eri"), ws_peers=["eri"], workspaces=["hermes", "cosmania-dex"],
        )
        assert written["cfg"]["hosts"]["hermes"]["workspace"] == "cosmania-dex"

    def test_browse_client_is_built_for_the_browsed_workspace(self, monkeypatch):
        """get_honcho_client keys its cache on workspace_id, so a browse must not reuse the profile's client."""
        import plugins.memory.honcho.client as client_mod

        seen = []
        base = client_mod.HonchoClientConfig(host="hermes", workspace_id="hermes", api_key="k")
        monkeypatch.setattr(client_mod.HonchoClientConfig, "from_global_config",
                            classmethod(lambda cls, host=None, config_path=None: base))
        monkeypatch.setattr(client_mod, "get_honcho_client", lambda cfg: seen.append(cfg) or object())
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")

        _, own = honcho_cli._peers_map_client()
        _, browsed = honcho_cli._peers_map_client(workspace="cosmania-dex")

        assert own.workspace_id == "hermes"
        assert browsed.workspace_id == "cosmania-dex"
        assert [c.workspace_id for c in seen] == ["hermes", "cosmania-dex"]


class TestSaveScope:
    @pytest.mark.parametrize("hosts, answers", [
        ({"hermes": {}}, ["222", "tek", ""]),
        ({"hermes": {}, "hermes.dreamer": {}}, ["222", "tek", "", "all"]),
    ], ids=["single-profile-no-prompt", "multi-profile-all"])
    def test_root_sourced_aliases_write_back_to_root(self, monkeypatch, tmp_path, hosts, answers):
        cfg = {"apiKey": "***", "userPeerAliases": {"111": "eri"}, "hosts": hosts}
        written = _run_map(monkeypatch, tmp_path, answers=answers, cfg=cfg)
        assert written["cfg"]["userPeerAliases"] == {"111": "eri", "222": "tek"}
        assert "userPeerAliases" not in written["cfg"]["hosts"]["hermes"]

    def test_scope_this_forks_host_block(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "userPeerAliases": {"111": "eri"}, "hosts": {"hermes": {}, "hermes.dreamer": {}}}
        written = _run_map(monkeypatch, tmp_path, answers=["222", "tek", "", "this"], cfg=cfg)
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"111": "eri", "222": "tek"}
        assert written["cfg"]["userPeerAliases"] == {"111": "eri"}  # root stays the other profiles' baseline


def test_classify_workspace_peers_labels_from_local_config(monkeypatch):
    monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
    monkeypatch.setattr(honcho_cli, "_all_profile_host_configs", lambda: [
        ("default", "hermes", {"peerName": "eri", "aiPeer": "hermetika"}),
        ("dreamer", "hermes.dreamer", {"aiPeer": "dreamer-ai"}),
    ])
    cfg = {"peerName": "eri", "hosts": {"claude_code": {"aiPeer": "clawd"}}}
    accounts = [{"platform": "telegram", "user_id": "7654321", "user_id_alt": ""}]
    labels = honcho_cli._classify_workspace_peers(
        ["eri", "hermetika", "dreamer-ai", "clawd", "7654321", "tg_7654321", "friend", "user-default-root", "meow"],
        cfg, accounts, {"999": "friend"}, "tg_",
    )
    assert labels == {
        "eri": "your peer (peerName)",
        "hermetika": "AI peer · this profile",
        "dreamer-ai": "AI peer · profile dreamer",
        "clawd": "AI peer of app 'claude_code'",
        "7654321": "runtime peer · telegram 7654321",
        "tg_7654321": "runtime peer · telegram 7654321",
        "friend": "alias target",
        "user-default-root": "fallback peer (pre-identity traffic)",
        "meow": "unrecognized",
    }
