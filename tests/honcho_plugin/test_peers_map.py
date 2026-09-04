"""Tests for ``hermes honcho peers map``.

The command lists gateway accounts recorded in state.db (the gateway stamps
each session row with its routing peer) and interactively edits
``userPeerAliases``. Discovery, the resolution preview, and the write-level
rules (host block vs root cascade) are covered here.
"""

import json
import sqlite3
from types import SimpleNamespace

import plugins.memory.honcho.cli as honcho_cli
from plugins.memory.honcho.cli import (
    _preview_peer_resolution,
    _seen_gateway_accounts,
)


def _make_state_db(path, rows):
    """Create a minimal sessions table with only the columns the query reads."""
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE sessions (
               id TEXT PRIMARY KEY,
               source TEXT,
               user_id TEXT,
               session_key TEXT,
               display_name TEXT,
               origin_json TEXT,
               started_at REAL
           )"""
    )
    conn.executemany(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)", rows,
    )
    conn.commit()
    conn.close()


def _origin(user_name=None, is_bot=False, user_id_alt=None):
    return json.dumps({
        "user_name": user_name, "is_bot": is_bot, "user_id_alt": user_id_alt,
    })


class TestSeenGatewayAccounts:
    def test_groups_rows_by_platform_and_user(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "telegram", "111", "telegram:dm:111", "Eri", _origin("eri"), 100.0),
            ("s2", "telegram", "111", "telegram:dm:111", "Eri", _origin("eri"), 200.0),
            ("s3", "discord", "222", "discord:dm:222", "Tek", _origin("teknium"), 50.0),
        ])
        accounts = _seen_gateway_accounts(db)
        assert len(accounts) == 2
        assert accounts[0] == {
            "platform": "telegram", "user_id": "111", "user_id_alt": "",
            "label": "eri", "sessions": 2, "profiles": [],
        }
        assert accounts[1]["user_id"] == "222"

    def test_orders_most_recent_first(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "telegram", "111", "k1", None, None, 100.0),
            ("s2", "discord", "222", "k2", None, None, 900.0),
        ])
        accounts = _seen_gateway_accounts(db)
        assert [a["user_id"] for a in accounts] == ["222", "111"]

    def test_skips_bots_and_rows_without_identity(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "discord", "333", "k1", "Bot", _origin("webhook", is_bot=True), 100.0),
            ("s2", "cli", None, None, None, None, 100.0),
            ("s3", "telegram", "111", "k2", None, _origin("eri"), 100.0),
        ])
        accounts = _seen_gateway_accounts(db)
        assert [a["user_id"] for a in accounts] == ["111"]

    def test_legacy_rows_without_session_key_still_count(self, tmp_path):
        """Gateway rows written before session_key was stamped are still accounts."""
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "telegram", "111", None, "Eri", None, 100.0),
            ("s2", "telegram", "111", "", "Eri", None, 200.0),
        ])
        accounts = _seen_gateway_accounts(db)
        assert [a["user_id"] for a in accounts] == ["111"]
        assert accounts[0]["sessions"] == 2

    def test_label_falls_back_to_display_name(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, [
            ("s1", "telegram", "111", "k1", "Eri DM", None, 100.0),
        ])
        accounts = _seen_gateway_accounts(db)
        assert accounts[0]["label"] == "Eri DM"

    def test_missing_db_returns_empty(self, tmp_path):
        assert _seen_gateway_accounts(tmp_path / "absent.db") == []

    def test_db_without_sessions_table_returns_empty(self, tmp_path):
        db = tmp_path / "state.db"
        sqlite3.connect(db).close()
        assert _seen_gateway_accounts(db) == []


class TestPreviewPeerResolution:
    def test_pin_wins_over_alias(self):
        out = _preview_peer_resolution(
            "111", pin=True, aliases={"111": "alice"}, prefix="tg_", peer_name="eri",
        )
        assert out == "eri (pinned)"

    def test_pin_without_peer_name_falls_through(self):
        out = _preview_peer_resolution(
            "111", pin=True, aliases={}, prefix="", peer_name="",
        )
        assert out == "111"

    def test_alias_hit(self):
        out = _preview_peer_resolution(
            "111", pin=False, aliases={"111": "alice"}, prefix="tg_", peer_name="eri",
        )
        assert out == "alice"

    def test_prefix_for_unknown_id(self):
        out = _preview_peer_resolution(
            "111", pin=False, aliases={}, prefix="tg_", peer_name="eri",
        )
        assert out == "tg_111 (prefixed)"

    def test_raw_sanitized_fallback(self):
        out = _preview_peer_resolution(
            "@you:matrix.org", pin=False, aliases={}, prefix="", peer_name="",
        )
        assert out == "-you-matrix-org"

    def test_alt_id_alias_hit(self):
        out = _preview_peer_resolution(
            "device-777", pin=False, aliases={"uuid-abc": "eri"}, prefix="",
            peer_name="", user_id_alt="uuid-abc",
        )
        assert out == "eri"

    def test_primary_alias_wins_over_alt(self):
        out = _preview_peer_resolution(
            "111", pin=False, aliases={"111": "alice", "uuid-abc": "bob"},
            prefix="", peer_name="", user_id_alt="uuid-abc",
        )
        assert out == "alice"


class TestCmdPeersMap:
    def _run(self, monkeypatch, tmp_path, *, answers, cfg, db_rows=(),
             ws_peers=None, workspaces=None, profiles=None):
        db = tmp_path / "state.db"
        if db_rows:
            _make_state_db(db, list(db_rows))

        written = {}
        monkeypatch.setattr(honcho_cli, "_read_config", lambda: cfg)
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(honcho_cli, "_active_profile_name", lambda: "default")
        monkeypatch.setattr(honcho_cli, "_state_db_path", lambda: db)
        monkeypatch.setattr(honcho_cli, "_local_config_path", lambda: tmp_path / "honcho.json")
        monkeypatch.setattr(
            honcho_cli, "_write_config",
            lambda c, path=None: written.update({"cfg": c}),
        )
        monkeypatch.setattr(
            honcho_cli, "_all_profile_host_configs",
            lambda: profiles if profiles is not None
            else [("default", "hermes", (cfg.get("hosts") or {}).get("hermes", {}))],
        )
        # API seams: offline by default; a fake client sentinel when ws_peers given.
        fake_client = object() if ws_peers is not None else None

        class _FakeCfg:
            workspace_id = "hermes"
        monkeypatch.setattr(
            honcho_cli, "_peers_map_client",
            lambda workspace=None: (fake_client, _FakeCfg() if fake_client else None),
        )
        monkeypatch.setattr(
            honcho_cli, "_api_workspace_peers",
            lambda client: (
                [{"id": p, "created": "2026-01-01"} for p in ws_peers]
                if client is not None and ws_peers is not None else None
            ),
        )
        monkeypatch.setattr(
            honcho_cli, "_api_workspaces",
            lambda client: list(workspaces) if workspaces is not None else None,
        )
        monkeypatch.setattr(
            honcho_cli, "_api_peer_detail",
            lambda client, pid: f"(card of {pid})",
        )

        answer_iter = iter(answers)
        def _scripted_prompt(label, default=None, secret=False):
            try:
                return next(answer_iter)
            except StopIteration:
                return default if default is not None else ""
        monkeypatch.setattr(honcho_cli, "_prompt", _scripted_prompt)

        honcho_cli.cmd_peers_map(SimpleNamespace())
        return written

    def test_maps_seen_account_to_host_block(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        rows = [("s1", "telegram", "111", "k1", None, _origin("eri"), 100.0)]
        written = self._run(
            monkeypatch, tmp_path,
            answers=["1", "eri", ""],
            cfg=cfg, db_rows=rows,
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"111": "eri"}

    def test_raw_runtime_id_entry(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["7654321", "eri", ""],
            cfg=cfg,
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"7654321": "eri"}

    def test_clear_alias_with_dash(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "hosts": {"hermes": {"userPeerAliases": {"111": "eri", "222": "tek"}}},
        }
        written = self._run(
            monkeypatch, tmp_path,
            answers=["111", "-", ""],
            cfg=cfg,
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"222": "tek"}

    def test_clearing_last_alias_removes_key(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"userPeerAliases": {"111": "eri"}}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["111", "-", ""],
            cfg=cfg,
        )
        assert "userPeerAliases" not in written["cfg"]["hosts"]["hermes"]

    def test_root_sourced_aliases_write_back_to_root(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "userPeerAliases": {"111": "eri"},
            "hosts": {"hermes": {}},
        }
        written = self._run(
            monkeypatch, tmp_path,
            answers=["222", "tek", ""],
            cfg=cfg,
        )
        assert written["cfg"]["userPeerAliases"] == {"111": "eri", "222": "tek"}
        assert "userPeerAliases" not in written["cfg"]["hosts"]["hermes"]

    def test_no_changes_no_write(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {}}}
        written = self._run(monkeypatch, tmp_path, answers=[""], cfg=cfg)
        assert written == {}

    def test_keeping_current_value_is_not_a_change(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"userPeerAliases": {"111": "eri"}}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["111", "eri", ""],
            cfg=cfg,
        )
        assert written == {}

    def test_pinned_declined_exits_without_write(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"pinUserPeer": True, "peerName": "eri"}}}
        written = self._run(monkeypatch, tmp_path, answers=["n"], cfg=cfg)
        assert written == {}

    def test_pinned_accepted_still_edits_aliases(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"pinUserPeer": True, "peerName": "eri"}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["y", "111", "eri", ""],
            cfg=cfg,
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"111": "eri"}

    def test_peers_dispatches_map_action(self, monkeypatch):
        called = {}
        monkeypatch.setattr(honcho_cli, "cmd_peers_map", lambda a: called.update({"map": True}))
        honcho_cli.cmd_peers(SimpleNamespace(peers_action="map"))
        assert called == {"map": True}

    def test_target_picked_from_workspace_peers(self, monkeypatch, tmp_path):
        """'p1' as the target resolves to the first workspace peer's id."""
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        rows = [("s1", "telegram", "111", "k1", None, _origin("eri"), 100.0)]
        written = self._run(
            monkeypatch, tmp_path,
            answers=["1", "p1", ""],
            cfg=cfg, db_rows=rows, ws_peers=["eri", "hermes"],
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"111": "eri"}

    def test_new_peer_and_history_consequences_printed(self, monkeypatch, tmp_path, capsys):
        """Mapping onto a name not in the workspace warns about peer creation,
        and moving an account off its existing runtime peer names the history
        that stays behind."""
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        rows = [("s1", "telegram", "111", "k1", None, _origin("bob"), 100.0)]
        self._run(
            monkeypatch, tmp_path,
            answers=["1", "fresh-name", ""],
            cfg=cfg, db_rows=rows, ws_peers=["eri", "111"],
        )
        out = capsys.readouterr().out
        assert "'fresh-name' is a new peer" in out
        assert "peer '111' keeps its existing history" in out

    def test_exists_markers_in_accounts_table(self, monkeypatch, tmp_path, capsys):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        rows = [
            ("s1", "telegram", "111", "k1", None, _origin("a"), 100.0),
            ("s2", "discord", "222", "k2", None, _origin("b"), 50.0),
        ]
        self._run(
            monkeypatch, tmp_path, answers=[""],
            cfg=cfg, db_rows=rows, ws_peers=["eri", "111"],
        )
        import re
        out = capsys.readouterr().out
        line_111 = next(ln for ln in out.splitlines() if re.match(r"\s+\d+\s+telegram\s", ln))
        line_222 = next(ln for ln in out.splitlines() if re.match(r"\s+\d+\s+discord\s", ln))
        assert "✓" in line_111
        assert "○ new" in line_222

    def test_offline_degrades_to_typed_targets(self, monkeypatch, tmp_path, capsys):
        cfg = {"apiKey": "***", "hosts": {"hermes": {}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["7654321", "eri", ""],
            cfg=cfg, ws_peers=None,
        )
        out = capsys.readouterr().out
        assert "peers unavailable" in out
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {"7654321": "eri"}

    def test_inspect_peer_prints_card(self, monkeypatch, tmp_path, capsys):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        self._run(
            monkeypatch, tmp_path, answers=["p2", ""],
            cfg=cfg, ws_peers=["eri", "meow"],
        )
        out = capsys.readouterr().out
        assert "(card of meow)" in out

    def test_empty_workspace_hints_at_workspace_list(self, monkeypatch, tmp_path, capsys):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        self._run(monkeypatch, tmp_path, answers=[""], cfg=cfg, ws_peers=[])
        out = capsys.readouterr().out
        assert "No peers here yet" in out
        assert "Wrong workspace?" in out

    def test_unrecognized_workspace_hints_wrong_workspace(self, monkeypatch, tmp_path, capsys):
        """Peers exist but none match local identity → wrong-workspace hint."""
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        self._run(
            monkeypatch, tmp_path, answers=[""],
            cfg=cfg, ws_peers=["stranger1", "stranger2"],
        )
        out = capsys.readouterr().out
        assert "None of these match your configured identity" in out


class TestWorkspaceSwitch:
    _run = TestCmdPeersMap._run

    def test_browse_and_confirmed_switch_writes_workspace(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["w", "2", "y", ""],
            cfg=cfg, ws_peers=["eri"], workspaces=["hermes", "cosmania-dex"],
        )
        assert written["cfg"]["hosts"]["hermes"]["workspace"] == "cosmania-dex"

    def test_browse_client_is_built_for_the_browsed_workspace(self, monkeypatch):
        """The browse client must carry the browsed workspace, not the configured one.
        get_honcho_client keys its cache on workspace_id, so this is what keeps a browse
        from reusing the profile's own client."""
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

    def test_declined_switch_leaves_config_untouched(self, monkeypatch, tmp_path):
        cfg = {"apiKey": "***", "hosts": {"hermes": {"peerName": "eri"}}}
        written = self._run(
            monkeypatch, tmp_path,
            answers=["w", "2", "n", ""],
            cfg=cfg, ws_peers=["eri"], workspaces=["hermes", "cosmania-dex"],
        )
        assert written == {}


class TestSaveScope:
    _run = TestCmdPeersMap._run

    def _multi_profiles(self, cfg):
        hosts = cfg.get("hosts") or {}
        return [
            ("default", "hermes", hosts.get("hermes", {})),
            ("dreamer", "hermes.dreamer", hosts.get("hermes.dreamer", {})),
        ]

    def test_root_sourced_multi_profile_prompts_scope_all(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "userPeerAliases": {"111": "eri"},
            "hosts": {"hermes": {}, "hermes.dreamer": {}},
        }
        written = self._run(
            monkeypatch, tmp_path,
            answers=["222", "tek", "", "all"],
            cfg=cfg, profiles=self._multi_profiles(cfg),
        )
        assert written["cfg"]["userPeerAliases"] == {"111": "eri", "222": "tek"}
        assert "userPeerAliases" not in written["cfg"]["hosts"]["hermes"]

    def test_root_sourced_multi_profile_scope_this_forks_host(self, monkeypatch, tmp_path):
        cfg = {
            "apiKey": "***",
            "userPeerAliases": {"111": "eri"},
            "hosts": {"hermes": {}, "hermes.dreamer": {}},
        }
        written = self._run(
            monkeypatch, tmp_path,
            answers=["222", "tek", "", "this"],
            cfg=cfg, profiles=self._multi_profiles(cfg),
        )
        assert written["cfg"]["hosts"]["hermes"]["userPeerAliases"] == {
            "111": "eri", "222": "tek",
        }
        # Root map stays as the other profiles' baseline.
        assert written["cfg"]["userPeerAliases"] == {"111": "eri"}

    def test_cross_workspace_root_write_warns(self, monkeypatch, tmp_path, capsys):
        cfg = {
            "apiKey": "***",
            "userPeerAliases": {},
            "hosts": {
                "hermes": {"workspace": "hermes"},
                "hermes.dreamer": {"workspace": "dreamland"},
            },
        }
        self._run(
            monkeypatch, tmp_path,
            answers=["222", "tek", "", "all"],
            cfg=cfg, profiles=self._multi_profiles(cfg),
        )
        out = capsys.readouterr().out
        assert "also apply in workspace 'dreamland'" in out

    def test_sibling_divergence_marked(self, monkeypatch, tmp_path, capsys):
        """A sibling profile resolving the same account differently is shown."""
        cfg = {
            "apiKey": "***",
            "hosts": {
                "hermes": {"peerName": "eri", "userPeerAliases": {"111": "eri"}},
                "hermes.dreamer": {"peerName": "eri", "userPeerAliases": {"111": "bob"}},
            },
        }
        rows = [("s1", "telegram", "111", "k1", None, _origin("x"), 100.0)]
        self._run(
            monkeypatch, tmp_path, answers=[""],
            cfg=cfg, db_rows=rows, profiles=self._multi_profiles(cfg),
        )
        out = capsys.readouterr().out
        assert "≠ dreamer→bob" in out


class TestClassifyWorkspacePeers:
    def test_labels_from_local_config(self, monkeypatch):
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(
            honcho_cli, "_all_profile_host_configs",
            lambda: [
                ("default", "hermes", {"peerName": "eri", "aiPeer": "hermetika"}),
                ("dreamer", "hermes.dreamer", {"aiPeer": "dreamer-ai"}),
            ],
        )
        cfg = {
            "peerName": "eri",
            "hosts": {"claude_code": {"aiPeer": "clawd"}},
        }
        accounts = [{"platform": "telegram", "user_id": "7654321", "user_id_alt": ""}]
        labels = honcho_cli._classify_workspace_peers(
            ["eri", "hermetika", "dreamer-ai", "clawd", "7654321",
             "user-default-root", "meow"],
            cfg, accounts, {"999": "friend"}, "",
        )
        assert labels["eri"] == "your peer (peerName)"
        assert labels["hermetika"] == "AI peer · this profile"
        assert labels["dreamer-ai"] == "AI peer · profile dreamer"
        assert labels["clawd"] == "AI peer of app 'claude_code'"
        assert labels["7654321"] == "runtime peer · telegram 7654321"
        assert labels["user-default-root"] == "fallback peer (pre-identity traffic)"
        assert labels["meow"] == "unrecognized"

    def test_alias_targets_and_prefix_recognized(self, monkeypatch):
        monkeypatch.setattr(honcho_cli, "_host_key", lambda: "hermes")
        monkeypatch.setattr(
            honcho_cli, "_all_profile_host_configs",
            lambda: [("default", "hermes", {})],
        )
        accounts = [{"platform": "telegram", "user_id": "42", "user_id_alt": ""}]
        labels = honcho_cli._classify_workspace_peers(
            ["friend", "tg_42"], {}, accounts, {"999": "friend"}, "tg_",
        )
        assert labels["friend"] == "alias target"
        assert labels["tg_42"] == "runtime peer · telegram 42"
