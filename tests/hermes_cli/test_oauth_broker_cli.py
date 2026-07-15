"""`hermes oauth-broker` CLI: parser tree, handlers, and safety gates.
Device flow, Keychain, launchctl, HTTP, and migration boundaries are all
patched — no real logins, secrets, services, or network."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from hermes_cli.auth import AuthError
import hermes_cli.oauth_broker as cli_mod
from hermes_cli.subcommands.oauth_broker import build_oauth_broker_parser
from agent.oauth_broker.models import GrantStoreError, OAuthGrant

SYNTHETIC_ACCESS = "synthetic-cli-access-token"
SYNTHETIC_REFRESH = "synthetic-cli-refresh-token"


class FakeStore:
    def __init__(self, grants=None):
        self.grants = dict(grants or {})
        self.replaced = []
        self.deleted = []

    def load(self, alias):
        try:
            return self.grants[alias]
        except KeyError:
            raise GrantStoreError(
                alias=alias, category="not_found", detail="no grant provisioned"
            ) from None

    def replace(self, alias, grant):
        self.replaced.append(alias)
        self.grants[alias] = grant

    def delete(self, alias):
        self.deleted.append(alias)
        self.grants.pop(alias, None)


def _grant(expires_at=4102444800.0):
    return OAuthGrant(
        access_token=SYNTHETIC_ACCESS,
        refresh_token=SYNTHETIC_REFRESH,
        expires_at=expires_at,
        account_id="acct-synthetic-a",
    )


def _valid_snapshot():
    return {
        "snapshot_schema_version": 2,
        "mode": "dry-run",
        "provider": "openai-codex",
        "port": 17880,
        "groups": {"p1": "A"},
        "group_counts": {"A": 1, "B": 0, "C": 0},
        "profiles": {
            "p1": {
                "group": "A",
                "auth_sha256": "0" * 64,
                "auth_canonical_sha256": "1" * 64,
                "added_entry_ids": ["broker-A", "broker-B", "broker-C"],
                "legacy": [{"id": "synthetic-legacy-p1"}],
            }
        },
    }


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    sentinel = object()
    build_oauth_broker_parser(subparsers, cmd_oauth_broker=sentinel)
    args = parser.parse_args(argv)
    assert args.func is sentinel
    return args


# ---------------------------------------------------------------------------
# Parser tree
# ---------------------------------------------------------------------------


def test_parser_covers_all_verbs():
    assert _parse(["oauth-broker", "run"]).oauth_broker_command == "run"
    args = _parse(["oauth-broker", "run", "--host", "127.0.0.1", "--port", "17999"])
    assert (args.host, args.port) == ("127.0.0.1", 17999)
    assert _parse(["oauth-broker", "status"]).oauth_broker_command == "status"
    assert _parse(["oauth-broker", "doctor"]).oauth_broker_command == "doctor"
    assert _parse(["oauth-broker", "install"]).apply is False
    assert _parse(["oauth-broker", "uninstall"]).apply is False
    login = _parse(["oauth-broker", "auth", "login", "B"])
    assert (login.oauth_broker_auth_command, login.alias) == ("login", "B")
    status = _parse(["oauth-broker", "auth", "status"])
    assert status.alias is None
    logout = _parse(["oauth-broker", "auth", "logout", "C", "--yes"])
    assert logout.yes is True
    migrate = _parse(
        [
            "oauth-broker",
            "migrate",
            "--profiles-root",
            "/tmp/synthetic-root",
            "--groups",
            "/tmp/groups.json",
            "--snapshot",
            "/tmp/snap.json",
        ]
    )
    assert migrate.apply is False
    rollback = _parse(
        [
            "oauth-broker",
            "rollback",
            "--profiles-root",
            "/tmp/synthetic-root",
            "--snapshot",
            "/tmp/snap.json",
        ]
    )
    assert rollback.oauth_broker_command == "rollback"


def test_parser_rejects_unknown_alias():
    with pytest.raises(SystemExit):
        _parse(["oauth-broker", "auth", "login", "D"])


def test_main_registers_oauth_broker_parser():
    main_source = Path("hermes_cli/main.py").read_text(encoding="utf-8")
    assert "from hermes_cli.subcommands.oauth_broker import build_oauth_broker_parser" in main_source
    assert "build_oauth_broker_parser(" in main_source
    assert "def cmd_oauth_broker(" in main_source


# ---------------------------------------------------------------------------
# auth login / status / logout
# ---------------------------------------------------------------------------


def _args(**kwargs):
    return argparse.Namespace(**kwargs)


def test_auth_login_stores_grant_without_printing_secrets(monkeypatch, capsys):
    events = []

    class LockCheckingStore(FakeStore):
        def replace(self, alias, grant):
            assert events == ["lock:B:enter"]
            events.append("replace:B")
            super().replace(alias, grant)

    @contextmanager
    def fake_account_lock(alias):
        events.append(f"lock:{alias}:enter")
        try:
            yield
        finally:
            events.append(f"lock:{alias}:exit")

    store = LockCheckingStore()
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_confirm", lambda prompt: True)
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(cli_mod, "_account_process_lock", fake_account_lock)
    monkeypatch.setattr(
        cli_mod,
        "_device_code_login",
        lambda: {
            "tokens": {
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "account_id": "acct-synthetic-b",
            }
        },
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="auth", oauth_broker_auth_command="login", alias="B")
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert store.replaced == ["B"]
    assert store.grants["B"].account_id == "acct-synthetic-b"
    assert events == ["lock:B:enter", "replace:B", "lock:B:exit"]
    assert "B" in out
    assert SYNTHETIC_ACCESS not in out
    assert SYNTHETIC_REFRESH not in out


def test_auth_login_fails_safely_off_macos(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: False)
    monkeypatch.setattr(cli_mod, "_device_code_login", lambda: called.append(1))
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="auth", oauth_broker_auth_command="login", alias="A")
    )
    assert rc == 1
    assert called == []
    assert "macos" in capsys.readouterr().out.lower()


def test_auth_login_handles_auth_error_without_traceback(monkeypatch, capsys):
    store = FakeStore()

    def fail_login():
        raise AuthError(
            "OpenAI is rate-limiting Codex login requests (HTTP 429).",
            provider="openai-codex",
            code="codex_rate_limited",
        )

    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(cli_mod, "_device_code_login", fail_login)

    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="auth", oauth_broker_auth_command="login", alias="A")
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert store.replaced == []
    assert "oauth-broker: login for account A failed" in captured.out
    assert "HTTP 429" in captured.out
    assert captured.err == ""


def test_auth_status_prints_booleans_only(monkeypatch, capsys):
    store = FakeStore({"A": _grant()})
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(cli_mod, "_broker_health_detailed", lambda port: None)
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="auth",
            oauth_broker_auth_command="status",
            alias=None,
            port=17880,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "A present=True expiring=False healthy=unknown" in out
    assert "B present=False" in out
    assert "C present=False" in out
    assert SYNTHETIC_ACCESS not in out


def test_auth_logout_requires_yes(monkeypatch, capsys):
    events = []

    class LockCheckingStore(FakeStore):
        def delete(self, alias):
            assert events == ["lock:C:enter"]
            events.append("delete:C")
            super().delete(alias)

    @contextmanager
    def fake_account_lock(alias):
        events.append(f"lock:{alias}:enter")
        try:
            yield
        finally:
            events.append(f"lock:{alias}:exit")

    store = LockCheckingStore({"C": _grant()})
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(cli_mod, "_account_process_lock", fake_account_lock)
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="auth",
            oauth_broker_auth_command="logout",
            alias="C",
            yes=False,
        )
    )
    assert rc == 1
    assert store.deleted == []
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="auth",
            oauth_broker_auth_command="logout",
            alias="C",
            yes=True,
        )
    )
    assert rc == 0
    assert store.deleted == ["C"]
    assert events == ["lock:C:enter", "delete:C", "lock:C:exit"]


# ---------------------------------------------------------------------------
# run / status / doctor
# ---------------------------------------------------------------------------


def test_run_builds_abc_slots_and_fails_closed_without_client_key(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(
        cli_mod,
        "_grant_store",
        lambda: FakeStore({"A": _grant(), "B": _grant(), "C": _grant()}),
    )
    monkeypatch.setattr(cli_mod, "_broker_state_dir", lambda: tmp_path)

    def missing_key():
        raise GrantStoreError(alias="local", category="not_found")

    monkeypatch.setattr(cli_mod, "_read_client_key", missing_key)
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="run", host="127.0.0.1", port=17880)
    )
    assert rc == 1
    assert "install" in capsys.readouterr().out.lower()

    captured = {}
    monkeypatch.setattr(cli_mod, "_read_client_key", lambda: "synthetic-local-key")
    monkeypatch.setattr(
        cli_mod, "_run_broker", lambda **kwargs: captured.update(kwargs)
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="run", host="127.0.0.1", port=17999)
    )
    assert rc == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 17999
    assert sorted(captured["slots"]) == ["A", "B", "C"]
    assert captured["local_key"] == "synthetic-local-key"


def test_run_refuses_to_bind_when_preloaded_grant_matches_recovery_marker(
    monkeypatch, capsys, tmp_path
):
    from agent.oauth_broker.account_slot import AccountSlot

    store = FakeStore({"A": _grant(), "B": _grant(), "C": _grant()})
    marker_writer = AccountSlot("A", grant_store=store, state_dir=tmp_path)
    marker_writer._write_recovery_marker(SYNTHETIC_REFRESH)

    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_read_client_key", lambda: "synthetic-local-key")
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(cli_mod, "_broker_state_dir", lambda: tmp_path)
    started = []
    monkeypatch.setattr(cli_mod, "_run_broker", lambda **kwargs: started.append(kwargs))

    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="run", host="127.0.0.1", port=17880)
    )

    assert rc == 1
    assert started == []
    out = capsys.readouterr().out
    assert "persistence_recovery_required" in out
    assert SYNTHETIC_ACCESS not in out
    assert SYNTHETIC_REFRESH not in out


def test_status_reports_health_and_fail_closed(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_mod,
        "_broker_health_detailed",
        lambda port: {"status": "ok", "accounts": []},
    )
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="status", port=17880))
    assert rc == 0
    assert "ok" in capsys.readouterr().out

    monkeypatch.setattr(
        cli_mod,
        "_broker_health_detailed",
        lambda port: {"status": "degraded", "accounts": []},
    )
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="status", port=17880))
    assert rc == 1
    assert "degraded" in capsys.readouterr().out

    monkeypatch.setattr(cli_mod, "_broker_health_detailed", lambda port: None)
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="status", port=17880))
    assert rc == 1
    assert "fail closed" in capsys.readouterr().out.lower()


def test_doctor_reports_pass_fail_lines(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_read_client_key", lambda: "synthetic-local-key")
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: FakeStore({"A": _grant()}))
    plist = tmp_path / "ai.hermes.oauth-broker.plist"
    monkeypatch.setattr(cli_mod, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli_mod, "_broker_health", lambda port: {"status": "ok"})
    monkeypatch.setattr(
        cli_mod,
        "_broker_health_detailed",
        lambda port: {"status": "degraded", "accounts": []},
    )
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="doctor", port=17880))
    out = capsys.readouterr().out
    assert rc == 1  # grants B/C missing, plist not installed
    assert "PASS macOS" in out
    assert "PASS client key in Keychain" in out
    assert "PASS grant A" in out
    assert "FAIL grant B" in out
    assert "FAIL launchd plist installed" in out
    assert "PASS broker /health" in out
    assert "FAIL broker readiness" in out
    assert SYNTHETIC_ACCESS not in out


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def test_install_render_only_generates_key_and_never_runs_launchctl(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    plist = tmp_path / "ai.hermes.oauth-broker.plist"
    monkeypatch.setattr(cli_mod, "_plist_path", lambda: plist)
    monkeypatch.setattr(cli_mod, "_render_plist", lambda port: b"synthetic-plist")
    keys = {}

    def read_key():
        if "value" not in keys:
            raise GrantStoreError(alias="local", category="not_found")
        return keys["value"]

    monkeypatch.setattr(cli_mod, "_read_client_key", read_key)
    monkeypatch.setattr(
        cli_mod, "_write_client_key", lambda value: keys.update(value=value)
    )
    ran = []
    monkeypatch.setattr(cli_mod, "_runner", lambda: ran.append)
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="install", port=17880, apply=False)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert plist.exists()
    assert ran == []  # render-only: launchctl untouched
    assert keys["value"]  # client key generated into the Keychain
    assert keys["value"] not in out  # and never printed


def test_uninstall_never_touches_grants(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    plist = tmp_path / "ai.hermes.oauth-broker.plist"
    plist.write_bytes(b"synthetic")
    monkeypatch.setattr(cli_mod, "_plist_path", lambda: plist)

    def forbidden():
        raise AssertionError("uninstall must never open the grant store")

    monkeypatch.setattr(cli_mod, "_grant_store", forbidden)
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="uninstall", apply=False)
    )
    assert rc == 0
    assert plist.exists()
    assert "bootout" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# migrate / rollback routing
# ---------------------------------------------------------------------------


def test_migrate_dry_run_and_apply_routing(monkeypatch, capsys, tmp_path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps({"p1": "A"}))
    snapshot_path = tmp_path / "snap.json"
    recorded = {}
    fake_snapshot = _valid_snapshot()
    monkeypatch.setattr(
        cli_mod,
        "_plan_migration",
        lambda root, groups, port: recorded.update(
            plan=(Path(root), groups, port)
        )
        or fake_snapshot,
    )
    monkeypatch.setattr(
        cli_mod,
        "_apply_migration",
        lambda root, snapshot, journal_path: recorded.update(applied=True)
        or {"applied": True, "written": ["p1"]},
    )
    monkeypatch.setattr(cli_mod, "_confirm", lambda prompt: True)

    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="migrate",
            profiles_root=str(tmp_path / "root"),
            groups=str(groups_path),
            snapshot=str(snapshot_path),
            port=17880,
            apply=False,
        )
    )
    assert rc == 0
    assert recorded["plan"] == (tmp_path / "root", {"p1": "A"}, 17880)
    assert "applied" not in recorded  # dry-run never applies
    assert snapshot_path.exists()

    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="migrate",
            profiles_root=str(tmp_path / "root"),
            groups=str(groups_path),
            snapshot=str(snapshot_path),
            port=17880,
            apply=True,
        )
    )
    assert rc == 0
    assert recorded["applied"] is True


# ---------------------------------------------------------------------------
# Security repair checkpoint: confirmations, --yes gates, run preload
# ---------------------------------------------------------------------------


def test_migrate_apply_requires_confirmation(monkeypatch, capsys, tmp_path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps({"p1": "A"}))
    snapshot_path = tmp_path / "snap.json"
    applied = []
    monkeypatch.setattr(
        cli_mod, "_plan_migration", lambda root, groups, port: _valid_snapshot()
    )
    monkeypatch.setattr(
        cli_mod,
        "_apply_migration",
        lambda root, snapshot, journal_path: applied.append(1)
        or {"applied": True, "written": []},
    )
    prompts = []
    monkeypatch.setattr(
        cli_mod, "_confirm", lambda prompt: prompts.append(prompt) or False
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="migrate",
            profiles_root=str(tmp_path / "root"),
            groups=str(groups_path),
            snapshot=str(snapshot_path),
            port=17880,
            apply=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert prompts, "apply must ask for confirmation"
    assert applied == []  # declined: nothing applied
    assert snapshot_path.exists()  # dry-run snapshot kept
    assert "dry-run" in out.lower()

    monkeypatch.setattr(cli_mod, "_confirm", lambda prompt: True)
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="migrate",
            profiles_root=str(tmp_path / "root"),
            groups=str(groups_path),
            snapshot=str(snapshot_path),
            port=17880,
            apply=True,
        )
    )
    assert rc == 0
    assert applied == [1]


def test_rollback_requires_explicit_yes(monkeypatch, capsys, tmp_path):
    snapshot_path = tmp_path / "snap.json"
    snapshot_path.write_text(json.dumps({"profiles": {}}))
    rolled = []
    monkeypatch.setattr(
        cli_mod,
        "_rollback_migration",
        lambda root, snapshot, journal: rolled.append(1) or {"restored": []},
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="rollback",
            profiles_root=str(tmp_path / "root"),
            snapshot=str(snapshot_path),
            yes=False,
        )
    )
    assert rc == 1
    assert rolled == []
    assert "--yes" in capsys.readouterr().out
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="rollback",
            profiles_root=str(tmp_path / "root"),
            snapshot=str(snapshot_path),
            yes=True,
        )
    )
    assert rc == 0
    assert rolled == [1]


def test_rollback_parser_has_yes_flag():
    args = _parse(
        [
            "oauth-broker",
            "rollback",
            "--profiles-root",
            "/tmp/synthetic-root",
            "--snapshot",
            "/tmp/snap.json",
        ]
    )
    assert args.yes is False


def test_auth_login_confirms_alias_and_prints_only_fingerprint(
    monkeypatch, capsys
):
    store = FakeStore()
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    monkeypatch.setattr(
        cli_mod,
        "_device_code_login",
        lambda: {
            "tokens": {
                "access_token": SYNTHETIC_ACCESS,
                "refresh_token": SYNTHETIC_REFRESH,
                "account_id": "acct-synthetic-b",
            }
        },
    )
    prompts = []
    monkeypatch.setattr(
        cli_mod, "_confirm", lambda prompt: prompts.append(prompt) or False
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="auth", oauth_broker_auth_command="login", alias="B")
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert store.replaced == []  # declined: nothing stored
    [prompt] = prompts
    assert "B" in prompt
    assert "sha256:" in prompt  # non-secret account-id fingerprint
    assert "acct-synthetic-b" not in prompt + out  # raw account id never shown
    assert SYNTHETIC_ACCESS not in prompt + out

    monkeypatch.setattr(cli_mod, "_confirm", lambda prompt: True)
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="auth", oauth_broker_auth_command="login", alias="B")
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert store.replaced == ["B"]
    assert "sha256:" in out
    assert "acct-synthetic-b" not in out


def test_run_preloads_grants_and_fails_closed_when_missing(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli_mod, "_is_darwin", lambda: True)
    monkeypatch.setattr(cli_mod, "_read_client_key", lambda: "synthetic-local-key")
    monkeypatch.setattr(cli_mod, "_broker_state_dir", lambda: tmp_path)
    store = FakeStore({"A": _grant(), "C": _grant()})  # B missing
    monkeypatch.setattr(cli_mod, "_grant_store", lambda: store)
    started = []
    monkeypatch.setattr(cli_mod, "_run_broker", lambda **kw: started.append(kw))
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="run", host="127.0.0.1", port=17880)
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert started == []
    assert "B" in out and "fail" in out.lower()

    store.grants["B"] = _grant()
    rc = cli_mod.cmd_oauth_broker(
        _args(oauth_broker_command="run", host="127.0.0.1", port=17880)
    )
    assert rc == 0
    [captured] = started
    for alias in "ABC":
        assert captured["slots"][alias].status().present is True


def test_status_authenticates_via_detailed_health(monkeypatch, capsys):
    def forbidden(port):
        raise AssertionError("status must not trust unauthenticated /health")

    monkeypatch.setattr(cli_mod, "_broker_health", forbidden)
    monkeypatch.setattr(
        cli_mod,
        "_broker_health_detailed",
        lambda port: {"status": "ok", "accounts": []},
    )
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="status", port=17880))
    assert rc == 0
    assert "ok" in capsys.readouterr().out

    monkeypatch.setattr(cli_mod, "_broker_health_detailed", lambda port: None)
    rc = cli_mod.cmd_oauth_broker(_args(oauth_broker_command="status", port=17880))
    assert rc == 1
    assert "fail closed" in capsys.readouterr().out.lower()


def test_rollback_routes_snapshot(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snap.json"
    snapshot_path.write_text(json.dumps({"profiles": {}}))
    recorded = {}
    monkeypatch.setattr(
        cli_mod,
        "_rollback_migration",
        lambda root, snapshot, journal: recorded.update(
            root=Path(root), journal=Path(journal)
        )
        or {"restored": []},
    )
    rc = cli_mod.cmd_oauth_broker(
        _args(
            oauth_broker_command="rollback",
            profiles_root=str(tmp_path / "root"),
            snapshot=str(snapshot_path),
            yes=True,
        )
    )
    assert rc == 0
    assert recorded["root"] == tmp_path / "root"
    assert recorded["journal"] == Path(str(snapshot_path) + ".rollback.journal")
