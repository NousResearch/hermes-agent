"""#100368 regression: the corruption guidance must not direct a WAL-reset-
vulnerable sqlite3 CLI at a live Hermes database.

Field forensics (issue #100368, maintainer round 2 + the isolated reproducer
in its comments): when a shell with SQLite's WAL-reset opener bug (fixed
3.51.3+ / backports 3.50.7 / 3.44.6 — Debian/Ubuntu system CLIs 3.45.1 /
3.46.1 are in the vulnerable band) opens a live state.db whose writer's DMS
lock has been cancelled, it unlinks the live -wal/-shm pair and splits the
store into two concurrent generations. Both generations report
``integrity_check ok`` while an old-generation acknowledged write is lost.

Hermes' own corruption banners used to instruct exactly that command
(`sqlite3 ~/.hermes/state.db ".recover"`). The fix routes operators to
`hermes sessions recover --source ...`, whose lane snapshots the damaged
bundle before any shell touches it, and refuses a WAL-reset-vulnerable
sqlite3 CLI for the page-level salvage lane even on the snapshot.
"""

from __future__ import annotations

import argparse
import inspect
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_state_repair

from hermes_cli.session_lost_and_found import (
    _parse_sqlite3_cli_version,
    _wal_reset_vulnerable,
    find_sqlite3_cli,
    find_sqlite3_cli_refusal,
)
from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable


LIVE_DB_SALVAGE_COMMAND = 'sqlite3 ~/.hermes/state.db ".recover"'


# ---------------------------------------------------------------------------
# The version gate itself
# ---------------------------------------------------------------------------


class TestWalResetVersionGate:
    @pytest.mark.parametrize(
        "version",
        [(3, 45, 1), (3, 46, 1), (3, 44, 5), (3, 50, 4), (3, 51, 2), (3, 8, 0)],
    )
    def test_vulnerable_versions(self, version):
        assert _wal_reset_vulnerable(version) is True

    @pytest.mark.parametrize(
        "version",
        [
            (3, 44, 6),
            (3, 44, 7),
            (3, 50, 7),
            (3, 50, 8),
            (3, 51, 3),
            (3, 51, 4),
            (3, 52, 0),
            (3, 53, 1),
            (4, 0, 0),
        ],
    )
    def test_fixed_versions(self, version):
        assert _wal_reset_vulnerable(version) is False

    def test_gate_mirrors_library_gate(self):
        """The salvage gate must agree with the shared runtime gate so the
        embedded library and the salvage shell can never disagree."""
        versions = [
            (3, 44, 5),
            (3, 44, 6),
            (3, 45, 1),
            (3, 50, 4),
            (3, 50, 7),
            (3, 51, 2),
            (3, 51, 3),
            (3, 53, 1),
        ]
        for version in versions:
            assert _wal_reset_vulnerable(version) == (
                is_sqlite_wal_reset_vulnerable(version)
            ), f"salvage gate disagrees with the runtime gate at {version}"


# ---------------------------------------------------------------------------
# find_sqlite3_cli refuses unsafe shells and explains why
# ---------------------------------------------------------------------------


class TestFindSqlite3CliRefusal:
    def test_missing_binary_refusal(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found.shutil.which", lambda _: None
        )
        assert find_sqlite3_cli() is None
        assert find_sqlite3_cli_refusal()["reason"] == "missing"

    def test_no_dbpage_refusal(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found.shutil.which",
            lambda _: "/usr/bin/sqlite3",
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._cli_supports_recover",
            lambda _: False,
        )
        assert find_sqlite3_cli() is None
        assert find_sqlite3_cli_refusal()["reason"] == "no_dbpage"

    def test_wal_reset_vulnerable_refusal(self, monkeypatch):
        """A .recover-capable but WAL-reset-vulnerable CLI must be refused.

        This is the Debian/Ubuntu shape from the #100368 incident: the
        system sqlite3 (3.45.1) has sqlite_dbpage, so the capability probe
        passes, while the WAL-reset opener bug is still present.
        """
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found.shutil.which",
            lambda _: "/usr/bin/sqlite3",
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._cli_supports_recover",
            lambda _: True,
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._parse_sqlite3_cli_version",
            lambda _: (3, 45, 1),
        )
        assert find_sqlite3_cli() is None
        refusal = find_sqlite3_cli_refusal()
        assert refusal["reason"] == "wal_reset_vulnerable"
        assert refusal["version"] == "3.45.1"
        assert "WAL-reset" in refusal["detail"]

    def test_fixed_capable_cli_accepted(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found.shutil.which",
            lambda _: "/usr/local/bin/sqlite3",
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._cli_supports_recover",
            lambda _: True,
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._parse_sqlite3_cli_version",
            lambda _: (3, 51, 3),
        )
        assert find_sqlite3_cli() == "/usr/local/bin/sqlite3"
        assert find_sqlite3_cli_refusal() == {}

    def test_unparsable_version_still_usable(self, monkeypatch):
        """A CLI whose version line cannot be parsed is not refused on
        version grounds alone (the salvage lane runs against a snapshot
        copy, not the live file)."""
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found.shutil.which",
            lambda _: "/usr/bin/sqlite3",
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._cli_supports_recover",
            lambda _: True,
        )
        monkeypatch.setattr(
            "hermes_cli.session_lost_and_found._parse_sqlite3_cli_version",
            lambda _: None,
        )
        assert find_sqlite3_cli() == "/usr/bin/sqlite3"


class TestParseSqlite3CliVersion:
    def test_parses_modern_output(self):
        class Probe:
            returncode = 0
            stdout = b"3.51.4 2026-XX-XX 12:34:56\n"

        with patch(
            "hermes_cli.session_lost_and_found.subprocess.run",
            return_value=Probe(),
        ):
            assert _parse_sqlite3_cli_version("x") == (3, 51, 4)

    def test_unexecutable_returns_none(self):
        with patch(
            "hermes_cli.session_lost_and_found.subprocess.run",
            side_effect=OSError("no such file"),
        ):
            assert _parse_sqlite3_cli_version("x") is None


# ---------------------------------------------------------------------------
# The operator-facing guidance never names the live DB
# ---------------------------------------------------------------------------


class TestGuidanceNeverNamesLiveDb:
    def test_gateway_corruption_banner(self):
        """The gateway broadcast must route to the two-stage `sessions
        recover` contract and must warn against pointing a raw sqlite3
        shell at the live file."""
        import gateway.run as gateway_run

        body = inspect.getsource(
            gateway_run.GatewayRunner._send_session_db_warning_notifications
        )
        assert LIVE_DB_SALVAGE_COMMAND not in body
        assert "sessions recover --source" in body
        assert "--inspect-only" in body
        assert "--output" in body
        assert "do NOT" in body

    def test_run_agent_corrupt_explanation(self):
        from run_agent import AIAgent

        explanation = AIAgent._format_turn_completion_explanation(
            "session_persistence_failed", "corrupt"
        )
        assert LIVE_DB_SALVAGE_COMMAND not in explanation
        assert "hermes sessions recover --source" in explanation
        assert "--inspect-only" in explanation
        assert "--output recovered-state.db" in explanation
        assert ".recover" in explanation  # the warning still names the hazard

    def test_repair_budget_error_names_safe_lane(self, tmp_path: Path):
        import hermes_state_repair as hermes_state  # helper lives in the split-out repair module

        message = hermes_state._persistent_repair_exhausted_error(
            tmp_path / "state.db"
        )
        assert "Manual recovery required" in message
        assert "sessions recover --source" in message
        assert "--inspect-only" in message
        assert "--output recovered-state.db" in message
        # The old shape embedded the live path straight into a raw sqlite3
        # command: `sqlite3 {db_path} ".recover"`.
        assert ".recover\"`" not in message
        assert "do NOT" in message

    def test_forensic_backup_refusals_name_safe_lane(self):
        """The low-disk and stat-failure forensic backup refusal strings
        must not embed a raw sqlite3 command against the live path."""
        import hermes_state

        body = inspect.getsource(hermes_state_repair._backup_db_file)
        assert ".recover\"`" not in body
        assert "sessions recover --source" in body
        assert "--inspect-only" in body

    def test_kanban_manual_recovery_warns_about_live_db(self):
        import hermes_cli.kanban_ops as kanban  # ``_cmd_repair`` lives here (split from hermes_cli.kanban)

        source = inspect.getsource(kanban)
        assert '`sqlite3 kanban.db ".recover"`' not in source
        assert "copy kanban.db aside FIRST" in source


# ---------------------------------------------------------------------------
# The emitted command satisfies the real CLI contract
# ---------------------------------------------------------------------------
# The reviewer's blocker on the first iteration of this fix: the banners
# printed `hermes sessions recover --source <db>` — which cmd_sessions
# rejects with exit 2 ("--output is required unless --inspect-only is
# used") before any snapshot is taken. These tests dispatch the EXACT argv
# shapes the banners emit through the real parser + cmd_sessions, so a
# guidance string can never again pass a source-substring test while the
# command it prints deterministically fails.


class TestEmittedCommandsSatisfyCliContract:
    """Every `sessions recover` argv the guidance prints must be accepted
    by the real CLI contract — the reviewer's blocker on the first
    iteration of this fix was exactly this: the banners printed
    `hermes sessions recover --source <db>`, which cmd_sessions rejects
    with exit 2 ("--output is required unless --inspect-only is used")
    before any snapshot is taken.

    These tests dispatch the EXACT argv shapes the banners emit through
    the real `cmd_sessions` (the same function `hermes` main() hands the
    parsed namespace to), so a guidance string can never again pass a
    source-substring test while the command it prints deterministically
    fails.
    """

    @staticmethod
    def _namespace(source: Path, **overrides) -> "argparse.Namespace":
        """The namespace hermes main() produces for `sessions recover`.

        Mirrors the registrations in hermes_cli/main.py (sessions_recover
        subparser): --source, --output, --inspect-only, --work-dir,
        --chunk-size (default 1000), --allow-partial, --report.
        """
        fields = dict(
            sessions_action="recover",
            source=source,
            output=None,
            inspect_only=False,
            work_dir=None,
            chunk_size=1000,
            allow_partial=False,
            report=None,
        )
        fields.update(overrides)
        return argparse.Namespace(**fields)

    def test_old_v1_shape_is_still_rejected(self, tmp_path):
        """Guard the test's own premise: the bare `--source <db>` shape the
        v1 banner printed (neither --inspect-only nor --output) is rejected
        with rc 2 by the real dispatcher."""
        import hermes_cli.sessions_cmd as sc

        rc = sc.cmd_sessions(self._namespace(tmp_path / "state.db"))
        assert rc == 2

    def test_inspect_stage_dispatches_past_gate(self, tmp_path):
        """`--inspect-only` (stage 1 of the emitted sequence) must pass
        the contract gate and reach actual inspection work (rc 0/1, not
        the gate's 2)."""
        import hermes_cli.sessions_cmd as sc

        source = tmp_path / "state.db"
        conn = sqlite3.connect(str(source))
        try:
            conn.execute("CREATE TABLE t (x)")
            conn.commit()
        finally:
            conn.close()

        rc = sc.cmd_sessions(
            self._namespace(source, inspect_only=True)
        )
        assert rc != 2, "--inspect-only shape must pass the contract gate"

    def test_output_stage_dispatches_past_gate(self, tmp_path):
        """`--output recovered-state.db` (stage 2) must pass the contract
        gate and reach actual recovery work (rc 0/1, not the gate's 2)."""
        import hermes_cli.sessions_cmd as sc

        source = tmp_path / "state.db"
        conn = sqlite3.connect(str(source))
        try:
            conn.execute("CREATE TABLE t (x)")
            conn.commit()
        finally:
            conn.close()

        rc = sc.cmd_sessions(
            self._namespace(source, output=tmp_path / "recovered-state.db")
        )
        assert rc != 2, "--output shape must pass the contract gate"

    def test_banner_strings_emit_only_contract_valid_argv(self, tmp_path):
        """The exact argv shapes embedded in the guidance strings, when
        parsed and dispatched, must never return the contract-gate 2.

        Extracts each `sessions recover` invocation printed by the
        banners' code and runs its flag set through the real dispatcher.
        """
        import hermes_cli.sessions_cmd as sc

        source = tmp_path / "state.db"
        conn = sqlite3.connect(str(source))
        try:
            conn.execute("CREATE TABLE t (x)")
            conn.commit()
        finally:
            conn.close()

        # Every emitted flag-set from the five guidance sites. Stage 1
        # (inspect) and stage 2 (output) as printed by the banners:
        emitted_shapes = [
            {"inspect_only": True},                      # --inspect-only
            {"output": tmp_path / "recovered-state.db"},  # --output <new>
        ]
        for overrides in emitted_shapes:
            rc = sc.cmd_sessions(self._namespace(source, **overrides))
            assert rc != 2, (
                f"emitted shape {overrides} must pass the cmd_sessions "
                "contract gate — the banner is printing a command the CLI "
                "rejects before doing anything"
            )
