"""Behavior tests for ``hermes migrate state-to-postgres``.

These tests cover:
- Argument parsing and defaults
- DSN resolution precedence (explicit > env > config)
- Error path when no DSN can be resolved anywhere
- Non-interactive refusal (not a TTY and no --yes)
- Count mismatch surface in output
- Successful migration reporting

``migrate_state_to_postgres.migrate`` is mocked throughout — no PostgreSQL or
Docker required.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_SUMMARY = {
    "sqlite_path": "/tmp/state.db",
    "source_sessions": 3,
    "source_messages": 12,
    "imported_sessions": 3,
    # Counts scoped to THIS migration. These are what success is judged on;
    # target_* are whole-table totals that also include pre-existing rows.
    "migrated_sessions": 3,
    "migrated_messages": 12,
    "target_sessions": 3,
    "target_messages": 12,
    "nul_rows": 0,
    # Field-value verification results.
    "field_check": {
        "sessions_checked": 3,
        "messages_checked": 12,
        "field_mismatches": [],
        "clean": True,
    },
    "complete": True,
}

_MISMATCH_SUMMARY = {
    **_GOOD_SUMMARY,
    "migrated_sessions": 2,  # one short
    "migrated_messages": 9,
    "target_sessions": 2,
    "target_messages": 9,
    "field_check": {
        "sessions_checked": 2,
        "messages_checked": 9,
        "field_mismatches": [],
        "clean": True,
    },
    "complete": False,
}


def _make_args(**kwargs: Any) -> SimpleNamespace:
    """Return a minimal argparse-like namespace for cmd_migrate_state_to_postgres."""
    defaults = {
        "dsn": None,
        "sqlite_path": None,
        "yes": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _run(args: SimpleNamespace, *, tty: bool, env: dict | None = None) -> tuple[int, str, str]:
    """
    Invoke cmd_migrate_state_to_postgres and capture stdout/stderr.

    Returns (exit_code, stdout, stderr).
    """
    from hermes_cli.migrate_postgres import cmd_migrate_state_to_postgres

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    env_patch = env or {}

    with (
        patch("sys.stdout", buf_out),
        patch("sys.stderr", buf_err),
        patch("sys.stdin", MagicMock(isatty=MagicMock(return_value=tty))),
        patch.dict("os.environ", env_patch, clear=False),
    ):
        rc = cmd_migrate_state_to_postgres(args)

    return rc, buf_out.getvalue(), buf_err.getvalue()


# ---------------------------------------------------------------------------
# DSN resolution
# ---------------------------------------------------------------------------


class TestDSNResolution:
    """explicit --dsn > env var > config; no-DSN-anywhere is a clean error."""

    def test_explicit_dsn_is_used(self, tmp_path: Path) -> None:
        """When --dsn is given it is passed straight through to migrate()."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        explicit = "postgresql://user:pw@localhost/db"
        args = _make_args(dsn=explicit, sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ) as mock_migrate, patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, _ = _run(args, tty=False)

        assert rc == 0
        mock_migrate.assert_called_once()
        _, call_dsn = mock_migrate.call_args.args
        assert call_dsn == explicit

    def test_env_var_beats_config(self, tmp_path: Path) -> None:
        """HERMES_STATE_DATABASE_URL takes priority over config resolution."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        env_dsn = "postgresql://env-host/envdb"
        args = _make_args(sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ) as mock_migrate, patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ), patch(
            "hermes_state_postgres.resolve_postgres_dsn",
            return_value="postgresql://config-host/configdb",
        ):
            rc, _, _ = _run(args, tty=False, env={"HERMES_STATE_DATABASE_URL": env_dsn})

        assert rc == 0
        _, call_dsn = mock_migrate.call_args.args
        assert call_dsn == env_dsn

    def test_explicit_dsn_beats_env_and_config(self, tmp_path: Path) -> None:
        """Explicit --dsn wins even when env vars and config both have values."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        explicit = "postgresql://explicit-host/explicitdb"
        args = _make_args(dsn=explicit, sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ) as mock_migrate, patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, _, _ = _run(
                args,
                tty=False,
                env={"HERMES_STATE_DATABASE_URL": "postgresql://env-host/envdb"},
            )

        assert rc == 0
        _, call_dsn = mock_migrate.call_args.args
        assert call_dsn == explicit

    def test_config_dsn_used_when_no_explicit_or_env(self, tmp_path: Path) -> None:
        """Falls back to resolve_postgres_dsn(config) when no explicit/env DSN."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        config_dsn = "postgresql://config-host/configdb"
        args = _make_args(sqlite_path=str(sqlite_file), yes=True)

        import os

        saved = {k: os.environ.pop(k, None) for k in ("HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN")}
        try:
            with patch(
                "migrate_state_to_postgres.migrate",
                return_value=_GOOD_SUMMARY,
            ) as mock_migrate, patch(
                "migrate_state_to_postgres._resolve_sqlite_path",
                return_value=sqlite_file,
            ), patch(
                "hermes_state_postgres.resolve_postgres_dsn",
                return_value=config_dsn,
            ):
                rc, _, _ = _run(args, tty=False)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

        assert rc == 0
        _, call_dsn = mock_migrate.call_args.args
        assert call_dsn == config_dsn

    def test_no_dsn_anywhere_exits_1_with_actionable_message(
        self, tmp_path: Path
    ) -> None:
        """When no DSN can be resolved, exit code is 1 and stderr names the fix."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(sqlite_path=str(sqlite_file), yes=True)

        import os

        saved = {k: os.environ.pop(k, None) for k in ("HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN")}
        try:
            with patch(
                "migrate_state_to_postgres._resolve_sqlite_path",
                return_value=sqlite_file,
            ), patch(
                "hermes_state_postgres.resolve_postgres_dsn",
                return_value=None,
            ):
                rc, _, err = _run(args, tty=False)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

        assert rc == 1
        # Error message must name both the env var and the config key.
        assert "HERMES_STATE_DATABASE_URL" in err or "HERMES_STATE_POSTGRES_DSN" in err
        assert "sessions.postgres_dsn" in err or "config.yaml" in err


# ---------------------------------------------------------------------------
# Non-interactive safety
# ---------------------------------------------------------------------------


class TestNonInteractiveSafety:
    """stdin not a TTY and no --yes must refuse, not hang."""

    def test_non_tty_without_yes_exits_nonzero(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=False)

        with patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, _, err = _run(args, tty=False)

        assert rc != 0
        assert "TTY" in err or "--yes" in err or "-y" in err

    def test_non_tty_with_yes_proceeds(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, _, _ = _run(args, tty=False)

        assert rc == 0

    def test_tty_without_yes_shows_prompt(self, tmp_path: Path) -> None:
        """With a TTY and no --yes a confirmation prompt is shown."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=False)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ), patch(
            "builtins.input",
            return_value="y",
        ) as mock_input, patch(
            "sys.stdin",
            MagicMock(isatty=MagicMock(return_value=True)),
        ):
            rc, out, _ = _run(args, tty=True)

        # input() must have been called (the prompt appeared).
        mock_input.assert_called_once()
        assert rc == 0


# ---------------------------------------------------------------------------
# Count mismatch reporting
# ---------------------------------------------------------------------------


class TestMismatchReporting:
    """A count mismatch must be visually obvious and return non-zero."""

    def test_mismatch_exits_nonzero(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_MISMATCH_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, _ = _run(args, tty=False)

        assert rc != 0

    def test_mismatch_surfaces_in_output(self, tmp_path: Path) -> None:
        """Output must contain something indicating the mismatch, not just a count."""
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_MISMATCH_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, _ = _run(args, tty=False)

        combined = out + _  # out is stdout from _run; _ is stderr
        # The word MISMATCH or a visual indicator must appear.
        assert "MISMATCH" in combined.upper() or "⚠" in combined


# ---------------------------------------------------------------------------
# Success reporting
# ---------------------------------------------------------------------------


class TestSuccessReporting:
    """On a clean migration, exit 0 and source counts visible in output."""

    def test_success_exits_zero(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, _ = _run(args, tty=False)

        assert rc == 0

    def test_success_shows_counts(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://h/db", sqlite_path=str(sqlite_file), yes=True)

        with patch(
            "migrate_state_to_postgres.migrate",
            return_value=_GOOD_SUMMARY,
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, _ = _run(args, tty=False)

        assert str(_GOOD_SUMMARY["source_sessions"]) in out
        assert str(_GOOD_SUMMARY["source_messages"]) in out


# ---------------------------------------------------------------------------
# Connection / credential failure handling
# ---------------------------------------------------------------------------


class TestConnectionFailure:
    """psycopg errors must not spew a raw traceback; they must be caught."""

    def test_psycopg_error_exits_nonzero_no_traceback(self, tmp_path: Path) -> None:
        sqlite_file = tmp_path / "state.db"
        sqlite_file.touch()
        args = _make_args(dsn="postgresql://bad/db", sqlite_path=str(sqlite_file), yes=True)

        class FakePsycopgError(Exception):
            pass

        with patch(
            "migrate_state_to_postgres.migrate",
            side_effect=FakePsycopgError("connection refused"),
        ), patch(
            "migrate_state_to_postgres._resolve_sqlite_path",
            return_value=sqlite_file,
        ):
            rc, out, err = _run(args, tty=False)

        assert rc != 0
        combined = out + err
        # "Traceback" must not appear; the error should be caught and prettified.
        assert "Traceback" not in combined
        # The underlying message should be somewhere in the output.
        assert "connection refused" in combined


# ---------------------------------------------------------------------------
# Argument parsing: verify the subcommand is reachable via the main parser
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    """Smoke-test that the CLI parser wires state-to-postgres correctly."""

    def test_subcommand_registered(self) -> None:
        """The real CLI parser dispatches the migration and forwards its flags."""
        from hermes_cli.main import _build_cli_parser
        from hermes_cli.migrate_postgres import cmd_migrate_state_to_postgres

        parser, _ = _build_cli_parser()
        args = parser.parse_args([
            "migrate", "state-to-postgres", "--dsn", "postgresql://host/state",
            "--sqlite-path", "/tmp/source.db", "--yes",
        ])

        assert args.func is cmd_migrate_state_to_postgres
        assert args.dsn == "postgresql://host/state"
        assert args.sqlite_path == "/tmp/source.db"
        assert args.yes is True


# ---------------------------------------------------------------------------
# Incomplete-migration detection
#
# Rows keep their original SQLite ids and are inserted with ON CONFLICT DO
# NOTHING, so a target whose id space overlaps the source's silently discards
# the colliding rows. The target's whole-table totals cannot detect that: a
# target that already holds rows satisfies any ">= source" comparison no matter
# how much was dropped. Success must be judged on counts scoped to the sessions
# this run actually migrated.
#
# Concretely: migrating a 3-session / 6-message database into a target that
# already held 12 messages imported all 3 sessions and 0 messages, and the
# command reported success with exit 0.
# ---------------------------------------------------------------------------


def test_dropped_messages_fail_the_command():
    """Every session arrives, every message is dropped -> must not report OK."""
    summary = {
        **_GOOD_SUMMARY,
        "source_sessions": 3,
        "source_messages": 6,
        "migrated_sessions": 3,
        "migrated_messages": 0,
        # Healthy-looking totals from rows that were already present.
        "target_sessions": 7,
        "target_messages": 12,
        "complete": False,
    }
    args = _make_args(dsn="postgresql://u:p@h:5432/d", yes=True)
    with patch("migrate_state_to_postgres.migrate", return_value=summary), patch(
        "migrate_state_to_postgres._resolve_sqlite_path",
        return_value=Path("/tmp/state.db"),
    ):
        rc, out, _err = _run(args, tty=False)

    assert rc != 0, f"incomplete migration reported success:\n{out}"
    assert "0/6" in out, f"missing-message count not surfaced:\n{out}"


def test_healthy_target_totals_do_not_mask_missing_rows():
    """Regression: totals far exceeding the source used to read as success."""
    summary = {
        **_GOOD_SUMMARY,
        "source_sessions": 2,
        "source_messages": 4,
        "migrated_sessions": 2,
        "migrated_messages": 1,
        "target_sessions": 99,
        "target_messages": 99,
        "complete": False,
    }
    args = _make_args(dsn="postgresql://u:p@h:5432/d", yes=True)
    with patch("migrate_state_to_postgres.migrate", return_value=summary), patch(
        "migrate_state_to_postgres._resolve_sqlite_path",
        return_value=Path("/tmp/state.db"),
    ):
        rc, out, _err = _run(args, tty=False)

    assert rc != 0, (
        "target totals exceeding the source must not be read as success when "
        f"the migrated counts fall short:\n{out}"
    )


def test_complete_migration_reports_scoped_counts():
    """The happy path reports migrated/source, not whole-table totals."""
    summary = {
        **_GOOD_SUMMARY,
        "source_sessions": 3,
        "source_messages": 6,
        "migrated_sessions": 3,
        "migrated_messages": 6,
        "target_sessions": 3,
        "target_messages": 6,
        "complete": True,
    }
    args = _make_args(dsn="postgresql://u:p@h:5432/d", yes=True)
    with patch("migrate_state_to_postgres.migrate", return_value=summary), patch(
        "migrate_state_to_postgres._resolve_sqlite_path",
        return_value=Path("/tmp/state.db"),
    ):
        rc, out, _err = _run(args, tty=False)

    assert rc == 0, out
    assert "3/3" in out and "6/6" in out, out


@pytest.mark.parametrize("dsn", [
    "postgresql://alice:private-password@localhost/state",
    "postgresql://localhost/state?user=alice&password=private-password",
    "host=localhost user=alice password=private-password dbname=state",
])
def test_confirmation_hides_dsn_credentials(dsn):
    """All supported DSN forms keep their password out of confirmation output."""
    args = _make_args(dsn=dsn)
    with patch("builtins.input", return_value="n"):
        rc, out, err = _run(args, tty=True)

    assert rc != 0
    assert "Target" in out
    assert "private-password" not in out + err


def test_matching_counts_do_not_mask_field_mismatches():
    """The command must respect the migrator's field verification result."""
    summary = {
        **_GOOD_SUMMARY,
        "field_check": {
            "sessions_checked": 3,
            "messages_checked": 12,
            "field_mismatches": ["session.title differs"],
            "clean": False,
        },
        "complete": False,
    }
    args = _make_args(dsn="postgresql://localhost/state", yes=True)
    with patch("migrate_state_to_postgres.migrate", return_value=summary):
        rc, out, err = _run(args, tty=False)

    assert rc != 0
    assert "field" in (out + err).lower()
    assert "Do not switch backends" in out + err
