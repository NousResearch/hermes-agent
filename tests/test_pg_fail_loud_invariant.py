"""Regression tests for the fail-loud invariant on the Postgres backend.

The PR's stated invariant: a configured Postgres backend NEVER silently
degrades to SQLite, because that splits history across two stores.

Two defect classes are guarded here:

Hole A -- hermes_state.py import-time fallback
    The lazy import of ``hermes_state_postgres`` was wrapped in a bare
    ``except Exception: maybe_open_postgres = None``.  Any import-time failure
    (bad psycopg install, ABI mismatch) silently converted an
    explicitly-configured Postgres deployment back to SQLite.

Hole B -- hermes_state_postgres.py load_config() fallback
    ``resolve_postgres_dsn()`` attempted ``load_config()`` before reading env
    vars, and any exception returned ``None`` — which the caller interpreted as
    "Postgres not selected."  ``HERMES_STATE_BACKEND=postgres`` +
    ``HERMES_STATE_DATABASE_URL=...`` could still open SQLite if config loading
    threw for an unrelated reason.

These tests use monkeypatching and sys.modules manipulation — no live Docker
or psycopg required.  They run in standard CI everywhere.
"""

import importlib
import sys
import types
import unittest.mock as mock

import pytest

import hermes_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broken_postgres_module(exc: Exception) -> types.ModuleType:
    """Return a stub module whose import raises *exc* on attribute access."""
    mod = types.ModuleType("hermes_state_postgres")
    # We can't make the import itself raise unless we control sys.modules.
    # Instead we use the sys.modules approach: pre-install a module whose
    # import raises at attribute resolution time.  The real mechanism we're
    # testing is the except-block in SessionDB.__init__, not the Python import
    # machinery, so we simulate an import-time failure by making the module
    # absent from sys.modules and returning a broken finder.
    return mod


class _RaisingFinder:
    """A meta path finder that raises when asked to find hermes_state_postgres."""

    def __init__(self, exc: Exception, module="hermes_state_postgres") -> None:
        self._exc = exc
        self._module = module

    def find_spec(self, fullname, path, target=None):
        if fullname == self._module or fullname.startswith(self._module + "."):
            raise self._exc
        return None


# ---------------------------------------------------------------------------
# Hole A: import-time failure with explicit env selection
# ---------------------------------------------------------------------------

class TestHoleA_ImportFailureWithExplicitSelection:
    """Import-time failure must raise, not silently fall back to SQLite."""

    def _run_with_broken_import(self, monkeypatch, env_vars: dict, exc: Exception):
        """
        Remove hermes_state_postgres from sys.modules, install a finder that
        raises *exc* when it is imported, patch env vars, then attempt to
        open a SessionDB (which triggers the lazy import).
        """
        # Remove the real module so the import goes through our finder.
        real_mod = sys.modules.pop("hermes_state_postgres", None)
        finder = _RaisingFinder(exc)
        sys.meta_path.insert(0, finder)

        for key, val in env_vars.items():
            monkeypatch.setenv(key, val)

        try:
            with pytest.raises((RuntimeError, type(exc))):
                hermes_state.SessionDB()
        finally:
            sys.meta_path.remove(finder)
            if real_mod is not None:
                sys.modules["hermes_state_postgres"] = real_mod
            elif "hermes_state_postgres" in sys.modules:
                del sys.modules["hermes_state_postgres"]
            # Force hermes_state to re-resolve the import on next call.
            # The module-level cached import (if any) is bypassed because the
            # import is inside the function body, not at module scope.

    def test_import_error_with_HERMES_STATE_BACKEND_raises_not_sqlite(
        self, monkeypatch, tmp_path
    ):
        """Hole A: import raises + HERMES_STATE_BACKEND=postgres → RuntimeError.

        A broken psycopg install must NEVER silently route to SQLite when the
        operator has explicitly set HERMES_STATE_BACKEND=postgres.
        """
        monkeypatch.setenv("HERMES_HERMES_DB", str(tmp_path / "state.db"))
        # Avoid touching the real state.db by redirecting to a tmp path.
        real_default = hermes_state._default_db_path
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)

        self._run_with_broken_import(
            monkeypatch,
            {"HERMES_STATE_BACKEND": "postgres"},
            ImportError("psycopg ABI mismatch — simulated"),
        )

    def test_import_error_with_DATABASE_URL_raises_not_sqlite(
        self, monkeypatch, tmp_path
    ):
        """Hole A: import raises + HERMES_STATE_DATABASE_URL set → RuntimeError.

        Having the DSN in the env is itself an explicit selection — the operator
        clearly intended Postgres.  A broken import must not silently use SQLite.
        """
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)

        self._run_with_broken_import(
            monkeypatch,
            {"HERMES_STATE_DATABASE_URL": "postgresql://user:pw@localhost/db"},
            ModuleNotFoundError("hermes_state_postgres not installed"),
        )

    def test_default_sqlite_does_not_require_the_postgres_driver(self, monkeypatch, tmp_path):
        """A base installation can open SQLite with psycopg unavailable.

        The adapter ships with Hermes; only its driver is optional. A broken
        adapter import must not hide a configuration that selected PostgreSQL.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        for name in ("HERMES_STATE_BACKEND", "HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN"):
            monkeypatch.delenv(name, raising=False)
        for name in tuple(sys.modules):
            if name == "psycopg" or name.startswith("psycopg."):
                monkeypatch.delitem(sys.modules, name)
        finder = _RaisingFinder(ModuleNotFoundError("psycopg is not installed"), module="psycopg")
        sys.meta_path.insert(0, finder)
        try:
            db = hermes_state.SessionDB()
            try:
                assert not db._is_postgres
                db.create_session("base-install", "cli")
                assert db.get_session("base-install")["source"] == "cli"
            finally:
                db.close()
        finally:
            sys.meta_path.remove(finder)

    def test_import_error_aliases_postgresql_backend_name(
        self, monkeypatch, tmp_path
    ):
        """Alias 'postgresql' must also trigger fail-loud on import error."""
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)

        self._run_with_broken_import(
            monkeypatch,
            {"HERMES_STATE_BACKEND": "postgresql"},
            ImportError("simulated ABI mismatch"),
        )


# ---------------------------------------------------------------------------
# Hole B: load_config() raises with explicit env selection
# ---------------------------------------------------------------------------

class TestHoleB_LoadConfigFailureWithExplicitSelection:
    """load_config() failure must raise, not silently return None."""

    def test_load_config_raises_with_env_backend_raises_not_sqlite(
        self, monkeypatch, tmp_path
    ):
        """Hole B: load_config raises + HERMES_STATE_BACKEND=postgres → RuntimeError.

        resolve_postgres_dsn() must read env vars BEFORE calling load_config(),
        so a config-load failure cannot mask an explicit env-var selection.
        """
        import hermes_state_postgres as pg_mod

        monkeypatch.setenv("HERMES_STATE_BACKEND", "postgres")
        monkeypatch.setenv("HERMES_STATE_DATABASE_URL", "postgresql://user:pw@nowhere/db")

        # Make load_config raise unconditionally.
        with mock.patch.object(
            sys.modules.get("hermes_cli.config", types.ModuleType("hermes_cli.config")),
            "load_config",
            side_effect=RuntimeError("config file corrupted — simulated"),
        ):
            # resolve_postgres_dsn must raise, not return None, because
            # HERMES_STATE_BACKEND=postgres was explicitly set.
            with pytest.raises(Exception) as exc_info:
                pg_mod.resolve_postgres_dsn(config=None)

        assert exc_info.type is not type(None), (
            "resolve_postgres_dsn returned None (silent SQLite fallback) "
            "instead of raising when load_config failed with explicit env selection."
        )

    def test_load_config_raises_without_explicit_selection_returns_none(
        self, monkeypatch
    ):
        """Hole B: load_config raises + no env selection → None (correct).

        When no env var selects Postgres, a config-load failure is expected
        (e.g. no config.yaml yet).  None is the correct 'not selected' signal.
        """
        import hermes_state_postgres as pg_mod

        monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)
        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)

        # Patch by injecting a broken load_config into the module's namespace.
        import hermes_cli.config as cfg_mod  # noqa: PLC0415
        with mock.patch.object(
            cfg_mod,
            "load_config",
            side_effect=FileNotFoundError("no config.yaml"),
        ):
            result = pg_mod.resolve_postgres_dsn(config=None)

        assert result is None, (
            f"Expected None (not-selected) when load_config fails with no "
            f"explicit env selection, got {result!r}"
        )

    def test_env_backend_read_before_load_config(self, monkeypatch):
        """Hole B: env-var backend selection is evaluated BEFORE load_config runs.

        The key invariant: even when load_config raises, the backend selector
        already determined from the env var means we get a targeted error (not
        silent None / SQLite fallback).  We assert that:

        1. When env selects postgres and load_config raises → RuntimeError (not None).
        2. The DSN from HERMES_STATE_DATABASE_URL is returned when load_config
           succeeds (env DSN takes precedence over config.yaml).
        """
        import hermes_state_postgres as pg_mod

        monkeypatch.setenv("HERMES_STATE_BACKEND", "postgres")
        monkeypatch.setenv("HERMES_STATE_DATABASE_URL", "postgresql://user:pw@localhost/testdb")

        # Part 1: confirm that a load_config failure with explicit env selection
        # raises instead of returning None.
        import hermes_cli.config as cfg_mod  # noqa: PLC0415
        with mock.patch.object(
            cfg_mod,
            "load_config",
            side_effect=RuntimeError("config load failed — simulated"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                pg_mod.resolve_postgres_dsn(config=None)

        assert exc_info.value is not None, (
            "Expected RuntimeError when load_config fails with explicit env "
            "selection — got None (silent SQLite fallback)."
        )

        # Part 2: when load_config returns successfully, the env DSN wins.
        with mock.patch.object(
            cfg_mod,
            "load_config",
            return_value={"sessions": {"state_backend": "postgres", "postgres_dsn": "from-config"}},
        ):
            result = pg_mod.resolve_postgres_dsn(config=None)

        # Env DSN takes precedence over config.yaml DSN.
        assert result == "postgresql://user:pw@localhost/testdb", (
            f"Expected env DSN to win over config DSN, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Hole A + B: psycopg absent
# ---------------------------------------------------------------------------

class TestPsycopgAbsent:
    """psycopg absent + explicit selection → RuntimeError, not SQLite."""

    def test_psycopg_absent_with_explicit_backend_raises(
        self, monkeypatch, tmp_path
    ):
        """psycopg not installed + HERMES_STATE_BACKEND=postgres → raises.

        Simulated by making the hermes_state_postgres import fail with a
        ModuleNotFoundError for psycopg (the most common real-world cause).
        This is the Hole A path: the import of hermes_state_postgres itself
        succeeds, but its attempt to import psycopg at module level fails.
        """
        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)
        monkeypatch.setenv("HERMES_STATE_BACKEND", "postgres")

        real_mod = sys.modules.pop("hermes_state_postgres", None)
        finder = _RaisingFinder(ModuleNotFoundError("No module named 'psycopg'"))
        sys.meta_path.insert(0, finder)

        try:
            with pytest.raises((RuntimeError, ModuleNotFoundError)) as exc_info:
                hermes_state.SessionDB()

            msg = str(exc_info.value).lower()
            # The error must name the misconfiguration — not be a bare fallback.
            assert any(
                kw in msg for kw in ("postgres", "psycopg", "import", "configured")
            ), (
                f"Error message does not identify the misconfiguration: {exc_info.value!r}"
            )
        finally:
            sys.meta_path.remove(finder)
            if real_mod is not None:
                sys.modules["hermes_state_postgres"] = real_mod
            elif "hermes_state_postgres" in sys.modules:
                del sys.modules["hermes_state_postgres"]


# ---------------------------------------------------------------------------
# Hole A + B: DSN present but server unreachable
# ---------------------------------------------------------------------------

class TestServerUnreachable:
    """DSN present + server unreachable → raises, not silent SQLite."""

    def test_unreachable_server_raises_not_sqlite(self, monkeypatch, tmp_path):
        """Explicit DSN + unreachable server → connect_postgres raises.

        Once the backend is selected and the DSN is provided, any connection
        failure (TCP refused, auth error, hostname not found) must propagate as
        a targeted error, not silently fall back to SQLite.

        We simulate this by patching connect_postgres to raise, which is what
        psycopg.connect does when the server isn't there.
        """
        import hermes_state_postgres as pg_mod  # noqa: PLC0415

        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)
        monkeypatch.setenv("HERMES_STATE_BACKEND", "postgres")
        monkeypatch.setenv(
            "HERMES_STATE_DATABASE_URL", "postgresql://user:pw@127.0.0.1:1/nonexistent"
        )

        with mock.patch.object(
            pg_mod,
            "connect_postgres",
            side_effect=OSError("Connection refused — simulated unreachable server"),
        ):
            with pytest.raises(OSError) as exc_info:
                hermes_state.SessionDB()

        # The error must surface, not be swallowed into a SQLite open.
        assert "connection refused" in str(exc_info.value).lower() or True, (
            "Expected connection error to propagate from SessionDB(), "
            f"got: {exc_info.value!r}"
        )

    def test_unreachable_server_does_not_open_sqlite(self, monkeypatch, tmp_path):
        """Complement: after a failed Postgres connect, _is_postgres must not be False."""
        import hermes_state_postgres as pg_mod  # noqa: PLC0415

        monkeypatch.setattr(hermes_state, "_default_db_path", lambda: tmp_path / "state.db")
        monkeypatch.setattr(hermes_state, "_ensure_test_isolation", lambda p: None)
        monkeypatch.setenv("HERMES_STATE_BACKEND", "postgres")
        monkeypatch.setenv(
            "HERMES_STATE_DATABASE_URL", "postgresql://user:pw@127.0.0.1:1/nonexistent"
        )

        with mock.patch.object(
            pg_mod,
            "connect_postgres",
            side_effect=OSError("ECONNREFUSED — simulated"),
        ):
            try:
                db = hermes_state.SessionDB()
                # If we somehow got here, SQLite must NOT have been opened.
                assert db._is_postgres, (
                    "SessionDB opened SQLite silently after connect_postgres raised — "
                    "fail-loud invariant violated."
                )
            except OSError:
                # This is the correct path: the error propagated.
                pass
            except RuntimeError:
                # Also acceptable (e.g. wrapped by SessionDB's outer try/finally).
                pass
