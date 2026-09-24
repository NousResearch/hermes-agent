"""Selected-PostgreSQL foreign-profile SessionDB guards refuse before any state.db open.

The two leaf (c) guards under test:
- ``APIServerAdapter._open_and_cache_session_db`` (gateway/platforms/api_server.py)
- ``_workdir_owner_db`` (tui_gateway/session_workdir.py)

Both call ``require_legacy_state_db_runtime(home=<target home>)`` BEFORE ``acquire(.../state.db)``,
so a foreign profile that selects PostgreSQL must raise ``PostgreSQLRuntimeActivationError`` with
zero ``state.db`` opens and no ``state.db`` created.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server
from state_store_runtime_readiness import PostgreSQLRuntimeActivationError, trap_state_db_opens

_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


def test_open_and_cache_session_db_foreign_selected_pg_raises_typed(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    obj = SimpleNamespace(
        _session_db_cache_lock=threading.Lock(),
        _session_db_cache_closed=False,
        _session_dbs={},
    )

    with trap_state_db_opens(home) as opens:
        with pytest.raises(PostgreSQLRuntimeActivationError) as excinfo:
            api_server.APIServerAdapter._open_and_cache_session_db(obj, home)

    assert opens == []
    assert not (home / "state.db").exists()
    assert "tui-api-session-runtime" in excinfo.value.report.missing_capabilities


def test_workdir_owner_db_foreign_selected_pg_raises_typed(tmp_path, monkeypatch):
    import tui_gateway.session_workdir as sw

    home = _selected_pg_home(tmp_path, monkeypatch)
    # session_workdir bodies are rebound onto server.py globals at install time;
    # its module globals lack `Path`/`logger`, so supply them for a direct call.
    monkeypatch.setattr(sw, "Path", Path, raising=False)
    monkeypatch.setattr(sw, "logger", logging.getLogger("test-session-workdir"), raising=False)

    with trap_state_db_opens(home) as opens:
        with pytest.raises(PostgreSQLRuntimeActivationError) as excinfo:
            with sw._workdir_owner_db({"profile_home": str(home)}, "fail"):
                pass

    assert opens == []
    assert not (home / "state.db").exists()
    assert "tui-api-session-runtime" in excinfo.value.report.missing_capabilities
