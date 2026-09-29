"""The CLI's session store must be the registry's shared handle for state.db.

``_init_session_store`` used to construct a bare ``SessionDB()``. The goal/loop/heartbeat
managers acquire the same path through ``hermes_state_registry`` from the REPL thread a
moment later, so a bare handle meant a SECOND full open — including the /proc-wide
deleted-WAL sidecar scan (~4k readlinks, each a GIL round-trip against the busy startup
threads) — which showed up as the post-banner freeze before the first prompt.
"""

from types import SimpleNamespace

import hermes_cli.goals as goals
from cli import HermesCLI


def test_cli_session_store_is_the_registry_handle_goals_reuse(monkeypatch):
    import hermes_state_registry

    monkeypatch.setattr(goals, "_DB_CACHE", {})
    constructed = []
    real_open = hermes_state_registry._open_session_db

    def recording_open(path):
        constructed.append(path)
        return real_open(path)

    monkeypatch.setattr(hermes_state_registry, "_open_session_db", recording_open)

    cli = SimpleNamespace()
    try:
        HermesCLI._init_session_store(cli)
        assert cli._session_db is not None and not cli._session_db_unavailable
        # goals/loops/heartbeat go through the registry: same object, no second open.
        assert goals._get_session_db() is cli._session_db
        assert len(constructed) == 1
    finally:
        hermes_state_registry.close_all()
