"""Regression tests for the second-round review blockers.

Each pins a specific failure the reviewer identified at head 7aa43776:

1. ``open_store_for_profile`` mutated process-global ``HERMES_STATE_*`` env
   vars around ``SessionDB()``. A lock serialized seam callers but could not
   stop an unrelated ``SessionDB()`` on another thread from observing the
   pinned DSN and opening the wrong physical store.

2. An existing-but-unparseable ``config.yaml`` was treated as "SQLite", even
   though that file may be the only source selecting Postgres. Absent config
   legitimately means SQLite; unreadable config does not.

3. The Postgres search predicate used a bare ``active = 1``, dropping
   compaction-archived rows (``active=0, compacted=1``) that the SQLite
   contract keeps searchable.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. The seam must not mutate process-global backend selectors
# ---------------------------------------------------------------------------


class TestSeamDoesNotMutateGlobalEnv:
    def test_no_env_mutation_symbol_remains(self):
        """The seam-env lock is gone because the technique it guarded is gone.

        Behavioural proxy: the module must not expose a lock whose only purpose
        was serializing env mutation. Its presence would mean the env-pinning
        approach came back.
        """
        import hermes_state_postgres as hsp

        assert not hasattr(hsp, "_SEAM_ENV_LOCK"), (
            "_SEAM_ENV_LOCK exists only to serialize process-global env "
            "mutation in open_store_for_profile; the DSN is now an explicit "
            "constructor argument, so the lock (and the technique) must be gone"
        )

    def test_session_db_accepts_explicit_dsn(self):
        """SessionDB takes the DSN as an argument, not via the environment."""
        import inspect

        from hermes_state import SessionDB

        sig = inspect.signature(SessionDB.__init__)
        assert "postgres_dsn" in sig.parameters, (
            "SessionDB must accept an explicit postgres_dsn so the profile "
            "seam can pin a store per-instance without touching os.environ"
        )
        param = sig.parameters["postgres_dsn"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None, (
            "default None keeps normal env+config resolution unchanged"
        )

    def test_concurrent_open_cannot_observe_another_profiles_dsn(self, monkeypatch):
        """A racing SessionDB() must never see a seam caller's target DSN.

        Drives the real resolution helper on a second thread while the seam
        path runs, and asserts the racing thread's view of the backend
        selectors is unchanged throughout.
        """
        import os

        from hermes_state_postgres import maybe_open_postgres

        monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)
        monkeypatch.delenv("HERMES_STATE_DATABASE_URL", raising=False)
        monkeypatch.delenv("HERMES_STATE_POSTGRES_DSN", raising=False)

        observed: list[tuple] = []
        stop = threading.Event()

        def _watch():
            while not stop.is_set():
                observed.append(
                    (
                        os.environ.get("HERMES_STATE_BACKEND"),
                        os.environ.get("HERMES_STATE_DATABASE_URL"),
                        os.environ.get("HERMES_STATE_POSTGRES_DSN"),
                    )
                )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        try:
            # dsn_override is the seam's mechanism. Connecting will fail (no
            # server), which is fine — the point is that reaching the connect
            # attempt never required mutating the environment.
            for _ in range(50):
                try:
                    maybe_open_postgres(
                        False, 1, dsn_override="postgresql://unreachable/x"
                    )
                except Exception:
                    pass
        finally:
            stop.set()
            watcher.join(timeout=5)

        assert observed, "watcher thread did not sample the environment"
        assert set(observed) == {(None, None, None)}, (
            "a concurrent thread observed backend selectors set by the seam; "
            f"distinct samples: {sorted(set(observed))}"
        )


# ---------------------------------------------------------------------------
# 2. Unreadable config must fail closed, not resolve to SQLite
# ---------------------------------------------------------------------------


class TestUnreadableConfigFailsClosed:
    def _make_profile(self, tmp_path: Path, monkeypatch, body: str) -> str:
        """Create a profile whose config.yaml contains *body*."""
        from hermes_cli import profiles as profiles_mod

        name = "cfgtest"
        pdir = tmp_path / "profiles" / name
        pdir.mkdir(parents=True)
        (pdir / "config.yaml").write_text(body, encoding="utf-8")

        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "validate_profile_name", lambda n: None)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: str(pdir))
        return name

    def test_malformed_yaml_raises_rather_than_selecting_sqlite(
        self, tmp_path, monkeypatch
    ):
        from hermes_state_postgres import open_store_for_profile

        name = self._make_profile(
            tmp_path, monkeypatch, "sessions:\n  state_backend: [unclosed\n"
        )

        with pytest.raises(RuntimeError, match="not usable|could not be read or parsed"):
            open_store_for_profile(name)

    def test_malformed_yaml_selector_raises_rather_than_returning_false(
        self, tmp_path, monkeypatch
    ):
        from hermes_state_postgres import profile_selects_postgres

        name = self._make_profile(
            tmp_path, monkeypatch, "sessions:\n  state_backend: [unclosed\n"
        )

        with pytest.raises(RuntimeError, match="not usable|could not be read or parsed"):
            profile_selects_postgres(name)

    @pytest.mark.parametrize(
        "body", ("- just\n- a\n- list\n", "a-bare-scalar\n")
    )
    def test_non_mapping_config_fails_closed(self, tmp_path, monkeypatch, body):
        """A structurally invalid root cannot answer 'not Postgres'.

        Returning False here routes a cross-profile reader to SQLite. A list
        or scalar root means the operator wrote something unusable, so their
        intent is as unknown as it is for unparseable YAML — and both must
        fail closed rather than pick a physical store on the operator's behalf.
        """
        from hermes_state_postgres import profile_selects_postgres

        name = self._make_profile(tmp_path, monkeypatch, body)
        with pytest.raises(RuntimeError, match="not usable"):
            profile_selects_postgres(name)

    def test_empty_config_is_a_legitimate_sqlite_selection(
        self, tmp_path, monkeypatch
    ):
        """An empty file genuinely expresses no selection — SQLite is correct."""
        from hermes_state_postgres import profile_selects_postgres

        name = self._make_profile(tmp_path, monkeypatch, "")
        assert profile_selects_postgres(name) is False

    def test_absent_config_is_a_legitimate_sqlite_selection(
        self, tmp_path, monkeypatch
    ):
        """No config file at all also means no selection."""
        from hermes_cli import profiles as profiles_mod
        from hermes_state_postgres import profile_selects_postgres

        pdir = tmp_path / "profiles" / "nocfg"
        pdir.mkdir(parents=True)
        monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda n: n)
        monkeypatch.setattr(profiles_mod, "profile_exists", lambda n: True)
        monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda n: str(pdir))

        assert profile_selects_postgres("nocfg") is False


# ---------------------------------------------------------------------------
# 3. Compaction-archived rows stay searchable
# ---------------------------------------------------------------------------


class TestCompactedRowVisibilityParity:
    def test_pg_where_matches_sqlite_visibility_contract(self):
        """The Postgres predicate must keep compacted rows searchable.

        SQLite uses ``(m.active = 1 OR m.compacted = 1)``: rewind/undo rows are
        hidden, compaction-archived rows remain part of the searchable record.
        A bare ``active = 1`` silently drops every compacted message after a
        move to Postgres.
        """
        from hermes_state_postgres import _build_where

        params: list = []
        where = _build_where(
            source_filter=None,
            exclude_sources=None,
            role_filter=None,
            include_inactive=False,
            params=params,
        )
        joined = " ".join(where)

        assert "compacted" in joined, (
            "Postgres search drops compaction-archived rows; SQLite keeps them "
            f"via (active = 1 OR compacted = 1). Got: {joined}"
        )
        assert "active = 1 OR" in joined.replace("  ", " "), (
            f"expected an OR-form visibility predicate, got: {joined}"
        )

    def test_include_inactive_emits_no_visibility_predicate(self):
        """include_inactive=True still returns everything."""
        from hermes_state_postgres import _build_where

        params: list = []
        where = _build_where(
            source_filter=None,
            exclude_sources=None,
            role_filter=None,
            include_inactive=True,
            params=params,
        )
        joined = " ".join(where)
        assert "active" not in joined and "compacted" not in joined
