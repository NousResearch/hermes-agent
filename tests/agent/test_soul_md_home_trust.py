"""Regression (#136321): a wrong or unresolvable home must never silently
degrade to ambient SOUL.md identity.

#50233 scoped load_soul_md to the agent's own home, but two resolution paths
still leaked the launch profile's identity:

- PATH B: no HERMES_HOME override and no session db (a bare thread that lost
  the ContextVar) resolved ambient. In a multi-profile deployment ambient is
  the launch home — the WRONG identity for every other profile.
- PATH A (shared-db lane): a gateway multiplex hands every agent the shared
  launch-home state.db and binds profiles per turn; a thread that lost the
  binding resolves the launch home while its session row still records the
  bound profile. The disagreement marks the db-derived home as untrusted.

Both cases now skip SOUL.md (falling back to the built-in default identity)
instead of loading another profile's soul. A single-profile deployment keeps
ambient resolution: ambient == the only home there is.
"""

import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _RowDB:
    """Minimal SessionDB stand-in: db_path plus a read ctx over a real
    sessions(id, profile_name) table."""

    def __init__(self, db_path: Path, rows: dict[str, str]):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, profile_name TEXT)"
        )
        for sid, profile in rows.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?)", (sid, profile)
            )
        self._conn.commit()

    def _read_ctx(self):
        return closing(self._conn)


def _bare_agent(**overrides):
    base = dict(
        load_soul_identity=True,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _session_db=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _multi_profile_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "profiles" / "mybot").mkdir(parents=True)
    (root / "SOUL.md").write_text("LAUNCH SOUL — must not leak", encoding="utf-8")
    return root


def test_unresolvable_home_skips_soul_in_multi_profile_deployment(
    tmp_path, monkeypatch
):
    """PATH B: bare agent (no session db) on a thread that lost the HERMES_HOME
    ContextVar, in a deployment that actually has profiles — ambient resolution
    would read the launch profile's SOUL.md into an unknown-profile agent."""
    from agent import system_prompt

    root = _multi_profile_root(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))

    agent = _bare_agent()
    result = {}

    def load():
        result["soul"] = system_prompt._agent_soul_md(agent, None)

    t = threading.Thread(target=load)
    t.start()
    t.join()

    assert result["soul"] is None


def test_unresolvable_home_keeps_ambient_in_single_profile_deployment(
    tmp_path, monkeypatch
):
    """Single-profile deployment: ambient IS the only home, so an agent with no
    resolvable home still reads it — no behavior change for the common case."""
    from agent import system_prompt

    home = tmp_path / "solo"
    home.mkdir()
    (home / "SOUL.md").write_text("SOLO SOUL", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert system_prompt._agent_soul_md(_bare_agent(), None) == "SOLO SOUL"


def test_shared_db_profile_mismatch_skips_soul(tmp_path, monkeypatch):
    """PATH A, shared-db lane: no override bound, db-derived home is the launch
    (default) home, but the session row records profile 'mybot' — the row is
    the ground truth from row-creation time, so the launch home is untrusted."""
    from agent import system_prompt

    root = _multi_profile_root(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = _RowDB(root / "state.db", {"s1": "mybot"})

    agent = _bare_agent(_session_db=db, session_id="s1")
    assert system_prompt._agent_soul_md(agent, None) is None


def test_matching_session_profile_still_loads_soul(tmp_path, monkeypatch):
    """No false positives: db-derived home and the session row agree on the
    profile, so the SOUL.md of that home loads as before (#50233 semantics)."""
    from agent import system_prompt

    root = _multi_profile_root(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = _RowDB(root / "state.db", {"s1": "default"})

    agent = _bare_agent(_session_db=db, session_id="s1")
    assert system_prompt._agent_soul_md(agent, None) == "LAUNCH SOUL — must not leak"


def test_missing_session_row_keeps_db_home_semantics(tmp_path, monkeypatch):
    """A fresh session builds its prompt BEFORE its row exists (#45499): no row
    means no disagreement signal, and the db-derived home stands (#50233)."""
    from agent import system_prompt

    root = _multi_profile_root(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = _RowDB(root / "state.db", {})

    agent = _bare_agent(_session_db=db, session_id="fresh")
    assert system_prompt._agent_soul_md(agent, None) == "LAUNCH SOUL — must not leak"


def test_full_prompt_uses_default_identity_when_home_untrusted(tmp_path, monkeypatch):
    """Wiring: the fail-closed path must reach build_system_prompt — a
    multi-profile deployment with an unresolvable home yields the built-in
    default identity, never the launch profile's SOUL.md."""
    from agent import prompt_builder
    from agent.system_prompt import build_system_prompt, DEFAULT_AGENT_IDENTITY

    root = _multi_profile_root(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=False)

    agent = _bare_agent()
    result = {}

    def build():
        with patch("agent.prompt_builder.build_environment_hints", return_value=""):
            result["prompt"] = build_system_prompt(agent)

    t = threading.Thread(target=build)
    t.start()
    t.join()

    assert "LAUNCH SOUL" not in result["prompt"]
    assert DEFAULT_AGENT_IDENTITY in result["prompt"]
