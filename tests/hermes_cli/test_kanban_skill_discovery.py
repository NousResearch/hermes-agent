"""Phase-1 pre-dispatch skill discovery (ADR-001 Spec v1.0).

Covers the hook contract end to end, isolated from the LIVE board DB via
``HERMES_KANBAN_DB``/``HERMES_KANBAN_WORKSPACES_ROOT`` (the same seam the
``kanban_home``-fixture family uses — a scratch repro must never write into
the operator's real kanban.db).

Acceptance criteria under test:
- Hook fires on 100% of claims (every dispatched claim carries discovery
  evidence in its run metadata).
- ``tasks.skills`` is a populated string array after dispatch, OR the run
  metadata proves discovery ran and found nothing relevant (empty [] is
  valid only WITH an evidence record — no placeholder fakes).
- A simulated total tool failure (all sources) hard-blocks the spawn: the
  claim is rolled back, the task stays ``ready``, the failure breaker is
  NOT charged (infrastructure, not worker failure) and
  ``DispatchResult.skipped_discovery_failed`` names the task.
- Degraded-first: one failed source never blocks the dispatch, the vault
  FAISS fallback covers the obsidian outage.
- Kill switch: ``kanban.skill_discovery.enabled: false`` leaves tasks.skills
  untouched and writes no discovery metadata (fully off).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_skill_discovery as ksd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB, pinned via HERMES_KANBAN_DB
    so even code paths resolving the LIVE board's DB cannot escape the sandbox."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    db = home / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "kanban" / "workspaces"))
    return home


@pytest.fixture
def discovery_cfg(monkeypatch):
    """Deterministic hook config: both sources are FAKE-injectable via the
    module attributes the hook reads (search seams), timeouts tiny."""
    cfg = dict(ksd._DISCOVERY_DEFAULTS)
    cfg.update({"enabled": True, "timeout_s": 1})
    monkeypatch.setattr(ksd, "_discovery_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# Source fakes
# ---------------------------------------------------------------------------


def _ok_session(rows=None):
    def _fake(query, limit):
        return (rows or [{"title": "past: skill was kanban-worker",
                          "snippet": "t_01020304 did the thing",
                          "score": 0.9, "session_id": "s1"}]), None
    return _fake


def _ok_vault(rows=None):
    rows = rows if rows is not None else [
        {"title": "07-Incident-Reports/hermes-agent-incident.md",
         "snippet": "the kanban-codex-lane skill solved the deadlock",
         "score": 0.8, "path": "07-Incident-Reports/x.md"}]
    def _fake(query, limit, timeout_s):
        return rows, None
    return _fake


def _fail_source(name):
    def _fail(*args, **kwargs):
        return [], f"{name} down"
    return _fail


@pytest.fixture
def with_skill_dirs(kanban_home, monkeypatch):
    """A registered skill dir set the vault mapping can match names against."""
    for name in ("kanban-orchestrator", "vault-semantic-search", "kanban-codex-lane"):
        (kanban_home / "skills" / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ksd, "collect_skill_names", lambda home=None: frozenset({
        "kanban-orchestrator", "vault-semantic-search", "kanban-codex-lane"}))
    # The auto_load dedup consulted the LIVE profile config; pin it empty so the
    # fixture's three skills are all eligible regardless of this machine's config.
    monkeypatch.setattr("agent.skill_commands.resolve_auto_load_skills",
                        list, raising=False)


def _spawn_record(spawned):
    def _spawn(task, workspace, board=None):
        spawned.append(task)
        return 424242
    return _spawn


# ---------------------------------------------------------------------------
# Hook fires on every claim + skills population
# ---------------------------------------------------------------------------


def test_hook_fires_on_every_claim_and_populates_skills(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """Both sources answer, the vault hit names a registered skill →
    tasks.skills carries it, the run metadata carries the full evidence."""
    monkeypatch.setattr(ksd, "session_search", _ok_session())
    vault_rows = [{
        "title": "03-Projects/skill-refactor.md",
        "snippet": "use the vault-semantic-search skill for this exact query",
        "score": 0.86, "path": "03-Projects/skill-refactor.md",
    }]
    monkeypatch.setattr(ksd, "search_obsidian", _ok_vault(vault_rows))

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Refactor the vault search skill wiring",
                             body="mirror the token pattern", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned, res
        assert res.auto_blocked == []

        trow = conn.execute("SELECT skills FROM tasks WHERE id = ?", (tid,)).fetchone()
        skills = json.loads(trow["skills"])
        assert skills == ["vault-semantic-search"]

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        disc = json.loads(run["metadata"])["discovery"]
        assert disc["status"] == "ok"
        assert disc["queries"]["session"] and disc["queries"]["vault"]
        assert disc["audit"]["q1_sha256"] and disc["audit"]["q2_sha256"]
        assert disc["skills_written"] == skills
        required = {"skill", "score", "source", "ref", "snippet"}
        assert all(required <= set(h) for h in disc["hits"])
        assert len(disc["hits"]) <= 8


def test_empty_skills_valid_only_with_evidence_record(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """Both sources answer but name NO registered skill → skills may be []
    — but the evidence record must exist and carry the hits."""
    monkeypatch.setattr(ksd, "session_search", _ok_session())
    monkeypatch.setattr(ksd, "search_obsidian", _ok_vault([
        {"title": "02-Areas/trip-planning.md",
         "snippet": "completely unrelated content about trains",
         "score": 0.5, "path": "02-Areas/trip-planning.md"}]))

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Book a train to Hamburg", body="weekend trip",
                             assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        trow = conn.execute("SELECT skills FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert json.loads(trow["skills"]) == []
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        disc = json.loads(run["metadata"])["discovery"]
        assert disc["status"] == "ok"
        assert disc["result_count"] if "result_count" in disc else len(disc["hits"]) > 0
        assert disc["skills_written"] == []


def test_max_three_skills_deduplicated_and_score_sorted(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """Four vault hits naming four registered skills → only top 3 by score
    land, deduped."""
    rows = [
        {"title": "a.md", "snippet": "kanban-orchestrator is relevant", "score": 0.4, "path": "a.md"},
        {"title": "b.md", "snippet": "kanban-orchestrator again (dup target)", "score": 0.9, "path": "b.md"},
        {"title": "c.md", "snippet": "vault-semantic-search mentioned", "score": 0.7, "path": "c.md"},
        {"title": "d.md", "snippet": "kanban-codex-lane run", "score": 0.6, "path": "d.md"},
    ]
    monkeypatch.setattr(ksd, "session_search", _ok_session(rows=None))
    monkeypatch.setattr(ksd, "search_obsidian", _ok_vault(rows))

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Mixed skill test", body=None, assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        trow = conn.execute("SELECT skills FROM tasks WHERE id = ?", (tid,)).fetchone()
        skills = json.loads(trow["skills"])
        assert len(skills) == 3
        # score-desc across the deduped map: kanban-orchestrator (0.9), vault-semantic-search (0.7), kanban-codex-lane (0.6)
        assert skills == ["kanban-orchestrator", "vault-semantic-search", "kanban-codex-lane"]


# ---------------------------------------------------------------------------
# Degraded-first failure handling
# ---------------------------------------------------------------------------


def test_one_source_down_degrades_and_dispatches(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """Session DB down, Obsidian up → dispatch happens, status degraded:session."""
    monkeypatch.setattr(ksd, "session_search", _fail_source("session"))
    monkeypatch.setattr(ksd, "search_obsidian", _ok_vault())

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Only vault knows this",
                             body="vault-semantic-search saves the day", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned  # NOT skipped/blocked
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        disc = json.loads(run["metadata"])["discovery"]
        assert disc["status"] == "degraded:session"
        assert disc["source_results"]["session"]["status"] == "failed"


def test_obsidian_down_falls_back_to_vault_search_cli(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs, tmp_path,
):
    """Obsidian timeout/error triggers the FAISS CLI fallback; a fallback hit
    still populates skills and the record names the fallback path."""
    # Fake vault_search.py CLI instead of hitting the real one: a stub that
    # the fallback module runs via python.
    cli = tmp_path / "vault_search.py"
    cli.write_text(
        "import json, sys\n"
        "q = sys.argv[1]\n"
        "print(json.dumps({'success': True, 'results': [{'rank': 1, 'score': 0.77,"
        " 'file_path': '03-Projects/fallback-hit.md', 'title': 'fallback hit',"
        " 'section': 'skills used: kanban-codex-lane here', 'excerpt': 'run kanban-codex-lane',"
        " 'line_start': 1, 'line_end': 2}]}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_VAULT_SEARCH_CLI", str(cli))
    monkeypatch.setattr(ksd, "session_search", _ok_session())
    monkeypatch.setattr(ksd, "search_obsidian", _fail_source("obsidian"))

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Fallback test", body=None, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        disc = json.loads(run["metadata"])["discovery"]
        assert disc["status"] == "degraded:vault"
        assert disc["source_results"]["vault_fallback"]["status"] == "ok"
        assert json.loads(
            conn.execute("SELECT skills FROM tasks WHERE id = ?", (tid,)).fetchone()["skills"]
        ) == ["kanban-codex-lane"]


# ---------------------------------------------------------------------------
# Total failure → hard skip, breaker untouched
# ---------------------------------------------------------------------------


def test_total_source_failure_skips_spawn_task_stays_ready(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """ALL sources down → no spawn this tick, claim rolled back, card ready,
    consecutive_failures NOT charged, dedicated result bucket filled."""
    monkeypatch.setattr(ksd, "session_search", _fail_source("session"))
    monkeypatch.setattr(ksd, "search_obsidian", _fail_source("obsidian"))
    monkeypatch.setenv("HERMES_VAULT_SEARCH_CLI", "/nonexistent/vault_search.py")

    spawn_called = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Nobody can look this up", body=None, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawn_called))
        assert res.skipped_discovery_failed == [tid]
        assert res.spawned == []
        assert res.auto_blocked == []
        assert spawn_called == []  # spawn NEVER ran

        row = conn.execute(
            "SELECT status, claim_lock, claim_expires, consecutive_failures "
            "FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None and row["claim_expires"] is None
        assert row["consecutive_failures"] == 0  # infra failure, not the card's

        # The skipped run carried the failure evidence.
        runs = conn.execute(
            "SELECT outcome, status FROM task_runs WHERE task_id = ?", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["skipped_discovery_failed"]

        events = [e["kind"] for e in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (tid,),
        ).fetchall()]
        assert "discovery_total_failure" in events
        assert "skipped_discovery_failed" in events

        # Next tick with sources recovered: dispatches normally (retry path).
    with kbc.connect() as conn:
        monkeypatch.setattr(ksd, "session_search", _ok_session())
        monkeypatch.setattr(ksd, "search_obsidian", _ok_vault())
        spawned = []
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned and spawned[0].id == tid


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_leaves_skills_untouched_and_no_evidence(
    all_assignees_spawnable, kanban_home, monkeypatch, with_skill_dirs,
):
    """``kanban.skill_discovery.enabled: false`` → no search, no write,
    board keeps dispatching."""
    cfg = dict(ksd._DISCOVERY_DEFAULTS)
    cfg["enabled"] = False
    monkeypatch.setattr(ksd, "_discovery_config", lambda: cfg)

    def _must_not_run(*a, **kw):
        raise AssertionError("discovery sources must not run when disabled")
    monkeypatch.setattr(ksd, "session_search", _must_not_run)
    monkeypatch.setattr(ksd, "search_obsidian", _must_not_run)

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Kill switch test", body=None, assignee="alice",
                             skills=["preexisting-skill"])
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned
        assert spawned[0].skills == ["preexisting-skill"]  # untouched by discovery
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert not (run["metadata"] and json.loads(run["metadata"]).get("discovery"))


# ---------------------------------------------------------------------------
# Hook hardening: it must NEVER crash the dispatch loop
# ---------------------------------------------------------------------------


def test_hook_never_crashes_dispatch_on_source_exception(
    all_assignees_spawnable, kanban_home, monkeypatch, discovery_cfg, with_skill_dirs,
):
    """A source that raises (not returns an error) degrades into the result;
    dispatch proceeds when the other source answers."""
    def _boom(*a, **kw):
        raise RuntimeError("connection exploded")
    monkeypatch.setattr(ksd, "session_search", _boom)
    monkeypatch.setattr(ksd, "search_obsidian", _ok_vault())

    spawned = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Source raises", body=None, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_record(spawned))
        assert res.spawned
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        disc = json.loads(run["metadata"])["discovery"]
        assert disc["source_results"]["session"]["status"] == "failed"


def test_config_read_failure_defaults_to_enabled(monkeypatch):
    """A corrupt config must never silently widen the kill switch."""
    import hermes_cli.kanban_db as _kdd
    from hermes_cli import kanban_db_dispatch
    # point the loader at a broken config source
    def _broken():
        raise RuntimeError("config exploded")
    import hermes_cli.kanban_skill_discovery as ksd2
    monkeypatch.setattr(
        "hermes_cli.config_effective.load_user_config_effective", _broken, raising=False)
    cfg = ksd._discovery_config()
    # Unreadable config → defaults → hook stays enabled
    assert cfg["enabled"] is True