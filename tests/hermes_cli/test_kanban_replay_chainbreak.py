"""ADR-0020 §5.2 replay — the Q8 chain-break specimen on a synthetic board.

Deterministic proof that today's dead-quiet incident cannot recur silently: the
kernel functions run against a temp ``kanban.db`` with no gateway, and the
notification half (Fix D) is checked on the real subscription cursor, so the
``chain_broke`` event provably travels the same route ``completed`` already
uses. Run in ``hermes-upstream:fresh`` via ``ci/hermes-upstream/run-branch.sh``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

Q8_SUMMARY = "preparing handoff successor — filing t_next tomorrow"
WALL_SAMPLE_SUMMARY = (
    "bisect closed: wall sample wall_mid_turn_successor mislabeled; "
    "regenerated wall_mid_turn_successor.parquet, 41/41 rows match baseline"
)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _chain_break_defaults(monkeypatch):
    """Real code defaults, without touching any shared-root config.yaml."""
    def _cfg():
        return {
            "gate": True, "wake": False, "wake_require_idle_queue": True,
            "fresh_window_s": 21600, "wake_patterns": [], "routers": {},
            "chain_break": True, "intent_gate": False, "structural_break_check": False,
            "intent_patterns": None, "intent_denylist": None,
        }
    monkeypatch.setattr(kb, "_successor_guard_config", _cfg)


def _kinds(conn, tid):
    return [e.kind for e in kb.list_events(conn, tid)]


def _one(conn, tid, kind):
    rows = [e for e in kb.list_events(conn, tid) if e.kind == kind]
    assert len(rows) == 1, f"{kind} events on {tid}: {_kinds(conn, tid)}"
    return rows[0]


def _print_container_receipt():
    """CI-CONTAINERS-GUIDANCE receipt: the hermes-upstream:fresh pin, verbatim."""
    for candidate in (Path("ci/hermes-upstream/status.json"),
                      Path("/opt/hermes/ci/hermes-upstream/status.json")):
        if candidate.is_file():
            print("[replay-chainbreak] container pin:", candidate.read_text(encoding="utf-8"))
            return
    print("[replay-chainbreak] container pin: ci/hermes-upstream/status.json not in this tree")


# ---------- scenario 1: the Q8 specimen (BUILD-A §5.2) ----------


def test_replay_q8_chain_break_fires_and_triages(conn, monkeypatch):
    _print_container_receipt()
    _chain_break_defaults(monkeypatch)
    parent = kb.create_task(conn, title="Q8 bisect", body="hunt the divergence", assignee="dev")

    assert kb.complete_task(conn, parent, summary=Q8_SUMMARY, result="", metadata={},
                            created_cards=[]) is True

    row = _one(conn, parent, "chain_broke")
    payload = dict(row.payload or {})
    assert payload["matched_phrase"]
    assert payload["completing_task"] == parent
    triage = kb.get_task(conn, str(payload["triage_task"]))
    assert triage is not None and triage.status == "triage"
    assert triage.title.startswith("CHAIN-BROKE:")
    assert "preparing handoff successor" in triage.body
    # Phase 1 is warn-only: the completion still lands.
    assert _one(conn, parent, "completed")
    assert kb.get_task(conn, parent).status == "done"
    print(f"[replay-chainbreak] PASS event={row.id} triage={triage.id}")


def test_replay_wall_sample_stays_silent(conn, monkeypatch):
    """Negative arm: the known false-positive corpus must produce nothing."""
    _chain_break_defaults(monkeypatch)
    tid = kb.create_task(conn, title="wall bisect", body="relabel samples", assignee="dev")
    assert kb.complete_task(conn, tid, summary=WALL_SAMPLE_SUMMARY) is True
    assert "chain_broke" not in _kinds(conn, tid)
    assert conn.execute(
        "SELECT count(*) c FROM tasks WHERE created_by = ?", (kb._CHAIN_BREAK_CREATOR,)
    ).fetchone()["c"] == 0


def test_replay_explicit_declaration_regression(conn, monkeypatch):
    """G-55 unchanged: SUCCESSOR: required with no successor still refuses, and the
    explicit gate answers first — no triage card, no chain_broke stamp."""
    _chain_break_defaults(monkeypatch)
    tid = kb.create_task(conn, title="declared", body="work\nSUCCESSOR: required — next peel\n",
                         assignee="dev")
    with pytest.raises(kb.MissingSuccessorError):
        kb.complete_task(conn, tid, summary=Q8_SUMMARY, created_cards=[])
    assert kb.get_task(conn, tid).status != "done"
    assert _one(conn, tid, "completion_blocked_missing_successor")
    assert "chain_broke" not in _kinds(conn, tid)


def test_replay_clean_completion_is_silent(conn, monkeypatch):
    """False-positive arm: no intent, discharged chain → zero new signal."""
    _chain_break_defaults(monkeypatch)
    parent = kb.create_task(conn, title="root", body="b", assignee="pm")
    child = kb.create_task(conn, title="leaf", body="b", assignee="dev", parents=[parent],
                           created_by="pm")
    assert kb.complete_task(conn, parent, summary="kicked off, child card minted",
                            created_cards=[child]) is True
    promoted = kb.get_task(conn, child)
    assert promoted is not None and promoted.status == "ready"  # promoted, chain intact
    assert kb.complete_task(conn, child, summary="contrast gate measured 15.70:1, 23 tests green",
                            result="restyled to OutlinedButton") is True
    assert _one(conn, child, "completed")
    assert "chain_broke" not in _kinds(conn, child)
    assert "chain_broke" not in _kinds(conn, parent)


# ---------- scenario 2: notification parity with completions (Fix D) ----------


def test_chain_broke_notifies_on_the_completion_route(conn, monkeypatch):
    """Acceptance arm 5: the same subscription cursor that carries ``completed``
    claims ``chain_broke`` — same kinds allowlist, same route, one claim."""
    from gateway.kanban_watchers_notifier import (
        TERMINAL_KINDS, _EVENT_FORMATTERS, diagnostic_event,
    )

    assert "chain_broke" in TERMINAL_KINDS, "chain-break must be claimed like completions"
    assert "chain_broke" in _EVENT_FORMATTERS, "chain-break must render like completions"
    _chain_break_defaults(monkeypatch)

    tid = kb.create_task(conn, title="notify me", body="b", assignee="dev")
    kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    assert kb.complete_task(conn, tid, summary=Q8_SUMMARY) is True

    _, _, events = kbn.claim_unseen_events_for_sub(
        conn, task_id=tid, platform="telegram", chat_id="chat-1", kinds=TERMINAL_KINDS,
    )
    claimed = [e.kind for e in events]
    assert "completed" in claimed and "chain_broke" in claimed
    # A second claim delivers nothing: per-subscription cursor dedup, unchanged.
    _, _, again = kbn.claim_unseen_events_for_sub(
        conn, task_id=tid, platform="telegram", chat_id="chat-1", kinds=TERMINAL_KINDS,
    )
    assert [e.kind for e in again] == []

    broke = [e for e in events if e.kind == "chain_broke"][0]
    # Primary attention, never diagnostic-suppressed (ADR-0020 §2.4).
    assert diagnostic_event(broke) is False

    class _N:  # the render fields _fmt_chain_broke reads
        head = "[default] Kanban " + tid
        title = "notify me"

    msg, wake_handoff, _detail = _EVENT_FORMATTERS["chain_broke"](broke, _N())
    assert "CHAIN BROKE" in msg
    assert str(dict(broke.payload or {})["triage_task"]) in msg
    assert wake_handoff
