"""Four-stage probe for #96811 — per-response session ids churn every
conversation-affinity key.

Run: ``python scratch/repro_96811.py`` (no pytest, no network, temp state.db).

The probe walks the causal chain end to end on a real ``SessionDB``:

  S1  reproduce the churn on current main's ``/v1/responses`` shape
  S2  show why the durable key->session mapping cannot help today
  S3  show the two-line repair restores affinity across replies
  S4  isolation: /new rotates, distinct keys never collide, no-key unchanged

Every stage prints PASS/FAIL; the process exits non-zero if any stage fails.
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompt_cache_scope import resolve_prompt_cache_scope  # noqa: E402
from hermes_state import SessionDB  # noqa: E402

SOURCE = "api_server"
DECLARED_KEY = "agent:main:api_server:room-42:member-7"
OTHER_KEY = "agent:main:api_server:room-42:member-8"

_failures: list[str] = []


def check(stage: str, label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {stage} {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _failures.append(f"{stage} {label} {detail}".strip())


def agent_for(session_id: str, db: SessionDB) -> types.SimpleNamespace:
    """Minimal stand-in for the attributes resolve_prompt_cache_scope reads."""
    return types.SimpleNamespace(session_id=session_id, _session_db=db)


def responses_turn(
    db: SessionDB,
    *,
    declared_key: str | None,
    resolve_declared: bool,
    stamp_peer: bool,
) -> str:
    """One POST /v1/responses turn with client-managed conversation_history.

    ``resolve_declared`` / ``stamp_peer`` toggle the two proposed repairs so
    the same code path reproduces both the defect and the fix.
    """
    # gateway/platforms/api_server.py::_handle_responses — client supplies its
    # own history, so the previous_response_id chain yields nothing.
    stored_session_id = None

    declared_session_id = None
    if resolve_declared and declared_key:
        row = db.find_latest_gateway_session_for_peer(
            source=SOURCE, session_key=declared_key
        )
        declared_session_id = row["id"] if row else None

    # main today: `session_id = stored_session_id or str(uuid.uuid4())`
    session_id = stored_session_id or declared_session_id or str(uuid.uuid4())

    # AIAgent._ensure_db_session() — note: no session_key is written.
    if db.get_session(session_id) is None:
        db.create_session(session_id=session_id, source=SOURCE, model="m")

    if stamp_peer and declared_key:
        db.record_gateway_session_peer(
            session_id, source=SOURCE, session_key=declared_key
        )
    return session_id


def affinity_surfaces(session_id: str, db: SessionDB) -> str:
    """The single value all four wire surfaces are derived from.

    prompt_cache_key (codex + chat_completions transports) reads
    resolve_prompt_cache_scope; the OpenRouter/Nous sticky ``session_id`` and
    xAI's ``x-grok-conv-id`` read the conversation root, which is the same
    lineage walk over the same physical id. One value, four consumers.
    """
    return resolve_prompt_cache_scope(agent_for(session_id, db))


def stage_1_reproduce(db: SessionDB) -> tuple[str, str]:
    print("\nS1  reproduce: two replies on one declared conversation")
    a = responses_turn(db, declared_key=DECLARED_KEY, resolve_declared=False, stamp_peer=False)
    time.sleep(0.01)
    b = responses_turn(db, declared_key=DECLARED_KEY, resolve_declared=False, stamp_peer=False)
    check("S1", "physical ids differ per reply", a != b, f"{a[:8]} != {b[:8]}")
    check(
        "S1",
        "affinity scope churns per reply",
        affinity_surfaces(a, db) != affinity_surfaces(b, db),
        "prompt_cache_key / sticky session_id / x-grok-conv-id all re-key",
    )
    return a, b


def stage_2_mapping_unreachable(db: SessionDB, session_id: str) -> None:
    print("\nS2  why the durable mapping cannot help today")
    row = db.get_session(session_id)
    check("S2", "row is created unkeyed", not (row or {}).get("session_key"),
          f"session_key={(row or {}).get('session_key')!r}")
    found = db.find_latest_gateway_session_for_peer(
        source=SOURCE, session_key=DECLARED_KEY
    )
    check("S2", "reset-fenced peer lookup finds nothing", found is None,
          "unkeyed rows are invisible to recovery")


def stage_3_repair(db: SessionDB) -> str:
    print("\nS3  repair: stamp the routing key, resolve the id from it")
    a = responses_turn(db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True)
    time.sleep(0.01)
    b = responses_turn(db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True)
    c = responses_turn(db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True)
    check("S3", "physical id is stable across replies", a == b == c, f"{a[:8]}")
    scopes = {affinity_surfaces(x, db) for x in (a, b, c)}
    check("S3", "affinity scope is stable across replies", len(scopes) == 1,
          next(iter(scopes))[:16])
    return a


def stage_4_isolation(db: SessionDB, live_id: str) -> None:
    print("\nS4  isolation: rotation, collision, opt-out")

    # /new — SessionStore.reset_session ends the row with 'session_reset',
    # which is inside _RESET_END_REASONS, so the recovery fence blocks it.
    live_scope = affinity_surfaces(live_id, db)
    db.end_session(live_id, "session_reset")
    after_reset = responses_turn(
        db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True
    )
    check("S4", "/new rotates the physical id", after_reset != live_id,
          f"{live_id[:8]} -> {after_reset[:8]}")
    check(
        "S4",
        "/new rotates the affinity scope",
        affinity_surfaces(after_reset, db) != live_scope,
        "no ABA: the boundary is durable in sessions.end_reason",
    )

    # idle / daily / suspended auto-resets use the same fence set.
    for reason in ("idle", "daily", "suspended"):
        prev = responses_turn(
            db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True
        )
        db.end_session(prev, reason)
        nxt = responses_turn(
            db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True
        )
        check("S4", f"'{reason}' auto-reset rotates", nxt != prev,
              f"{prev[:8]} -> {nxt[:8]}")

    # A second declared channel never lands on the first one's conversation.
    mine = responses_turn(
        db, declared_key=DECLARED_KEY, resolve_declared=True, stamp_peer=True
    )
    theirs = responses_turn(
        db, declared_key=OTHER_KEY, resolve_declared=True, stamp_peer=True
    )
    check("S4", "distinct declared keys stay isolated", mine != theirs,
          f"{mine[:8]} != {theirs[:8]}")

    # A client that declares nothing keeps today's per-request identity.
    n1 = responses_turn(db, declared_key=None, resolve_declared=True, stamp_peer=True)
    n2 = responses_turn(db, declared_key=None, resolve_declared=True, stamp_peer=True)
    check("S4", "no declared key -> behavior unchanged", n1 != n2,
          "undeclared clients keep per-request ids")


def main() -> int:
    # ignore_cleanup_errors: SessionDB holds the SQLite handle open, and
    # Windows refuses to unlink a mapped file (WinError 32).
    with tempfile.TemporaryDirectory(
        prefix="repro96811_", ignore_cleanup_errors=True
    ) as tmp:
        db = SessionDB(Path(tmp) / "state.db")
        churn_a, _ = stage_1_reproduce(db)
        stage_2_mapping_unreachable(db, churn_a)
        live = stage_3_repair(db)
        stage_4_isolation(db, live)

    print("\n" + "=" * 62)
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all four stages PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
