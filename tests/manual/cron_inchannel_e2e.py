"""
Offline E2E for continuable in-channel cron (specs/cron-inchannel-continuable).

Exercises the REAL create→persist→find→append path end-to-end against a REAL
SessionStore + REAL mirror_to_session + REAL _find_session_id + REAL
build_session_key — NO mocking of the session layer. This is the harness that
would have caught the shipped bug (the first version mocked mirror_to_session and
so never exercised the fact that the mirror only APPENDS to a pre-existing
session; the flat channel row was never created and the brief was silently lost).

Two scenarios, each asserting the brief actually lands in the SAME session the
inbound reply resolves to:

  CHANNEL: cron in_channel delivery → _seed_cron_channel_session CREATES the flat
    (slack, C, None) session (chat_type=group, keyed to the origin user) and
    mirrors the brief in. Then a plain channel reply (reply_in_thread:false →
    thread_id=None) keys to the SAME session → the brief is in its transcript.

  1:1 DM: same, chat_type=dm. The DM session key ignores user_id, so the reply
    resolves regardless; assert the brief lands and the key matches.

Run from INSIDE the worktree (so the worktree code loads, not the editable
main-checkout install):

    cd <worktree>
    PYTHONPATH="$PWD" ../../.venv/bin/python tests/manual/cron_inchannel_e2e.py

Uses a throwaway HERMES_HOME so it never touches ~/.hermes. No real names.
"""

import os
import sys
import tempfile
from pathlib import Path


def _fresh_home():
    """Point HERMES_HOME at a throwaway dir BEFORE importing gateway modules
    (mirror.py binds _SESSIONS_INDEX from get_hermes_home() at import time)."""
    d = tempfile.mkdtemp(prefix="cron_inchannel_e2e_")
    os.environ["HERMES_HOME"] = d
    return Path(d)


HOME = _fresh_home()

# Import AFTER HERMES_HOME is set.
import cron.scheduler as sched  # noqa: E402
import gateway.mirror as mirror  # noqa: E402
from gateway.config import GatewayConfig, Platform  # noqa: E402
from gateway.session import SessionStore, SessionSource, build_session_key  # noqa: E402

# Force mirror.py's module-level index path to our temp home (it may have bound
# a different get_hermes_home() at import if something imported it earlier).
mirror._SESSIONS_DIR = HOME / "sessions"
mirror._SESSIONS_INDEX = HOME / "sessions" / "sessions.json"

BRIEF = "brief: PRs need review\n- Harden: session lifecycle teardown"


def _real_store():
    cfg = GatewayConfig()
    store = SessionStore(HOME / "sessions", cfg)
    return store


def _run_scenario(name, chat_id, is_dm, reply_chat_type):
    print(f"\n=== {name} (chat_id={chat_id}, is_dm={is_dm}) ===")
    store = _real_store()

    # A real Slack-like adapter exposing only what the seeder needs: the live
    # session store. (We call the seeder directly — the delivery leg's flat-post
    # is covered by the unit tests; here we prove the SESSION plumbing works.)
    class _Adapter:
        _session_store = store

    ok = sched._seed_cron_channel_session(
        {"id": "brief-job", "name": "PR review brief"},
        _Adapter(), "slack", chat_id, BRIEF,
        is_dm=is_dm, user_id="U_HUMAN", chat_name="test",
    )
    assert ok, f"{name}: seeder returned False — session not created/mirrored"

    # LEG 1: what session key did the seed create?
    seeded_source = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id,
        chat_type="dm" if is_dm else "group",
        user_id="U_HUMAN", thread_id=None,
    )
    seed_key = build_session_key(seeded_source)

    # LEG 2: what does a plain inbound reply (reply_in_thread:false → thread None)
    # from the same user resolve to?
    inbound = SessionSource(
        platform=Platform.SLACK, chat_id=chat_id, chat_type=reply_chat_type,
        user_id="U_HUMAN", thread_id=None,
    )
    reply_key = build_session_key(inbound)
    print(f"  seed key : {seed_key}")
    print(f"  reply key: {reply_key}")
    assert seed_key == reply_key, f"{name}: KEY MISMATCH — reply won't continue the seed"

    # GROUND TRUTH: the brief must actually be in that session's transcript, and
    # discoverable via the same _find_session_id the inbound reply path uses.
    sid = mirror._find_session_id("slack", chat_id, thread_id=None, user_id="U_HUMAN")
    assert sid, f"{name}: _find_session_id found NO session — the reply would dead-end"
    # Read the session transcript back and confirm the brief text is present.
    idx = mirror._SESSIONS_INDEX
    import json
    data = json.loads(idx.read_text())
    entry = next((e for e in data.values() if isinstance(e, dict) and e.get("session_id") == sid), None)
    assert entry, f"{name}: session {sid} not in index"
    # transcript lives in the JSONL / SQLite; verify via the store's own read.
    found = _brief_in_transcript(store, sid)
    assert found, f"{name}: brief NOT found in session {sid} transcript"
    print(f"  ✓ session {sid} created, brief present, reply resolves here")
    return True


def _brief_in_transcript(store, sid):
    """Best-effort read of the session transcript to confirm the brief landed."""
    # Try the SQLite DB first (the mirror writes both JSONL + SQLite).
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        msgs = db.get_messages(sid)
        for m in msgs:
            if "PRs need review" in str(m.get("content", "")):
                return True
    except Exception:
        pass
    # Fallback: scan the JSONL transcript file.
    for p in (HOME / "sessions").glob("*.json*"):
        try:
            if "PRs need review" in p.read_text():
                return True
        except Exception:
            continue
    return False


def main():
    print(f"scheduler module: {sched.__file__}")
    print(f"HERMES_HOME (throwaway): {HOME}")
    if "cron-inchannel" not in sched.__file__:
        print("WARNING: not the worktree scheduler — set PYTHONPATH=$PWD", file=sys.stderr)

    _run_scenario("CHANNEL", "C_TEST", is_dm=False, reply_chat_type="group")
    _run_scenario("1:1 DM", "D_TEST", is_dm=True, reply_chat_type="dm")

    print(
        "\nPASS: in_channel cron seeds the flat session for BOTH a channel and a "
        "1:1 DM; the brief lands in the transcript and a plain reply resolves to "
        "the same session (continuation works)."
    )


if __name__ == "__main__":
    main()
