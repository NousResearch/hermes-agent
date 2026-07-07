"""E2E for topic backfill with BOTH SessionDB siblings AND the raw Bot-API log.

Real temp HERMES_HOME, real SessionDB, real sessions.json, real log file. Covers
the six required assertions plus the B1 regression (pristine home, log only, NO
state.db) and a process-safe concurrency test for the writer's fcntl lock.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway import topic_backfill
from gateway.session import SessionSource, is_shared_multi_user_session
from gateway.platforms.base import Platform

# The pure-stdlib writer lives under ~/.hermes/scripts. Tests import it by path.
_SCRIPTS_DIR = Path.home() / ".hermes" / "scripts"


def _import_writer():
    import importlib.util

    p = _SCRIPTS_DIR / "tg_topic_recent_posts.py"
    spec = importlib.util.spec_from_file_location("tg_topic_recent_posts_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed_session(home, *, session_id, chat_id, thread_id, user_id, user_name, messages):
    index_path = home / "sessions" / "sessions.json"
    data = {}
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    key = f"agent:main:telegram:group:{chat_id}:{thread_id or ''}:{user_id}"
    data[key] = {
        "session_key": key,
        "session_id": session_id,
        "origin": {
            "platform": "telegram", "chat_id": chat_id, "thread_id": thread_id,
            "user_id": user_id, "user_name": user_name, "chat_type": "group",
        },
    }
    index_path.write_text(json.dumps(data), encoding="utf-8")

    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id=session_id, source="telegram")
    for m in messages:
        db.append_message(
            session_id=session_id, role=m["role"], content=m.get("content"),
            timestamp=m.get("timestamp"),
        )


def _write_log(home, chat_id, thread_id, posts):
    d = home / "state" / "topic-recent-posts" / "telegram" / str(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{thread_id}.json").write_text(json.dumps({"posts": posts}), encoding="utf-8")


# (a) log-only post visible
def test_log_only_post_visible(temp_home):
    now = time.time()
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "cron digest landed", "timestamp": now - 30,
         "label": "orchestration-profiler", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None and "cron digest landed" in block
    assert "orchestration-profiler" in block


# B1 REGRESSION: pristine home, log only, NO state.db file at all.
def test_log_only_with_no_state_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Deliberately do NOT create state.db or any session. Only the log exists.
    assert not (home / "state.db").exists()
    now = time.time()
    _write_log(home, "42", "7", [
        {"role": "assistant", "text": "pure bot-api post no session", "timestamp": now - 30,
         "label": "watchdog", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None
    assert "pure bot-api post no session" in block


# (b) dedup across log + session (bot post mirrored into a sibling assistant row)
def test_dedup_across_log_and_session(temp_home):
    now = time.time()
    _seed_session(temp_home, session_id="S1", chat_id="42", thread_id="7",
                  user_id="a", user_name="Alice",
                  messages=[{"role": "assistant", "content": "deploy finished", "timestamp": now - 50}])
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "deploy finished", "timestamp": now - 50,
         "label": "cron", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None
    assert block.count("deploy finished") == 1


def test_dedup_preserves_log_context_text(temp_home):
    now = time.time()
    _seed_session(temp_home, session_id="S1", chat_id="42", thread_id="7",
                  user_id="a", user_name="Alice",
                  messages=[{"role": "assistant", "content": "idea card", "timestamp": now - 50}])
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "idea card", "context_text": "SOURCE PACKET",
         "timestamp": now - 50, "label": "yt-disc-idea", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None
    assert block.count("idea card") == 1
    assert "SOURCE PACKET" in block


def test_dedup_context_timestamp_participates_in_tail_cap(temp_home):
    """A duplicate log row with context must not stay in an old slot and get capped away."""
    now = time.time()
    _seed_session(temp_home, session_id="S1", chat_id="42", thread_id="7",
                  user_id="a", user_name="Alice",
                  messages=[
                      {"role": "assistant", "content": "idea card", "timestamp": now - 100},
                      {"role": "assistant", "content": "middle one", "timestamp": now - 20},
                      {"role": "assistant", "content": "middle two", "timestamp": now - 10},
                  ])
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "idea card", "context_text": "SOURCE PACKET",
         "timestamp": now - 1, "label": "yt-disc-idea", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=2, max_age_hours=24)
    assert block is not None
    assert "SOURCE PACKET" in block
    assert "idea card" in block
    assert "middle two" in block
    assert "middle one" not in block


# (c) cache-prefix stability / single user turn  (d) role alternation
def test_single_user_turn_and_alternation(temp_home):
    now = time.time()
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "bot post one", "timestamp": now - 40,
         "label": "cron", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None
    # Simulate the run.py fold (run.py:8832-8833): block rides ONE user turn,
    # AFTER the sender prefix, never a synthetic assistant/tool message.
    user_text = "[Alice] what happened here?"
    folded = f"{block}\n\n[New message]\n{user_text}"
    history = [{"role": "user", "content": folded}]  # new session => empty history + 1 turn
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert all(h["role"] == "user" for h in history)
    # The block is inert READ-ONLY context text, not a real assistant turn.
    assert "READ-ONLY" in block


# (e) age-prune + cap on the COMBINED set
def test_age_prune_and_cap(temp_home):
    now = time.time()
    posts = [{"role": "assistant", "text": f"p{i}", "timestamp": now - (20 - i),
              "label": "cron", "source": "bot-api"} for i in range(10)]
    posts.append({"role": "assistant", "text": "ancient", "timestamp": now - 99 * 3600,
                  "label": "cron", "source": "bot-api"})
    _write_log(temp_home, "42", "7", posts)
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=2, max_age_hours=24)
    assert block is not None
    assert "p9" in block and "p8" in block
    assert "p0" not in block       # cap to most recent 2
    assert "ancient" not in block  # age-filtered


# combined merge: session + log interleave chronologically
def test_session_and_log_merged_chronologically(temp_home):
    now = time.time()
    _seed_session(temp_home, session_id="S1", chat_id="42", thread_id="7",
                  user_id="a", user_name="Alice",
                  messages=[{"role": "user", "content": "human asked first", "timestamp": now - 100}])
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "bot posted later", "timestamp": now - 50,
         "label": "cron", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24)
    assert block is not None
    assert "human asked first" in block and "bot posted later" in block
    assert block.index("human asked first") < block.index("bot posted later")


# (f) DM-skip + established-session gate (adapter-level invariants)
def test_dm_skipped():
    dm = SessionSource(platform=Platform.TELEGRAM, chat_id="500", chat_type="dm",
                       user_id="7", user_name="Carol")
    assert is_shared_multi_user_session(dm) is False
    shared = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="group",
                           user_id="7", user_name="Carol", thread_id="7")
    assert is_shared_multi_user_session(shared) is True


def test_disabled_flag_skips_log_source(temp_home):
    now = time.time()
    _write_log(temp_home, "42", "7", [
        {"role": "assistant", "text": "should not appear", "timestamp": now - 30,
         "label": "cron", "source": "bot-api"},
    ])
    block = topic_backfill.build_topic_backfill(
        platform="telegram", chat_id="42", thread_id="7",
        exclude_session_id="NEW", max_messages=15, max_age_hours=24,
        include_bot_posts=False)
    assert block is None


# writer concurrency: parallel subprocess appends must not drop posts (fcntl lock)
def test_concurrent_appends_do_not_drop_posts(temp_home):
    now = time.time()
    code = (
        "import sys, os\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "os.environ['HERMES_HOME'] = sys.argv[2]\n"
        "from tg_topic_recent_posts import append_recent_post\n"
        "ok = append_recent_post('42', '7', sys.argv[3], 'worker', timestamp=float(sys.argv[4]))\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    procs = [
        subprocess.run(
            [sys.executable, "-c", code, str(_SCRIPTS_DIR), str(temp_home),
             f"post-{i}", str(now + i)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        for i in range(12)
    ]
    assert [p.returncode for p in procs] == [0] * 12, [p.stderr for p in procs]

    writer = _import_writer()
    rows = writer.read_recent_posts("42", "7")
    texts = {r["text"] for r in rows}
    assert texts == {f"post-{i}" for i in range(12)}
