"""User images survive turns rebuilt from the session DB (``agent.image_store``): stored on flush,
replayed on the wire copy for native-vision requests, and cleaned up with their session."""

import base64
import copy
import hashlib
import os
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import image_store
from agent.session_persistence import _durable_content


def _data_url(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def _user(text, *payloads):
    parts = [{"type": "text", "text": text}]
    parts += [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in payloads]
    return {"role": "user", "content": parts}


def _stored_row(live):
    """What a turn rebuilt from the DB sees: the text-only projection plus the row's message_uid."""
    return {"role": "user", "content": _durable_content(live["content"]), "message_uid": live["message_uid"]}


def _pairs(rows):
    return [(row, copy.deepcopy(row)) for row in rows]


def _sha_name(payload: bytes) -> str:
    return f"{hashlib.sha256(payload).hexdigest()}.png"


def _files(home):
    root = home / "image_store"
    return sorted(p.name for p in root.iterdir() if p.is_file()) if root.is_dir() else []


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def db(home):
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=home / "state.db")
    yield session_db
    session_db.close()


def _add(db, sid, payload, ts):
    """One user turn with one image, written as the agent does: image store first, then the row."""
    live = _user(f"photo {payload!r}", payload)
    image_store.persist_message_images(live, session_id=sid, ts=ts)
    db.ensure_session(sid, "api_server")
    db.append_message(sid, "user", content=_durable_content(live["content"]),
                      message_uid=live["message_uid"], timestamp=ts)
    return live["message_uid"]


def test_replayed_row_is_byte_identical_to_the_live_message(home):
    live = _user("What do you see?", b"img-A")
    assert image_store.persist_message_images(live) == 1
    row = _stored_row(live)
    assert row["content"] == "What do you see?\n[screenshot]"  # the DB row stays text-only
    pairs = _pairs([row])
    assert image_store.rehydrate_wire_images(pairs, keep_recent=3) == 1
    # Same bytes as the turn that first sent it, so the provider's prompt-cache prefix still matches.
    assert pairs[0][1]["content"] == live["content"]
    assert row["content"] == "What do you see?\n[screenshot]"  # the source history is never modified


def test_only_the_most_recent_images_are_replayed_deterministically(home):
    rows = []
    for i in range(5):
        live = _user(f"photo {i}", f"img-{i}".encode())
        image_store.persist_message_images(live)
        rows += [_stored_row(live), {"role": "assistant", "content": f"seen {i}"}]
    pairs = _pairs(rows)
    assert image_store.rehydrate_wire_images(pairs, keep_recent=3) == 3
    users = [api["content"] for src, api in pairs if src["role"] == "user"]
    assert users[:2] == [f"photo {i}\n{image_store.OLDER_IMAGE_TEXT}" for i in (0, 1)]
    assert all([p["type"] for p in content] == ["text", "image_url"] for content in users[2:])
    again = _pairs(rows)
    image_store.rehydrate_wire_images(again, keep_recent=3)
    assert [api for _, api in again] == [api for _, api in pairs]


def test_live_images_count_toward_the_window(home):
    old = _user("older", b"img-old")
    image_store.persist_message_images(old)
    current = _user("newer", b"n1", b"n2", b"n3")
    pairs = [(_stored_row(old), _stored_row(old)), (current, copy.deepcopy(current))]
    assert image_store.rehydrate_wire_images(pairs, keep_recent=3) == 0
    assert pairs[0][1]["content"] == f"older\n{image_store.OLDER_IMAGE_TEXT}"
    assert pairs[1][1]["content"] == current["content"]


def test_marker_without_a_stored_image_reads_as_an_older_image(home):
    # A row stored before this feature, or whose image was pruned: same bytes either way.
    row = {"role": "user", "content": "old message\n[screenshot]", "message_uid": "abc123"}
    pairs = _pairs([row])
    assert image_store.rehydrate_wire_images(pairs, keep_recent=3) == 0
    assert pairs[0][1]["content"] == f"old message\n{image_store.OLDER_IMAGE_TEXT}"


def test_only_stored_bytes_replay_and_each_image_keeps_its_marker(home):
    # A remote URL is re-fetched by the provider on every request (an expired signed URL 400s the
    # turn); an undecodable part stores nothing. Neither may shift later images onto earlier markers.
    live = {"role": "user", "content": [
        {"type": "text", "text": "compare"},
        {"type": "image_url", "image_url": {"url": "https://cdn.example/signed.png?X-Amz-Expires=60"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {"type": "image_url", "image_url": {"url": _data_url(b"third")}},
    ]}
    assert image_store.persist_message_images(live) == 1
    pairs = _pairs([_stored_row(live)])
    assert image_store.rehydrate_wire_images(pairs, keep_recent=3) == 1
    older = {"type": "text", "text": image_store.OLDER_IMAGE_TEXT}
    assert pairs[0][1]["content"] == [live["content"][0], older, older, live["content"][3]]


def test_a_late_flush_never_recreates_a_deleted_profile(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles" / ".deleted").mkdir(parents=True)
    (root / "profiles" / ".deleted" / "gone").touch()  # what `hermes profile delete` leaves behind
    gone = root / "profiles" / "gone"
    monkeypatch.setenv("HERMES_HOME", str(gone))
    assert image_store.persist_message_images(_user("late", b"img"), session_id="S", keep=3) == 0
    assert not gone.exists()


def test_each_session_keeps_only_its_newest_images(home, db):
    uids = [_add(db, "S1", f"s1-{i}".encode(), 1000 + i) for i in range(5)]
    other = _add(db, "S2", b"s2-0", 900)  # older than all of S1, other session: untouched
    assert [bool(image_store.load_refs(u)) for u in uids] == [False, False, True, True, True]
    assert image_store.load_refs(other)
    assert _files(home) == sorted([_sha_name(f"s1-{i}".encode()) for i in (2, 3, 4)] + [_sha_name(b"s2-0")])
    # A late re-flush of an older message never evicts newer images.
    late = _user("late", b"late")
    image_store.persist_message_images(late, session_id="S1", ts=1)
    assert not image_store.load_refs(late["message_uid"]) and _sha_name(b"late") not in _files(home)


def test_a_shared_file_survives_until_its_last_reference(home, db):
    _add(db, "S1", b"shared", 1000)
    _add(db, "S2", b"shared", 1001)
    for i in range(3):  # pushes S1's copy out of S1's window
        _add(db, "S1", f"s1-{i}".encode(), 1100 + i)
    assert _sha_name(b"shared") in _files(home)  # still referenced by S2
    assert db.delete_session("S2", sessions_dir=home / "sessions")
    assert _sha_name(b"shared") not in _files(home)


def test_deleting_or_pruning_sessions_removes_their_images(home, db):
    keep = _add(db, "KEEP", b"keep", 1000)  # never ended: prune leaves it alone
    gone = [_add(db, "GONE", f"g{i}".encode(), 1001 + i) for i in range(2)]
    bulk = [_add(db, "A", b"a", 1003), _add(db, "B", b"b", 1004)]
    pruned = _add(db, "OLD", b"old", 1005)
    assert db.delete_session("GONE", sessions_dir=home / "sessions")
    assert db.delete_sessions(["A", "B"], sessions_dir=home / "sessions") == 2
    db.end_session("OLD", "user_exit")
    assert db.prune_sessions(older_than_days=0) == 1
    assert not any(image_store.load_refs(u) for u in [*gone, *bulk, pruned])
    assert image_store.load_refs(keep) and _files(home) == [_sha_name(b"keep")]


def test_sweep_drops_images_of_missing_or_compacted_messages(home, db):
    live = _add(db, "S", b"live", 1000)
    compacted = _add(db, "S", b"compacted", 999)
    conn = sqlite3.connect(home / "state.db")
    conn.execute("UPDATE messages SET active = 0 WHERE message_uid = ?", (compacted,))
    conn.commit()
    conn.close()
    never = _user("never committed", b"never")  # stored, then the process died before the insert
    image_store.persist_message_images(never, session_id="S", ts=998)
    root = home / "image_store"
    (root / ("0" * 64 + ".jpg")).write_bytes(b"orphan file")
    stale_tmp = root / ".x.jpg.1.tmp"
    stale_tmp.write_bytes(b"half written")
    old = time.time() - 7200
    os.utime(stale_tmp, (old, old))
    stats = image_store.gc_store(db)
    assert stats["refs_removed"] == 2 and stats["files_removed"] == 4
    assert image_store.load_refs(live) and _files(home) == [_sha_name(b"live")]


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _agent():
    return SimpleNamespace(provider="openrouter", model="test/model")


def test_replay_follows_the_image_input_mode(home):
    live = _user("look", b"img")
    image_store.persist_message_images(live)
    _write_config(home, "agent:\n  image_input_mode: text\n")
    text_pairs = _pairs([_stored_row(live)])
    image_store.rehydrate_for_agent(_agent(), text_pairs)
    assert text_pairs[0][1]["content"] == "look\n[screenshot]"  # untouched: pixels never go out in text mode
    _write_config(home, "agent:\n  image_input_mode: native\n")
    native_pairs = _pairs([_stored_row(live)])
    image_store.rehydrate_for_agent(_agent(), native_pairs)
    assert native_pairs[0][1]["content"] == live["content"]


def test_replay_recent_images_zero_stores_replays_and_keeps_nothing(home, db):
    stored = _add(db, "S", b"before", 1000)  # stored while the feature was on
    _write_config(home, "agent:\n  image_input_mode: native\nvision:\n  replay_recent_images: 0\n")
    live = _user("look", b"img")
    assert image_store.persist_message_images(live, session_id="S") == 0
    pairs = _pairs([{"role": "user", "content": "photo\n[screenshot]", "message_uid": stored}])
    image_store.rehydrate_for_agent(_agent(), pairs)
    assert pairs[0][1]["content"] == "photo\n[screenshot]"  # exactly the behaviour without the store
    image_store.gc_store(db)
    assert _files(home) == [] and not image_store.load_refs(stored)


def test_replay_recent_images_setting_sizes_the_window(home):
    _write_config(home, "agent:\n  image_input_mode: native\nvision:\n  replay_recent_images: 1\n")
    rows = []
    for i in range(2):
        live = _user(f"photo {i}", f"img-{i}".encode())
        image_store.persist_message_images(live)
        rows.append(_stored_row(live))
    pairs = _pairs(rows)
    image_store.rehydrate_for_agent(_agent(), pairs)
    assert pairs[0][1]["content"] == f"photo 0\n{image_store.OLDER_IMAGE_TEXT}"
    assert [p["type"] for p in pairs[1][1]["content"]] == ["text", "image_url"]


def _reply(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model")
    response.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    return response


def _fresh_agent(db):
    """A new agent per turn over the same session, as the API server builds one per request."""
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id="sid", skip_context_files=True, skip_memory=True)
    agent.client, agent.tool_delay, agent.save_trajectories = MagicMock(), 0, False
    agent.client.chat.completions.create.side_effect = [_reply("ok")]
    return agent


def test_a_turn_rebuilt_from_the_db_still_shows_the_earlier_image(home, db, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _write_config(home, "model:\n  supports_vision: true\n")
    photo = _user("What colour is this car?", b"car-pixels")
    first = _fresh_agent(db)
    first.run_conversation(user_message=photo["content"], conversation_history=[], task_id="sid")
    sent_first = first.client.chat.completions.create.call_args.kwargs["messages"]

    second = _fresh_agent(db)
    history = db.get_messages_as_conversation("sid", repair_alternation=True)
    assert history[0]["content"] == "What colour is this car?\n[screenshot]"  # the DB stays text-only
    second.run_conversation(user_message="And its brand?", conversation_history=history, task_id="sid")
    sent_second = second.client.chat.completions.create.call_args.kwargs["messages"]

    first_user = next(m for m in sent_first if m["role"] == "user")
    replayed = next(m for m in sent_second if m["role"] == "user")
    assert [p["type"] for p in first_user["content"]] == ["text", "image_url"]
    assert replayed["content"] == first_user["content"]  # same pixels, same bytes: the cached prefix holds
