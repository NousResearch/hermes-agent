"""Image-bearing history must replay the exact wire prefix after a SQLite reload."""

import base64
import json
import os
from types import SimpleNamespace

import pytest

from agent.session_persistence import _db_flush_row
from agent.turn_context import build_api_messages
from hermes_state import SessionDB
from tools.vision_tools import _build_native_vision_tool_result

IMAGE = b"a durable image payload"
DATA_URL = "data:image/png;base64," + base64.b64encode(IMAGE).decode()


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session("vision", "cli")
    yield store
    store.close()


def _history(kind):
    envelope = _build_native_vision_tool_result("shot.png", "Read it", DATA_URL, len(IMAGE))
    if kind == "user":
        return [{"role": "user", "content": envelope["content"]}]
    return [
        {"role": "user", "content": "Read the screenshot"},
        {"role": "assistant", "content": "Looking", "tool_calls": [
            {"id": "vision_call", "type": "function", "function": {
                "name": "vision_analyze", "arguments": '{"image_url":"shot.png"}'}}]},
        {"role": "tool", "content": envelope if kind == "envelope" else envelope["content"],
         "tool_call_id": "vision_call"},
    ]


def _wire(messages, current):
    agent = SimpleNamespace(
        _current_turn_timestamp=1000, ephemeral_system_prompt=None,
        _copy_reasoning_content_for_api=lambda msg, api: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    from agent.agent_runtime_helpers import sanitize_api_messages

    api_messages = build_api_messages(
        agent, messages, current_turn_user_idx=current, ext_prefetch_cache=None,
        plugin_user_context=None, moa_config=None, active_system_prompt="Stable system",
    )[0]
    from agent.transports.chat_completions import _sanitize_message

    return json.dumps([_sanitize_message(m, True) or m for m in sanitize_api_messages(api_messages)],
                      ensure_ascii=False)


def _flush(db, messages):
    agent = SimpleNamespace(_session_db=db)
    db.append_messages_batch("vision", [_db_flush_row(agent, msg, False) for msg in messages])


@pytest.mark.parametrize("kind", ["envelope", "tool_parts", "user"])
def test_next_turn_wire_prefix_survives_reload(db, kind):
    live = _history(kind)
    before = _wire(live, 0)
    _flush(db, live)
    reloaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    # Historical prefix on the next turn, not just a structural content comparison.
    next_turn = [{"role": "assistant", "content": "Done"}, {"role": "user", "content": "Continue"}]
    after = json.loads(_wire(reloaded + next_turn, len(reloaded) + 1))[:-2]
    assert json.dumps(after, ensure_ascii=False) == before
    assert json.dumps(reloaded[-1]["content"]) == json.dumps(live[-1]["content"])


@pytest.mark.parametrize("kind", ["envelope", "user"])
def test_reference_store_is_deduplicated_and_text_search_stays_clean(db, tmp_path, kind):
    from agent.session_persistence import _durable_content

    live = _history(kind)
    _flush(db, live)
    files = list((tmp_path / "cache/transcript_media").iterdir())
    assert [p.read_bytes() for p in files] == [IMAGE]
    import hashlib
    assert files[0].name == hashlib.sha256(IMAGE).hexdigest()
    import os
    os.utime(files[0], (1, 1))
    original_stat = files[0].stat()
    db.create_session("copy", "cli")
    db.append_messages_batch("copy", live)
    # A dedupe hit renews the grace window until its new DB reference commits.
    assert files[0].stat().st_mtime_ns > original_stat.st_mtime_ns
    # Every SQLite column, not just content, must be free of the image payload.
    dump = "\n".join(db._conn.iterdump())
    assert base64.b64encode(IMAGE).decode() not in dump
    assert db.search_messages("Image")
    assert not db.search_messages(base64.b64encode(IMAGE).decode())
    assert db.get_messages("vision")[-1]["content"] == _durable_content(live[-1]["content"])
    assert db.get_messages_as_conversation("vision")[-1]["content"] == _durable_content(live[-1]["content"])


@pytest.mark.parametrize("failure", ["missing", "unreadable", "corrupt"])
def test_unavailable_media_falls_back_to_text(db, tmp_path, monkeypatch, caplog, failure):
    from agent.session_persistence import _durable_content

    live = _history("envelope")
    _flush(db, live)
    path, = (tmp_path / "cache/transcript_media").iterdir()
    if failure == "missing":
        path.rename(tmp_path / "removed-image")
    elif failure == "corrupt":
        path.write_bytes(b"damaged")
    else:
        real_open = os.open
        def denied(name, flags, *args, **kwargs):
            if os.path.basename(name) == path.name:
                raise PermissionError("unreadable")
            return real_open(name, flags, *args, **kwargs)
        monkeypatch.setattr(os, "open", denied)
    with caplog.at_level("DEBUG", logger="hermes_state"):
        restored = db.get_messages_as_conversation("vision", repair_alternation=True)
    assert restored[-1]["content"] == _durable_content(live[-1]["content"])
    assert "Cannot restore transcript media" in caplog.text


def test_gateway_sweeps_only_transient_media(db, tmp_path, monkeypatch):
    import os
    import time
    from gateway.platforms import base

    _flush(db, _history("user"))
    path, = (tmp_path / "cache/transcript_media").iterdir()
    old = time.time() - 25 * 3600
    os.utime(path, (old, old))
    for kind in ("image", "video"):
        cache = tmp_path / "cache" / (kind + "s")
        cache.mkdir()
        transient = cache / "old.bin"
        transient.write_bytes(IMAGE)
        os.utime(transient, (old, old))
        monkeypatch.setattr(base, kind.upper() + "_CACHE_DIR", cache)
        assert getattr(base, "cleanup_" + kind + "_cache")(24) == 1
    assert path.read_bytes() == IMAGE
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == _history("user")[0]["content"]


@pytest.mark.parametrize("kind", ["envelope", "tool_parts", "user"])
@pytest.mark.parametrize("writer", ["flush", "compact"])
def test_stripped_images_never_rehydrate(db, kind, writer):
    from agent.context_compressor import _strip_historical_media

    live = _history(kind)
    _flush(db, live)
    restored = db.get_messages_as_conversation("vision", repair_alternation=True)
    newest = _history("tool_parts")[1:]
    newest[0]["tool_calls"][0]["id"] = "new_call"
    newest[1]["tool_call_id"] = "new_call"
    stripped = _strip_historical_media(restored + newest)
    target = len(restored) - 1
    assert stripped[target]["content"] != live[target]["content"]
    if writer == "flush":
        _flush(db, stripped)
    else:
        db.archive_and_compact("vision", stripped)
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    assert DATA_URL not in json.dumps(loaded[target]["content"])
    row = db.get_messages("vision")[target]
    assert row["media_content"] is None


def test_submit_row_rewrite_preserves_parts_then_clears_references(db):
    row_id = db.append_message("vision", "user", "raw submit")
    content = _history("user")[0]["content"]
    assert db.set_user_message_content("vision", row_id, content) == 1
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content
    assert db.set_user_message_content("vision", row_id, "stripped") == 1
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == "stripped"


@pytest.mark.parametrize("url", [DATA_URL + "\n", "data:image/png;base64,x"])
def test_noncanonical_data_url_replays_verbatim(db, url):
    live = _history("user")
    live[0]["content"][1]["image_url"]["url"] = url
    _flush(db, live)
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    assert _wire(loaded, None) == _wire(live, None)


def test_reopen_and_foreign_profile_scope_keep_the_same_media(db, tmp_path, monkeypatch):
    live = _history("envelope")
    _flush(db, live)
    db.close()
    # The DB owns the profile, not the ambient scope of a deferred load.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
    with SessionDB(db_path=tmp_path / "state.db") as reopened:
        from gateway.session import SessionStore
        store = SessionStore.__new__(SessionStore)
        store._db_for_session_id = lambda sid: reopened
        loaded = store.load_transcript("vision")
        assert _wire(loaded, None) == _wire(live, None)
    assert not (tmp_path / "other-profile/cache/transcript_media").exists()


def test_text_only_provider_does_not_gain_image_references(db, tmp_path):
    from agent.vision_message_prep import VisionMessagePrepMixin

    agent = VisionMessagePrepMixin()
    agent._model_supports_vision = lambda: True
    agent._provider_supports_vision_tool_messages = lambda: False
    live = _history("envelope")
    live[-1]["content"] = agent._tool_result_content_for_active_model("vision_analyze", live[-1]["content"])
    _flush(db, live)
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    assert _wire(loaded, None) == _wire(live, None)
    assert not (tmp_path / "cache/transcript_media").exists()


def test_atomic_deduplicated_writes(db, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_state_media import prepare_media_content, restore_media_content

    content = _history("user")[0]["content"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: prepare_media_content(db.db_path, content), range(12)))
    assert all(restore_media_content(db.db_path, refs, text) == content for text, refs in rows)
    assert [p.read_bytes() for p in (tmp_path / "cache/transcript_media").iterdir()] == [IMAGE]


def test_anthropic_user_image_source_stays_out_of_sqlite(db):
    content = [{"type": "text", "text": "Read this"}, {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": base64.b64encode(IMAGE).decode()}}]
    _flush(db, [{"role": "user", "content": content}])
    assert base64.b64encode(IMAGE).decode() not in "\n".join(db._conn.iterdump())
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content


def test_crash_persist_and_context_backfill_keep_wire_bytes(db):
    from agent.session_persistence import SessionPersistenceMixin
    from agent.turn_context import _append_multimodal_context, _persist_turn_start

    agent = SessionPersistenceMixin()
    agent._session_db = db
    agent.session_id = "vision"
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent._ensure_db_session = lambda: None
    live = _history("user")
    _persist_turn_start(agent, live, [], None)
    _append_multimodal_context(agent, live[0], "Recall note", "Plugin note", preflight_compressed=False)
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    assert _wire(live, 0) == _wire(loaded, None)


def _age_media(tmp_path):
    import os
    paths = list((tmp_path / "cache/transcript_media").iterdir())
    for path in paths:
        os.utime(path, (1, 1), follow_symlinks=False)
    return paths


@pytest.mark.parametrize("operation", ["delete", "bulk", "moved", "prune", "clear", "replace", "rewrite",
                                       "compact", "rewind", "archive_dropped", "deactivate"])
def test_retired_media_is_reclaimed(db, tmp_path, operation):
    _flush(db, _history("user"))
    row_id = db.get_active_message_ids("vision")[0]
    paths = _age_media(tmp_path)
    actions = {
        "delete": lambda: db.delete_session("vision"),
        "bulk": lambda: db.delete_sessions(["vision"]),
        "moved": lambda: db.delete_moved_session("vision"),
        "prune": lambda: (db.end_session("vision", "done"), db.prune_sessions(older_than_days=None)),
        "clear": lambda: db.clear_messages("vision"),
        "replace": lambda: db.replace_messages("vision", []),
        "rewrite": lambda: db.set_user_message_content("vision", row_id, "text only"),
        "compact": lambda: db.archive_and_compact("vision", [{"role": "user", "content": "summary"}]),
        "rewind": lambda: db.rewind_to_message("vision", row_id),
        "archive_dropped": lambda: db.replace_messages("vision", [], archive_dropped=True),
        "deactivate": lambda: db.deactivate_message("vision", row_id),
    }
    actions[operation]()
    assert all(not path.exists() for path in paths)
    assert all(row["media_content"] is None for row in db.get_messages("vision", include_inactive=True))


def test_shared_media_lives_until_last_session_is_deleted(db, tmp_path):
    _flush(db, _history("user"))
    db.create_session("copy", "cli")
    db.append_messages_batch("copy", _history("user"))
    paths = _age_media(tmp_path)
    db.delete_session("vision")
    assert all(path.exists() for path in paths)
    assert db.get_messages_as_conversation("copy", repair_alternation=True)[0]["content"] == _history("user")[0]["content"]
    db.delete_session("copy")
    assert all(not path.exists() for path in paths)


def test_gc_grace_and_crash_temps(db, tmp_path):
    from hermes_state_media import prepare_media_content, sweep_transcript_media

    prepare_media_content(db.db_path, _history("user")[0]["content"])
    root = tmp_path / "cache/transcript_media"
    temp = root / ".write-crashed"
    temp.write_bytes(b"partial")
    paths = list(root.iterdir())
    sweep_transcript_media(db)
    assert all(path.exists() for path in paths)
    _age_media(tmp_path)
    sweep_transcript_media(db)
    assert all(not path.exists() for path in paths)


@pytest.mark.platforms("posix")
def test_gc_never_touches_symlinks_or_unowned_names(db, tmp_path):
    from hermes_state_media import sweep_transcript_media

    root = tmp_path / "cache/transcript_media"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    (root / ("a" * 64)).symlink_to(outside)
    (root / ".write-link").symlink_to(outside)
    (root / "notes").write_bytes(b"keep")
    (root / ("b" * 64)).mkdir()
    paths = _age_media(tmp_path)
    sweep_transcript_media(db)
    assert all(path.exists() for path in paths)
    assert outside.read_bytes() == b"keep"


def test_gc_failure_does_not_fail_delete(db, tmp_path, monkeypatch, caplog):
    import hermes_state_media

    _flush(db, _history("user"))
    _age_media(tmp_path)
    def fail(_db):
        raise OSError("sweep unavailable")
    monkeypatch.setattr(hermes_state_media, "sweep_transcript_media", fail)
    with caplog.at_level("DEBUG", logger="hermes_state"):
        assert db.delete_session("vision")
    assert db.get_session("vision") is None
    assert "sweep unavailable" in caplog.text


def test_startup_maintenance_reclaims_crash_orphans(db, tmp_path):
    from hermes_state_media import prepare_media_content

    prepare_media_content(db.db_path, _history("user")[0]["content"])
    paths = _age_media(tmp_path)
    db.maybe_auto_prune_and_vacuum(vacuum=False)
    assert all(not path.exists() for path in paths)


def test_compaction_clone_keeps_live_image_only(db, tmp_path):
    db.append_message("vision", "user", "summarize")
    watermark = db.get_active_message_watermark("vision")
    _flush(db, _history("user"))
    _age_media(tmp_path)
    db.archive_and_compact("vision", [{"role": "user", "content": "summary"}], watermark=watermark)
    rows = db.get_messages("vision", include_inactive=True)
    assert all(row["media_content"] is None for row in rows if not row["active"])
    assert DATA_URL in json.dumps(db.get_messages_as_conversation("vision", repair_alternation=True))


@pytest.mark.parametrize("reuse", [False, True])
def test_other_connection_sweep_preserves_uncommitted_media(db, tmp_path, reuse):
    from hermes_state_media import prepare_media_content, sweep_transcript_media

    content = _history("user")[0]["content"]
    if reuse:
        prepare_media_content(db.db_path, content)
        _age_media(tmp_path)
    text, sidecar = prepare_media_content(db.db_path, content, display=False)
    # A second profile/process sees no row yet; the publication grace protects it.
    with SessionDB(db_path=db.db_path) as other:
        sweep_transcript_media(other)
    db.append_messages_batch("vision", [{"role": "user", "content": text, "media_content": sidecar}])
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content


def test_encoded_media_is_referenced_and_reclaimed(db, tmp_path):
    content = _history("user")[0]["content"]
    content[1]["image_url"]["url"] += "\n"
    _flush(db, [{"role": "user", "content": content}])
    paths = _age_media(tmp_path)
    from hermes_state_media import sweep_transcript_media
    sweep_transcript_media(db)
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    db.delete_session("vision")
    assert all(not path.exists() for path in paths)


def test_rewind_prefill_materializes_before_gc(db, tmp_path):
    content = _history("user")[0]["content"]
    db.append_messages_batch("vision", [{"role": "user", "content": content}])
    row_id = db.get_active_message_ids("vision")[0]
    paths = _age_media(tmp_path)
    result = db.rewind_to_message("vision", row_id)
    assert result["target_message"]["content"] == content
    assert all(not path.exists() for path in paths)


def test_raw_reference_reuse_renews_publication_grace(db, tmp_path):
    from hermes_state_media import prepare_media_content, sweep_transcript_media

    content = _history("user")[0]["content"]
    text, sidecar = prepare_media_content(db.db_path, content)
    paths = _age_media(tmp_path)
    text, sidecar = prepare_media_content(db.db_path, text, sidecar)
    sweep_transcript_media(db)
    assert all(path.exists() for path in paths)
    db.append_messages_batch("vision", [{"role": "user", "content": text, "media_content": sidecar}])
    assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content


def test_profile_move_preserves_images_before_source_cleanup(db, tmp_path):
    content = _history("user")[0]["content"]
    _flush(db, [{"role": "user", "content": content}])
    paths = _age_media(tmp_path)
    payload = db.export_session_for_move("vision")
    with SessionDB(db_path=tmp_path / "target/state.db") as target:
        assert target.import_moved_session(payload, profile_name="target") == "imported"
        assert db.delete_moved_session("vision")
        assert all(not path.exists() for path in paths)
        assert target.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content
        target_paths = _age_media(tmp_path / "target")
        target.delete_session("vision")
        assert all(not path.exists() for path in target_paths)


@pytest.mark.platforms("posix")
def test_gc_skips_a_symlinked_media_directory(db, tmp_path):
    from hermes_state_media import sweep_transcript_media

    outside = tmp_path / "outside"
    outside.mkdir()
    path = outside / ("a" * 64)
    path.write_bytes(b"keep")
    import os
    os.utime(path, (1, 1))
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "transcript_media").symlink_to(outside, target_is_directory=True)
    assert sweep_transcript_media(db) == 0
    assert path.read_bytes() == b"keep"


def test_rollback_keeps_committed_media_and_later_reclaims_orphans(db, tmp_path):
    from hermes_state_media import prepare_media_content, sweep_transcript_media

    _flush(db, _history("user"))
    paths = _age_media(tmp_path)
    def fail(conn):
        conn.execute("DELETE FROM messages WHERE session_id = 'vision'")
        raise RuntimeError("rollback")
    with pytest.raises(RuntimeError, match="rollback"):
        db._execute_write(fail, sweep_media=True)
    sweep_transcript_media(db)
    assert all(path.exists() for path in paths)
    assert DATA_URL in json.dumps(db.get_messages_as_conversation("vision", repair_alternation=True))
    content = _history("user")[0]["content"]
    content[1]["image_url"]["url"] += "\n"
    prepare_media_content(db.db_path, content)
    all_paths = _age_media(tmp_path)
    assert len(all_paths) == 2
    sweep_transcript_media(db)
    assert [path for path in all_paths if path.exists()] == paths


def test_existing_database_gains_nullable_media_column(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "old.db"
    with SessionDB(db_path=path) as store:
        store.create_session("vision", "cli")
        store.append_message("vision", "user", "old text")
        store._conn.execute("DROP INDEX idx_messages_media")
        store._conn.execute("ALTER TABLE messages DROP COLUMN media_content")
    with SessionDB(db_path=path) as store:
        assert store.get_messages_as_conversation("vision")[0]["content"] == "old text"
        store.set_user_message_content("vision", store.get_messages("vision")[0]["id"], _history("user")[0]["content"])
        assert store.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == _history("user")[0]["content"]


def test_media_bookkeeping_never_scans_the_messages_table(tmp_path, monkeypatch):
    """Both per-write statements must hit the partial index: on a 1.4M-row state.db the scan form of
    each costs ~30 s, inside the write lock for the retired-row clear. The index is created through
    DEFERRED_INDEX_SQL, so a legacy store that just gained the column must get it too."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "old.db"
    with SessionDB(db_path=path) as store:
        store._conn.execute("DROP INDEX idx_messages_media")
        store._conn.execute("ALTER TABLE messages DROP COLUMN media_content")
    with SessionDB(db_path=path) as store:
        for sql in ("UPDATE messages SET media_content = NULL WHERE active = 0 AND media_content IS NOT NULL",
                    "SELECT media_content FROM messages WHERE media_content IS NOT NULL"):
            plan = " ".join(row[3] for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql))
            assert "idx_messages_media" in plan, plan


@pytest.mark.parametrize("kind", ["envelope", "tool_parts", "user"])
def test_rewind_from_reloaded_view_keeps_image_prefix_rows(db, kind):
    """A rewind issued from the rehydrated view must recognise image-bearing rows as the kept prefix (#82956)."""
    live = _history(kind) + [{"role": "assistant", "content": "Done"}, {"role": "user", "content": "Again"}]
    _flush(db, live)
    ids = [m["id"] for m in db.get_messages("vision")]
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    db.replace_messages("vision", [dict(m) for m in loaded[:-1]], active_only=True, archive_dropped=True)
    assert [m["id"] for m in db.get_messages("vision")] == ids[:-1]
    assert len([m for m in db.get_messages("vision", include_inactive=True) if not m["active"]]) == 1


def test_stale_writer_never_resurrects_a_deleted_profile(tmp_path, monkeypatch):
    """A writer still holding a deleted named profile's db_path must not recreate its home."""
    from hermes_state_media import prepare_media_content

    root = tmp_path / ".hermes"
    home = root / "profiles" / "work"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    db_path = home / "state.db"
    (root / "profiles" / ".deleted").mkdir()
    home.rename(root / "profiles" / ".deleted" / "work")
    content = _history("user")[0]["content"]
    assert prepare_media_content(db_path, content) == (prepare_media_content(db_path, content)[0], None)
    assert not home.exists()


@pytest.mark.platforms("posix")
def test_symlinked_media_store_is_refused_for_write_and_read(db, tmp_path):
    from hermes_state_media import prepare_media_content

    content = _history("user")[0]["content"]
    _flush(db, [{"role": "user", "content": content}])
    store = tmp_path / "cache" / "transcript_media"
    outside = tmp_path / "outside"
    store.rename(outside)
    store.symlink_to(outside, target_is_directory=True)
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"]
    assert DATA_URL not in json.dumps(loaded)  # read refused: text projection, not the outside file
    before = sorted(p.name for p in outside.iterdir())
    other = _history("user")[0]["content"]
    other[1]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(b"other").decode()
    assert prepare_media_content(db.db_path, other)[1] is None
    assert sorted(p.name for p in outside.iterdir()) == before


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_profile_move_refuses_to_flatten_unavailable_media(db, tmp_path, failure):
    content = _history("user")[0]["content"]
    _flush(db, [{"role": "user", "content": content}])
    path, = (tmp_path / "cache/transcript_media").iterdir()
    path.unlink() if failure == "missing" else path.write_bytes(b"damaged")
    payload = db.export_session_for_move("vision")
    with SessionDB(db_path=tmp_path / "target/state.db") as target:
        with pytest.raises(OSError, match="unavailable in the source store"):
            target.import_moved_session(payload, profile_name="target")
        assert target.get_session("vision") is None
    assert db.get_messages("vision")[0]["media_content"]


def test_media_file_io_runs_outside_the_write_transaction(db, monkeypatch):
    """Image reads/hashes/writes never hold BEGIN IMMEDIATE, so a slow store cannot block every writer."""
    import hermes_state_media

    in_write = []
    original_write = db._execute_write
    def tracked_write(fn, *args, **kwargs):
        def run(conn):
            in_write.append(True)
            try:
                return fn(conn)
            finally:
                in_write.pop()
        return original_write(run, *args, **kwargs)
    monkeypatch.setattr(db, "_execute_write", tracked_write)
    calls = []
    for name in ("_get", "_put"):
        real = getattr(hermes_state_media, name)
        def guarded(*args, _real=real, _name=name):
            calls.append(_name)
            assert not in_write, f"{_name} ran inside the SQLite write transaction"
            return _real(*args)
        monkeypatch.setattr(hermes_state_media, name, guarded)

    live = _history("envelope") + [{"role": "assistant", "content": "Done"}, {"role": "user", "content": "Again"}]
    db.append_messages_batch("vision", [dict(m) for m in live])
    loaded = db.get_messages_as_conversation("vision", repair_alternation=True)
    db.replace_messages("vision", [dict(m) for m in loaded[:-1]], active_only=True, archive_dropped=True)
    db.append_messages_batch("vision", [{"role": "user", "content": _history("user")[0]["content"]}])
    db.rewind_to_message("vision", db.get_active_message_ids("vision")[-1])
    payload = db.export_session_for_move("vision")
    db.create_session("copy", "cli")
    db.append_messages_batch("copy", [dict(m) for m in loaded])
    target = SessionDB(db_path=db.db_path.parent / "target/state.db")
    try:
        target.import_moved_session(payload, profile_name="target")
    finally:
        target.close()
    assert "_get" in calls and "_put" in calls


@pytest.mark.platforms("posix")
def test_first_publication_fsyncs_the_store_directory(db, tmp_path, monkeypatch):
    import stat
    from hermes_state_media import prepare_media_content

    real_fsync, synced = os.fsync, []
    def spy(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", spy)
    prepare_media_content(db.db_path, _history("user")[0]["content"])
    assert synced == [False, True]  # the image file, then its directory entry


@pytest.mark.platforms("posix")
def test_symlinked_cache_never_creates_the_store_outside(db, tmp_path):
    from hermes_state_media import prepare_media_content

    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "cache"
    if cache.exists():
        cache.rename(tmp_path / "old-cache")
    cache.symlink_to(outside, target_is_directory=True)
    assert prepare_media_content(db.db_path, _history("user")[0]["content"])[1] is None
    assert list(outside.iterdir()) == []


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", ["write", "read"])
def test_store_swapped_after_validation_is_not_followed(db, tmp_path, monkeypatch, op):
    """A directory swapped for a symlink between the store check and the file I/O must not redirect it."""
    import hermes_state_media
    from hermes_state_media import prepare_media_content

    content = _history("user")[0]["content"]
    _flush(db, [{"role": "user", "content": content}])
    store = tmp_path / "cache" / "transcript_media"
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in store.iterdir():  # the outside copy a redirected read would return
        (outside / path.name).write_bytes(path.read_bytes())
    real_enter = hermes_state_media._open_store.__enter__
    def swap_then_enter(self):
        store.rename(tmp_path / "moved")
        store.symlink_to(outside, target_is_directory=True)
        return real_enter(self)
    monkeypatch.setattr(hermes_state_media._open_store, "__enter__", swap_then_enter)
    if op == "write":
        other = _history("user")[0]["content"]
        other[1]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(b"other").decode()
        before = sorted(p.name for p in outside.iterdir())
        prepare_media_content(db.db_path, other)
        assert sorted(p.name for p in outside.iterdir()) == before
        assert any((tmp_path / "moved").iterdir())
    else:
        real_enter_once = swap_then_enter
        def swap_and_hide(self):
            result = real_enter_once(self)
            for path in (tmp_path / "moved").iterdir():  # only the outside copy remains readable by path
                path.unlink()
            return result
        monkeypatch.setattr(hermes_state_media._open_store, "__enter__", swap_and_hide)
        loaded = db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"]
        assert DATA_URL not in json.dumps(loaded)
