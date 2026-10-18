"""Retained-media collection stays off the owner loop, costs messages + candidates, and never fails a
committed delete (#140297)."""
import asyncio
import base64
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from gateway.session_api_turn import admit_api_turn, recover_api_turns
from gateway.session_ingress_media import _media_root, release_unheld_media
from hermes_state_runtime import claim_session_input, settle_session_input


def _plant(count, start=0):
    """Retained API image files no admission holds (their chats were deleted)."""
    planted = []
    for index in range(start, start + count):
        digest = f'{index:064x}'
        path = _media_root() / digest / f'api_{digest[:32]}.png'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'img' + index.to_bytes(4, 'big'))
        planted.append({'path': str(path), 'sha256': digest})
    return planted


def _image_turn(api, owner, session_id, request_id='turn', images=1):
    """A completed API turn with images: its admission holds them as history context."""
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(
        b'\x89PNG\r\n\x1a\n' + os.urandom(64)).decode()}} for _ in range(images)]
    _, _, row = admit_api_turn(api, session_id=session_id, request_id=request_id, conversation_history=[],
                               user_message=content)
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=session_id)
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')
    return [Path(reference['path']) for reference in row['payload']['api_turn_v1']['media']]


@contextmanager
def _on_every_connection(db, monkeypatch, install):
    """Run ``install(conn)`` on the writer and on every read connection checked out meanwhile."""
    original = db._read_ctx

    @contextmanager
    def instrumented(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            install(conn)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
                conn.set_progress_handler(None, 0)
    monkeypatch.setattr(db, '_read_ctx', instrumented)
    install(db._conn)
    try:
        yield
    finally:
        db._conn.set_trace_callback(None)
        db._conn.set_progress_handler(None, 0)
        monkeypatch.setattr(db, '_read_ctx', original)


@pytest.mark.asyncio
async def test_startup_image_collection_runs_off_the_owner_loop(api, owner, monkeypatch):
    """Boot, reconnect and hot-serve publish the API adapter on the owner loop: the transcript scan
    behind image collection must not run there, or every session of the gateway freezes meanwhile."""
    import gateway.session_ingress_media as media
    admit_api_turn(api, session_id='queued', request_id='queued-turn', user_message='pending',
                   conversation_history=[])
    scheduled, threads, done = [], [], threading.Event()
    monkeypatch.setattr(owner, '_schedule', lambda ref: scheduled.append(ref.session_id))
    collect = media.collect_unheld_api_images

    def recorded(db):
        threads.append(threading.get_ident())
        try:
            return collect(db)
        finally:
            done.set()
    monkeypatch.setattr(media, 'collect_unheld_api_images', recorded)
    orphan = Path(_plant(1)[0]['path'])
    pending = recover_api_turns(api)
    assert scheduled == ['queued']
    assert await asyncio.to_thread(done.wait, 10)
    assert threads and threads[0] != threading.get_ident(), 'image collection ran on the owner loop'
    await asyncio.gather(*pending)
    assert not orphan.exists()


@pytest.mark.asyncio
async def test_retirement_keeps_ownership_while_a_collection_can_still_unlink(api, owner, monkeypatch):
    """The off-loop collection unlinks files: until its thread returns, retirement must not release
    the profile (a successor's in-flight capture of the same bytes is invisible to this process) and
    shutdown must count it, even once the task awaiting it is cancelled."""
    import gateway.session_ingress_media as media
    from gateway.run_runtime import _retire_profile_authority, managed_turn_count
    orphan = Path(_plant(1)[0]['path'])
    loop_thread, paused, resume = threading.get_ident(), threading.Event(), threading.Event()
    scan = media._history_mentions

    def held(*args, **kwargs):
        paused.set()
        if threading.get_ident() != loop_thread:  # a synchronous collection would block the test
            resume.wait(30)
        return scan(*args, **kwargs)
    monkeypatch.setattr(media, '_history_mentions', held)
    pending = recover_api_turns(api) or []
    try:
        assert await asyncio.to_thread(paused.wait, 10)
        for task in pending:
            task.cancel()  # cancelling the waiter does not end the thread
        counted = managed_turn_count(owner.runner)
        retired = await _retire_profile_authority(owner, timeout=0.2)
        assert (counted, retired) == (1, False), 'ownership released while the collection could still unlink'
    finally:
        resume.set()
        await asyncio.gather(*pending, return_exceptions=True)
    assert not orphan.exists()
    assert await _retire_profile_authority(owner, timeout=5) is True


def test_history_scan_work_does_not_grow_with_the_candidate_count(api, owner, monkeypatch):
    """Many unheld images (a big chat deleted, or leftovers of a failed sweep) cost one read of each
    row, never one test per row and candidate: that took 8 s at 1,000 images on 20,000 messages."""
    owner.db.create_session('chat', source='api_server')
    for index in range(300):
        owner.db.append_message('chat', 'user', f'turn {index}: a few words of ordinary history')

    def sqlite_steps(count, start):
        candidates = _plant(count, start)
        steps = [0]

        def tick():
            steps[0] += 1
            return 0
        with _on_every_connection(owner.db, monkeypatch, lambda conn: conn.set_progress_handler(tick, 100)):
            assert release_unheld_media(owner.db, candidates) == count
        return steps[0]
    few, many = sqlite_steps(20, 0), sqlite_steps(400, 20)
    assert many < 2 * few, f'20 candidates cost {few} and 400 cost {many} (hundreds of SQLite VM steps)'


def test_admission_held_images_are_never_scanned(api, owner, monkeypatch):
    """Every image of a live API chat is held by its admission, so a boot or an API delete must not
    pay a transcript scan for it (each one used to offer every retained image to the scan)."""
    from gateway.session_ingress_media import collect_unheld_api_images
    held = _image_turn(api, owner, 'chat', 'first', images=10) + _image_turn(api, owner, 'chat', 'second', images=10)
    owner.db.append_message('chat', 'user', 'unrelated history')
    scans = []

    def trace(sql):
        if 'FROM messages' in sql and 'content' in sql:
            scans.append(sql)
    with _on_every_connection(owner.db, monkeypatch, lambda conn: conn.set_trace_callback(trace)):
        assert collect_unheld_api_images(owner.db) == 0
    assert all(path.exists() for path in held)
    assert scans == []


@pytest.mark.parametrize('error', [sqlite3.OperationalError('database is locked'),
                                   sqlite3.DatabaseError('database disk image is malformed')], ids=type)
def test_a_storage_error_after_a_committed_delete_keeps_its_candidates(api, owner, monkeypatch, error):
    """``delete_session`` collects retired media after its commit: a storage error there must not
    turn the finished delete into an error (a 500 with the chat already gone) or lose the candidates."""
    import gateway.session_ingress_media as media
    from hermes_state_media import PREFIX, collect_retired_media
    [image] = _image_turn(api, owner, 'chat')
    scan = media._history_mentions

    def failing(conn, references, after_id=0):
        raise error
    monkeypatch.setattr(media, '_history_mentions', failing)
    assert owner.db.delete_session('chat') is True
    assert owner.db.get_session('chat') is None
    assert image.exists()
    monkeypatch.setattr(media, '_history_mentions', scan)
    collect_retired_media(owner.db)  # the next delete or prune pass
    assert not image.exists()
    assert owner.db._read_all('SELECT key FROM state_meta WHERE key>=? AND key<?', (PREFIX, PREFIX[:-1] + '/')) == []


def test_a_referenced_image_survives_and_an_unreferenced_one_is_collected(api, owner, monkeypatch):
    """Above the ``instr`` prefilter's size: each spelling the transcript can carry still holds its
    image (attachment note, Windows or sandbox path raw and JSON-escaped, a row written after the
    scan's snapshot), and an unmentioned image, or one whose digest names another file, goes."""
    candidates = _plant(24)
    note, raw_windows, escaped_windows, late, other_name = candidates[:5]

    def windows(reference):
        return 'C:\\hermes\\native-inputs\\{}\\{}'.format(reference['sha256'], Path(reference['path']).name)
    owner.db.create_session('chat', source='api_server')
    owner.db.append_message('chat', 'user', 'look\n\n[Image attached at: {}]'.format(note['path']))
    owner.db.append_message('chat', 'assistant', 'saved at ' + windows(raw_windows))
    owner.db.append_message('chat', 'assistant', json.dumps({'path': windows(escaped_windows)}))
    owner.db.append_message('chat', 'assistant', other_name['sha256'] + '/api_other.png')
    write = owner.db._execute_write

    def mention_after_the_snapshot(fn, *args, **kwargs):
        monkeypatch.setattr(owner.db, '_execute_write', write)
        owner.db.append_message('chat', 'user', 'copied: [Image attached at: {}]'.format(late['path']))
        return write(fn, *args, **kwargs)
    monkeypatch.setattr(owner.db, '_execute_write', mention_after_the_snapshot)
    assert release_unheld_media(owner.db, candidates) == 20
    survivors = [reference for reference in candidates if Path(reference['path']).exists()]
    assert survivors == [note, raw_windows, escaped_windows, late]
