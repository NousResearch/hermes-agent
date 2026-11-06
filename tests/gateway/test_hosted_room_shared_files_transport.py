"""A bot on another profile of this host shares files through the private owner transport."""
import asyncio
import base64
import hashlib
import threading
from dataclasses import asdict
from pathlib import Path

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_outputs import HostedTurn, _PUSH_CHUNK_BYTES, bind_outputs, output_prefix, release_outputs, stage
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_hosted_transport import _server, _SourceTask


def _chunks(source, data, name="report.txt", **overrides):
    digest = hashlib.sha256(data).hexdigest()
    params = {**source.params, "name": name, "size": len(data), "sha256": digest, **overrides}
    params.pop("prompt", None)
    params.pop("attachments", None)
    for offset in range(0, len(data), _PUSH_CHUNK_BYTES):
        yield {**params, "offset": offset,
               "data_base64": base64.b64encode(data[offset:offset + _PUSH_CHUNK_BYTES]).decode("ascii")}


def _push(source, data, **kwargs):
    result = None
    for chunk in _chunks(source, data, **kwargs):
        result = source.service.attest(source.selector, "output", chunk)
    return result


def test_the_owner_stages_a_pushed_file_for_the_exact_running_attempt(tmp_path, monkeypatch):
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        data = b"named profile result\n" * 5000
        result = _push(source, data)
        stored = result["stored"]
        assert (stored["name"], stored["size"], stored["sha256"]) == (
            "report.txt", len(data), hashlib.sha256(data).hexdigest())
        assert result["owner"] == "alice"
        # Publication finds it like a file the owner's own bot staged.
        outputs = bind_outputs(source.service, source.service._room("room"), source.task)
        assert [item["attachment_id"] for item in outputs["attachments"]] == [stored["attachment_id"]]
        # The same file again is the same staged file.
        assert _push(source, data)["stored"]["attachment_id"] == stored["attachment_id"]
        assert not list((Path(tmp_path) / "hosted-room-attachments" / "incoming").glob("*.part"))


def test_pushed_chunks_are_fenced_to_the_running_attempt_and_their_order(tmp_path, monkeypatch):
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        data = b"x" * (_PUSH_CHUNK_BYTES + 10)
        first, second = list(_chunks(source, data))
        for bad in ({"execution_generation": 99}, {"task": {**first["task"], "task_id": "other"}},
                    {"_target_home": str(tmp_path / "elsewhere")}):
            with pytest.raises(RuntimeStoreError, match="permission_denied"):
                source.service.attest(source.selector, "output", {**first, **bad})
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            source.service.attest(source.selector, "output", second)  # out of order
        assert source.service.attest(source.selector, "output", first) == {
            "owner": "alice", "target_home": source.homes["other"], "received": _PUSH_CHUNK_BYTES}
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            source.service.attest(source.selector, "output", {**second, "sha256": "0" * 64})
        # Stop fences the attempt: a later chunk is refused before any byte is written.
        tasks.begin_task_cancel(source.db.db_path, source.task["identity"], cancel_id="stop",
                                expected_cancel_generation=0, clock=lambda: 1.0)
        with pytest.raises(RuntimeStoreError, match="permission_denied"):
            source.service.attest(source.selector, "output", second)
        assert bind_outputs(source.service, source.service._room("room"), source.task) == {}


def test_a_named_profile_turn_pushes_its_file_over_the_owner_socket(tmp_path, monkeypatch):
    from gateway.session_hosted_transport import install_hosted_transport
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()
        server = _server(tmp_path)
        sizes = []
        handle = server.handle_request_line

        def measured(raw, *args):
            sizes.append(len(raw) + 1)
            return handle(raw, *args)
        server.handle_request_line = measured
        install_hosted_transport(server, source.authority, loop, attest=source.service.attest)
        assert asyncio.run_coroutine_threadsafe(server.start(), loop).result()
        try:
            identity = source.task["identity"]
            turn = HostedTurn(identity.room_id, identity.task_id, source.task["execution_generation"],
                              tmp_path / "profiles" / "other" / "state.db", task=asdict(identity),
                              transport=source.binding)
            data = b"\x89PNG\r\n\x1a\n" + b"pixels" * 20000
            stored = stage(turn, name="chart.png", data=data)
            assert (stored["kind"], stored["mime"], stored["size"]) == ("image", "image/png", len(data))
            chunk_requests = [size for size in sizes if size > 1024]  # besides each request's identify probe
            assert len(chunk_requests) == -(-len(data) // _PUSH_CHUNK_BYTES) and max(sizes) < 65536
            row, = source.store.staged_uploads(
                room_id="room", upload_prefix=output_prefix(identity.task_id, source.task["execution_generation"]))
            assert row["attachment_id"] == stored["attachment_id"]
            # The target never opened the owner's database or a path of its own store.
            assert not (tmp_path / "profiles" / "other" / "hosted-room-attachments").exists()
        finally:
            asyncio.run_coroutine_threadsafe(server.stop(), loop).result()
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()


def test_a_member_turn_binds_its_owner_transport_when_it_has_one(tmp_path):
    import json
    from types import SimpleNamespace
    from gateway.hosted_room_outputs import hosted_turn_for
    from gateway.session_hosted_transport import _BINDING
    from hermes_state import SessionDB
    with SessionDB(tmp_path / "state.db") as db:
        authority = SimpleNamespace(db=db)
        identity = {"room_id": "room", "task_id": "dtask:1", "thread_id": "t", "turn_id": "u"}
        row = {"request_id": "hosted:" + json.dumps([identity, 2], sort_keys=True, separators=(",", ":"))}
        local = hosted_turn_for(authority, SimpleNamespace(session_id="local"), row)
        assert (local.room_id, local.task_id, local.execution_generation, local.transport) == (
            "room", "dtask:1", 2, None)
        binding = {"source_home": "/owner", "selector": {"room_id": "room", "member_id": "two", "profile": "other"},
                   "target_home": "/other", "owner": "alice"}
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?)", (_BINDING + "named", json.dumps(binding))))
        assert hosted_turn_for(authority, SimpleNamespace(session_id="named"), row).transport == binding
        assert hosted_turn_for(authority, SimpleNamespace(session_id="x"), {"request_id": "native:1"}) is None


def test_an_interrupted_transfer_is_reclaimed_by_the_stores_own_pruning(tmp_path, monkeypatch):
    """Seam: a partial file's mtime moved back past the uncommitted-upload expiry (time passing)."""
    import os
    import time
    from gateway import hosted_room_attachments as store_module
    from gateway.hosted_room_attachments import UNCOMMITTED_TTL_SECONDS
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        for name in ("old.txt", "live.txt"):
            first = next(_chunks(source, b"y" * (_PUSH_CHUNK_BYTES + 1), name=name))
            assert "received" in source.service.attest(source.selector, "output", first)
        store = HostedRoomAttachmentStore(source.db.db_path)
        old, = store.incoming_root.glob(f"*{_name_digest('old.txt')}*.part")
        stale = time.time() - UNCOMMITTED_TTL_SECONDS - 60
        os.utime(old, (stale, stale))
        store.prune()
        assert not old.exists()
        assert len(list(store.incoming_root.glob("*.part"))) == 1  # the live transfer keeps going
        live, = store.incoming_root.glob("*.part")
        os.utime(live, (stale, stale))
        store_module._STARTED.clear()  # a restart: the first store of the process prunes
        HostedRoomAttachmentStore(source.db.db_path)
        assert not live.exists()


def test_a_turns_limits_count_staged_files_and_transfers_in_flight_which_end_with_it(tmp_path, monkeypatch):
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        # The owner checks a pushed file as its own staging does, before it is staged.
        assert "refused" in _push(source, b"not a png", name="picture.png")
        for index in range(4):
            assert "stored" in _push(source, b"%d\n" % index, name=f"f{index}.txt")
        for index in range(4):
            first = next(_chunks(source, b"z" * (_PUSH_CHUNK_BYTES + 1), name=f"part{index}.txt"))
            assert "received" in source.service.attest(source.selector, "output", first)
        # Four staged and four arriving: a ninth file is refused before any of its bytes move.
        ninth = next(_chunks(source, b"ninth\n", name="ninth.txt"))
        assert source.service.attest(source.selector, "output", ninth)["refused"] == (
            "A turn can share at most 8 files.")
        store = HostedRoomAttachmentStore(source.db.db_path)
        assert len(list(store.incoming_root.glob("*.part"))) == 4
        # The task's outcome ends its transfers: nothing still arriving can be published.
        release_outputs(source.service, source.service._room("room"), source.task)
        assert list(store.incoming_root.glob("*.part")) == []


def _name_digest(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("swap", ["symlink", "hard link"])
def test_a_chunk_is_appended_only_to_the_part_file_its_transfer_created(tmp_path, monkeypatch, swap):
    """The member process runs as the same OS user, so it can swap the owner's part file between chunks."""
    import os
    source = _SourceTask(tmp_path, monkeypatch, b"input")
    with source.db:
        data = b"B" * (_PUSH_CHUNK_BYTES + 10)
        first, second = list(_chunks(source, data))
        assert source.service.attest(source.selector, "output", first)["received"] == _PUSH_CHUNK_BYTES
        part, = HostedRoomAttachmentStore(source.db.db_path).incoming_root.glob("*.part")
        victim = tmp_path / "victim.bin"
        victim.write_bytes(b"V" * _PUSH_CHUNK_BYTES)  # exactly as long as the bytes received so far
        part.unlink()
        if swap == "symlink":
            part.symlink_to(victim)
        else:
            os.link(victim, part)
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            source.service.attest(source.selector, "output", second)
        assert victim.read_bytes() == b"V" * _PUSH_CHUNK_BYTES
        # The transfer starts over from its first chunk and completes.
        assert _push(source, data)["stored"]["sha256"] == hashlib.sha256(data).hexdigest()
        assert victim.read_bytes() == b"V" * _PUSH_CHUNK_BYTES
