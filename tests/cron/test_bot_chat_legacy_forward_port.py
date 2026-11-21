"""Legacy receipt boundaries and shared SQL/JSON ownership."""
import json
from pathlib import Path

import pytest

from cron import bot_chat_legacy as legacy, delivery_queue as queue


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron" / "deliveries.db")
    (tmp_path / "state.db").touch()
    return tmp_path


@pytest.mark.parametrize("key", ["../escape", "nested/escape", "/absolute", "A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 64 + "\n", "", None])
def test_noncanonical_caller_key_is_rejected_before_path_use(store, key):
    if key == "/absolute":
        key = str(store / "absolute")
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        legacy.defer(key, {"id": "job"}, "body", store, for_failure=False)
    assert not (store / "cron").exists()


@pytest.mark.parametrize("bad_id", ["../escape", "nested/escape", "/absolute", "A" * 64, "a" * 63])
def test_invalid_file_identity_never_reaches_transport_or_blocks_peers(store, monkeypatch, bad_id):
    from tools import bot_live_delivery as live
    folder = store / "cron" / "bot_chat_pending"
    folder.mkdir(parents=True)
    good, bad = "a" * 64, "b" * 64
    for key, identity in [(good, good), (bad, bad_id)]:
        (folder / f"{key}.json").write_text(json.dumps(dict(
            id=identity, status="queued", job={"id": "job"}, content="body", home=str(store), sequence=1)))
    sent = []
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {})
    monkeypatch.setattr(live, "deliver_to_live_owner", lambda *args, **kw: sent.append(kw["delivery_id"]) or {"status": "queued"})
    legacy.drain_legacy_pending()
    assert sent == [good]
    assert (folder / f"{bad}.json").exists()
    assert json.loads((folder / f"{bad}.json").read_text())["status"] == "ambiguous"


@pytest.mark.parametrize("change", ["name", "body", "home", "category"])
def test_defer_hash_binds_render_destination_category_without_overwrite(store, change):
    import hashlib
    key = "c" * 64
    job = {"id": "job", "name": "Daily", "execution_id": "run"}
    legacy.defer(key, job, "body", store, for_failure=False)
    path = store / "cron" / "bot_chat_pending" / f"{key}.json"
    original = path.read_bytes()
    record = json.loads(original)
    message = '[Cronjob "Daily" output — scheduled job, not the user. Review it, act on anything that needs action, and summarize for the chat.]\n\nbody'
    expected = hashlib.sha256(json.dumps(dict(message=message, destination=str(store.resolve()), category="result"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert record["payload_hash"] == expected
    legacy.defer(key, job, "body", store, for_failure=False)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="different payload"):
        legacy.defer(key, dict(job, name="Renamed") if change == "name" else job,
                     "changed" if change == "body" else "body",
                     store / "other" if change == "home" else store,
                     for_failure=change == "category")
    assert path.read_bytes() == original


@pytest.mark.parametrize("terminal", ["transferred", "suppressed", "ambiguous"])
@pytest.mark.parametrize("old_hash", [False, True])
def test_json_terminalization_scrubs_payload_retains_hash_and_disposition(store, monkeypatch, terminal, old_hash):
    from tools import bot_live_delivery as live
    from gateway import warning_notifications
    key = "d" * 64
    legacy.defer(key, {"id": "job", "name": "Daily"}, "body", store, for_failure=True)
    path = store / "cron" / "bot_chat_pending" / f"{key}.json"
    record = json.loads(path.read_text())
    digest = record["payload_hash"]
    if old_hash:
        record.pop("payload_hash")
        path.write_text(json.dumps(record))
    if terminal == "ambiguous":
        (store / "state.db").unlink()
    monkeypatch.setattr(warning_notifications, "warning_notifications_enabled", lambda *args: terminal != "suppressed")
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {})
    monkeypatch.setattr(live, "deliver_to_live_owner", lambda *args, **kw: {"status": "queued"})
    legacy.drain_legacy_pending()
    result = json.loads(path.read_text())
    assert result["status"] == terminal
    assert "content" not in result and "job" not in result
    assert result["payload_hash"] == digest
    with pytest.raises(ValueError, match="different payload"):
        legacy.defer(key, {"id": "job", "name": "Changed"}, "body", store, for_failure=True)


@pytest.mark.parametrize("state", ["queued", "claimed", "ambiguous", "settled", "suppressed"])
def test_existing_json_owner_fences_sql_and_arbitration_checks_hash(store, state):
    key = "e" * 64
    job = {"id": "job"}
    legacy.defer(key, job, "body", store, for_failure=False)
    path = store / "cron" / "bot_chat_pending" / f"{key}.json"
    record = json.loads(path.read_text())
    record["status"] = state
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    result = queue.enqueue(key, job, "body", payload_hash=record["payload_hash"])
    assert result["status"] == state
    assert queue.get_status(key) is None
    decision = legacy.arbitrate(key, record["payload_hash"], origin_home=store)
    assert decision["owned"] and decision["owner"] == "legacy"
    assert decision["record"]["status"] == state
    with pytest.raises(ValueError, match="different payload"):
        legacy.arbitrate(key, "changed", origin_home=store)
    assert path.read_bytes() == original


@pytest.mark.parametrize("terminal", [False, True])
def test_sql_owner_or_tombstone_fences_legacy_creation(store, monkeypatch, terminal):
    key = "f" * 64
    job = {"id": "job"}
    digest = legacy.payload_hash(job, "body", store)
    queue.enqueue(key, job, "body", payload_hash=digest)
    if terminal:
        monkeypatch.setattr(queue, "MAX_TERMINAL_DELIVERIES", 0)
        queue.claim_next()
        queue._finish(key, error=None)
    result = legacy.defer(key, job, "body", store, for_failure=False)
    assert result["status"] == ("delivered" if terminal else "pending")
    assert not (store / "cron" / "bot_chat_pending" / f"{key}.json").exists()
    decision = legacy.arbitrate(key, digest, origin_home=store)
    assert decision["owned"] and decision["owner"] == "sql"
    with pytest.raises(ValueError, match="different payload"):
        legacy.defer(key, job, "changed", store, for_failure=False)


def test_concurrent_sql_and_json_producers_publish_one_owner(store):
    import subprocess
    import sys
    # Initialize before the processes race; the tested exclusion is admission,
    # not platform-dependent first-open journal-mode negotiation.
    queue.get_status("initialize")
    code = """
import sys
from pathlib import Path
from cron import bot_chat_legacy as legacy, delivery_queue as queue
home, kind, key = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
queue.DELIVERY_DB = home / 'cron' / 'deliveries.db'
job = {'id': 'job'}
print('ready', flush=True)
sys.stdin.readline()
if kind == 'sql':
    queue.enqueue(key, job, 'body', payload_hash=legacy.payload_hash(job, 'body', home))
else:
    legacy.defer(key, job, 'body', home, for_failure=False)
"""
    for index in range(4):
        key = f"{index:064x}"
        processes = [subprocess.Popen([sys.executable, "-c", code, str(store), kind, key],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for kind in ("sql", "json")]
        try:
            for proc in processes:
                assert proc.stdout.readline().strip() == "ready"
            for proc in processes:
                proc.stdin.write("go\n")
                proc.stdin.flush()
            for proc in processes:
                out, err = proc.communicate(timeout=20)
                assert proc.returncode == 0, (out, err)
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
        sql_owned = queue.get_status(key) is not None
        json_owned = (store / "cron" / "bot_chat_pending" / f"{key}.json").exists()
        assert sql_owned != json_owned



def test_json_publication_uses_exclusive_create_and_preserves_racing_owner(store, monkeypatch):
    import os
    key = "9" * 64
    path = store / "cron" / "bot_chat_pending" / f"{key}.json"
    rival = dict(id=key, status="claimed", payload_hash=legacy.payload_hash({"id": "job"}, "body", store))
    original_open = os.open
    seen = []
    def race_open(filename, flags, *args, **kwargs):
        if Path(filename) == path and flags & os.O_EXCL:
            seen.append(flags)
            path.write_text(json.dumps(rival))
        return original_open(filename, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", race_open)
    legacy.defer(key, {"id": "job"}, "body", store, for_failure=False)
    assert seen, "receipt admission must use O_EXCL, not exists-then-replace"
    assert json.loads(path.read_text()) == rival


@pytest.mark.parametrize("defect", ["filename", "id_mismatch", "sequence"])
def test_malformed_scan_metadata_is_isolated(store, monkeypatch, defect):
    from tools import bot_live_delivery as live
    folder = store / "cron" / "bot_chat_pending"
    folder.mkdir(parents=True)
    good, bad = "a" * 64, "b" * 64
    record = dict(id=bad, status="queued", home=str(store), job={"id": "job"}, content="body", sequence=1)
    if defect == "sequence":
        record["sequence"] = {"bad": "order"}
    if defect == "id_mismatch":
        record["id"] = "c" * 64
    path = folder / ("NONCANONICAL.json" if defect == "filename" else f"{bad}.json")
    path.write_text(json.dumps(record))
    (folder / f"{good}.json").write_text(json.dumps(dict(record, id=good, sequence=2)))
    calls = []
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda home: {})
    monkeypatch.setattr(live, "deliver_to_live_owner", lambda *args, **kw: calls.append(kw["delivery_id"]) or {})
    legacy.drain_legacy_pending()
    assert calls == [good]
    assert path.exists()
    assert json.loads(path.read_text())["status"] == "ambiguous"


def test_corrupt_exact_owner_fences_admission_without_blocking_other_keys(store):
    folder = store / "cron" / "bot_chat_pending"
    folder.mkdir(parents=True)
    key = "1" * 64
    path = folder / f"{key}.json"
    path.write_text("{")
    with pytest.raises(ValueError):
        queue.enqueue(key, {}, "body", payload_hash="digest")
    assert queue.get_status(key) is None
    assert path.read_text() == "{"
    assert queue.enqueue("2" * 64, {}, "body", payload_hash="digest")["status"] == "pending"


def test_explicit_origin_arbitration_routes_both_stores_to_origin(store, monkeypatch):
    monkeypatch.setattr(queue, "DELIVERY_DB", None)
    key = "3" * 64
    digest = legacy.payload_hash({}, "body", store)
    legacy.defer(key, {}, "body", store, for_failure=False, origin_home=store)
    monkeypatch.setenv("HERMES_HOME", str(store / "other"))
    decision = legacy.arbitrate(key, digest, origin_home=store)
    assert decision["owner"] == "legacy"
    # The shared SQLite helper loads ambient WAL config (and initializes home
    # directories); neither ownership store may be created there.
    assert not (store / "other" / "cron" / "deliveries.db").exists()
    assert not (store / "other" / "cron" / "bot_chat_pending").exists()
    result = queue.enqueue(key, {}, "body", payload_hash=digest, origin_home=store)
    assert result["status"] == "queued"


@pytest.mark.parametrize("producer", ["sql", "legacy"])
def test_producer_can_publish_inside_existing_arbitration_reservation(store, producer):
    key = "4" * 64
    digest = legacy.payload_hash({}, "body", store)
    with legacy.ownership_arbitration(key, origin_home=store) as conn:
        assert not legacy.arbitrate(key, digest, origin_home=store, conn=conn)["owned"]
        if producer == "sql":
            result = queue.enqueue(key, {}, "body", payload_hash=digest, origin_home=store, conn=conn)
        else:
            result = legacy.defer(key, {}, "body", store, for_failure=False, origin_home=store, conn=conn)
        assert conn.in_transaction
        assert result["status"] in {"pending", "queued"}
    assert legacy.arbitrate(key, digest, origin_home=store)["owner"] == producer
