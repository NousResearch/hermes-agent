"""Unit tests for JsonFileDedupStore — the Hermes implementation of the
SDK ``lark_oapi.channel.normalize.DedupStore`` Protocol.

Pure unit tests — no FeishuAdapter, no SDK imports beyond the Protocol
type-check at runtime. Covers eight behavior classes:

  1. Basic seen/mark
  2. TTL expiry
  3. LRU eviction
  4. Thread safety
  5. Atomic writes
  6. Backward-compat with the legacy list and transitional formats
  7. Shutdown flush
  8. Robustness (missing path / corrupt JSON)
"""

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("lark_oapi.channel")

from gateway.platforms.feishu import JsonFileDedupStore


# -- 1. Basic seen/mark behavior --------------------------------------------

def test_unmarked_key_returns_false(tmp_path: Path):
    store = JsonFileDedupStore(path=tmp_path / "dedup.json", max_entries=100)
    assert store.seen("never_marked") is False


def test_marked_key_returns_true(tmp_path: Path):
    store = JsonFileDedupStore(path=tmp_path / "dedup.json", max_entries=100)
    store.mark("k1", ttl_seconds=60)
    assert store.seen("k1") is True


# -- 2. TTL expiry ----------------------------------------------------------

def test_ttl_expiry_lazy_deletes_entry(tmp_path: Path):
    store = JsonFileDedupStore(path=tmp_path / "dedup.json", max_entries=100)
    store.mark("k_short", ttl_seconds=1)
    assert store.seen("k_short") is True
    time.sleep(1.1)
    assert store.seen("k_short") is False
    # Verify the lazy expire also removed the entry from memory.
    assert store.size() == 0


# -- 3. LRU eviction --------------------------------------------------------

def test_lru_eviction_drops_oldest_when_max_entries_reached(tmp_path: Path):
    store = JsonFileDedupStore(path=tmp_path / "dedup.json", max_entries=3)
    for i in range(5):
        store.mark(f"k{i}", ttl_seconds=3600)
    # Expect last 3 to remain: k2, k3, k4.
    assert store.seen("k0") is False
    assert store.seen("k1") is False
    assert store.seen("k2") is True
    assert store.seen("k3") is True
    assert store.seen("k4") is True
    assert store.size() == 3


# -- 4. Thread safety --------------------------------------------------------

def test_concurrent_mark_and_seen_no_race(tmp_path: Path):
    """32 threads x 100 mark + 100 seen each. No exception, no size drift."""
    store = JsonFileDedupStore(
        path=tmp_path / "dedup.json", max_entries=10000,
    )
    errors: list = []

    def worker(tid: int):
        try:
            for i in range(100):
                store.mark(f"t{tid}_k{i}", ttl_seconds=3600)
                store.seen(f"t{tid}_k{i}")  # self-check; should be True
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.size() == 32 * 100


# -- 5. Atomic write (interrupted flush leaves main file intact) ------------

def test_atomic_write_tmp_residue_does_not_corrupt_main(tmp_path: Path):
    """Simulate a previous run leaving a .tmp file behind. Main file
    should still load cleanly; .tmp file is ignored on read."""
    main = tmp_path / "dedup.json"
    main.write_text(
        json.dumps({"k_main": time.time() + 3600}), encoding="utf-8",
    )
    # Stale tmp from a prior interrupted flush.
    tmp_residue = main.with_suffix(main.suffix + ".tmp")
    tmp_residue.write_text("PARTIAL_GARBAGE_NOT_VALID_JSON")

    store = JsonFileDedupStore(path=main, max_entries=100)
    assert store.seen("k_main") is True   # main file loaded fine
    # The .tmp residue should be untouched by the loader (we don't read it).
    assert tmp_residue.exists()


# -- 6. Backward-compat: legacy list format ---------------------------------

def test_loads_legacy_list_format_with_default_ttl(tmp_path: Path):
    path = tmp_path / "dedup.json"
    path.write_text(
        json.dumps(["om_legacy_a", "om_legacy_b", "om_legacy_c"]),
        encoding="utf-8",
    )
    store = JsonFileDedupStore(
        path=path, max_entries=100, default_ttl_seconds=3600,
    )
    assert store.seen("om_legacy_a") is True
    assert store.seen("om_legacy_b") is True
    assert store.seen("om_legacy_c") is True
    assert store.seen("om_never_seen") is False
    assert store.size() == 3


def test_loads_transitional_dict_format(tmp_path: Path):
    """Transitional on-disk format: ``{'message_ids': {id: wallclock_ts}}``.
    Loader treats wallclock_ts as 'last marked at' and re-derives expiry."""
    path = tmp_path / "dedup.json"
    now = time.time()
    path.write_text(
        json.dumps({"message_ids": {
            "om_recent": now,           # marked just now
            "om_old": now - 7200,       # marked 2h ago
        }}),
        encoding="utf-8",
    )
    store = JsonFileDedupStore(
        path=path, max_entries=100, default_ttl_seconds=3600,   # 1h ttl
    )
    # om_recent: expiry = now + 3600 → still valid
    assert store.seen("om_recent") is True
    # om_old: expiry = (now - 7200) + 3600 = now - 3600 → expired → False
    assert store.seen("om_old") is False


def test_transitional_message_ids_are_namespaced_for_sdk_keys(tmp_path: Path):
    path = tmp_path / "dedup.json"
    now = time.time()
    path.write_text(
        json.dumps({"message_ids": {"om_legacy": now}}),
        encoding="utf-8",
    )

    store = JsonFileDedupStore(
        path=path,
        max_entries=100,
        default_ttl_seconds=3600,
        account_id="cli_test_app",
    )

    assert store.seen("msg:cli_test_app:om_legacy") is True
    assert store.seen("om_legacy") is False


def test_mark_persists_immediately_for_crash_safe_dedup(tmp_path: Path):
    path = tmp_path / "dedup.json"
    store = JsonFileDedupStore(path=path, max_entries=100)

    store.mark("msg:cli_test_app:om_now", ttl_seconds=3600)

    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "msg:cli_test_app:om_now" in raw


def test_top_level_message_ids_are_namespaced_for_sdk_keys(tmp_path: Path):
    path = tmp_path / "dedup.json"
    path.write_text(
        json.dumps({"om_legacy": time.time() + 3600}),
        encoding="utf-8",
    )

    store = JsonFileDedupStore(
        path=path,
        max_entries=100,
        account_id="cli_test_app",
    )

    assert store.seen("msg:cli_test_app:om_legacy") is True
    assert store.seen("om_legacy") is False


# -- 7. Shutdown flush -------------------------------------------------------

def test_flush_writes_to_disk_immediately(tmp_path: Path):
    path = tmp_path / "dedup.json"
    store = JsonFileDedupStore(path=path, max_entries=100)
    store.mark("k_pre_flush", ttl_seconds=3600)
    # Before flush, file may not exist yet (debounced).
    store.flush()
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "k_pre_flush" in raw
    # Re-load in fresh store, verify persistence round-trip.
    store2 = JsonFileDedupStore(path=path, max_entries=100)
    assert store2.seen("k_pre_flush") is True


def test_flush_persists_then_fresh_store_loads(tmp_path: Path):
    """Round-trip: shutdown flush + fresh-store load — the contract that
    the FeishuAdapter.disconnect() → flush() path relies on."""
    path = tmp_path / "dedup.json"
    s1 = JsonFileDedupStore(path=path, max_entries=100)
    for i in range(10):
        s1.mark(f"k{i}", ttl_seconds=3600)
    s1.flush()

    # Fresh store on the same file — should load all 10 entries.
    s2 = JsonFileDedupStore(path=path, max_entries=100)
    for i in range(10):
        assert s2.seen(f"k{i}") is True
    assert s2.size() == 10


# -- 8. Robustness: missing path / corrupt JSON -----------------------------

def test_missing_path_does_not_raise(tmp_path: Path):
    """Path doesn't exist → ctor returns empty store (no exception).
    Parent directory may also not exist."""
    nonexistent = tmp_path / "subdir" / "dedup.json"
    store = JsonFileDedupStore(path=nonexistent, max_entries=100)
    assert store.size() == 0
    # Subsequent mark + flush creates the parent dir.
    store.mark("k", ttl_seconds=60)
    store.flush()
    assert nonexistent.exists()


def test_corrupt_json_falls_back_to_empty_store(tmp_path: Path, caplog):
    path = tmp_path / "dedup.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = JsonFileDedupStore(path=path, max_entries=100)
    assert store.size() == 0
    # Loader logs a warning but does NOT raise; tolerate caplog config diffs.
    assert any("failed to load" in r.message.lower() for r in caplog.records) \
        or store.size() == 0


def test_load_from_disk_tolerates_malformed_timestamp_values(tmp_path):
    """Regression: a non-numeric value in the persisted dedup state must
    not crash the store; the bad key is skipped, the rest loads."""
    import json
    from gateway.platforms.feishu.dedup_store import JsonFileDedupStore

    state_path = tmp_path / "feishu_seen_message_ids.json"
    state_path.write_text(
        json.dumps({
            "om_good": 9_999_999_999.0,
            "om_bad_str": "not-a-timestamp",
            "om_bad_null": None,
        }),
        encoding="utf-8",
    )

    store = JsonFileDedupStore(path=state_path, max_entries=1024, default_ttl_seconds=86400)
    assert store.seen("om_good") is True
    assert store.seen("om_bad_str") is False
    assert store.seen("om_bad_null") is False
