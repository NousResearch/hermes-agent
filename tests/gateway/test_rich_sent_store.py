import threading

from gateway import rich_sent_store


def test_lookup_degrades_when_index_path_is_unreadable(tmp_path, monkeypatch):
    path = tmp_path / "rich_sent_index.json"
    path.mkdir()
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(path))

    assert rich_sent_store.lookup("chat", "message") is None
    assert rich_sent_store.lookup_media("chat", "message") == []


def test_concurrent_records_preserve_both_entries(tmp_path, monkeypatch):
    path = tmp_path / "rich_sent_index.json"
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(path))
    original_load = rich_sent_store._load
    first_loaded = threading.Event()
    release_first = threading.Event()
    load_count = 0
    count_lock = threading.Lock()

    def delayed_first_load(store_path):
        nonlocal load_count
        data = original_load(store_path)
        with count_lock:
            load_count += 1
            first = load_count == 1
        if first:
            first_loaded.set()
            assert release_first.wait(timeout=5)
        return data

    monkeypatch.setattr(rich_sent_store, "_load", delayed_first_load)
    first = threading.Thread(target=rich_sent_store.record, args=("chat", "first", "one"))
    second = threading.Thread(target=rich_sent_store.record, args=("chat", "second", "two"))
    first.start()
    assert first_loaded.wait(timeout=5)
    second.start()
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert rich_sent_store.lookup("chat", "first") == "one"
    assert rich_sent_store.lookup("chat", "second") == "two"
