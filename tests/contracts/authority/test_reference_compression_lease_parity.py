"""Compression lock boundaries shared by the reference and SQLite backends."""

def test_holder_is_visible_at_exact_expiry(backend, scope_a, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1000.0)
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id="s", holder="owner", ttl_seconds=2) is True
    monkeypatch.setattr("time.time", lambda: 1002.0)
    assert coord.get_compression_lock_holder(session_id="s") == "owner"
    assert coord.try_acquire_compression_lock(session_id="s", holder="other", ttl_seconds=2) is False
    monkeypatch.setattr("time.time", lambda: 1002.001)
    assert coord.get_compression_lock_holder(session_id="s") is None
