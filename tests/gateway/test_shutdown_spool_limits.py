"""Recovery must not retain every decoded spool payload while sorting."""

import tracemalloc
from pathlib import Path

import pytest

from gateway import shutdown_flush


def test_recovery_payload_memory_does_not_scale_with_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path)
    content = "x" * (256 * 1024)
    for seq in range(24):
        shutdown_flush.spool_dropped_transcript_message(
            "sess", {"role": "user", "content": content, "timestamp": seq},
        )

    class Messages:
        def append(self, session_id: str, message) -> None:
            assert message.content == content

    class Sink:
        messages = Messages()

    tracemalloc.start()
    try:
        assert shutdown_flush.recover_pending_to_db(Sink()) == 24
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Six MiB of payloads on disk; allow ample room for one decoded payload,
    # its source text and the small ordering index, but not the entire backlog.
    assert peak < 4 * 1024 * 1024, f"recovery retained {peak} bytes"
    assert not list(tmp_path.glob("*.json"))


def test_same_second_shutdown_slot_precedes_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    monkeypatch.setattr(shutdown_flush, "_get_flush_dir", lambda: tmp_path)
    monkeypatch.setattr(shutdown_flush.time, "time", lambda: 100)
    names = iter(["zzz", "aaa", "bbb"])
    monkeypatch.setattr(shutdown_flush.uuid, "uuid4", lambda: SimpleNamespace(hex=next(names)))
    assert shutdown_flush.flush_pending_to_file({"session": {"text": "A", "session_id": "sess"}}) == 1
    assert shutdown_flush.flush_overflow_to_file({"session": [
        {"text": "B", "session_id": "sess"}, {"text": "C", "session_id": "sess"},
    ]}) == 2
    store = MagicMock()
    assert shutdown_flush.recover_pending_to_db(store) == 3
    assert [call.args[1].content for call in store.messages.append.call_args_list] == ["A", "B", "C"]
    assert not list(tmp_path.glob("*.json"))
