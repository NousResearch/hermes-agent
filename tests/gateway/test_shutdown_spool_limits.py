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

    class Sink:
        def append_message(self, **kwargs: object) -> None:
            assert kwargs["content"] == content

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
