"""A successful compaction hands allocator pages back to the OS.

The compressed-away message dicts are the largest allocation a long session
ever frees, but Python's arena allocator keeps those pages in the process heap
— RSS retains the pre-compaction high-water mark until exit. #76905's
trim_memory lifecycle covers the gateway/TUI housekeeping loops but not the
CLI compression path, so compress() now calls
``trim_memory(reason="post-compression")`` after a successful pass.

trim_memory itself is glibc/Linux-gated (a fast no-op on macOS), so these
tests monkeypatch the seam rather than asserting on RSS. Salvaged in spirit
from #70782 (which reached for a bare gc.collect(); trim_memory is the
house mechanism and already wraps a collect).
"""
import hermes_cli.mem_trim as mem_trim
from agent.context_compressor import ContextCompressor


def _compressor(threshold_tokens: int = 24_576) -> ContextCompressor:
    cc = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=5,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    cc.threshold_tokens = threshold_tokens  # pin; don't couple to window math
    cc._generate_summary = lambda *a, **k: "Summary of earlier turns."
    return cc


def _messages(n: int, size: int = 1500) -> list:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i} " + "z" * size})
    return msgs


def test_successful_compression_trims_memory_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mem_trim, "trim_memory", lambda *a, **kw: calls.append(kw) or False
    )

    cc = _compressor()
    out = cc.compress(_messages(14), current_tokens=100_000)

    assert len(out) < 15, "sanity: compaction should have made progress"
    assert len(calls) == 1, "trim_memory must run exactly once per compaction"
    assert calls[0].get("reason") == "post-compression"


def test_trim_failure_does_not_break_compression(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("allocator says no")

    monkeypatch.setattr(mem_trim, "trim_memory", boom)

    cc = _compressor()
    out = cc.compress(_messages(14), current_tokens=100_000)

    assert cc._last_compression_made_progress is True
    assert isinstance(out, list) and out, "compress() must still return messages"
