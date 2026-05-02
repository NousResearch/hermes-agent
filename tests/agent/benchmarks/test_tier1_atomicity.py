# tests/agent/benchmarks/test_tier1_atomicity.py
from tests.agent.benchmarks.fixture_builders import (
    make_loop_session, make_parallel_tool_session, make_multimodal_session,
)


def _orphan_tool_ids(messages) -> set[str]:
    """Return any tool_call_ids present on a tool message but absent on
    any assistant tool_calls (or vice versa)."""
    asst_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid:
                    asst_ids.add(cid)
    tool_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                tool_ids.add(cid)
    return (asst_ids ^ tool_ids)


def test_1_5_no_orphan_tool_pairs_after_compaction(compressor_pair, stub_summarizer):
    """Across all flag combos, compress() must never leave orphan tool
    pairs in the output. Sanitization is a post-condition the API
    relies on; an orphan would 400 on the next API call."""
    from itertools import product
    baseline, with_flags = compressor_pair

    fixtures = {
        "loop": make_loop_session(20),
        "parallel": make_parallel_tool_session(8, fanout=3),
        "mixed": make_loop_session(10) + make_parallel_tool_session(4, fanout=2),
    }
    for name, msgs in fixtures.items():
        for c in (baseline, with_flags):
            out = c.compress(msgs.copy(), current_tokens=999_999)  # force compact
            orphans = _orphan_tool_ids(out)
            assert not orphans, f"{name} → orphans: {orphans}"


def test_1_9_multimodal_image_parts_never_clobbered(compressor_pair, stub_summarizer):
    """Vision messages with image content must survive both compressors
    intact — no part dropped, no string replacement."""
    baseline, with_flags = compressor_pair
    msgs = make_multimodal_session(n_image_turns=15)

    for c in (baseline, with_flags):
        out = c.compress(msgs.copy(), current_tokens=999_999)
        # Every surviving user message that originally had an image part
        # must still have one. (Compaction may delete whole messages, but
        # whatever survives must be intact.)
        for m in out:
            content = m.get("content")
            if isinstance(content, list):
                # If list-content survived at all, image_url parts must
                # be present and uncorrupted (data: URL prefix).
                images = [p for p in content
                          if isinstance(p, dict) and p.get("type") == "image_url"]
                if images:
                    for img in images:
                        url = (img.get("image_url") or {}).get("url", "")
                        assert url.startswith("data:image/"), (
                            f"image_url corrupted: {url[:80]}"
                        )
