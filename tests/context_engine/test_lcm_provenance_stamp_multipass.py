"""P2 — multi-pass provenance stamp (spec 2026-07-02 §0.6/§5B, PR-B).

The Option B `_src_idx` stamp was gated to single-pass compactions. The §0.6
five-step proof establishes the fresh tail is a contiguous suffix of the ORIGINAL
`messages` for ANY number of leaf passes (front-only removal, summaries → DAG never
into working_messages, frozen-K tail, stub-insertion strictly post-stamp), so the
end-anchored map is valid for `leaf_passes >= 1`.

These tests exercise the widened gate + the hardened per-row structural guard
(role + tool_call_id + tool_calls arity), including a multi-pass fixture with real
chunk removal in an early pass and a boundary-shift fixture (different fold amounts
per pass). Structural test: summary rows never appear in the working list.
"""
from __future__ import annotations

from plugins.context_engine.lcm.config import LCMConfig
from plugins.context_engine.lcm.engine import LCMEngine


def _engine():
    e = LCMEngine.__new__(LCMEngine)
    e._config = LCMConfig()
    e._session_id = None
    e._session_ignored = False
    e._session_stateless = False
    e.compression_count = 0
    e.last_leaf_passes = 0
    e._pending_context_anchor_messages = None

    class _DAG:
        def get_session_nodes(self, sid):
            return []

        def get_uncondensed_at_depth(self, sid, d):
            return []

    e._dag = _DAG()
    return e


def _stamp_and_assemble(e, messages, *, leaf_passes, working_messages=None):
    """Replicate the compress() Step-7 stamp + assemble + final-sanitize slice,
    mirroring the LIVE gate (leaf_passes >= 1) and the hardened structural guard."""
    wm = working_messages if working_messages is not None else list(messages)
    lac = e._leading_anchor_count(wm)
    e.last_leaf_passes = leaf_passes
    tail_rows = wm[lac:]
    if leaf_passes >= 1 and tail_rows and len(tail_rows) <= len(messages):
        n_msgs = len(messages)
        n_tail = len(tail_rows)
        stamped = []
        for off, row in enumerate(tail_rows):
            src_idx = n_msgs - (n_tail - off)
            if (
                isinstance(row, dict)
                and 0 <= src_idx < n_msgs
                and isinstance(messages[src_idx], dict)
                and messages[src_idx].get("role") == row.get("role")
                and messages[src_idx].get("tool_call_id") == row.get("tool_call_id")
                and len(messages[src_idx].get("tool_calls") or [])
                == len(row.get("tool_calls") or [])
            ):
                row = dict(row, **{"_src_idx": src_idx})
            stamped.append(row)
        tail_rows = stamped
    compressed = e._assemble_context(wm[0] if lac else None, tail_rows)
    compressed = e._sanitize_active_context_messages(compressed)
    return compressed


def _msgs(n=60):
    rows = [{"role": "system", "content": "sys"}]
    for i in range(1, n):
        rows.append(
            {"role": "user" if i % 2 else "assistant", "content": f"m{i} " + "w" * 40}
        )
    return rows


def _kept_stamps(compressed):
    return [m["_src_idx"] for m in compressed if isinstance(m, dict) and "_src_idx" in m]


def test_gate_source_uses_ge_1():
    """The live engine gate is leaf_passes >= 1 (not == 1)."""
    import inspect

    src = inspect.getsource(LCMEngine._compress_lossless)
    assert "leaf_passes >= 1" in src
    assert "leaf_passes == 1 and tail_rows" not in src


def test_multipass_two_folds_stamps_exact_suffix():
    """Two leaf passes, each removing a front chunk → the surviving tail rows are
    stamped with their TRUE original indices (end-anchored)."""
    messages = _msgs(60)
    # pass 1 folded rows 1..20, pass 2 folded rows 21..40 → working = [sys] + rows 41..59
    working = [messages[0]] + [dict(r) for r in messages[41:]]
    compressed = _stamp_and_assemble(_engine(), messages, leaf_passes=2, working_messages=working)
    stamps = _kept_stamps(compressed)
    assert stamps, "multi-pass must now stamp"
    assert stamps == list(range(41, 60))
    # content round-trip: each stamp maps to the row it copies
    for m in compressed:
        if isinstance(m, dict) and "_src_idx" in m and m.get("role") != "assistant":
            assert messages[m["_src_idx"]]["content"] == m["content"]


def test_multipass_boundary_shift_still_exact():
    """Different fold amounts per pass (front shrinks unevenly) — end-anchored
    indexing is immune to how much each pass removed."""
    messages = _msgs(50)
    for folded in (5, 17, 30, 44):
        working = [messages[0]] + [dict(r) for r in messages[folded + 1:]]
        compressed = _stamp_and_assemble(
            _engine(), messages, leaf_passes=3, working_messages=working
        )
        stamps = _kept_stamps(compressed)
        assert stamps == list(range(folded + 1, 50)), f"folded={folded}"


def test_multipass_scaffold_drop_shape():
    """Scaffold-drop + fold in the same compress(): the tail is still the original
    suffix; stamps exact."""
    messages = _msgs(40)
    # scaffold-drop removed rows 1..3, folds removed 4..25 → tail = 26..39
    working = [messages[0]] + [dict(r) for r in messages[26:]]
    compressed = _stamp_and_assemble(_engine(), messages, leaf_passes=2, working_messages=working)
    assert _kept_stamps(compressed) == list(range(26, 40))


def test_structural_guard_rejects_toolcall_mismatch():
    """A tail row whose tool_call_id doesn't match its mapped origin is NOT stamped
    (harvest will then fall to A-floor via any_stamped/partial accounting)."""
    messages = _msgs(20)
    working = [messages[0]] + [dict(r) for r in messages[10:]]
    # corrupt one working row: same role, different tool_call_id
    working[3] = dict(working[3], tool_call_id="tc-mismatch")
    compressed = _stamp_and_assemble(_engine(), messages, leaf_passes=2, working_messages=working)
    stamps = _kept_stamps(compressed)
    expected = [i for i in range(10, 20) if i != 12]  # row at origin 12 unstamped
    assert stamps == expected


def test_singlepass_unchanged():
    """Single-pass behavior byte-identical to PR #110."""
    messages = _msgs(30)
    working = [messages[0]] + [dict(r) for r in messages[12:]]
    compressed = _stamp_and_assemble(_engine(), messages, leaf_passes=1, working_messages=working)
    assert _kept_stamps(compressed) == list(range(12, 30))


def test_last_leaf_passes_exposed():
    e = _engine()
    _stamp_and_assemble(e, _msgs(20), leaf_passes=2,
                        working_messages=[_msgs(20)[0]] + [dict(r) for r in _msgs(20)[10:]])
    assert e.last_leaf_passes == 2
