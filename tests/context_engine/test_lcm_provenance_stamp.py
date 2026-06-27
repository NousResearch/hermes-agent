"""Option B engine-side provenance stamp (2026-06-27, P1-B).

The LCM engine stamps ``_src_idx`` (origin index into the original ``messages``)
onto each tail row handed to ``_assemble_context``, ONLY on a single-pass compaction
(the 1:1 messages→working_messages mapping holds only then). This guards:
  * the stamp survives the assembly + final sanitize pipeline onto ``compressed``;
  * every stamped kept row maps back to the same-role origin row;
  * multi-pass (rebuilt working_messages) does NOT stamp (would mis-index).
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
    e._pending_context_anchor_messages = None

    class _DAG:
        def get_session_nodes(self, sid):
            return []

        def get_uncondensed_at_depth(self, sid, d):
            return []

    e._dag = _DAG()
    return e


def _stamp_and_assemble(e, messages, *, leaf_passes, working_messages=None):
    """Replicate the compress() Step-7 stamp + assemble + final-sanitize slice.

    Stamp uses SUFFIX-from-end indexing (the fresh tail is a suffix of the original
    messages even after a front-fold), guarded to single-pass + per-row identity."""
    wm = working_messages if working_messages is not None else list(messages)
    lac = e._leading_anchor_count(wm)
    tail_rows = wm[lac:]
    if leaf_passes == 1 and tail_rows and len(tail_rows) <= len(messages):
        n_msgs = len(messages)
        n_tail = len(tail_rows)
        stamped = []
        for off, row in enumerate(tail_rows):
            src_idx = n_msgs - (n_tail - off)
            if (isinstance(row, dict) and 0 <= src_idx < n_msgs
                    and isinstance(messages[src_idx], dict)
                    and messages[src_idx].get("role") == row.get("role")):
                row = dict(row, **{"_src_idx": src_idx})
            stamped.append(row)
        tail_rows = stamped
    compressed = e._assemble_context(wm[0] if lac else None, tail_rows)
    compressed = e._sanitize_active_context_messages(compressed)
    return compressed


def _msgs(n=60):
    out = [{"role": "system", "content": "You are Apollo."}]
    for i in range(n):
        out.append({"role": "user" if i % 2 else "assistant", "content": f"m{i} " + ("w" * 40)})
    return out


def test_stamp_survives_assembly_and_sanitize():
    e = _engine()
    messages = _msgs(60)
    compressed = _stamp_and_assemble(e, messages, leaf_passes=1)
    kept = [m for m in compressed if m.get("role") != "system" and not m.get("_lcm_summary")]
    with_key = [m for m in kept if "_src_idx" in m]
    assert with_key, "stamp must survive onto compressed kept rows"
    assert len(with_key) == len(kept), "every real kept row carries _src_idx"
    # every src_idx maps to the same-role origin row
    assert all(messages[m["_src_idx"]].get("role") == m.get("role") for m in with_key)


def test_stamp_engages_after_front_fold():
    """The Greptile #110 fix: a real single-pass leaf-fold removes a chunk from the
    FRONT, so working_messages is shorter than messages — but the fresh tail is still
    a SUFFIX of the original. Suffix-from-end indexing must still stamp + map exactly."""
    e = _engine()
    messages = _msgs(120)
    # simulate a front-fold + ingest SHALLOW-COPY: anchor + COPIES of rows after a
    # 60-row folded chunk (ingest copies rows, so working rows are NOT identity-equal
    # to messages — the role-guard, not `is`, must let the stamp fire). Greptile #110.
    working = [dict(messages[0])] + [dict(m) for m in messages[61:]]
    compressed = _stamp_and_assemble(e, messages, leaf_passes=1, working_messages=working)
    kept = [m for m in compressed if m.get("role") != "system" and not m.get("_lcm_summary")]
    with_key = [m for m in kept if "_src_idx" in m]
    assert with_key, "stamp MUST engage after a front-fold (was the Greptile bug)"
    assert len(with_key) == len(kept)
    # every src_idx maps to the exact same-role original row
    assert all(messages[m["_src_idx"]].get("role") == m.get("role") for m in with_key)
    # and they are the trailing rows of messages
    assert max(m["_src_idx"] for m in with_key) == len(messages) - 1


def test_multipass_does_not_stamp():
    """Multi-pass (leaf_passes>1) → no stamp (the suffix assumption isn't guaranteed
    across multiple folds; fall back to A-floor)."""
    e = _engine()
    messages = _msgs(60)
    working = messages[:1] + messages[20:]
    compressed = _stamp_and_assemble(e, messages, leaf_passes=2, working_messages=working)
    kept = [m for m in compressed if m.get("role") != "system" and not m.get("_lcm_summary")]
    assert all("_src_idx" not in m for m in kept), "multi-pass must NOT stamp"
