"""Byte-parity + benchmark harness for the per-iteration cursor optimizations.

Drives the pure functions directly (no AIAgent):
  1. sanitize_tool_call_arguments (with/without cursor)
  2. estimate_messages_tokens_rough (memoized) vs a reference reimplementation
  3. _flush_messages_to_session_db bounded scan — simulated via a stub agent

Run: HERMES worktree venv python parity_harness.py
"""
import copy
import json
import random
import statistics
import sys
import time

sys.path.insert(0, ".")

random.seed(1234)

UNI = "日本語テキスト🎉 café Ω ≈ 中文字符串"


def build_history(n):
    """Synthetic conversation: user/assistant/tool cycles, malformed args, unicode."""
    msgs = []
    i = 0
    while len(msgs) < n:
        msgs.append({"role": "user", "content": f"question {i} {UNI} " + "x" * random.randint(10, 400)})
        if i % 3 == 0:
            args = json.dumps({"q": f"val {i}", "u": UNI, "n": i})
            if i % 9 == 0:
                args = '{"broken": tru'  # malformed
            elif i % 6 == 0:
                args = ""  # empty
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                                "function": {"name": "web_search", "arguments": args}}],
            })
            msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                         "name": "web_search", "content": f"result {i} {UNI}"})
        else:
            msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * random.randint(10, 600),
                         "reasoning_content": f"thinking {i}"})
        if i % 7 == 0 and msgs:
            msgs[-1]["content"] = [{"type": "text", "text": f"part {i}"},
                                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        i += 1
    return msgs[:n]


# ---------- reference (pre-optimization) implementations ----------

from agent.model_metadata import (
    estimate_messages_tokens_rough,
    _estimate_message_tokens_without_images,
    _count_image_tokens,
    _MSG_TOKENS_CACHE,
)


def estimate_messages_tokens_rough_OLD(messages):
    _IMAGE_TOKEN_COST = 1500
    text_tokens = 0
    image_tokens = 0
    for msg in messages:
        text_tokens += _estimate_message_tokens_without_images(msg)
        image_tokens += _count_image_tokens(msg, _IMAGE_TOKEN_COST)
    return text_tokens + image_tokens


from agent.agent_runtime_helpers import sanitize_tool_call_arguments


def simulate_compression(msgs):
    """Rewrite the middle of the history with fresh dict copies + a summary."""
    head, mid, tail = msgs[:2], msgs[2:-6], msgs[-6:]
    summary = {"role": "user", "content": "SUMMARY OF DROPPED CONTEXT " + UNI}
    new = [dict(m) if isinstance(m, dict) else m for m in head]
    new.append(summary)
    new.extend(dict(m) if isinstance(m, dict) else m for m in tail)
    msgs[:] = new


def test_parity_sanitize_cursor():
    print("=== parity: sanitize_tool_call_arguments cursor ===")
    for n in (50, 200, 500):
        base = build_history(n)
        old_list = copy.deepcopy(base)
        new_list = copy.deepcopy(base)
        cursor = {}
        for iteration in range(3):
            r_old = sanitize_tool_call_arguments(old_list)
            r_new = sanitize_tool_call_arguments(new_list, cursor=cursor)
            assert r_old == r_new, (n, iteration, r_old, r_new)
            assert old_list == new_list, f"list mismatch n={n} it={iteration}"
            # append a new exchange (one malformed) between iterations
            for lst in (old_list, new_list):
                lst.append({"role": "assistant", "content": "",
                            "tool_calls": [{"id": f"c{iteration}", "type": "function",
                                            "function": {"name": "t", "arguments": '{"bad": '}}]})
                lst.append({"role": "tool", "tool_call_id": f"c{iteration}", "content": "ok"})
            if iteration == 1:
                simulate_compression(old_list)
                simulate_compression(new_list)
        # after mutations, one more full compare
        r_old = sanitize_tool_call_arguments(old_list)
        r_new = sanitize_tool_call_arguments(new_list, cursor=cursor)
        assert r_old == r_new and old_list == new_list
        print(f"  n={n}: OK (element-wise equal across 3 iterations + compression)")


def test_parity_token_memo():
    print("=== parity: estimate_messages_tokens_rough memo ===")
    for n in (50, 200, 500):
        msgs = build_history(n)
        _MSG_TOKENS_CACHE.clear()
        for iteration in range(3):
            # simulate api_messages copies each iteration (shallow copies)
            api = [m.copy() for m in msgs]
            old = estimate_messages_tokens_rough_OLD(api)
            new = estimate_messages_tokens_rough(api)
            assert old == new, (n, iteration, old, new)
            msgs.append({"role": "user", "content": f"followup {iteration} {UNI}"})
            if iteration == 1:
                simulate_compression(msgs)
        # mutate a string in place-ish: replace content of an existing dict
        msgs[0]["content"] = "EDITED " + UNI
        api = [m.copy() for m in msgs]
        assert estimate_messages_tokens_rough_OLD(api) == estimate_messages_tokens_rough(api)
        # odd types fall through the memo
        weird = [{"role": "user", "content": {"_multimodal": True, "text_summary": "s"}},
                 {"role": "user", "content": None}, "not-a-dict",
                 {"role": "tool", "content": [{"type": "text", "text": UNI}, "raw"], "meta": (1, 2)}]
        assert estimate_messages_tokens_rough_OLD(weird) == estimate_messages_tokens_rough(weird)
        print(f"  n={n}: OK (equal across 3 iterations + compression + in-place edit + odd types)")


def test_parity_persist_bounded_scan():
    print("=== parity: _flush_messages_to_session_db bounded scan ===")
    import run_agent as ra

    class FakeDB:
        def __init__(self):
            self.rows = []
        def append_message(self, **kw):
            self.rows.append({k: copy.deepcopy(v) for k, v in kw.items()})
        def append_messages_batch(self, session_id, messages, **kw):
            for m in messages:
                row = {k: copy.deepcopy(v) for k, v in m.items()}
                row["session_id"] = session_id
                self.rows.append(row)
            return list(range(1, len(messages) + 1))

    def make_agent(bounded):
        a = ra.AIAgent.__new__(ra.AIAgent)
        a.session_id = "s1"
        a._session_db = FakeDB()
        a._session_db_created = True
        a._last_flushed_db_idx = 0
        a._flushed_db_message_ids = set()
        a._persist_disabled = False
        a._session_persist_lock = None
        if not bounded:
            # neutralize the cursor: force full scan every time
            a._db_flush_scan_prefix = None
        return a

    for n in (50, 200, 500):
        base = build_history(n)
        la, lb = copy.deepcopy(base), copy.deepcopy(base)
        A, B = make_agent(False), make_agent(True)
        for iteration in range(3):
            A._db_flush_scan_prefix = None  # baseline: always full scan
            ra_ok = A._flush_messages_to_session_db_unlocked(la, None)
            rb_ok = B._flush_messages_to_session_db_unlocked(lb, None)
            assert ra_ok is True and rb_ok is True
            assert A._session_db.rows == B._session_db.rows, f"rows diverge n={n} it={iteration}"
            assert la == lb
            for lst in (la, lb):
                lst.append({"role": "user", "content": f"turn {iteration} {UNI}"})
                lst.append({"role": "assistant", "content": f"reply {iteration}",
                            "_empty_recovery_synthetic": iteration == 0})  # scaffolding once
            if iteration == 1:
                # compression-style rewrite: fresh copies without markers
                for lst in (la, lb):
                    head = [dict(m) for m in lst[:3]]
                    for m in head:
                        m.pop(ra._DB_PERSISTED_MARKER, None)
                    tail = [dict(m) for m in lst[-4:]]
                    for m in tail:
                        m.pop(ra._DB_PERSISTED_MARKER, None)
                    lst[:] = head + [{"role": "user", "content": "SUMMARY"}] + tail
        A._db_flush_scan_prefix = None
        A._flush_messages_to_session_db_unlocked(la, None)
        B._flush_messages_to_session_db_unlocked(lb, None)
        assert A._session_db.rows == B._session_db.rows and la == lb
        print(f"  n={n}: OK (identical DB rows + marker stamps across 3 flushes + compression rewrite)")


def bench():
    print("=== benchmarks (median of 5, per call) ===")

    def timeit(fn, reps=5):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3  # ms

    for n in (50, 200, 500):
        msgs = build_history(n)
        sanitize_tool_call_arguments(msgs)  # settle repairs first

        # sanitize: old (no cursor) vs new (warm cursor)
        old_ms = timeit(lambda: sanitize_tool_call_arguments(msgs))
        cur = {}
        sanitize_tool_call_arguments(msgs, cursor=cur)
        new_ms = timeit(lambda: sanitize_tool_call_arguments(msgs, cursor=cur))

        # tokens: old walk vs warm memo (on fresh shallow copies, like api_messages)
        api = [m.copy() for m in msgs]
        told = timeit(lambda: estimate_messages_tokens_rough_OLD([m.copy() for m in msgs]))
        _MSG_TOKENS_CACHE.clear()
        estimate_messages_tokens_rough([m.copy() for m in msgs])  # warm
        tnew = timeit(lambda: estimate_messages_tokens_rough([m.copy() for m in msgs]))

        # persist scan: fully-flushed list, old full walk vs bounded skip
        import run_agent as ra
        flushed = copy.deepcopy(msgs)
        for m in flushed:
            if isinstance(m, dict):
                m[ra._DB_PERSISTED_MARKER] = True

        def old_scan():
            for _idx, m in enumerate(flushed):
                if not isinstance(m, dict):
                    continue
                if ra._is_ephemeral_scaffolding(m):
                    continue
                if m.get(ra._DB_PERSISTED_MARKER):
                    continue

        prefix = flushed[:]

        def new_scan():
            s = 0
            lim = min(len(prefix), len(flushed))
            while s < lim and flushed[s] is prefix[s]:
                s += 1
            for _idx in range(s, len(flushed)):
                m = flushed[_idx]
                if not isinstance(m, dict):
                    continue
                if ra._is_ephemeral_scaffolding(m):
                    continue
                if m.get(ra._DB_PERSISTED_MARKER):
                    continue

        pold = timeit(old_scan)
        pnew = timeit(new_scan)

        print(f"  n={n:3d}: sanitize {old_ms:.3f}ms -> {new_ms:.3f}ms | "
              f"tokens {told:.3f}ms -> {tnew:.3f}ms | "
              f"persist-scan {pold*1000:.1f}us -> {pnew*1000:.1f}us")


if __name__ == "__main__":
    test_parity_sanitize_cursor()
    test_parity_token_memo()
    test_parity_persist_bounded_scan()
    bench()
    print("ALL PARITY CHECKS PASSED")
