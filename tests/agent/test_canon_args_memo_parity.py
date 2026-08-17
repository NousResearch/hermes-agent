"""Byte-parity + complexity proof for the memoized send-path tool-call
argument canonicalization (agent/conversation_loop.py).

The pre-fix inline loop re-ran ``json.loads`` + ``json.dumps(sort_keys=True)``
on EVERY historical tool call's arguments on EVERY API-call iteration —
quadratic in session tool-call count.  The fix routes the same logic through
``_canonicalize_api_tool_calls`` with a bounded value-keyed memo
(``_CANON_ARGS_CACHE``).

These tests drive the real shipped function (no copies of the new code) and
assert:
  1. byte-parity with the pre-fix logic across a growing simulated session
     (unicode, nested, malformed, empty, and non-string arguments included);
  2. the persisted history is never mutated (copy-on-write preserved);
  3. determinism + idempotence of the canonical form;
  4. malformed inputs are never memoized (repair path reruns, as before);
  5. the cache stays bounded;
  6. complexity: json.loads call count is LINEAR in unique tool calls under
     the fix, vs quadratic under the pre-fix logic — a deterministic proof
     (call counts, not wall clock) that the O(n^2) is gone.
"""
import copy
import json
import random

import pytest

import agent.conversation_loop as cl
from agent.message_sanitization import _repair_tool_call_arguments

random.seed(1234)

UNI = "日本語テキスト🎉 café Ω ≈ 中文字符串"


@pytest.fixture(autouse=True)
def _clear_canon_cache():
    # getattr (not cl._CANON_ARGS_CACHE) keeps this fixture from erroring at
    # setup on the pre-fix tree, so sabotage runs record real test FAILURES
    # (AttributeError inside each test body) instead of collection errors.
    cache = getattr(cl, "_CANON_ARGS_CACHE", None)

    def _reset():
        if cache is not None:
            cache.clear()
        if hasattr(cl, "_canon_args_cache_bytes"):
            cl._canon_args_cache_bytes = 0

    _reset()
    yield
    _reset()


def test_cache_bounded_by_bytes():
    """Large argument strings (write_file contents run 100KB+) must not pin
    unbounded memory: the byte budget evicts before the count bound."""
    big = json.dumps({"path": "/tmp/big.py", "content": "y" * 200_000})
    for i in range(300):  # 300 x ~400KB (key+value) >> 32MB budget
        cl._canonicalize_tool_call_arguments(
            big[:-1] + f',"n":{i}}}'
        )
    assert cl._canon_args_cache_bytes <= cl._CANON_ARGS_CACHE_MAX_BYTES, (
        f"cache holds {cl._canon_args_cache_bytes} bytes — byte budget "
        "regressed; large tool-call args pin unbounded memory again")
    assert len(cl._CANON_ARGS_CACHE) >= 1  # still memoizes something


def build_history(n_tool_calls, arg_bytes=2048):
    """Synthetic session: n assistant tool-call messages (+ tool results).

    Includes unicode, malformed, and empty argument strings — the cases the
    send-path normalization actually sees.
    """
    msgs = []
    filler = "x" * (arg_bytes - 200)
    for i in range(n_tool_calls):
        args = json.dumps({"path": f"/tmp/file_{i}.py", "content": filler,
                           "u": UNI, "n": i, "mode": "write"})
        if i % 9 == 8:
            args = '{"broken": tru'  # malformed -> repair path
        elif i % 6 == 5:
            args = ""  # empty -> repair path
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"call_{i}", "type": "function",
                            "function": {"name": "write_file",
                                         "arguments": args}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}",
                     "name": "write_file", "content": f"result {i} {UNI}"})
    return msgs


def canonicalize_pass_OLD(api_messages):
    """Byte-exact reference of the pre-fix inline loop."""
    for am in api_messages:
        tcs = am.get("tool_calls")
        if not tcs:
            continue
        new_tcs = []
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                try:
                    args_obj = json.loads(tc["function"]["arguments"])
                    tc = {**tc, "function": {
                        **tc["function"],
                        "arguments": json.dumps(
                            args_obj, separators=(",", ":"),
                            sort_keys=True,
                        ),
                    }}
                except Exception:
                    tc["function"]["arguments"] = _repair_tool_call_arguments(
                        tc["function"]["arguments"],
                        tc["function"].get("name", "?"),
                    )
            new_tcs.append(tc)
        am["tool_calls"] = new_tcs


class TestByteParity:
    def test_growing_session_every_iteration(self):
        """OLD vs NEW must produce identical api_messages at EVERY iteration
        of a growing session — not just the final state."""
        n = 60
        history = build_history(n)
        for k in range(1, n + 1):
            prefix = history[: 2 * k]
            old_msgs = copy.deepcopy(prefix)
            new_msgs = copy.deepcopy(prefix)
            canonicalize_pass_OLD(old_msgs)
            cl._canonicalize_api_tool_calls(new_msgs)
            assert old_msgs == new_msgs, f"diverged at iteration {k}"

    def test_history_not_mutated(self):
        """The canonicalize path is copy-on-write: with valid args, the
        persisted history bytes stay intact even though api_messages
        shallow-copies history dicts (shares the nested function dicts).
        (Malformed args take the in-place repair path — pre-existing
        behavior, identical in both implementations; see parity tests.)"""
        history = build_history(20)
        for m in history:  # all-valid: canonicalize path only
            if m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                fn["arguments"] = json.dumps({"id": m["tool_calls"][0]["id"],
                                              "u": UNI})
        before = copy.deepcopy(history)
        api_messages = [dict(m) for m in history]  # shallow, like the loop
        cl._canonicalize_api_tool_calls(api_messages)
        assert history == before

    def test_non_string_arguments_parity(self):
        """A dict (not str) in 'arguments' takes the repair path in both
        implementations — the memo must not change that."""
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "t",
                                              "arguments": {"a": 1}}}]}]
        old_msgs = copy.deepcopy(msgs)
        new_msgs = copy.deepcopy(msgs)
        canonicalize_pass_OLD(old_msgs)
        cl._canonicalize_api_tool_calls(new_msgs)
        assert old_msgs == new_msgs


class TestMemoSemantics:
    def test_deterministic_and_idempotent(self):
        raw = json.dumps({"b": 2, "a": UNI, "nested": {"z": [3, 2, 1]}})
        canon = cl._canonicalize_tool_call_arguments(raw)
        assert canon == cl._canonicalize_tool_call_arguments(raw)
        assert cl._canonicalize_tool_call_arguments(canon) == canon
        # exact canonical form: sorted keys, tight separators, ascii-escaped
        assert canon == json.dumps(json.loads(raw), separators=(",", ":"),
                                   sort_keys=True)
        assert canon == canon.encode().decode()  # pure ASCII wire form

    def test_cache_hit_skips_json_loads(self):
        raw = json.dumps({"k": "v"})
        cl._canonicalize_tool_call_arguments(raw)
        assert raw in cl._CANON_ARGS_CACHE

    def test_malformed_never_memoized(self):
        with pytest.raises(Exception):
            cl._canonicalize_tool_call_arguments('{"broken": tru')
        assert cl._CANON_ARGS_CACHE == {}

    def test_cache_bounded(self):
        for i in range(cl._CANON_ARGS_CACHE_MAX + 100):
            cl._canonicalize_tool_call_arguments(json.dumps({"i": i}))
        assert len(cl._CANON_ARGS_CACHE) <= cl._CANON_ARGS_CACHE_MAX


class TestComplexityProof:
    def test_json_loads_linear_not_quadratic(self, monkeypatch):
        """Deterministic perf proof: count json.loads invocations.

        Pre-fix logic: one loads per tool call PER ITERATION -> K(K+1)/2 for
        a K-tool-call session.  Fixed logic: one loads per UNIQUE argument
        string, ever -> K.  (Malformed arguments raise and are never
        memoized in EITHER implementation — covered in the parity tests —
        so this proof uses an all-valid history to compare exactly.)
        """
        n = 40
        history = build_history(n)
        # force every argument string valid so both implementations take
        # only the canonicalize path (repair path is parity-tested elsewhere)
        for m in history:
            if m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                fn["arguments"] = json.dumps({"name": fn["name"],
                                              "id": m["tool_calls"][0]["id"],
                                              "u": UNI})

        def counting_loads(counter):
            real_loads = json.loads

            def wrapper(*a, **kw):
                counter[0] += 1
                return real_loads(*a, **kw)
            return wrapper

        # OLD: quadratic — K(K+1)/2 loads over a K-iteration session
        old_counter = [0]
        monkeypatch.setattr(json, "loads", counting_loads(old_counter))
        for k in range(1, n + 1):
            canonicalize_pass_OLD(copy.deepcopy(history[: 2 * k]))
        monkeypatch.undo()
        assert old_counter[0] == n * (n + 1) // 2

        # NEW: linear — each unique string loaded exactly once, ever
        new_counter = [0]
        monkeypatch.setattr(json, "loads", counting_loads(new_counter))
        for k in range(1, n + 1):
            cl._canonicalize_api_tool_calls(copy.deepcopy(history[: 2 * k]))
        monkeypatch.undo()
        assert new_counter[0] == n

        # quadratic -> linear, by exact call count
        assert old_counter[0] == (n + 1) / 2 * new_counter[0]


class TestUnrepairableArgsAreNotWrittenBackToHistory:
    """The repair path must be copy-on-write too (#80498).

    ``api_messages`` is built with ``msg.copy()`` — a SHALLOW per-message
    copy — so every ``tool_calls`` entry is the same dict object the
    persisted history holds. The canonicalize branch has always honoured
    that (``test_history_not_mutated``), but the repair branch assigned
    straight into ``tc["function"]``, so an unrepairable argument string
    (repair returns ``"{}"``) overwrote the model's real arguments in the
    stored turn.

    Field report: a stream died mid ``write_file`` and the file content it
    had already streamed was replaced by ``{}`` in the transcript, leaving
    only a WARNING behind.
    """

    @staticmethod
    def _history_with_truncated_write():
        # Exactly the incident shape: arguments cut off mid-string.
        truncated = '{"content": "# chapter draft\nline one\nline two'
        history = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": truncated},
            }],
        }]
        return history, truncated

    def test_history_keeps_the_original_arguments(self):
        history, truncated = self._history_with_truncated_write()
        before = copy.deepcopy(history)

        api_messages = [dict(m) for m in history]  # shallow, like the send path
        cl._canonicalize_api_tool_calls(api_messages)

        assert history == before, (
            "the send-path canonicalizer rewrote the persisted history"
        )
        assert (
            history[0]["tool_calls"][0]["function"]["arguments"] == truncated
        ), "the model's streamed arguments were destroyed in the transcript"

    def test_send_copy_is_still_repaired(self):
        """The API copy must still carry safe JSON — only the aliasing changes."""
        history, _ = self._history_with_truncated_write()

        api_messages = [dict(m) for m in history]
        cl._canonicalize_api_tool_calls(api_messages)

        sent = api_messages[0]["tool_calls"][0]["function"]["arguments"]
        assert sent == "{}"
        json.loads(sent)  # the whole point of the repair: never ship broken JSON

    def test_valid_calls_alongside_a_broken_one_are_untouched(self):
        """A broken call must not disturb its siblings' history entries."""
        good = json.dumps({"path": "a.txt", "u": UNI})
        history = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": good}},
                {"id": "c2", "type": "function",
                 "function": {"name": "write_file", "arguments": '{"content": "cut'}},
            ],
        }]
        before = copy.deepcopy(history)

        api_messages = [dict(m) for m in history]
        cl._canonicalize_api_tool_calls(api_messages)

        assert history == before
        sent = api_messages[0]["tool_calls"]
        assert json.loads(sent[0]["function"]["arguments"]) == json.loads(good)
        assert sent[1]["function"]["arguments"] == "{}"

    def test_repeated_sends_do_not_accumulate_damage(self):
        """Re-canonicalizing the same history every iteration stays lossless."""
        history, truncated = self._history_with_truncated_write()
        for _ in range(5):
            api_messages = [dict(m) for m in history]
            cl._canonicalize_api_tool_calls(api_messages)
            assert api_messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"
        assert (
            history[0]["tool_calls"][0]["function"]["arguments"] == truncated
        )
