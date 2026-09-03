"""Unit tests for relay-level empty tool_calls stripping.

Strict OpenAI-compatible providers (DeepSeek v4, Console Go) reject
``tool_calls: []`` with HTTP 400 "Invalid 'messages[N].tool_calls': empty
array". Empty arrays reach the final relay chokepoint from session-resume /
compression rebuild paths that re-materialize assistant turns with a
vacuous tool_calls key. ``_strip_empty_tool_calls`` removes the key in
place; non-empty lists are never touched.

Self-contained on purpose: tests only the relay helper, no other layer.
"""

from agent.relay_llm import _strip_empty_tool_calls


def _assistant(content="hi", tool_calls=None, **extra):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    msg.update(extra)
    return msg


def test_non_dict_returns_zero():
    assert _strip_empty_tool_calls(None) == 0
    assert _strip_empty_tool_calls("nope") == 0
    assert _strip_empty_tool_calls(42) == 0


def test_empty_request_returns_zero():
    assert _strip_empty_tool_calls({}) == 0


def test_empty_tool_calls_is_stripped():
    msg = _assistant(tool_calls=[])
    req = {"messages": [msg]}
    assert _strip_empty_tool_calls(req) == 1
    assert "tool_calls" not in msg


def test_non_empty_tool_calls_preserved():
    tc = [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    msg = _assistant(tool_calls=tc)
    req = {"messages": [msg]}
    assert _strip_empty_tool_calls(req) == 0
    assert req["messages"][0]["tool_calls"] == tc


def test_moa_prepared_request_messages_stripped():
    msgs = [_assistant(tool_calls=[])]
    req = {"messages": [], "_moa_prepared_request": {"messages": msgs}}
    assert _strip_empty_tool_calls(req) == 1
    assert "tool_calls" not in msgs[0]


def test_aliased_list_stripped_once():
    msgs = [_assistant(tool_calls=[])]
    req = {"messages": msgs, "_moa_prepared_request": {"messages": msgs}}
    assert _strip_empty_tool_calls(req) == 1  # id-dedup: only once
    assert "tool_calls" not in msgs[0]


def test_non_list_tool_calls_are_stripped():
    """Strings / None tool_calls are stripped too (they'd also 400).

    Implementation semantics: only a non-empty list is preserved;
    ``[]``, strings, and ``None`` are all popped.
    """
    msgs = [
        _assistant(tool_calls="not-a-list"),
        {"role": "assistant", "content": "hi", "tool_calls": None},  # 显式建键
    ]
    req = {"messages": msgs}
    assert _strip_empty_tool_calls(req) == 2
    assert "tool_calls" not in msgs[0]
    assert "tool_calls" not in msgs[1]


def test_non_assistant_and_plain_messages_untouched():
    """User-role messages keep their tool_calls; plain strings don't crash."""
    req = {"messages": [
        {"role": "user", "content": "hi", "tool_calls": []},
        "plain-string-message",
    ]}
    assert _strip_empty_tool_calls(req) == 0
    assert "tool_calls" in req["messages"][0]  # user role untouched


if __name__ == "__main__":
    # Local runner (venv has no pytest): python tests/agent/test_relay_strip_empty_tool_calls.py
    import sys
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{sum(1 for n in globals() if n.startswith('test_')) - failed}/{sum(1 for n in globals() if n.startswith('test_'))} passed")
    sys.exit(1 if failed else 0)
