"""Tests for hermes_cli.partial_compress — the pure split/parse helpers
behind ``/compress here [N]`` (boundary-aware "summarize up to here").

Inspired by Claude Code's Rewind "Summarize up to here" action.
"""

from hermes_cli.partial_compress import (
    DEFAULT_KEEP_LAST,
    MAX_KEEP_LAST,
    parse_partial_compress_args,
    rejoin_compressed_head_and_tail,
    split_history_for_partial_compress,
)


def _history(n_pairs: int) -> list[dict[str, str]]:
    """Build n_pairs of (user, assistant) exchanges."""
    h: list[dict[str, str]] = []
    for i in range(n_pairs):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


# ── parse_partial_compress_args ──────────────────────────────────────


def test_empty_args_is_full_compress():
    partial, keep, focus = parse_partial_compress_args("")
    assert partial is False
    assert keep == DEFAULT_KEEP_LAST
    assert focus is None


def test_here_defaults_keep_last():
    partial, keep, focus = parse_partial_compress_args("here")
    assert partial is True
    assert keep == DEFAULT_KEEP_LAST
    assert focus is None


def test_here_with_count():
    partial, keep, focus = parse_partial_compress_args("here 4")
    assert partial is True
    assert keep == 4
    assert focus is None


def test_up_to_here_alias():
    partial, keep, focus = parse_partial_compress_args("up to here 3")
    assert partial is True
    assert keep == 3
    assert focus is None


def test_keep_flag_forms():
    for arg in ("--keep 5", "-k 5", "--keep=5"):
        partial, keep, focus = parse_partial_compress_args(arg)
        assert partial is True, arg
        assert keep == 5, arg
        assert focus is None, arg


def test_focus_topic_when_not_boundary_form():
    partial, keep, focus = parse_partial_compress_args("database schema")
    assert partial is False
    assert focus == "database schema"


def test_here_count_clamped_low_and_high():
    _, keep_low, _ = parse_partial_compress_args("here 0")
    assert keep_low == 1
    _, keep_high, _ = parse_partial_compress_args(f"here {MAX_KEEP_LAST + 50}")
    assert keep_high == MAX_KEEP_LAST


def test_here_garbage_count_falls_back_to_default():
    partial, keep, focus = parse_partial_compress_args("here lots")
    assert partial is True
    assert keep == DEFAULT_KEEP_LAST


# ── split_history_for_partial_compress ───────────────────────────────


def test_split_keeps_last_n_exchanges():
    h = _history(5)  # 10 messages: u0 a0 u1 a1 u2 a2 u3 a3 u4 a4
    head, tail = split_history_for_partial_compress(h, keep_last=2)
    # Keep last 2 user-starts → tail begins at u3 (index 6).
    assert tail == h[6:]
    assert head == h[:6]
    # Tail must begin on a user turn (role-alternation safety).
    assert tail[0]["role"] == "user"


def test_split_default_keep():
    h = _history(4)  # 8 messages
    head, tail = split_history_for_partial_compress(h, keep_last=DEFAULT_KEEP_LAST)
    assert tail[0]["role"] == "user"
    assert head + tail == h
    assert len(head) > 0


def test_split_tail_always_starts_on_user():
    # Tool messages interleaved — tail must still snap to a user turn.
    h = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a1b"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    head, tail = split_history_for_partial_compress(h, keep_last=1)
    assert tail[0]["role"] == "user"
    assert tail[0]["content"] == "u2"
    assert head + tail == h


def test_split_degenerate_returns_no_tail():
    # keep_last larger than the number of exchanges → nothing to compress.
    h = _history(2)  # 4 messages, 2 user turns
    head, tail = split_history_for_partial_compress(h, keep_last=5)
    # Boundary lands at the first user turn → head empty → signal full.
    assert tail == []
    assert head == h


def test_split_empty_history():
    head, tail = split_history_for_partial_compress([], keep_last=2)
    assert head == []
    assert tail == []


def test_split_rejoin_preserves_all_messages():
    h = _history(6)
    head, tail = split_history_for_partial_compress(h, keep_last=3)
    assert head + tail == h


# ── rejoin_compressed_head_and_tail (seam-alternation guard) ─────────


def _roles(msgs):
    return [m["role"] for m in msgs if m["role"] in ("user", "assistant")]


def _no_consecutive_dupes(msgs):
    r = _roles(msgs)
    return all(r[i] != r[i + 1] for i in range(len(r) - 1))


def test_rejoin_valid_seam_assistant_then_user():
    # Normal case: head ends on assistant, tail starts on user → valid.
    head = [{"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "ack"}]
    tail = [{"role": "user", "content": "next"},
            {"role": "assistant", "content": "reply"}]
    out = rejoin_compressed_head_and_tail(head, tail)
    assert out == head + tail
    assert _no_consecutive_dupes(out)


def test_rejoin_user_user_seam_merges():
    # Degenerate head ending on a user summary; tail starts on user.
    head = [{"role": "user", "content": "[summary of head]"}]
    tail = [{"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "answer"}]
    out = rejoin_compressed_head_and_tail(head, tail)
    assert _no_consecutive_dupes(out), out
    # The two user messages were merged into one.
    assert out[0]["content"] == "[summary of head]\n\nlatest question"
    assert out[1] == {"role": "assistant", "content": "answer"}


def test_rejoin_assistant_assistant_seam_merges():
    head = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "head end"}]
    tail = [{"role": "assistant", "content": "tail start"},
            {"role": "user", "content": "u"}]
    out = rejoin_compressed_head_and_tail(head, tail)
    assert _no_consecutive_dupes(out), out
    assert out[-2]["content"] == "head end\n\ntail start"


def test_rejoin_empty_tail_returns_head():
    head = [{"role": "user", "content": "x"}]
    assert rejoin_compressed_head_and_tail(head, []) == head


def test_rejoin_tool_seam_left_alone():
    # tool->tool is the one legal repetition; don't merge.
    head = [{"role": "user", "content": "q"}, {"role": "tool", "content": "t1"}]
    tail = [{"role": "user", "content": "u"}]
    out = rejoin_compressed_head_and_tail(head, tail)
    assert out == head + tail
