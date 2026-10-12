"""Algorithmic reproduction and regression for issue #61932.

After several in-place compactions a tool-heavy session can be short enough
that nearly every remaining message sits inside the protected recent tail,
yet those messages are huge completed ``read_file`` / tool outputs.  The
middle compress window is then empty or tiny, preflight makes no material
token progress, and the turn dies with::

    Context length exceeded (174,833 tokens). Cannot compress further.

This is the core compressor contract — not Desktop/Windows-specific.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _MAX_TAIL_MESSAGE_FLOOR,
    _PRESSURE_KEEP_RECENT_MESSAGES,
    _summarize_tool_result,
    _tool_content_has_images,
)
from agent.model_metadata import estimate_messages_tokens_rough
from agent.prompt_builder import steer_user_row
from agent.turn_context import compression_made_progress


def _unique_tool_pair(i: int, chars: int) -> list[dict]:
    """Assistant tool_call + unique tool result (no dedupe shortcut)."""
    body = f"FILE_{i}_START\n" + (f"line {i} unique payload " * (chars // 22))[:chars]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path":"f{i}.py"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": body,
        },
    ]


def _already_compacted_session(
    *,
    n_pairs: int,
    tool_chars: int,
    user_chars: int,
) -> list[dict]:
    """Shape after multiple in-place compactions: head + handoff + heavy tail."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Investigate thoroughly"},
        {"role": "assistant", "content": "OK"},
        {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
                + ("Prior findings. " * 200)
            ),
        },
        {"role": "assistant", "content": "Continuing from compacted context."},
    ]
    for i in range(n_pairs):
        msgs.extend(_unique_tool_pair(i, tool_chars))
    msgs.append(
        {
            "role": "user",
            "content": "Full structured report:\n" + ("U" * user_chars),
        }
    )
    return msgs


@pytest.fixture()
def compressor_128k():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=128_000,
    ):
        c = ContextCompressor(
            model="openai-codex/gpt-test",
            threshold_percent=0.50,
            summary_target_ratio=0.20,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            config_context_length=128_000,
        )
    c._generate_summary = lambda *a, **k: "compact summary of earlier investigation"
    return c


class TestProtectedTailPressure61932:



    def test_compress_escapes_cannot_compress_further_dead_end(
        self, compressor_128k
    ):
        """Full compress path must materially reduce an over-context tail.

        Reproduces the #61932 failure class: multipass compression previously
        dropped a couple of message rows while leaving ~170k tokens intact,
        then reported no further progress.
        """
        c = compressor_128k
        msgs = _already_compacted_session(
            n_pairs=4, tool_chars=200_000, user_chars=80_000
        )
        rough0 = estimate_messages_tokens_rough(msgs)
        assert rough0 > c.context_length

        cur = msgs
        tok = rough0
        last_progress = False
        for _pass in range(3):
            o_len, o_tok = len(cur), tok
            out = c.compress(list(cur), current_tokens=tok)
            n_tok = estimate_messages_tokens_rough(out)
            last_progress = compression_made_progress(
                o_len, len(out), o_tok, n_tok
            )
            cur, tok = out, n_tok
            if n_tok < c.threshold_tokens and n_tok < c.context_length:
                break

        assert tok < c.context_length, (
            f"still over context after compression: {tok:,} >= {c.context_length:,}"
        )
        assert tok < rough0 * 0.5, (
            f"compression did not reclaim enough headroom: {rough0:,} → {tok:,}"
        )
        # Either we recovered under threshold, or the last pass still made
        # progress (never a pure no-op dead-end above the window).
        assert tok < c.threshold_tokens or last_progress

    def test_all_oversized_tail_dead_end_shape_now_compresses(
        self, compressor_128k
    ):
        """Exact #61932 dead-end: the protected tail ALONE holds everything.

        Head (3 messages) + an 8-message tail of exclusively oversized tool
        pairs.  The tail token budget + the ``_MAX_TAIL_MESSAGE_FLOOR`` (8)
        floor protect every non-head message, so ``compress_start >=
        compress_end`` — pre-fix ``compress()`` returned the transcript
        UNCHANGED, incremented ``_ineffective_compression_count``, and the
        retry loop died with "Cannot compress further".  Post-fix the Phase-1
        pressure pass demotes the oversized tool bodies even though the
        summary window is empty, so the same call materially shrinks the
        transcript below the context window.
        """
        c = compressor_128k
        msgs: list[dict] = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Investigate thoroughly"},
            {"role": "assistant", "content": "OK"},
        ]
        for i in range(4):
            msgs.extend(_unique_tool_pair(i, 200_000))
        assert len(msgs) == 11  # 3 head + 8-message all-oversized tail

        before = estimate_messages_tokens_rough(msgs)
        assert before > c.context_length, "fixture must start over-context"

        out = c.compress(list(msgs), current_tokens=before)
        after = estimate_messages_tokens_rough(out)

        # The dead-end is broken: one pass reclaims the bulk of the tail.
        assert after < c.context_length, (
            f"still over context: {after:,} >= {c.context_length:,}"
        )
        assert after < before * 0.25, (
            f"expected the oversized tail to demote: {before:,} → {after:,}"
        )

        # tool_call/tool_result pairing must survive demotion — never orphan
        # a tool result or a tool call (provider 400s otherwise).  Whole
        # pairs may legitimately be summarized away together.
        call_ids = {
            tc["id"]
            for m in out
            if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
            if isinstance(tc, dict)
        }
        tool_result_ids = [
            m.get("tool_call_id") for m in out if m.get("role") == "tool"
        ]
        assert tool_result_ids, "expected surviving tool pairs in the tail"
        for rid in tool_result_ids:
            assert rid in call_ids, f"orphaned tool result {rid!r}"
        for cid in call_ids:
            assert cid in tool_result_ids, f"orphaned tool call {cid!r}"


def _terminal_call(call_id: str, command: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": command})},
    }


def _terminal_result(call_id: str, tag: str, lines: int) -> dict:
    """A multi-line terminal result in the tool's own JSON shape (newlines escaped)."""
    output = "\n".join(
        f"node-{tag}-{i:03d} cpu={i % 97} lease={tag.upper()}{i:06X} zone=eu-west-2a status=HEALTHY "
        f"mem=41% disk=73% uptime=31d kernel=6.8.0-45 owner=okafor rack=R-{i % 40:02d} note=nominal"
        for i in range(lines)
    )
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"output": output, "exit_code": 0})}


def _image_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "vision_analyze", "arguments": json.dumps({"image_url": f"shot-{call_id}.png"})},
    }


def _image_result(call_id: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [
            {"type": "text", "text": f"Image {call_id}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        ],
    }


@pytest.mark.parametrize("steers", [0, 1, 2], ids=["round_last", "steer_after_round", "two_steers_after_round"])
def test_mid_turn_compaction_keeps_the_pending_tool_round_verbatim(steers):
    """Regression: compaction after a tool round stubbed the output the model had just asked for.

    Lean mode's 10K tail budget puts pass 4's soft ceiling at 15K tokens. Long user messages (no pass
    may shrink them) filled it, so the last resort stubbed the pending round. The model has not read
    that round and would re-run the command or answer blind, so it must survive verbatim while older
    rounds still give way to the budget. A /steer sent during the tools lands as a user row after the
    round before preflight compaction runs; the round is still unread then. Two steers can land in one
    iteration: one drained when the tool batch ends, another drained before the next request and
    inserted right after the newest tool result, so the round is followed by two steer rows.
    """
    ctx = 272_000
    with patch("agent.context_compressor.get_model_context_length", return_value=ctx):
        c = ContextCompressor(
            model="test/model", threshold_percent=0.50, protect_first_n=3, protect_last_n=8,
            quiet_mode=True, config_context_length=ctx,
        )
    c._generate_summary = lambda *a, **k: "compact summary of earlier turns"
    prose = ("the quarterly fleet review keeps drifting between regions and owners " * 360)[:24_000]
    msgs: list[dict] = [{"role": "system", "content": "You are Hermes."}]
    for t in range(13):
        msgs += [
            {"role": "user", "content": f"{prose}\nRun probe {t} and name the hottest node."},
            {"role": "assistant", "content": None, "tool_calls": [_terminal_call(f"old_{t}", f"python3 probe.py c{t} 3")]},
            _terminal_result(f"old_{t}", f"o{t}", 70),
            {"role": "assistant", "content": f"node-o{t}-000 is the hottest."},
        ]
    # The pending call's own args (well over pass 3's 500-char floor) belong to the unread round too.
    long_command = "python3 probe.py lyra 1 " + " ".join(f"--node node-nb-{i:03d}" for i in range(120))
    pending_calls = [_terminal_call("new_a", "python3 probe.py indus 3"), _terminal_call("new_b", long_command)]
    msgs += [
        {"role": "user", "content": "Run both probes and compare them."},
        {"role": "assistant", "content": None, "tool_calls": pending_calls},
        _terminal_result("new_a", "na", 70),
        # Last and bigger than the soft ceiling on its own, still well inside the hard window share.
        _terminal_result("new_b", "nb", 420),
    ]
    pending = {m["tool_call_id"]: m["content"] for m in msgs[-2:]}
    previous = msgs[-6]
    for text in ["Also flag any node above 90% cpu.", "And list the racks they sit in."][:steers]:
        msgs.append(steer_user_row(text))
    assert previous["tool_call_id"] == "old_12"

    out = c.compress(list(msgs), current_tokens=estimate_messages_tokens_rough(msgs))

    by_id = {m.get("tool_call_id"): m.get("content") for m in out if m.get("role") == "tool"}
    for call_id, content in pending.items():
        assert by_id.get(call_id) == content, f"pending result {call_id} was not kept verbatim: {by_id.get(call_id)!r:.120}"
    owning = [m for m in out if m.get("role") == "assistant" and m.get("tool_calls")]
    assert owning[-1]["tool_calls"] == pending_calls, "the pending round's tool-call args were truncated"
    # The budget still binds older rounds: the previous turn's result does not survive verbatim.
    assert by_id.get("old_12") != previous["content"]


def test_compaction_keeps_four_images_in_a_spared_pending_round(compressor_128k):
    pending_ids = [f"new_{i}" for i in range(4)]
    msgs = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Inspect the first image."},
        {"role": "assistant", "content": None, "tool_calls": [_image_call("old")]},
        _image_result("old"),
        {"role": "assistant", "content": "The first image is clear."},
        {"role": "user", "content": "Inspect these four images."},
        {"role": "assistant", "content": None, "tool_calls": [_image_call(call_id) for call_id in pending_ids]},
        *[_image_result(call_id) for call_id in pending_ids],
        steer_user_row("Compare their labels."),
        steer_user_row("Check the colors too."),
    ]

    out = compressor_128k.compress(msgs, current_tokens=estimate_messages_tokens_rough(msgs))

    results = {m["tool_call_id"]: m for m in out if m.get("role") == "tool"}
    assert [_tool_content_has_images(results[call_id]["content"]) for call_id in pending_ids] == [True] * 4
    assert not _tool_content_has_images(results["old"]["content"])


def _heavy_round_transcript(trailer: dict | None) -> list[dict]:
    """Older terminal rounds plus one unread round, optionally followed by a non-assistant row."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Run the fleet probes."},
    ]
    for t in range(12):
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [_terminal_call(f"old_{t}", f"python3 probe.py c{t} 3")]},
            _terminal_result(f"old_{t}", f"o{t}", 60),
        ]
    msgs += [
        {"role": "assistant", "content": None, "tool_calls": [_terminal_call("new_a", "cat big.json")]},
        _terminal_result("new_a", "na", 375),
    ]
    if trailer is not None:
        msgs.append(trailer)
    return msgs


def _pressure_compressor_200k():
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        c = ContextCompressor(model="test/model", quiet_mode=True, config_context_length=200_000)
    c.context_length = 200_000
    c.max_tokens = 32_000
    return c


@pytest.mark.parametrize(
    "trailer_kind",
    ["none", "notice_user_row", "steer_row"],
    ids=["round_last", "notice_after_round", "steer_after_round"],
)
def test_pressure_spares_pending_round_behind_a_non_steer_row(trailer_kind):
    """Regression (#137105): a background-process notice lands as a plain user row after the unread
    tool round. Only trailing steer rows were skipped, so the round read as answered and the pressure
    pass's last resort replaced the newest, still-unread result with a one-line stub. Rows after the
    round do not answer it — only an assistant row does."""
    c = _pressure_compressor_200k()
    trailers = {
        "none": None,
        "notice_user_row": {"role": "user", "content": "[IMPORTANT: Background process proc_x completed (exit code 0).]"},
        "steer_row": steer_user_row("Also flag any node above 90% cpu."),
    }
    msgs = _heavy_round_transcript(trailers[trailer_kind])
    pending = next(m["content"] for m in msgs if m.get("tool_call_id") == "new_a")

    out, _ = c._prune_old_tool_results(msgs, protect_tail_count=20, protect_tail_tokens=5_000)

    kept = next(m for m in out if m.get("tool_call_id") == "new_a")
    assert kept["content"] == pending, f"unread round was stubbed: {kept['content']!r:.90}"


def test_pressure_may_stub_the_round_once_an_assistant_row_answered_it():
    """The fix must not over-protect: once an assistant row has answered the round, the result is
    history the model already read, and the pressure pass's last resort may stub it again."""
    c = _pressure_compressor_200k()
    msgs = _heavy_round_transcript({"role": "assistant", "content": "node-na-000 is the hottest."})

    out, _ = c._prune_old_tool_results(msgs, protect_tail_count=20, protect_tail_tokens=5_000)

    answered = next(m for m in out if m.get("tool_call_id") == "new_a")
    assert answered["content"].startswith("[terminal] ran `cat big.json` -> exit 0, 375 lines output")


def test_terminal_stub_counts_output_lines_not_envelope_lines():
    """Regression (#137105): the terminal result is a JSON envelope whose ``output`` newlines are
    escaped, so counting the envelope always reported "1 lines output" for any real output."""
    stub = _summarize_tool_result(
        "terminal",
        json.dumps({"command": "cat big.json"}),
        json.dumps({"output": "\n".join(f"line-{i}" for i in range(2000)), "exit_code": 0}),
    )
    assert stub == "[terminal] ran `cat big.json` -> exit 0, 2000 lines output"
    # Malformed historical content keeps the envelope count instead of failing compression.
    assert _summarize_tool_result("terminal", '{"command": "ls"}', "raw non-json output\nsecond line") == (
        "[terminal] ran `ls` -> exit ?, 2 lines output"
    )
