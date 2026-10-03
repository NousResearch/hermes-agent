"""Trim of stale ``reasoning``/``reasoning_content`` text at compaction boundaries.

DeepSeek / Kimi / MiMo enforce reasoning echo-back, so every retained assistant row's
thinking text is re-sent (and billed) on every call — 48.5% of this install's active
transcript chars were reasoning text. Only the current turn's replay needs the text
(the same rule the overflow-salvage path already applies via
``_NEWEST_TURN_ONLY_BUDGET_KEYS``), and the compaction boundary has already broken the
prompt-cache prefix, so the NORMAL compaction assembly trims it there (never on the
per-turn hot path, where a trim would start a cache miss from the first trimmed
message).

Behaviour contract pinned here: old assistant rows lose the keys, the newest turn
(everything after the last USER message — a turn spans several assistant rows) keeps
them, no other row type moves, a transcript shorter than the protected tail is
untouched, and the DeepSeek echo-family wire build still sees a valid
``reasoning_content`` on every assistant row afterwards (trimmed rows get the same
single-space pad the sanitizer already writes for reasoning-free turns).
"""

import copy

# Pre-imported at COLLECTION on purpose: ``compress()`` lazily imports ``agent.conversation_loop``
# (message-classification constants, avoiding an import cycle), and that import runs Hermes' dependency
# activation, which probes for a sealed-payload ``manifest.json`` BESIDE this checkout — under the real
# home when the checkout is a git worktree, where the suite's per-test home-I/O guard refuses the probe
# mid-test. Collection imports run before the guard fixture arms (exactly as in a full-suite run, which
# already pulls this chain in through other files), so a focused run behaves the same.
import agent.conversation_loop  # noqa: F401

from agent.context_compressor import ContextCompressor, is_compaction_summary_message
from agent.message_sanitization import apply_reasoning_content_policy, needs_reasoning_echo

THINKING_OLD = "dead-end hypothesis " * 60
THINKING_MID = "interim tool-round plan " * 40
THINKING_NEW = "live hypothesis " * 60
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}


def _thinking_assistant(content, thinking):
    msg = {"role": "assistant", "content": content}
    if thinking is not None:
        msg["reasoning"] = thinking
        msg["reasoning_content"] = thinking
    return msg


def _transcript(turns=30, size=1500):
    """Every assistant row carries thinking text; size keeps the tail cut mid-transcript."""
    messages = [{"role": "system", "content": "sys " + "z" * size}]
    for t in range(turns):
        messages.append({"role": "user", "content": f"question {t} " + "z" * size})
        messages.append(_thinking_assistant(f"answer {t} " + "z" * size, f"thinking {t} " + "t" * size))
    return messages


def _compressor():
    cc = ContextCompressor(
        model="test-model", threshold_percent=0.75, protect_first_n=1, protect_last_n=12,
        quiet_mode=True, config_context_length=40960, provider="test",
    )
    cc._generate_summary = lambda *a, **k: "Summary of earlier turns."
    return cc


def _active_turn_split(compressed):
    """(rows before the last user row, assistant rows of the active turn) in the output."""
    last_user = max(
        i for i, m in enumerate(compressed)
        if isinstance(m, dict) and m.get("role") == "user" and not is_compaction_summary_message(m)
    )
    older = [m for m in compressed[:last_user] if isinstance(m, dict) and m.get("role") == "assistant"]
    active = [
        m for m in compressed[last_user + 1:]
        if isinstance(m, dict) and m.get("role") == "assistant" and not is_compaction_summary_message(m)
    ]
    return older, active


class TestCompactionTrimsStaleThinking:
    """Through the production entry point: ``ContextCompressor.compress()``."""

    def test_old_assistant_rows_lose_thinking_text(self):
        cc = _compressor()
        messages = _transcript()
        compressed = cc.compress(messages, current_tokens=200_000, force=True)

        assert len(compressed) < len(messages), "compaction must have run"
        older, active = _active_turn_split(compressed)
        assert older, "the protected tail must carry completed earlier turns"
        for msg in older:
            assert "reasoning" not in msg, f"old assistant row kept reasoning: {msg}"
            assert "reasoning_content" not in msg, f"old assistant row kept reasoning_content: {msg}"
        assert active, "the newest turn must survive in the tail"

    def test_newest_turn_keeps_its_thinking_text(self):
        cc = _compressor()
        messages = _transcript()
        compressed = cc.compress(messages, current_tokens=200_000, force=True)

        older, active = _active_turn_split(compressed)
        assert active, "the newest turn must survive in the tail"
        assert all(m.get("reasoning_content") for m in active)
        # Verbatim, not a placeholder: the newest turn's replay is the one the next call needs.
        last_thinking = messages[-1]["reasoning_content"]
        assert active[-1]["reasoning_content"] == last_thinking

    def test_live_transcript_is_never_mutated(self):
        cc = _compressor()
        messages = _transcript()
        before = copy.deepcopy(messages)
        cc.compress(messages, current_tokens=200_000, force=True)
        assert messages == before, "compaction assembly must only ever trim its own copies"

    def test_short_transcript_shorter_than_protected_tail_is_unchanged(self):
        """Fewer messages than protect_last_n: no compaction runs, so no trim happens —
        thinking text survives on every row and the input list is returned as-is."""
        cc = ContextCompressor(
            model="test-model", threshold_percent=0.75, protect_first_n=3, protect_last_n=20,
            quiet_mode=True, config_context_length=40960, provider="test",
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question 1"},
            _thinking_assistant("answer 1", THINKING_OLD),
            {"role": "user", "content": "question 2"},
            _thinking_assistant("answer 2", THINKING_NEW),
        ]
        assert len(messages) < cc.protect_last_n
        before = copy.deepcopy(messages)
        out = cc.compress(messages, current_tokens=200_000, force=True)

        assert out == before
        assert out[2]["reasoning_content"] == THINKING_OLD
        assert messages == before


class TestDeepSeekEchoFamilyStaysValid:
    """The trim must not strand an echo-family route without its required field."""

    def test_route_is_detected_as_echo_family(self):
        assert needs_reasoning_echo("openrouter", "deepseek/deepseek-v4.1-flash", "https://openrouter.ai/api/v1")

    def test_wire_build_pads_trimmed_rows_and_keeps_live_thinking(self):
        cc = _compressor()
        compressed = cc.compress(_transcript(), current_tokens=200_000, force=True)

        wire_rows = []
        for msg in compressed:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                api_msg = {}
                apply_reasoning_content_policy(msg, api_msg, needs_thinking_pad=True)
                wire_rows.append(api_msg)

        assert wire_rows, "compaction output must still carry assistant rows"
        # Every assistant row must reach the wire with a non-empty field: DeepSeek V4 Pro
        # rejects empty string ("The reasoning content in the thinking mode must be passed back").
        for api_msg in wire_rows:
            assert isinstance(api_msg.get("reasoning_content"), str)
            assert api_msg["reasoning_content"] != ""
        padded = [m for m in wire_rows if m["reasoning_content"] == " "]
        verbatim = [m for m in wire_rows if m["reasoning_content"] != " "]
        assert padded, "trimmed rows must receive the single-space pad"
        assert verbatim, "the newest turn must carry its real thinking text"
        assert all(m["reasoning_content"].strip() for m in verbatim)


class TestTrimBoundary:
    """Unit contract of the pass itself: boundary = last USER message (a turn spans
    several assistant rows; cutting at the last assistant row would strip mid-chain)."""

    def _trim(self, messages):
        from agent.context_compressor import _prune_stale_reasoning_text

        return _prune_stale_reasoning_text(messages)

    def test_old_turn_trimmed_newest_turn_kept(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            _thinking_assistant("old answer", THINKING_OLD),
            {"role": "user", "content": "latest question"},
            _thinking_assistant("latest answer", THINKING_NEW),
        ]
        assert self._trim(messages) == 1
        assert "reasoning" not in messages[2] and "reasoning_content" not in messages[2]
        assert messages[4]["reasoning"] == THINKING_NEW
        assert messages[4]["reasoning_content"] == THINKING_NEW

    def test_multi_row_active_turn_keeps_every_row_of_its_chain(self):
        messages = [
            {"role": "user", "content": "old question"},
            _thinking_assistant("old answer", THINKING_OLD),
            {"role": "user", "content": "current question"},
            {**_thinking_assistant("", THINKING_MID), "tool_calls": [TOOL_CALL]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            _thinking_assistant("final answer", THINKING_NEW),
        ]
        assert self._trim(messages) == 1
        assert "reasoning_content" not in messages[1]
        assert messages[3]["reasoning_content"] == THINKING_MID
        assert messages[5]["reasoning_content"] == THINKING_NEW

    def test_non_assistant_rows_are_never_touched(self):
        messages = [
            {"role": "user", "content": "question", "reasoning_content": "user-side note"},
            _thinking_assistant("old answer", THINKING_OLD),
            {"role": "tool", "content": "result", "tool_call_id": "c1", "reasoning_content": "tool-side note"},
            {"role": "user", "content": "latest question"},
            _thinking_assistant("latest answer", THINKING_NEW),
        ]
        assert self._trim(messages) == 1
        assert messages[0]["reasoning_content"] == "user-side note"
        assert messages[2]["reasoning_content"] == "tool-side note"

    def test_rows_without_thinking_text_do_not_count(self):
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "plain old answer"},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "plain latest answer"},
        ]
        assert self._trim(messages) == 0

    def test_no_user_row_trims_nothing(self):
        """Fail open toward replay correctness: without a turn boundary, nothing is stripped."""
        messages = [
            _thinking_assistant("a1", THINKING_OLD),
            _thinking_assistant("a2", THINKING_NEW),
        ]
        assert self._trim(messages) == 0
        assert messages[0]["reasoning_content"] == THINKING_OLD
