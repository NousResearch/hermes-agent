# Fix: Context Compression Duplicate Messages

## Problem

Session hygiene (auto-compression) introduces duplicate assistant messages into the transcript. Each subsequent compression amplifies the duplication — a cascading bug.

**User-facing symptom**: Multiple identical messages appear in chat after long tool-calling sessions.

## Call Chain

```
gateway/run.py:_hyg_msgs (strips tool_calls/timestamp)
  → run_agent.py:_compress_context()
    → context_compressor.py:compress() (tail copy preserves duplicates)
      → gateway/session.py:rewrite_transcript()
        → hermes_state.py:replace_messages() (no dedup check)
```

## Plan

### Task 1: Write regression test (TDD — test first)

**File**: `tests/agent/test_context_compressor.py`
**Location**: Add `TestTailDedup` class at end of file

**Test case**: Create a mock compressor with messages containing consecutive identical assistant content in the tail. Verify `compress()` produces only one copy.

```python
class TestTailDedup:
    """Regression: compression should not duplicate consecutive identical assistant messages."""

    @staticmethod
    def _make_compressor():
        """Create a minimal ContextCompressor for testing compress() tail dedup."""
        from agent.context_compressor import ContextCompressor
        c = ContextCompressor.__new__(ContextCompressor)
        c.context_length = 1_000_000
        c.threshold_percent = 0.5
        c.threshold_tokens = 500_000
        c.tail_token_budget = 20000
        c.protect_last_n = 2
        c.quiet_mode = True
        c.compression_count = 0
        c._previous_summary = None
        c._summary_failure_cooldown_until = 0.0
        c._ineffective_compression_count = 0
        c._last_compress_aborted = False
        c._last_summary_error = None
        c._last_summary_dropped_count = 0
        c._last_summary_fallback_used = False
        c._last_aux_model_failure_error = None
        c._last_aux_model_failure_model = None
        c.abort_on_summary_failure = False
        c.last_prompt_tokens = 10000
        c.last_compression_rough_tokens = 0
        c.last_completion_tokens = 0
        c.awaiting_real_usage_after_compression = False
        c._last_compression_savings_pct = 0.0
        c._last_compression_summary_warning = None
        c._last_aux_fallback_warning_key = None
        # Mock methods
        c._generate_summary = lambda turns, focus_topic=None: "Summary of work done."
        c._protect_head_size = lambda msgs: 1
        c._align_boundary_forward = lambda msgs, idx: idx
        c._find_tail_cut_by_tokens = lambda msgs, start: len(msgs) - 1
        c._find_latest_context_summary = lambda msgs, start, end: (None, None)
        c._derive_auto_focus_topic = lambda msgs: None
        c._sanitize_tool_pairs = lambda msgs: msgs
        return c

    def test_consecutive_identical_tail_deduplicated(self):
        """Two consecutive identical assistant messages (no tool_calls) in tail → one survives."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "Running tests...", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": "103 passed", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "103 passed! All good."},
            {"role": "user", "content": "Run full suite"},
            {"role": "assistant", "content": "Running full suite...", "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": "490 passed", "tool_call_id": "tc2"},
            {"role": "assistant", "content": "103 passed! All good."},  # duplicate of index 4
        ]

        compressor = self._make_compressor()
        result = compressor.compress(messages, current_tokens=10000)

        # Count assistant messages with "103 passed! All good."
        dups = [m for m in result if m.get("role") == "assistant" and m.get("content") == "103 passed! All good."]
        assert len(dups) == 1, f"Expected 1 copy, got {len(dups)}: {dups}"

    def test_non_consecutive_identical_preserved(self):
        """Non-consecutive identical assistant messages should NOT be deduplicated."""
        from agent.context_compressor import ContextCompressor

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "Running tests...", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": "103 passed", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "103 passed! All good."},  # first occurrence
            {"role": "user", "content": "Now run lint"},               # user message separates them
            {"role": "assistant", "content": "Running lint...", "tool_calls": [{"id": "tc3", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": "0 errors", "tool_call_id": "tc3"},
            {"role": "assistant", "content": "103 passed! All good."},  # second occurrence, NOT consecutive
        ]

        compressor = self._make_compressor()
        result = compressor.compress(messages, current_tokens=10000)

        dups = [m for m in result if m.get("role") == "assistant" and m.get("content") == "103 passed! All good."]
        assert len(dups) == 2, f"Expected 2 copies (non-consecutive), got {len(dups)}"

    def test_identical_with_tool_calls_preserved(self):
        """Assistant messages with tool_calls should NOT be deduplicated even if identical."""
        from agent.context_compressor import ContextCompressor

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "Running tests...", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "content": "103 passed", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Running tests...", "tool_calls": [{"id": "tc4", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},  # identical content BUT has tool_calls
            {"role": "tool", "content": "490 passed", "tool_call_id": "tc4"},
            {"role": "user", "content": "Done"},
        ]

        compressor = self._make_compressor()
        result = compressor.compress(messages, current_tokens=10000)

        # Both should survive because they have tool_calls
        running = [m for m in result if m.get("role") == "assistant" and m.get("content") == "Running tests..."]
        assert len(running) == 2, f"Expected 2 copies (with tool_calls), got {len(running)}"
```

**Acceptance criteria**:
- [ ] Test file compiles
- [ ] Test fails before fix (RED phase)
- [ ] Test passes after fix (GREEN phase)

### Task 2: Fix `compress()` tail dedup (core fix)

**File**: `agent/context_compressor.py`
**Location**: `compress()` method, line ~2378 (tail copy loop)

**Exact insertion**: Before the `_merge_summary_into_tail` check, add dedup guard.

```python
# Line 2378, BEFORE the existing for loop:
_prev_tail_content = None
for i in range(compress_end, n_messages):
    msg = messages[i].copy()
    # Deduplicate consecutive identical assistant messages in tail.
    # In tool-calling loops, the model often produces identical text
    # ("103 passed!") across turns. After _hyg_msgs strips tool_calls,
    # these become consecutive identical entries that compound across
    # compression cycles.
    _msg_content = msg.get("content")
    if (msg.get("role") == "assistant"
            and _msg_content == _prev_tail_content
            and not msg.get("tool_calls")):
        continue
    _prev_tail_content = _msg_content
    # --- existing merge logic below (unchanged) ---
    if _merge_summary_into_tail and i == compress_end:
        merged_prefix = summary + "\n\n" + _SUMMARY_END_MARKER + "\n\n"
        msg["content"] = _append_text_to_content(
            msg.get("content"),
            merged_prefix,
            prepend=True,
        )
        msg[COMPRESSED_SUMMARY_METADATA_KEY] = True
        _merge_summary_into_tail = False
    compressed.append(msg)
```

**Acceptance criteria**:
- [ ] Dedup guard is BEFORE the merge block
- [ ] Only consecutive identical assistant messages (no tool_calls) are skipped
- [ ] Non-consecutive, user messages, and tool-call messages are preserved

### Task 3: Add defense-in-depth dedup in `replace_messages`

**File**: `hermes_state.py`
**Location**: `replace_messages()` method, line ~2608 (insert loop top)

**Exact insertion**: After extracting `role` and before the main field processing.

```python
# Line 2608, inside the for loop, AFTER role = msg.get("role", "unknown"):
role = msg.get("role", "unknown")
content = msg.get("content")
# Defense-in-depth: skip consecutive identical assistant messages
# that slipped through upstream dedup (context_compressor.py).
if (role == "assistant"
        and content == _prev_dedup_content
        and not msg.get("tool_calls")):
    continue
_prev_dedup_content = content

# --- existing field processing continues below ---
```

Add `_prev_dedup_content = None` before the loop (near line 2605).

**Acceptance criteria**:
- [ ] Dedup check is at the TOP of the loop body
- [ ] Only consecutive identical assistant messages (no tool_calls) are skipped
- [ ] `_prev_dedup_content` is initialized before the loop

### Task 4: Run full test suite

**Command**: `python -m pytest tests/agent/test_context_compressor.py -v -x` then `python -m pytest tests/ -x -q --timeout=60`

**Acceptance criteria**:
- [ ] New tests pass
- [ ] All existing tests pass
- [ ] No regressions

## Files to Modify

1. `tests/agent/test_context_compressor.py` — new `TestTailDedup` class
2. `agent/context_compressor.py` — tail dedup in `compress()`
3. `hermes_state.py` — defense-in-depth dedup in `replace_messages()`

## Verification

1. `python -m pytest tests/agent/test_context_compressor.py::TestTailDedup -v` — new tests pass
2. `python -m pytest tests/ -x -q --timeout=60` — full suite passes
3. Manual: grep for consecutive duplicate assistant messages in compressed output
