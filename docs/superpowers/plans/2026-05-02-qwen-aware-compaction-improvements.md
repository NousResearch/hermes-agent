# Qwen-Aware Compaction Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Companion benchmark/regression plan:** `2026-05-02-qwen-aware-compaction-benchmarks.md` in this directory. Unit tests in this plan verify *correctness*; the benchmark plan provides numeric ship/no-ship signal on **performance / reliability / accuracy** changes after compaction. Tier 1 of the benchmark plan should run as part of the Task 8 verification step before merging.

**Goal:** Add five low-risk, feature-flagged compaction improvements to `agent/context_compressor.py` that materially reduce per-turn wall clock for local Qwen3.6 servers (no llama.cpp partial-prefix KV cache) while keeping cloud users unaffected.

**Architecture:** All changes are additive behind a new `compression.qwen_aware:` config block (off by default). The existing `ContextCompressor` is patched in place — no new engine class, no new plugin. Each feature has its own flag so it can be disabled independently if it regresses. The recommended local-Qwen flag set is documented and applied to `~/.hermes/config.yaml`.

**Tech Stack:** Python 3.11+, pytest, no new runtime dependencies. Tests follow the existing `tests/agent/test_context_compressor.py` and `tests/agent/test_compress_focus.py` pytest patterns.

---

## Background — what the research changes vs. the original report

Findings from the docs + code review on 2026-05-02 reshape the original recommendations:

1. **Hermes already exposes most of forgecode's config knobs** (`compression.{threshold,target_ratio,protect_last_n,enabled,hygiene_hard_message_limit}`), so the gap is not "configure compaction" but "extend it for local-Qwen-shaped sessions." The recommended config block is a **superset** of the existing one, not a replacement.

2. **The existing tool-atomicity helpers (`_align_boundary_backward`, `_sanitize_tool_pairs`) are correctness-grade.** What's missing is a **first-assistant anchor** for the start boundary (forgecode's invariant — Hermes' `protect_first_n` count can land mid-user-block) and explicit tests for parallel tool calls.

3. **Local Qwen prefix-cache constraint is a llama.cpp-side limitation, not a model-side one.** SGLang/vLLM with RadixAttention handle hybrid models differently. Plan should not bake "no prefix cache" assumptions into the persistent code path — only into the *defaults* for the local-Qwen config.

4. **The dedup win is smaller than the report claimed.** Hermes' existing `_prune_old_tool_results` already runs (a) content-hash dedup of identical tool results (Pass 1), (b) `_summarize_tool_result` 1-line replacement of all pre-boundary tool results (Pass 2), and (c) tool-call argument truncation (Pass 3). The op-keyed dedup we add (Pass 1.5) catches **tail-region duplicates** that Pass 2 misses (Pass 2 only walks `range(prune_boundary)`) and replaces opaque "[read_file] read X (5,000 chars)" summaries with structured "[Superseded by later read_file on file=/X — see message N]" back-references. Real but modest improvement, not a 10× win.

## Background — what we considered but rejected

Two ideas from the original report turned out to be redundant or unsafe; documenting why so future-us doesn't re-propose them:

**REJECTED: stripping `<think>` blocks from older messages (originally P1e).**
Hermes' `AIAgent._build_assistant_message` (`run_agent.py:8590-8602`) already strips inline reasoning tags at the **storage boundary** — every assistant message reaches the compressor with `content` already cleaned of `<think>`/`<thinking>`/`<reasoning>`/`<REASONING_SCRATCHPAD>`/`<thought>` blocks (5 variants, case-insensitive, plus unterminated-tag handling). The reasoning text moves to the separate `reasoning` and `reasoning_content` message fields. So a compaction-time strip would find nothing to strip.

The local jinja template at `~/llama-stack/templates/qwen3.6-fixed.jinja:150-160` then decides whether to emit historical `<think>` based on `enable_thinking` and `preserve_thinking`:

- **`qwen-instruct` mode** (`<|think_off|>` prelude → `enable_thinking=false`): the template never emits historical `<think>` regardless. A strip would do nothing.
- **`qwen-thinking` mode** (`<|think_on|>` prelude → `enable_thinking=true`): the template DOES emit historical `<think>` from `reasoning_content` because `preserve_thinking is undefined → True branch` (this template is more permissive than the upstream Qwen template). The model is trained to use this context. Stripping would degrade reasoning continuity — exactly the opposite of what we want.

Either way: dropping the strip task removes ~150 lines of redundant code with zero functional regression.

**REJECTED: `on_turn_end` trigger.**
Hermes has exactly one `should_compress` callsite (`run_agent.py:13280`), inside the tool loop **after** tool results are appended. At that point `messages[-1]` is always a tool result, never a user message — `on_turn_end` would never fire. Adding a preflight callsite (so compaction could fire while the user is typing) is an architectural change beyond this plan's scope. The ABC's `should_compress_preflight` exists at `agent/context_engine.py:100` but isn't wired in production. We keep `message_threshold` and `turn_threshold` (which fire usefully at the existing callsite) and skip `on_turn_end` until a preflight hook lands.

---

## File Structure

Files created or modified, in dependency order:

| Path | Status | Responsibility |
|---|---|---|
| `agent/context_compressor.py` | Modify | Add `_dedup_by_operation`, `_anchor_to_first_assistant`, `_compute_threshold_tokens`, multi-trigger `should_compress`, `last_compaction_result` attribute |
| `agent/compaction_result.py` | Create | Lightweight `CompactionResult` dataclass exposing per-event metrics |
| `hermes_cli/config.py` | Modify | Parse new `compression.qwen_aware.*` keys (additive — no migration) |
| `run_agent.py` | Modify | Read new config keys, pass through to `ContextCompressor.__init__` |
| `tests/agent/test_context_compressor.py` | Modify | Add tests for each new method (one per task) |
| `tests/agent/test_compaction_atomicity.py` | Create | Focused parallel-tool-call + first-assistant-anchor tests |
| `tests/agent/test_compaction_result.py` | Create | Tests for the metrics dataclass + compress() exposing it |
| `~/.hermes/config.yaml` | Modify (user config — outside repo) | Add the recommended local-Qwen flag set |
| `docs/research/2026-05-02-forgecode-context-compaction-takeaways.md` | (already exists) | Background reference |
| `docs/research/2026-05-02-qwen-aware-compaction-config.md` | Create | User-facing doc for the new `compression.qwen_aware.*` knobs |

Each task lands in 1-3 files. Tests are co-committed with their implementation — no "land code now, tests later."

---

## Safety net (applies to every task)

- **Branch:** all tasks happen on `feat/qwen-aware-compaction` off `main`. Don't merge to `main` until Task 9 (final smoke) passes.
- **Feature flag:** every new behavior gates on a `compression.qwen_aware.<flag>` boolean. The defaults are all `False` so cloud users see zero change until they opt in.
- **No deletion of existing logic.** Every change is additive. The new code paths run alongside the old, and the flag chooses which result wins.
- **Rollback procedure:** if a regression appears in production, set `compression.qwen_aware.enabled: false` in `~/.hermes/config.yaml` and restart Hermes. No code revert needed.

---

## Task 0: Branch setup and config schema scaffold

**Files:**
- Create: `agent/compaction_result.py`
- Modify: `hermes_cli/config.py` (config schema)

The first task creates the `CompactionResult` dataclass and the config-parsing scaffold. All later tasks depend on these, but neither has any user-visible behavior on its own — they just exist for downstream tasks to land into. This isolates the "introduce new attribute / new config keys" change from the behavior changes.

- [ ] **Step 1: Create the working branch**

```bash
cd "/home/admin/Projects/Generative AI/hermes-agent"
git checkout -b feat/qwen-aware-compaction
```

Expected: `Switched to a new branch 'feat/qwen-aware-compaction'`

- [ ] **Step 2: Write the failing test for `CompactionResult`**

Create `tests/agent/test_compaction_result.py`:

```python
"""Tests for the CompactionResult metrics dataclass."""

from agent.compaction_result import CompactionResult


def test_compaction_result_records_basic_fields():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    assert r.original_messages == 50
    assert r.compacted_messages == 12
    assert r.token_reduction_pct == 90.0
    assert r.message_reduction_pct == 76.0


def test_compaction_result_handles_zero_original():
    r = CompactionResult(
        original_messages=0,
        compacted_messages=0,
        original_tokens=0,
        compacted_tokens=0,
        operations_deduped=0,
        triggered_by="token",
    )
    assert r.token_reduction_pct == 0.0
    assert r.message_reduction_pct == 0.0


def test_compaction_result_summary_line():
    r = CompactionResult(
        original_messages=50,
        compacted_messages=12,
        original_tokens=80_000,
        compacted_tokens=8_000,
        operations_deduped=4,
        triggered_by="token",
    )
    line = r.summary_line()
    assert "80,000 → 8,000" in line
    assert "90%" in line
    assert "deduped 4" in line
    assert "trigger=token" in line
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd "/home/admin/Projects/Generative AI/hermes-agent"
.venv/bin/pytest tests/agent/test_compaction_result.py -v
```

Expected: 3 tests fail, all with `ModuleNotFoundError: No module named 'agent.compaction_result'`.

- [ ] **Step 4: Create `agent/compaction_result.py`**

```python
"""Lightweight metrics for one context-compaction event.

Stored on the engine as ``last_compaction_result`` so the gateway,
status line, and ``/usage`` can surface what just happened without
changing the ``compress()`` return signature (which the
ContextEngine ABC and any plugin implementations rely on).
"""

from dataclasses import dataclass
from typing import Literal

TriggerReason = Literal["token", "turn", "message", "manual"]


@dataclass(frozen=True)
class CompactionResult:
    original_messages: int
    compacted_messages: int
    original_tokens: int
    compacted_tokens: int
    operations_deduped: int
    triggered_by: TriggerReason

    @property
    def token_reduction_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        delta = self.original_tokens - self.compacted_tokens
        return round(delta / self.original_tokens * 100, 1)

    @property
    def message_reduction_pct(self) -> float:
        if self.original_messages <= 0:
            return 0.0
        delta = self.original_messages - self.compacted_messages
        return round(delta / self.original_messages * 100, 1)

    def summary_line(self) -> str:
        return (
            f"compaction: {self.original_tokens:,} → {self.compacted_tokens:,} tokens "
            f"({self.token_reduction_pct:.0f}% saved), "
            f"{self.original_messages} → {self.compacted_messages} msgs, "
            f"deduped {self.operations_deduped}, "
            f"trigger={self.triggered_by}"
        )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/pytest tests/agent/test_compaction_result.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Add the `qwen_aware` config block schema**

Find the existing compression config parsing in `run_agent.py:1804-1810`. Add a sibling block after `compression_protect_last`:

```python
# Read the qwen-aware extensions (all optional, defaults preserve current behavior)
_qa_cfg = _compression_cfg.get("qwen_aware") or {}
if not isinstance(_qa_cfg, dict):
    _qa_cfg = {}
qwen_aware_enabled = bool(_qa_cfg.get("enabled", False))
qwen_aware_dedup_operations = bool(_qa_cfg.get("dedup_operations", False))
qwen_aware_anchor_first_assistant = bool(_qa_cfg.get("anchor_first_assistant", False))
qwen_aware_threshold_absolute_max = _qa_cfg.get("threshold_absolute_max")  # None or int
qwen_aware_message_threshold = _qa_cfg.get("message_threshold")  # None or int
qwen_aware_turn_threshold = _qa_cfg.get("turn_threshold")  # None or int
```

- [ ] **Step 7: Plumb the new args through the constructor**

In `agent/context_compressor.py:376` the `__init__` signature, add (with defaults that preserve current behavior):

```python
def __init__(
    self,
    model: str,
    threshold_percent: float = 0.50,
    protect_first_n: int = 3,
    protect_last_n: int = 20,
    summary_target_ratio: float = 0.20,
    quiet_mode: bool = False,
    summary_model_override: str = None,
    base_url: str = "",
    api_key: str = "",
    config_context_length: int | None = None,
    provider: str = "",
    api_mode: str = "",
    # ── qwen_aware extensions ────────────────────────────────────────
    qwen_aware_enabled: bool = False,
    dedup_operations: bool = False,
    anchor_first_assistant: bool = False,
    threshold_absolute_max: int | None = None,
    message_threshold: int | None = None,
    turn_threshold: int | None = None,
):
```

**Field ordering matters.** Place the qwen_aware field assignments **immediately after the existing `self.threshold_percent = threshold_percent` block (around line 396) and BEFORE the `self.context_length = get_model_context_length(...)` call (around line 402)**. Task 3 will refactor the threshold computation to read `self.threshold_absolute_max`, which must already be set when that line runs. Concretely, insert this block right after `self.summary_target_ratio = max(...)` at line 399:

```python
# qwen_aware extensions (must precede self.threshold_tokens calculation)
self.qwen_aware_enabled = qwen_aware_enabled
self.dedup_operations = dedup_operations
self.anchor_first_assistant = anchor_first_assistant
self.threshold_absolute_max = threshold_absolute_max
self.message_threshold = message_threshold
self.turn_threshold = turn_threshold

# Per-event metrics scratch + result handle
from agent.compaction_result import CompactionResult  # noqa: E402 — top-level import in Task 6
self.last_compaction_result: "CompactionResult | None" = None
self._last_trigger: str | None = None
self._last_op_deduped: int = 0
```

In `update_model` (lines 348-374), no changes are needed for the qwen_aware fields themselves — they persist from `__init__`. Task 3 will refactor the threshold line in both methods.

Pass each new kwarg through from `run_agent.py:1981-1994`:

```python
self.context_compressor = ContextCompressor(
    model=self.model,
    threshold_percent=compression_threshold,
    protect_first_n=3,
    protect_last_n=compression_protect_last,
    summary_target_ratio=compression_target_ratio,
    summary_model_override=None,
    quiet_mode=self.quiet_mode,
    base_url=self.base_url,
    api_key=getattr(self, "api_key", ""),
    config_context_length=_config_context_length,
    provider=self.provider,
    api_mode=self.api_mode,
    # qwen_aware
    qwen_aware_enabled=qwen_aware_enabled,
    dedup_operations=qwen_aware_dedup_operations,
    anchor_first_assistant=qwen_aware_anchor_first_assistant,
    threshold_absolute_max=qwen_aware_threshold_absolute_max,
    message_threshold=qwen_aware_message_threshold,
    turn_threshold=qwen_aware_turn_threshold,
)
```

- [ ] **Step 8: Smoke-check the existing test suite still passes**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py tests/agent/test_compress_focus.py tests/agent/test_context_engine.py -v
```

Expected: all green. New kwargs are off by default; existing behavior unchanged.

- [ ] **Step 9: Commit**

```bash
git add agent/compaction_result.py tests/agent/test_compaction_result.py \
        agent/context_compressor.py run_agent.py
git commit -m "feat(compaction): scaffold CompactionResult + qwen_aware config (no behavior change)"
```

Expected: pre-commit hooks pass, single commit lands.

---

## Task 1 (P0a): Operation-keyed dedup in pre-LLM pruning

**Files:**
- Modify: `agent/context_compressor.py` (add `_dedup_by_operation`, hook into `_prune_old_tool_results`)
- Modify: `tests/agent/test_context_compressor.py` (one new test class)

The existing `_prune_old_tool_results` runs three passes:

- **Pass 1**: content-hash dedup of identical tool results — replaces older duplicates with `"[Duplicate tool output ...]"`.
- **Pass 2**: replaces all pre-`prune_boundary` tool results with `_summarize_tool_result` 1-liners (e.g., `"[read_file] read X (3,400 chars)"`).
- **Pass 3**: truncates large `tool_call.arguments` JSON in pre-boundary assistant messages.

We add **Pass 1.5** (operation-keyed dedup) between Pass 1 and Pass 2. It catches two cases the existing passes miss:

1. **Tail-region duplicates** — Pass 2 only walks `range(prune_boundary)`, leaving the protected tail untouched. If a recent `read_file` re-fetched a file that was already read earlier in the session, the older copy survives Pass 2 untouched. Pass 1.5 supersedes it with a back-reference to the live copy.
2. **Different-content duplicates of same resource** — Pass 1's hash dedup only catches *identical* contents. Two reads of the same path with different content (because the file changed) survive Pass 1. Pass 2 will summarize the older one, but to an opaque "[read_file] read X (5,000 chars)" line. Pass 1.5 produces a structured back-reference instead: "[Superseded by later read_file on file=/X — see message N]" — the model can follow the pointer to find the current state.

The win is real but **not** the "biggest local-Qwen wall-clock saver" the original report implied. Pass 2 already shrinks pre-boundary tool results to 1-liners. Pass 1.5's incremental savings come from tail-region dedup + better information preservation. For local Qwen with no llama.cpp partial-prefix KV reuse, every byte saved is real per-turn savings, but the magnitude is modest (10-30% of conversations show meaningful tail dedup; many show none).

We mirror forgecode's `TrimContextSummary.transform` semantics: read/write/patch on the same path share a key, and the LAST call wins regardless of intervening writes — the latest tool result reflects the post-state, so older ones are redundant.

- [ ] **Step 1: Write the failing test**

Tool names match the actual Hermes registry: `read_file`, `write_file`, `patch`, `terminal`, `search_files`. Verified via `grep -rnE 'registry\.register\(name="' tools/`.

Add to `tests/agent/test_context_compressor.py`:

```python
class TestDedupByOperation:
    """Operation-keyed last-wins dedup of same-resource tool calls.

    Mirrors forgecode's TrimContextSummary semantics: reads/writes on the
    same path share a key, and the LAST call wins regardless of intervening
    writes. This is intentional — the latest tool result reflects the
    post-state, so older results are redundant for the model's purposes.
    """

    def _compressor(self):
        c = ContextCompressor.__new__(ContextCompressor)
        c.protect_first_n = 1
        c.protect_last_n = 2
        c.tail_token_budget = 5000
        c.context_length = 200_000
        c.dedup_operations = True
        c.quiet_mode = True
        return c

    def test_consecutive_reads_same_path_collapse_to_last(self):
        c = self._compressor()
        msgs = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "v1 contents"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "v2 contents"},
            {"role": "user", "content": "next"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        # Both tool_calls remain (we don't break tool_call/tool_result pairs)
        # but the FIRST tool_result is replaced with a back-reference.
        assert deduped == 1
        first_tool = next(m for m in out if m.get("tool_call_id") == "1")
        assert first_tool["content"].startswith("[Superseded by later")
        last_tool = next(m for m in out if m.get("tool_call_id") == "2")
        assert last_tool["content"] == "v2 contents"

    def test_read_and_patch_on_same_path_dedup_to_last(self):
        """Read → patch → read on /a.py: keep only the final read result.

        Matches forgecode's File(path) keying — read/write share a key.
        Older result becomes a back-reference even though a write
        happened between them; the latest read is what matters.
        """
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "before edit"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "patch",
                 "arguments": '{"mode":"replace","path":"/a.py",'
                              '"old_string":"x","new_string":"y"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "patched"},
            {"role": "assistant", "tool_calls": [
                {"id": "3", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "3", "content": "after edit"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        # Both earlier results superseded → deduped == 2
        assert deduped == 2
        first = next(m for m in out if m.get("tool_call_id") == "1")
        assert first["content"].startswith("[Superseded by later")
        second = next(m for m in out if m.get("tool_call_id") == "2")
        assert second["content"].startswith("[Superseded by later")
        last = next(m for m in out if m.get("tool_call_id") == "3")
        assert last["content"] == "after edit"

    def test_different_paths_never_dedup(self):
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "a contents"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "read_file",
                 "arguments": '{"path":"/b.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "b contents"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert deduped == 0

    def test_v4a_patch_without_path_is_skipped(self):
        """Patch in V4A multi-file mode has no `path` arg → return None key → skip."""
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "patch",
                 "arguments": '{"mode":"patch","patch":"@@ ..."}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "applied"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "patch",
                 "arguments": '{"mode":"patch","patch":"@@ different"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "applied2"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert deduped == 0  # neither call has a `path` → not dedupable

    def test_disabled_when_flag_off(self):
        c = self._compressor()
        c.dedup_operations = False
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "v1"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "v2"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert out == msgs
        assert deduped == 0

    def test_terminal_command_dedup_keys_on_command_string(self):
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "terminal",
                 "arguments": '{"command":"ls -la"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "listing v1"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "terminal",
                 "arguments": '{"command":"ls -la"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "listing v2"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert deduped == 1

    def test_terminal_with_different_command_does_not_dedup(self):
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "terminal",
                 "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "out1"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "terminal",
                 "arguments": '{"command":"pwd"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "out2"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert deduped == 0

    def test_multimodal_tool_result_is_never_clobbered(self):
        """Defense in depth: if the older tool result has list (multimodal)
        content, the supersession must be skipped to avoid silently dropping
        image data. The operation-key whitelist already excludes vision
        tools (vision_analyze, browser_*) so this case is unreachable in
        practice, but we lock the invariant for future robustness."""
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.png"}'}},
            ]},
            # Imagine a future tool that returned multimodal content for a
            # path-keyed call. The dedup must not stomp on the image part.
            {"role": "tool", "tool_call_id": "1", "content": [
                {"type": "text", "text": "An image of a cat."},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,..."}},
            ]},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.png"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "second read text"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        # The multimodal first result was NOT replaced.
        assert deduped == 0
        first = next(m for m in out if m.get("tool_call_id") == "1")
        assert isinstance(first["content"], list)
        assert any(
            p.get("type") == "image_url" for p in first["content"]
        ), "Image part must survive the dedup pass"

    def test_skips_already_pass1_deduped_entries(self):
        """If Pass 1 (content-hash dedup) already replaced a tool result
        with `[Duplicate tool output ...]`, Pass 1.5 must not re-stamp it
        with a `[Superseded ...]` message — that would double-process and
        confuse the metric counter."""
        c = self._compressor()
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "1",
             "content": "[Duplicate tool output — same content as a more recent call]"},
            {"role": "assistant", "tool_calls": [
                {"id": "2", "function": {"name": "read_file",
                 "arguments": '{"path":"/a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "2", "content": "fresh content"},
        ]
        out, deduped = c._dedup_by_operation(msgs)
        assert deduped == 0  # already-Duplicate entry is left alone
        first = next(m for m in out if m.get("tool_call_id") == "1")
        assert first["content"].startswith("[Duplicate tool output")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py::TestDedupByOperation -v
```

Expected: 9 failures, all `AttributeError: 'ContextCompressor' object has no attribute '_dedup_by_operation'`.

- [ ] **Step 3: Implement `_dedup_by_operation`**

Add this method to `ContextCompressor` in `agent/context_compressor.py`, immediately after `_prune_old_tool_results`:

```python
# Operation-key derivation. The key identifies "the resource the call acts on"
# so that earlier calls on the same resource can collapse to the latest one.
# Mirrors forgecode's `Operation` enum (transformers/trim_context_summary.rs).
# Tool names match the Hermes registry verified at this commit:
# read_file / write_file / patch in tools/file_tools.py, terminal in
# tools/terminal_tool.py. New file ops added later should be added here too.
_FILE_OPS = {"read_file", "write_file", "patch"}

@staticmethod
def _operation_key(tool_name: str, args_json: str) -> tuple[str, str] | None:
    """Return ``(category, identifier)`` for a tool call, or ``None`` if not dedupable.

    File ops on the same path share a key (read/write/patch all collapse).
    Terminal commands key on the command string. Anything else returns
    ``None`` and is never deduped — including ``patch`` in V4A multi-file
    mode (no ``path`` arg) and ``web_extract`` (multi-URL ``urls`` arg).
    """
    try:
        args = json.loads(args_json) if args_json else {}
    except (ValueError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    if tool_name in ContextCompressor._FILE_OPS:
        path = args.get("path")
        return ("file", path) if isinstance(path, str) and path else None
    if tool_name == "terminal":
        cmd = args.get("command")
        return ("shell", cmd) if isinstance(cmd, str) and cmd else None
    return None

def _dedup_by_operation(
    self, messages: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    """Replace earlier tool results from same-key operations with a
    back-reference, keeping the most recent result intact.

    Semantics match forgecode's TrimContextSummary: read/write/patch on
    the same path share a key. Walking forward, when we see a tool result
    whose key matches a prior tool result, the prior one is replaced
    with a `[Superseded ...]` back-reference. The latest tool result
    holds the post-state and is what the model needs.

    Already-Pass-1-deduped entries (content starts with ``[Duplicate``)
    are left alone — they were marked by a different mechanism and
    re-stamping would double-count.

    Returns ``(new_messages, deduped_count)``.
    """
    if not getattr(self, "dedup_operations", False) or not messages:
        return messages, 0

    result = [m.copy() for m in messages]

    # Build index: tool_call_id -> (tool_name, args_json)
    # Mirrors the same shape as _prune_old_tool_results' call_id_to_tool index.
    call_index: Dict[str, tuple[str, str]] = {}
    for msg in result:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            cid = self._get_tool_call_id(tc)
            if not cid:
                continue
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                args = fn.get("arguments", "") if isinstance(fn, dict) else ""
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn else ""
                args = getattr(fn, "arguments", "") if fn else ""
            call_index[cid] = (name or "", args or "")

    # Walk tool results in order. For each, compute its op key. If we've
    # seen this key before, supersede the earlier result with a back-ref.
    deduped = 0
    last_seen: Dict[tuple[str, str], int] = {}  # op_key -> message idx

    for i, msg in enumerate(result):
        if msg.get("role") != "tool":
            continue
        cid = msg.get("tool_call_id") or ""
        info = call_index.get(cid)
        if not info:
            continue
        tool_name, args_json = info
        key = self._operation_key(tool_name, args_json)
        if key is None:
            continue

        prev_idx = last_seen.get(key)
        if prev_idx is not None:
            older = result[prev_idx]
            older_content = older.get("content")
            # Defense in depth (vision safety): if the previous tool
            # result is multimodal (list-of-content-blocks), leave it
            # alone. Replacing it with a string back-reference would
            # silently drop image data. Mirrors the
            # `if isinstance(content, list): continue` skip that
            # _prune_old_tool_results' Pass 1 and Pass 2 already use.
            # In practice the operation-key whitelist (file ops +
            # terminal) makes this unreachable, but the check costs
            # nothing and prevents future foot-guns when new tools
            # are added to the whitelist.
            if isinstance(older_content, list):
                last_seen[key] = i
                continue

            older_str = older_content or ""
            # Skip if Pass 1 (content-hash dedup) already marked this,
            # or if we already superseded it.
            if not isinstance(older_str, str) or older_str.startswith(
                ("[Duplicate tool output", "[Superseded by later")
            ):
                last_seen[key] = i
                continue

            result[prev_idx] = {
                **older,
                "content": (
                    f"[Superseded by later {tool_name} call on same "
                    f"{key[0]}={key[1]} — see message {i + 1}]"
                ),
            }
            deduped += 1
        last_seen[key] = i

    return result, deduped
```

- [ ] **Step 4: Wire into `_prune_old_tool_results`**

In the existing `_prune_old_tool_results` method, after the existing Pass 1 (content-hash dedup, around line 580) but before Pass 2 (summarize, around line 583), add:

```python
# Pass 1.5: operation-keyed dedup (qwen_aware extension)
if getattr(self, "dedup_operations", False):
    result, op_deduped = self._dedup_by_operation(result)
    pruned += op_deduped
    # Track separately so CompactionResult.operations_deduped (Task 6)
    # reports only this pass, not Pass 1's content-hash dedup or Pass 2's
    # summarize-pass.
    self._last_op_deduped = op_deduped
else:
    self._last_op_deduped = 0
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py::TestDedupByOperation -v
```

Expected: 9 passed.

- [ ] **Step 6: Run the full compressor test suite to confirm no regressions**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py -v
```

Expected: all green (existing tests untouched because `dedup_operations` defaults to False).

- [ ] **Step 7: Commit**

```bash
git add agent/context_compressor.py tests/agent/test_context_compressor.py
git commit -m "feat(compaction): operation-keyed last-write-wins dedup (P0a, flag-gated)"
```

---

## Task 2 (P0b): First-assistant anchor for the start boundary + parallel tool-call tests

**Files:**
- Modify: `agent/context_compressor.py` (add `_anchor_to_first_assistant`)
- Create: `tests/agent/test_compaction_atomicity.py`
- Modify: `agent/context_compressor.py` `compress()` to use anchor when `anchor_first_assistant=True`

Hermes' existing `_align_boundary_backward` and `_sanitize_tool_pairs` already enforce tool-call/result atomicity at the END boundary. The forgecode invariant we don't yet enforce is "compaction always starts at the first assistant message after the protected head" — Hermes uses `protect_first_n=3` (count) which can land mid-user-message. When the start boundary lands at a user message, that user message becomes orphaned in the summary frame and is hard for the model to attribute. The anchor patch slides the start forward until it hits an assistant, never crossing into the protected tail.

We also add explicit tests for parallel tool calls (one assistant → multiple tool_calls → multiple tool_results) — a real shape Hermes' `delegate_task` and parallel `read_file` flows produce.

- [ ] **Step 1: Write the failing tests**

Create `tests/agent/test_compaction_atomicity.py`:

```python
"""Boundary atomicity for compaction: parallel tool calls + first-assistant anchor."""

from agent.context_compressor import ContextCompressor


def _bare_compressor(**overrides):
    c = ContextCompressor.__new__(ContextCompressor)
    c.protect_first_n = 1
    c.protect_last_n = 2
    c.tail_token_budget = 5000
    c.context_length = 200_000
    c.threshold_percent = 0.50
    c.threshold_tokens = 100_000
    c.anchor_first_assistant = True
    c.quiet_mode = True
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class TestFirstAssistantAnchor:
    def test_anchor_skips_leading_user_block(self):
        """If protect_first_n=1 lands on a user msg, anchor slides to first assistant."""
        c = _bare_compressor(protect_first_n=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},      # idx 1, would be start
            {"role": "user", "content": "u2"},      # idx 2
            {"role": "assistant", "content": "a1"}, # idx 3 — should anchor here
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a2"},
        ]
        anchored = c._anchor_to_first_assistant(msgs, start_idx=1)
        assert anchored == 3

    def test_anchor_no_op_when_already_at_assistant(self):
        c = _bare_compressor(protect_first_n=2)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}, # idx 2, start_idx=2
            {"role": "user", "content": "u2"},
        ]
        assert c._anchor_to_first_assistant(msgs, start_idx=2) == 2

    def test_anchor_does_not_cross_tail(self):
        """If no assistant exists between start and tail, return unchanged."""
        c = _bare_compressor()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "user", "content": "u3"},  # tail starts here
        ]
        # Caller's logic still handles "no compress region" — anchor just doesn't lie.
        result = c._anchor_to_first_assistant(msgs, start_idx=1, tail_start=3)
        assert result >= 3  # signals "nothing to anchor"; caller must check

    def test_anchor_disabled_when_flag_off(self):
        c = _bare_compressor(anchor_first_assistant=False)
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        # With flag off, returns start_idx unchanged
        assert c._anchor_to_first_assistant(msgs, start_idx=0) == 0


class TestParallelToolCallsAtomicity:
    """Existing helpers must keep parallel tool_call/result groups together."""

    def test_aligned_backward_pulls_assistant_with_parallel_results(self):
        """Tail boundary inside a 3-tool-result block must walk back to the assistant."""
        c = _bare_compressor()
        msgs = [
            {"role": "user", "content": "do three things"},
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "2", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "3", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "content": "r2"},  # boundary lands here
            {"role": "tool", "tool_call_id": "3", "content": "r3"},
            {"role": "user", "content": "thanks"},
        ]
        # Call boundary at idx=4 (between r2 and r3). Must walk back to idx=1 (assistant).
        aligned = c._align_boundary_backward(msgs, idx=4)
        assert aligned == 1, (
            f"Expected boundary to pull back to assistant at idx=1, got {aligned}. "
            f"Splitting parallel tool_results would orphan tool_call_ids."
        )

    def test_sanitize_removes_orphan_results_from_split_parallel_group(self):
        """If sanitization sees a tool result whose call_id was summarized away, drop it."""
        c = _bare_compressor()
        # Compressed list missing the assistant with tool_calls
        compressed = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summary placeholder"},
            {"role": "tool", "tool_call_id": "orphan_1", "content": "stale result"},
            {"role": "user", "content": "next question"},
        ]
        out = c._sanitize_tool_pairs(compressed)
        roles = [m.get("role") for m in out]
        assert "tool" not in roles, "Orphaned tool result must be dropped"
```

- [ ] **Step 2: Run tests to confirm failures (parallel ones may already pass)**

```bash
.venv/bin/pytest tests/agent/test_compaction_atomicity.py -v
```

Expected: 4 of `TestFirstAssistantAnchor` fail with `AttributeError: ... '_anchor_to_first_assistant'`. The 2 in `TestParallelToolCallsAtomicity` should already pass (they're characterization tests for existing behavior — they document the invariant so a future change can't break it silently).

- [ ] **Step 3: Implement `_anchor_to_first_assistant`**

Add to `ContextCompressor`, near `_align_boundary_forward`:

```python
def _anchor_to_first_assistant(
    self,
    messages: List[Dict[str, Any]],
    start_idx: int,
    tail_start: int | None = None,
) -> int:
    """Slide ``start_idx`` forward to the first assistant message.

    When ``anchor_first_assistant`` is enabled, this preserves the forgecode
    invariant that the compressed range always begins at an assistant turn.
    Without the anchor, ``protect_first_n`` (a count) can land mid-user-block
    and produce a summary that begins with an orphan user message.

    If no assistant exists between ``start_idx`` and ``tail_start``, returns
    a value at or beyond ``tail_start`` so the caller's "compress_start >=
    compress_end" guard kicks in and skips compaction.
    """
    if not getattr(self, "anchor_first_assistant", False):
        return start_idx
    upper = tail_start if tail_start is not None else len(messages)
    for i in range(start_idx, upper):
        if messages[i].get("role") == "assistant":
            return i
    return upper
```

- [ ] **Step 4: Wire the anchor into `compress()` and `has_content_to_compress`**

Two callsites, both around the existing `_align_boundary_forward` calls.

**In `compress()` (around line 1283-1287):** after the existing `compress_end = self._find_tail_cut_by_tokens(...)` line, insert one line:

```python
compress_start = self.protect_first_n
compress_start = self._align_boundary_forward(messages, compress_start)

# Use token-budget tail protection instead of fixed message count
compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
compress_start = self._anchor_to_first_assistant(  # ← NEW
    messages, compress_start, tail_start=compress_end,
)

if compress_start >= compress_end:
    return messages
```

**In `has_content_to_compress` (around line 1227-1229):** add the same anchor call:

```python
def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
    compress_start = self._align_boundary_forward(messages, self.protect_first_n)
    compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
    compress_start = self._anchor_to_first_assistant(  # ← NEW
        messages, compress_start, tail_start=compress_end,
    )
    return compress_start < compress_end
```

The anchor is a no-op (returns `start_idx` unchanged) when `anchor_first_assistant=False`, so this insertion is safe for cloud users.

- [ ] **Step 5: Run tests to confirm pass**

```bash
.venv/bin/pytest tests/agent/test_compaction_atomicity.py tests/agent/test_context_compressor.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add agent/context_compressor.py tests/agent/test_compaction_atomicity.py
git commit -m "feat(compaction): first-assistant anchor + parallel-tool atomicity tests (P0b)"
```

---

## Task 3 (P0c): Absolute token cap alongside threshold_percent

**Files:**
- Modify: `agent/context_compressor.py` (`_compute_threshold_tokens`)
- Modify: `tests/agent/test_context_compressor.py`

Hermes computes `threshold_tokens = max(int(context_length * threshold_percent), MINIMUM_CONTEXT_LENGTH)`. With Qwen3.6-35B-A3B at 256K and `threshold=0.50`, that's 128K — and at our local decode rate of ~60 tok/s with NO partial-prefix KV cache reuse, every turn after compaction re-processes that prompt from scratch. An absolute ceiling lets local-Qwen sessions cap effective per-turn cost while leaving cloud users unaffected.

The pattern matches forgecode's `min(token_threshold, token_threshold_percentage * ctx)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/agent/test_context_compressor.py`:

```python
class TestThresholdAbsoluteMax:
    def _make(self, **kw):
        defaults = dict(
            threshold_percent=0.50,
            context_length=262_144,
            threshold_absolute_max=None,
        )
        defaults.update(kw)
        c = ContextCompressor.__new__(ContextCompressor)
        for k, v in defaults.items():
            setattr(c, k, v)
        return c

    def test_no_cap_uses_pure_percentage(self):
        c = self._make(threshold_absolute_max=None)
        # 262144 * 0.50 = 131072
        assert c._compute_threshold_tokens() == 131_072

    def test_cap_lower_than_percentage_wins(self):
        c = self._make(threshold_absolute_max=80_000)
        # min(131072, 80000) = 80000
        assert c._compute_threshold_tokens() == 80_000

    def test_cap_higher_than_percentage_is_no_op(self):
        c = self._make(threshold_absolute_max=200_000)
        # min(131072, 200000) = 131072
        assert c._compute_threshold_tokens() == 131_072

    def test_cap_below_minimum_floor_is_clamped(self):
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        c = self._make(threshold_absolute_max=10_000)
        # never go below MINIMUM_CONTEXT_LENGTH
        assert c._compute_threshold_tokens() == MINIMUM_CONTEXT_LENGTH
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py::TestThresholdAbsoluteMax -v
```

Expected: 4 failures with `AttributeError: '_compute_threshold_tokens'`.

- [ ] **Step 3: Implement `_compute_threshold_tokens` and refactor**

Add to `ContextCompressor`:

```python
def _compute_threshold_tokens(self) -> int:
    """Compute the token threshold honoring both percentage and absolute cap.

    Formula: max(MINIMUM_CONTEXT_LENGTH, min(absolute_max, ctx * pct)).

    The absolute cap is opt-in. When unset (default), behavior is identical
    to the prior ``max(int(ctx * pct), MINIMUM_CONTEXT_LENGTH)`` formula.
    """
    base = int(self.context_length * self.threshold_percent)
    cap = getattr(self, "threshold_absolute_max", None)
    if isinstance(cap, int) and cap > 0:
        base = min(base, cap)
    return max(base, MINIMUM_CONTEXT_LENGTH)
```

Then in both `__init__` and `update_model`, replace the existing `self.threshold_tokens = max(...)` with:

```python
self.threshold_tokens = self._compute_threshold_tokens()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py -v
```

Expected: all green, including new `TestThresholdAbsoluteMax`.

- [ ] **Step 5: Commit**

```bash
git add agent/context_compressor.py tests/agent/test_context_compressor.py
git commit -m "feat(compaction): absolute token cap (P0c, flag-gated)"
```

---

## Task 4 (P1d): Multi-trigger thresholds (turn / message)

**Files:**
- Modify: `agent/context_compressor.py` (`should_compress`, helper methods)
- Modify: `tests/agent/test_context_compressor.py`

`should_compress` currently checks token count only. We add two OR'd triggers, each gated on its own optional config:

- `message_threshold` — total messages crosses this count (catches sessions with many short turns that stay under token threshold)
- `turn_threshold` — count of user-role messages crosses this count (catches long back-and-forth conversations)

We also add a `_last_trigger` field so `CompactionResult.triggered_by` (Task 6) can identify which threshold fired.

**Why no `on_turn_end`?** See "Background — what we considered but rejected." Hermes' single `should_compress` callsite (`run_agent.py:13280`) fires inside the tool loop after tool results are appended, so `messages[-1]` is always a tool result, never a user message. `on_turn_end` would never fire without an additional preflight callsite, which is out of scope here.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agent/test_context_compressor.py`:

```python
class TestMultiTriggerThresholds:
    def _make(self, **kw):
        c = ContextCompressor.__new__(ContextCompressor)
        c.threshold_tokens = 100_000
        c.last_prompt_tokens = 0
        c._ineffective_compression_count = 0
        c.message_threshold = None
        c.turn_threshold = None
        c.quiet_mode = True
        c._last_trigger = None
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def test_token_only_fallback(self):
        c = self._make(last_prompt_tokens=120_000)
        assert c.should_compress() is True
        assert c._last_trigger == "token"

    def test_message_threshold_fires_below_token_threshold(self):
        c = self._make(message_threshold=200, last_prompt_tokens=10)
        msgs = [{"role": "user", "content": str(i)} for i in range(200)]
        assert c.should_compress(messages=msgs) is True
        assert c._last_trigger == "message"

    def test_turn_threshold_counts_user_messages(self):
        c = self._make(turn_threshold=30, last_prompt_tokens=10)
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": "u"} for _ in range(30)]
            + [{"role": "assistant", "content": "a"} for _ in range(30)]
        )
        assert c.should_compress(messages=msgs) is True
        assert c._last_trigger == "turn"

    def test_token_takes_precedence_when_multiple_fire(self):
        c = self._make(
            last_prompt_tokens=120_000,
            message_threshold=10,
            turn_threshold=2,
        )
        msgs = [{"role": "user", "content": "u"}] * 50
        assert c.should_compress(messages=msgs) is True
        # Token threshold checked first → it wins
        assert c._last_trigger == "token"

    def test_anti_thrashing_clears_trigger(self):
        c = self._make(last_prompt_tokens=120_000)
        c._ineffective_compression_count = 2
        assert c.should_compress() is False
        assert c._last_trigger is None

    def test_no_trigger_no_fire(self):
        c = self._make(last_prompt_tokens=10)
        assert c.should_compress() is False
        assert c._last_trigger is None
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py::TestMultiTriggerThresholds -v
```

Expected: 6 failures (most because `should_compress` doesn't take `messages`).
After implementation, 6 passes.

- [ ] **Step 3: Update `should_compress` signature**

Replace the existing `should_compress` method with:

```python
def should_compress(
    self,
    prompt_tokens: int = None,
    messages: List[Dict[str, Any]] | None = None,
) -> bool:
    """Multi-trigger compaction check.

    Triggers (any one fires; order = priority):
    - token: prompt_tokens >= threshold_tokens
    - message: len(messages) >= message_threshold (if set)
    - turn: count of user messages >= turn_threshold (if set)

    Anti-thrashing back-off still applies after any trigger fires.
    Records which trigger fired in ``self._last_trigger`` so callers
    (CompactionResult) can attribute cause.
    """
    self._last_trigger = None
    tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
    if tokens >= self.threshold_tokens:
        self._last_trigger = "token"
    elif messages is not None and self._fires_message_threshold(messages):
        self._last_trigger = "message"
    elif messages is not None and self._fires_turn_threshold(messages):
        self._last_trigger = "turn"
    if self._last_trigger is None:
        return False
    if self._ineffective_compression_count >= 2:
        if not self.quiet_mode:
            logger.warning(
                "Compression skipped — last %d compressions saved <10%% each.",
                self._ineffective_compression_count,
            )
        self._last_trigger = None
        return False
    return True

def _fires_message_threshold(self, messages: List[Dict[str, Any]]) -> bool:
    threshold = getattr(self, "message_threshold", None)
    return isinstance(threshold, int) and len(messages) >= threshold

def _fires_turn_threshold(self, messages: List[Dict[str, Any]]) -> bool:
    threshold = getattr(self, "turn_threshold", None)
    if not isinstance(threshold, int):
        return False
    user_count = sum(1 for m in messages if m.get("role") == "user")
    return user_count >= threshold
```

- [ ] **Step 4: Update the ABC + the single production callsite**

**4a. Update the abstract signature** in `agent/context_engine.py:72-74` so
plugin engines (LCM, etc.) that don't override `should_compress` still
satisfy the new contract:

```python
@abstractmethod
def should_compress(
    self,
    prompt_tokens: int = None,
    messages: List[Dict[str, Any]] | None = None,  # ← NEW
) -> bool:
    """Return True if compaction should fire this turn.

    Subclasses may ignore ``messages`` if they only key on token count;
    Hermes' built-in compressor uses it for multi-trigger thresholds.
    """
```

**4b. Update the single production callsite** at `run_agent.py:13280`
(verified via `grep -nE "(\\.|_)should_compress\\(" run_agent.py` — the
other `should_compress` matches are unrelated `error_classifier.py`
attributes). The local variable is `messages`:

```python
# Before:
if self.compression_enabled and _compressor.should_compress(_real_tokens):

# After:
if self.compression_enabled and _compressor.should_compress(
    _real_tokens, messages=messages,
):
```

**4c. Audit any plugin context-engine implementations.** Run:

```bash
grep -rn "def should_compress" --include="*.py" plugins/ 2>/dev/null
```

If any plugin defines its own `should_compress`, add `messages=None` to
its signature. (At time of writing, no plugin under `plugins/`
overrides this method — the ABC's default behavior of "ignore unknown
kwargs because they default to None" is enough.)

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py -v
```

Expected: all green. Existing call sites that didn't pass `messages` stay backward-compatible because the new triggers are no-ops when `messages` is None or the threshold attrs are unset.

- [ ] **Step 6: Commit**

```bash
git add agent/context_compressor.py run_agent.py tests/agent/test_context_compressor.py
git commit -m "feat(compaction): multi-trigger thresholds turn/message (P1d)"
```

---

## Task 5: REJECTED — `<think>` block stripping is redundant with existing storage-boundary strip

**Status:** removed from this plan after the second-pass review on 2026-05-02. Kept as a stub here so subsequent task numbering stays stable and so reviewers can see the rationale.

**Why removed:**
- `AIAgent._build_assistant_message` at `run_agent.py:8590-8602` already strips `<think>` (and 4 other reasoning-tag variants) from `content` at the storage boundary. Every assistant message reaches the compressor with `content` already cleaned. A compaction-time strip would find nothing to strip in well-behaved sessions.
- The local jinja template (`~/llama-stack/templates/qwen3.6-fixed.jinja:150-160`) emits historical `<think>` blocks in `qwen-thinking` mode (its `preserve_thinking is undefined → True` branch). If our strip ever did find a stray `<think>` in `qwen-thinking` mode it would degrade reasoning continuity — exactly opposite the intended effect.
- In `qwen-instruct` mode (`enable_thinking=false`), the template never emits historical `<think>` blocks regardless. A strip would do nothing observable from the model's perspective.
- The two cases together mean the feature is at best redundant and at worst harmful. The original report's recommendation was based on a mistaken assumption that `<think>` blocks survive into compaction input. They don't.

**What remains valid from the original idea:**
Nothing actionable for the local Qwen + Hermes path. If a future provider/path bypasses `_build_assistant_message` and lets `<think>` blocks reach stored content, *that* path should be fixed at the storage boundary, not papered over here.

(Subsequent tasks renumber starting at Task 6.)


---

## Task 6 (P2): Expose CompactionResult as `last_compaction_result`

**Files:**
- Modify: `agent/context_compressor.py` (`compress()` constructs and stores result)
- Modify: `tests/agent/test_compaction_result.py` (add integration test)

The compress() return type stays `List[Dict]` so the `ContextEngine` ABC contract is unchanged and any plugin engines (LCM) keep working. Per-event metrics land on `self.last_compaction_result` and are read by the gateway / status line / `/usage` command.

- [ ] **Step 1: Write the failing integration test**

The factory must satisfy two competing constraints: it has to bypass
`__init__` (no live LLM client / model metadata lookup), AND the message
list must produce a non-empty middle region so `compress()` actually runs
to completion (not the early-return path). The original draft's
`tail_token_budget=5000` swallowed all 10 short messages → middle region
empty → `compress()` returned early → `last_compaction_result` never
populated. Fixed by setting `tail_token_budget=300` so even 5 short
assistant turns can't all fit.

Append to `tests/agent/test_compaction_result.py`:

```python
class TestCompressorPopulatesResult:
    """compress() must populate ``self.last_compaction_result`` after each run."""

    def _compressor(self):
        from agent.context_compressor import ContextCompressor
        c = ContextCompressor.__new__(ContextCompressor)
        c.protect_first_n = 1
        c.protect_last_n = 2
        c.tail_token_budget = 300       # ← tight budget forces real middle region
        c.context_length = 200_000
        c.threshold_percent = 0.50
        c.threshold_tokens = 100_000
        c.threshold_absolute_max = None
        c.summary_target_ratio = 0.20
        c.max_summary_tokens = 8000
        c.dedup_operations = False
        c.anchor_first_assistant = False
        c.message_threshold = None
        c.turn_threshold = None
        c.quiet_mode = True
        c.compression_count = 0
        c.last_prompt_tokens = 50_000
        c._previous_summary = None
        c._summary_failure_cooldown_until = 0.0
        c._last_compression_savings_pct = 100.0
        c._ineffective_compression_count = 0
        c._last_summary_dropped_count = 0
        c._last_summary_fallback_used = False
        c._last_summary_error = None
        c._last_aux_model_failure_error = None
        c._last_aux_model_failure_model = None
        c._last_trigger = "token"
        c._last_op_deduped = 0
        c.summary_model = None
        c.model = "test"
        c.provider = "test"
        c.base_url = ""
        c.api_key = ""
        c.api_mode = "chat_completions"
        c.last_compaction_result = None
        return c

    def test_compress_populates_last_result(self, monkeypatch):
        from agent.context_compressor import ContextCompressor
        from agent.compaction_result import CompactionResult

        c = self._compressor()
        # Stub the LLM call to return a deterministic summary
        monkeypatch.setattr(
            ContextCompressor, "_generate_summary",
            lambda self, turns, focus_topic=None: "## Goal\nTest summary.",
        )
        # 12 messages, mixed roles. With protect_first_n=1, protect_last_n=2,
        # tail_token_budget=300, the middle region spans roughly idx 1..9
        # — non-empty so compress() actually runs.
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [
                m for i in range(5)
                for m in (
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": f"reply {i}"},
                )
            ]
            + [{"role": "user", "content": "final"}]
        )
        assert len(msgs) == 12

        out = c.compress(msgs, current_tokens=120_000)

        assert isinstance(c.last_compaction_result, CompactionResult)
        assert c.last_compaction_result.original_messages == len(msgs)
        assert c.last_compaction_result.compacted_messages == len(out)
        assert c.last_compaction_result.triggered_by == "token"
        # Middle was actually compressed (out should be shorter than in)
        assert len(out) < len(msgs)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
.venv/bin/pytest tests/agent/test_compaction_result.py -v
```

Expected: `TestCompressorPopulatesResult::test_compress_populates_last_result` fails (`last_compaction_result` is None).

- [ ] **Step 3: Add a top-of-file import + populate the result in `compress()`**

**3a. Add the import** at the top of `agent/context_compressor.py` next
to the other agent-package imports (around line 27-34):

```python
from agent.auxiliary_client import call_llm
from agent.compaction_result import CompactionResult     # ← NEW
from agent.context_engine import ContextEngine
```

(The `__init__` block in Task 0 has a local import workaround; replace
it with this top-level one once the file lands.)

**3b. Populate the result** near the end of `compress()`, just after
the existing `compressed = self._sanitize_tool_pairs(compressed)` line
and the existing `new_estimate = estimate_messages_tokens_rough(...)`
calculation (the variable already exists). Insert before the existing
"Anti-thrashing: track compression effectiveness" block:

```python
new_estimate = estimate_messages_tokens_rough(compressed)
saved_estimate = display_tokens - new_estimate

# Store per-event metrics for gateway/status-line/usage callers.
# Note: operations_deduped is the op-keyed dedup count from Pass 1.5
# only — NOT total `pruned_count`, which mixes content-hash dedup,
# op dedup, summarize-pass, and arg-truncate.
self.last_compaction_result = CompactionResult(
    original_messages=n_messages,
    compacted_messages=len(compressed),
    original_tokens=display_tokens,
    compacted_tokens=new_estimate,
    operations_deduped=getattr(self, "_last_op_deduped", 0),
    triggered_by=getattr(self, "_last_trigger", None) or "manual",
)
```

**Why getattr for fields the engine "should" have:** existing tests
(`test_compress_focus.py`, `test_context_compressor.py`) construct
compressors via `ContextCompressor.__new__(ContextCompressor)` and only
set the fields they directly read. The `getattr` defaults keep those
tests working without modification.

**Cross-task dependency check:** the `_last_op_deduped` field is set by
Task 1's wiring inside `_prune_old_tool_results`. Reset to 0 must
happen at the top of `compress()` itself (since Task 5 was rejected
and its earlier reset block is gone). Add this at the top of
`compress()` right after the `_min_for_compress` length check:

```python
# Reset per-event metric scratch
self._last_op_deduped = 0
```

This ensures that even on a no-op compress (no dedup), the metric is 0
not stale from a prior call.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/agent/test_compaction_result.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add agent/context_compressor.py tests/agent/test_compaction_result.py
git commit -m "feat(compaction): expose per-event metrics via last_compaction_result (P2)"
```

---

## Task 7: Local Qwen recommended config + user-facing docs

**Files:**
- Create: `docs/research/2026-05-02-qwen-aware-compaction-config.md`
- Modify: `~/.hermes/config.yaml` (user config — outside repo)

This task ships the documentation and applies the recommended flag set. No test needed — documentation lives or dies by being read.

- [ ] **Step 1: Write the user-facing doc**

Create `docs/research/2026-05-02-qwen-aware-compaction-config.md`:

```markdown
# Qwen-Aware Compaction — Configuration Guide

When using Hermes with the local Qwen3.6 servers (`qwen-instruct`,
`qwen-thinking`, `qwen-27b-vision`), enable the `qwen_aware` extensions
in `~/.hermes/config.yaml` to reduce per-turn wall clock.

## Why local Qwen needs different defaults

- llama.cpp builds (b8994 and the buun fork) cannot do partial-prefix KV
  cache reuse for Qwen3.6's hybrid Gated DeltaNet + Gated Attention
  architecture. Every turn re-processes the full prompt from scratch.
- Decode is ~60 tok/s (35B-A3B MoE) and ~33 tok/s (27B vision-tcq).
  Compaction is a real interactive-latency event, so we want it to
  fire **earlier** (smaller threshold) and **less often** (deterministic
  pre-pass that shrinks the input).
- Qwen3.6 in default mode (no `preserve_thinking`) expects only the
  *latest* user-turn's `<think>` block in history. Stripping older
  `<think>` blocks both saves tokens and aligns with the model's
  trained-in expectations.

## Recommended config block

```yaml
compression:
  enabled: true
  threshold: 0.50              # existing default
  target_ratio: 0.20           # existing default
  protect_last_n: 20           # existing default
  qwen_aware:
    enabled: true              # master switch for everything below
    dedup_operations: true     # P0a: collapse consecutive read/edit/terminal on same key
    anchor_first_assistant: true # P0b: never start compaction at a user message
    threshold_absolute_max: 80000 # P0c: never grow the prompt past 80K tokens
    message_threshold: 200       # P1d: catch many-short-turn sessions
    turn_threshold: 30           # P1d: catch long back-and-forth conversations

auxiliary:
  compression:
    provider: local-qwen          # pin summarizer to the cheap, non-thinking model
    model: qwen-instruct
    base_url: http://127.0.0.1:8085/v1
```

## Per-feature toggles

Every feature can be turned off independently by setting its flag to
`false` (or omitting it). Master `qwen_aware.enabled: false` disables
everything regardless of the per-feature flags.

| Flag | What it does | Safe to disable? |
|---|---|---|
| `dedup_operations` | Collapses earlier same-resource tool calls into back-references | Yes — Pass 2 (`_summarize_tool_result`) still runs |
| `anchor_first_assistant` | Slides compress-start to the first assistant msg | Yes — `_align_boundary_*` still runs |
| `threshold_absolute_max` | Caps the trigger threshold at an absolute token count | Yes — `threshold_percent` still applies |
| `message_threshold` / `turn_threshold` | Extra compaction triggers | Yes — token threshold still fires |

## When NOT to use these flags

- **You're using SGLang/vLLM with RadixAttention instead of
  llama.cpp** — RadixAttention does prefix-cache reuse for hybrid
  models. The cost asymmetry that motivates `threshold_absolute_max`
  is much smaller. Either skip the absolute cap or raise it
  significantly (e.g., 200_000).
- **You're on a cloud model (Claude / GPT / Gemini)** — keep
  `qwen_aware.enabled: false`. None of these tradeoffs apply; the
  prompt cache amortizes prompt re-processing.

## What's NOT in the qwen_aware block (and why)

- **`<think>` block stripping** — Hermes' `_build_assistant_message`
  already strips reasoning tags at the storage boundary
  (`run_agent.py:8590-8602`). Adding a compaction-time strip would be
  redundant in `qwen-instruct` mode and harmful in `qwen-thinking`
  mode (where the local jinja template emits historical `<think>`
  blocks that the model is trained to use).
- **`on_turn_end` trigger** — Hermes' single `should_compress`
  callsite fires inside the tool loop after tool results are
  appended, so `messages[-1]` is never a user message. The trigger
  would never fire without an architectural change (a preflight
  callsite). Use `message_threshold` / `turn_threshold` instead.

## Verifying the flags landed

```bash
hermes dump | grep -A 12 "compression:"
```

Should show your `qwen_aware:` block populated with the values above.
```

- [ ] **Step 2: Apply the recommended config to `~/.hermes/config.yaml`**

This is a *user* config edit, outside the repo. Open `~/.hermes/config.yaml` and add the `qwen_aware:` block under the existing `compression:` block (creating `compression:` if it doesn't exist).

If `auxiliary.compression.*` already points at the local Qwen server, skip that part. Otherwise add the auxiliary block too.

- [ ] **Step 3: Smoke-test the live config in a fresh Hermes session**

```bash
qwen-server status   # ensure moe profile is up
hermes -z "say hi"
hermes dump | grep -A 16 "compression:"
```

Expected: the dump output mirrors the `qwen_aware` block we wrote.

- [ ] **Step 4: Commit the doc**

```bash
git add docs/research/2026-05-02-qwen-aware-compaction-config.md
git commit -m "docs: qwen-aware compaction config guide for local Qwen3.6 servers"
```

(The user-config edit isn't tracked by git — it's already outside `HERMES_WRITE_SAFE_ROOT`.)

---

## Task 8: End-to-end smoke test against the live Qwen server

**Files:**
- Create: `tests/agent/test_compaction_e2e_qwen.py` (skipped unless qwen-server is up)

A behavior smoke test that runs the full `compress()` path against a real Hermes session with the local Qwen server. Skipped automatically if the server is down so CI on other environments doesn't break.

- [ ] **Step 1: Write the e2e test**

Create `tests/agent/test_compaction_e2e_qwen.py`:

```python
"""End-to-end compaction smoke test against the live local Qwen server.

Skipped automatically when http://127.0.0.1:8085/v1 is unreachable.
This is a behavior gate, not a unit test — its purpose is to catch
regressions in the moe-profile path that pure-Python unit tests miss.
"""

import socket
import pytest

from agent.context_compressor import ContextCompressor


def _qwen_server_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8085), timeout=0.5):
            return True
    except OSError:
        return False


qwen_required = pytest.mark.skipif(
    not _qwen_server_up(),
    reason="local-qwen moe profile not running on :8085",
)


@qwen_required
def test_full_pipeline_with_qwen_aware_flags():
    """Compaction should produce a non-empty summary using qwen-instruct.

    Constructs a synthetic conversation that exercises:
      - Pass 1.5 op-keyed dedup (two read_file calls on same path with
        DIFFERENT content — Pass 1's hash dedup won't catch them)
      - The first-assistant anchor (boundary at user msg slides forward)
      - The CompactionResult metric population
    """
    c = ContextCompressor(
        model="qwen-instruct",
        threshold_percent=0.50,
        protect_first_n=1,            # tight enough to exercise the anchor
        protect_last_n=2,
        summary_target_ratio=0.20,
        quiet_mode=True,
        base_url="http://127.0.0.1:8085/v1",
        api_key="not-needed",
        provider="local-qwen",
        api_mode="chat_completions",
        config_context_length=262_144,
        # qwen-aware flags ON
        qwen_aware_enabled=True,
        dedup_operations=True,
        anchor_first_assistant=True,
        threshold_absolute_max=80_000,
        message_threshold=200,
        turn_threshold=30,
    )
    # Build a synthetic history with redundant reads (DIFFERENT content
    # so Pass 1's hash dedup misses them and Pass 1.5 catches them).
    # Note: assistant content here is already-stripped (no <think> tags) —
    # that mirrors what _build_assistant_message produces at storage
    # boundary, so it's the realistic shape the compressor sees.
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Inspect /etc/hostname twice and summarize."},
        {"role": "assistant", "content": "I'll read it.", "tool_calls": [
            {"id": "1", "function": {"name": "read_file",
             "arguments": '{"path":"/etc/hostname"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "first-read-content-v1\n"},
        {"role": "assistant",
         "content": "Got it; reading again.",
         "tool_calls": [
             {"id": "2", "function": {"name": "read_file",
              "arguments": '{"path":"/etc/hostname"}'}},
         ]},
        {"role": "tool", "tool_call_id": "2", "content": "second-read-content-v2\n"},
        {"role": "assistant", "content": "Both reads complete."},
        {"role": "user", "content": "Now compact and tell me the gist."},
    ]
    out = c.compress(msgs, current_tokens=85_000)
    assert len(out) <= len(msgs)
    assert c.last_compaction_result is not None

    # Pass 1.5 should have superseded the FIRST tool result with a
    # back-reference (different content from the second, so Pass 1's
    # hash dedup doesn't catch it).
    assert c.last_compaction_result.operations_deduped >= 1, (
        f"Expected op-keyed dedup ≥ 1; got {c.last_compaction_result}"
    )

    # CompactionResult should attribute the trigger to "token" since we
    # passed current_tokens=85_000 above the 80_000 absolute cap.
    assert c.last_compaction_result.triggered_by == "token", (
        f"Expected token trigger; got {c.last_compaction_result.triggered_by}"
    )
```

- [ ] **Step 2: Run it (server up)**

```bash
qwen-server status
.venv/bin/pytest tests/agent/test_compaction_e2e_qwen.py -v
```

Expected: 1 passed (or 1 skipped if the server is genuinely down — which would be a bug to fix before this task is "done").

- [ ] **Step 3: Run the entire compaction-related test suite**

```bash
.venv/bin/pytest tests/agent/test_context_compressor.py \
                 tests/agent/test_compaction_atomicity.py \
                 tests/agent/test_compaction_result.py \
                 tests/agent/test_compaction_e2e_qwen.py \
                 tests/agent/test_compress_focus.py \
                 tests/agent/test_context_engine.py \
                 tests/cli/test_compress_focus.py \
                 tests/cli/test_manual_compress.py -v
```

Expected: all green.

- [ ] **Step 4: Run the Tier-1 benchmark suite**

The benchmark suite (companion plan: `2026-05-02-qwen-aware-compaction-benchmarks.md`) supplies numeric ship/no-ship signal on whether the changes actually improved performance/reliability/accuracy. Tier 1 is deterministic and runs in <30s — gate the merge on it.

```bash
.venv/bin/pytest tests/agent/benchmarks/test_tier1_*.py -v
```

Expected: all green AND the report at `docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md` shows:
- Benchmark 1.1 token reduction ≥ 15% on the dedup-friendly fixture
- Benchmark 1.2 ratio in [0.95, 1.05] on the neutral fixture
- Benchmark 1.3 wall-clock ≤ 1.20× baseline
- Benchmarks 1.5, 1.7, 1.8, 1.9 all pass (atomicity, anti-thrashing, anchor, multimodal preservation)

If Tier 2 (live Qwen) is feasible, also run `pytest tests/agent/benchmarks/test_tier2_*.py -v -s` for fact-retention validation. Tier 2 is recommended pre-merge but not strictly required if Tier 1 passes cleanly.

- [ ] **Step 5: Commit**

```bash
git add tests/agent/test_compaction_e2e_qwen.py tests/agent/benchmarks/
git commit -m "test(compaction): e2e smoke + Tier-1 benchmarks for qwen-aware path"
```

---

## Task 9: Manual interactive verification + merge

**Files:** none (verification + merge step)

This task validates that the end-to-end behavior matches expectations under real session shapes. No automated tests because the failure modes (the model getting confused by post-compaction state) require human judgment.

- [ ] **Step 1: Drive a long agent session that crosses the threshold**

```bash
qwen-server start moe -d
hermes -z "Read every Python file under agent/ and tell me the top 3 longest functions and their LOC."
```

Watch the agent loop. It should:
- Run several `read_file` calls
- Cross either `turn_threshold=30` or `threshold_absolute_max=80000`
- Emit a `compaction:` log line that includes `deduped N` (>0)
- Continue answering coherently after compaction

- [ ] **Step 2: Run `/compact` manually mid-session**

In a fresh Hermes session, accumulate ~50 turns of light back-and-forth, then:

```text
/compress
```

Expected: the response shows the new `last_compaction_result.summary_line()` text in its output (or via `/usage`).

- [ ] **Step 3: Disable each flag one at a time and confirm graceful degradation**

```bash
# In ~/.hermes/config.yaml, set qwen_aware.dedup_operations: false
hermes -z "say hi"   # confirm Hermes still starts
# Repeat for each flag, restoring after each test
```

Expected: every flag is independently togglable without restart-affecting fallout.

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "feat(compaction): qwen-aware extensions for local Qwen servers" \
  --body "$(cat <<'EOF'
## Summary
- Adds five flag-gated compaction improvements optimized for local Qwen3.6 servers (no llama.cpp partial-prefix KV cache reuse).
- All flags default OFF; cloud users see zero behavior change.
- Recommended local-Qwen config + docs in `docs/research/2026-05-02-qwen-aware-compaction-config.md`.

## What's new (each behind its own flag under `compression.qwen_aware`)
- **P0a** `dedup_operations` — supersedes earlier same-resource tool results with back-references (read/write/patch on same path; terminal on same command). Catches tail-region duplicates that existing Pass 2 misses.
- **P0b** `anchor_first_assistant` — slides compaction start to the first assistant message
- **P0c** `threshold_absolute_max` — combined-with-percentage absolute token cap
- **P1d** `message_threshold` / `turn_threshold` — extra compaction triggers
- **P2** `last_compaction_result` — per-event metrics dataclass on the engine instance

## Rejected during review (see plan Background section)
- `<think>` block stripping — redundant with `_build_assistant_message`'s storage-boundary strip; harmful in `qwen-thinking` mode
- `on_turn_end` trigger — Hermes' single `should_compress` callsite is post-tool-result, so `messages[-1]` is never a user message

## Design notes
- Patches the existing `ContextCompressor` rather than introducing a new engine class — flags default off so cloud users are unaffected.
- Preserves the `ContextEngine` ABC contract: `compress()` still returns `List[Dict]`. Metrics surface via a new attribute, not the return signature.
- The `_dedup_by_operation` skips multimodal tool results defensively — vision tools are not in the operation-key whitelist, but the skip prevents future foot-guns when new tools are added.

## Test plan
- [x] Unit tests for each new method in `tests/agent/test_context_compressor.py`
- [x] Atomicity tests in `tests/agent/test_compaction_atomicity.py`
- [x] Metrics tests in `tests/agent/test_compaction_result.py`
- [x] E2E smoke against local Qwen in `tests/agent/test_compaction_e2e_qwen.py` (auto-skipped without local server)
- [x] Manual session walkthrough (Task 9, steps 1-3)
EOF
)"
```

- [ ] **Step 5: After review, merge to main**

```bash
gh pr merge --squash
git checkout main && git pull
git branch -d feat/qwen-aware-compaction
```

---

## Self-Review (run after writing the plan)

**Spec coverage:** Original report items mapped: P0(a)→Task 1, P0(b)→Task 2, P0(c)→Task 3, P1(d)→Task 4 (without `on_turn_end`, see Background), P1(e)→**Task 5 REJECTED** (redundant + harmful, see Background), P2 metrics→Task 6, P2 summarizer-pin→Task 7 (config doc).

**Placeholder scan:** No "TBD" or "implement later." Every step has either runnable code, a runnable command with expected output, or both. The only "user-config" edit is Task 7 step 2, which is explicitly outside the repo and won't be tracked.

**Type consistency:** `CompactionResult.triggered_by` is typed `TriggerReason = Literal[...]` in Task 0. Task 4 sets `_last_trigger` to one of those exact strings (`"token"`, `"message"`, `"turn"`, or `"manual"` as fallback). Task 6 reads it back. The `_dedup_by_operation` method returns `tuple[List[Dict], int]` consistently in Task 1 and is consumed by `_prune_old_tool_results`.

**Safety:** Every behavior is behind a flag. Every flag defaults OFF. Each task ends with a single commit. Rollback = flip the flag. No DB changes, no schema migrations, no shared-state mutations.

**Vision safety (verified in second-pass review):**
- `_dedup_by_operation` operation-key whitelist excludes all vision tools (`vision_analyze`, `image_generate`, `browser_snapshot`, `browser_get_images`, `browser_vision`).
- Defensive skip on multimodal (list-content) tool results in `_dedup_by_operation` (mirrors existing Pass 1 / Pass 2 skip pattern).
- `_anchor_to_first_assistant` walks role==assistant only — doesn't touch user messages with image content.
- `_compute_threshold_tokens`, multi-trigger thresholds, `CompactionResult` are pure metadata — never mutate message content.

**Performance regression risks weighed:**
- Op-keyed dedup adds an O(n) walk (`n` = message count); negligible vs. existing token-counting passes.
- Multi-trigger `should_compress` adds at most two short walks per call; called once per turn boundary.
- `_compute_threshold_tokens` is called once per `__init__`/`update_model` — fixed cost.
- No hot-path regex over large content (`<think>` strip task was rejected, so no per-message regex).
- Pass 1.5 dedup runs in `_prune_old_tool_results` which is already a per-compaction event, not per-turn.

**Cross-task dependency check:**
- `self.last_compaction_result`, `self._last_trigger`, `self._last_op_deduped` are all initialized in `__init__` (Task 0 step 7). Existing tests that bypass `__init__` rely on `getattr(..., default)` reads in `compress()` for forward compatibility.
- Task 1's wiring sets `self._last_op_deduped` inside `_prune_old_tool_results`. The reset to 0 happens at the top of `compress()` (added in Task 6 step 3b). Task 6 reads it via `getattr` for the `CompactionResult`.
- Task 0 step 7 places `self.threshold_absolute_max = ...` BEFORE the `self.threshold_tokens = ...` calculation in `__init__`. Task 3 refactors that line to call `self._compute_threshold_tokens()`, which reads `self.threshold_absolute_max`. Order is preserved.
- The ABC `should_compress(prompt_tokens=None, messages=None)` is updated in Task 4 step 4a so plugin engines that inherit the abstract signature get the new contract. Plugin overrides under `plugins/` should be audited (Task 4 step 4c) — none currently override it.

**Tool-name accuracy:** `_FILE_OPS = {"read_file", "write_file", "patch"}` matches the actual Hermes tool registry (verified with `grep -rnE 'registry\.register\(name="' tools/`). The `scrapling_fetch_*` plugin is not on `main` at this commit — if added later, extend `_FILE_OPS`/`_operation_key` to include them.

**Open questions for the implementer:**
1. The `_dedup_by_operation` follows forgecode's "last-wins" semantics: read/write/patch on the same path all share a key, so a `patch` between two `read_file`s causes the first read to be superseded too. If a regression appears where the model loses needed pre-edit state, the workaround is to add an "intervening write breaks the chain" guard — but ship the simple version first and measure.
2. The e2e test in Task 8 hardcodes `127.0.0.1:8085` (moe profile). If we want it to also exercise vision-tcq (`:8086`), parameterize the test — but the moe path is the high-traffic one, so leave it for a follow-up.
3. `web_extract` takes a `urls` array (not a single `url`), so `_operation_key` returns `None` for it (not dedupable). If multi-URL dedup ever becomes desirable, key on `tuple(sorted(args["urls"]))`. For now, leave it out — the keying ambiguity (subset vs superset URLs) doesn't have a clearly correct answer.
4. The `qwen-instruct` summarizer inherits `presence_penalty=1.5` from the model config. Summaries may show more variance than e.g. a Gemini Flash summarizer. The strong "do not respond to questions" preamble in `_generate_summary` mitigates this, but if quality regresses, consider adding a per-aux sampling override (separate feature, not in this plan).
