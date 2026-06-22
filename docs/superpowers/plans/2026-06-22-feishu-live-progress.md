# Implementation Plan: Feishu Live Progress Card

**Spec**: `docs/superpowers/specs/2026-06-22-feishu-live-progress-design.md`
**Branch**: `feat/feishu-live-progress`
**Test file**: `tests/gateway/test_feishu_live_card.py`

---

## File Change Scope

| File | What changes |
|------|-------------|
| `gateway/platforms/feishu_card.py` | +`CardElementValidator`, +`build_progress_card_json()` |
| `gateway/platforms/feishu.py` | +`LiveCardManager` class, modify `on_processing_start`, `on_processing_complete`, `send()`, `edit_message()` |
| `tests/gateway/test_feishu_live_card.py` | New test file (all tests) |
| `gateway/run.py` | **NO CHANGES** |

---

## Task Breakdown (TDD Cycles)

### Task 1: CardElementValidator — markdown block splitting

**Test** (`test_feishu_live_card.py`):
```python
class TestCardElementValidator:
    def test_short_markdown_unchanged(self):
        elements = [{"tag": "markdown", "content": "hello"}]
        result = CardElementValidator.validate(elements)
        assert result == elements

    def test_long_markdown_split_by_paragraphs(self):
        long_text = "\n\n".join([f"paragraph {i}" for i in range(50)])
        elements = [{"tag": "markdown", "content": long_text}]
        result = CardElementValidator.validate(elements)
        assert all(len(e["content"]) <= 4000 for e in result if e["tag"] == "markdown")
        rejoined = "\n\n".join(e["content"] for e in result if e["tag"] == "markdown")
        assert rejoined == long_text

    def test_unsplittable_long_block_hard_truncated(self):
        huge = "x" * 5000
        elements = [{"tag": "markdown", "content": huge}]
        result = CardElementValidator.validate(elements)
        assert len(result[0]["content"]) <= 4000
        assert result[0]["content"].endswith("...(内容过长已截断)")
```

**Implement** in `feishu_card.py`:
- `CardElementValidator` class with static `validate(elements: list[dict]) -> list[dict]`
- `_split_markdown(content: str, max_chars: int = 4000) -> list[str]` — split by `\n\n` paragraphs, join up to limit, hard-truncate if single paragraph exceeds
- Constants: `MAX_MARKDOWN_CHARS = 4000`

**Cycle**: write tests → `pytest tests/gateway/test_feishu_live_card.py::TestCardElementValidator::test_short_markdown_unchanged -x` (fail) → implement → run (pass) → commit

---

### Task 2: CardElementValidator — table count limit

**Test**:
```python
    def test_tables_within_limit_unchanged(self):
        elements = [{"tag": "table", "columns": [...], "rows": [...]}] * 5
        result = CardElementValidator.validate(elements)
        assert sum(1 for e in result if e["tag"] == "table") == 5

    def test_excess_tables_converted_to_markdown(self):
        elements = [{"tag": "table", "columns": [{"name": "a", "display_name": "A"}],
                      "rows": [{"a": "1"}]}] * 7
        result = CardElementValidator.validate(elements)
        assert sum(1 for e in result if e["tag"] == "table") <= 5
```

**Implement**: `_enforce_table_limit(elements, max_tables=5)` — converts excess tables to markdown-formatted text (pipe syntax).

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 3: CardElementValidator — element count & byte size

**Test**:
```python
    def test_elements_merged_when_over_limit(self):
        elements = [{"tag": "markdown", "content": f"line {i}"} for i in range(35)]
        result = CardElementValidator.validate(elements)
        assert len(result) <= 30

    def test_total_bytes_truncated(self):
        big = "A" * 20000
        elements = [{"tag": "markdown", "content": big}, {"tag": "markdown", "content": big}]
        result = CardElementValidator.validate(elements)
        total = len(json.dumps(result, ensure_ascii=False).encode())
        assert total <= 24000
```

**Implement**:
- `_merge_adjacent_markdown(elements, max_elements=30)` — merge adjacent `markdown` tags
- `_enforce_byte_limit(elements, max_bytes=24000)` — drop from end if needed, append truncation notice
- Constants: `MAX_ELEMENTS = 30`, `MAX_CARD_BYTES = 24000`, `MAX_TABLES = 5`
- `validate()` pipeline: `_enforce_table_limit` → `_split_markdown` (each element) → `_merge_adjacent_markdown` → `_enforce_byte_limit`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 4: build_progress_card_json

**Test**:
```python
class TestBuildProgressCardJson:
    def test_thinking_only(self):
        card = build_progress_card_json(
            accumulated_text="",
            tool_lines=[],
            status_line="⏳ 已思考 5s",
        )
        assert card["config"]["update_multi"] is True
        md_elements = [e for e in card["elements"] if e["tag"] == "markdown"]
        assert any("已思考 5s" in e["content"] for e in md_elements)

    def test_with_text_and_tools(self):
        card = build_progress_card_json(
            accumulated_text="Here is my analysis...",
            tool_lines=["📖 阅读文件", "💻 执行命令"],
            status_line="⏳ 已思考 12s · 执行命令",
        )
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "Here is my analysis" in md
        assert "📖 阅读文件" in md
        assert "已思考 12s" in md

    def test_tool_chain_separator(self):
        card = build_progress_card_json(
            accumulated_text="",
            tool_lines=["📖 阅读文件", "💻 执行命令", "🔍 搜索代码"],
            status_line="⏳ 已思考 8s",
        )
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "→" in md

    def test_validation_applied(self):
        huge_text = "x" * 5000
        card = build_progress_card_json(
            accumulated_text=huge_text,
            tool_lines=[],
            status_line="⏳ 已思考 3s",
        )
        for e in card["elements"]:
            if e["tag"] == "markdown":
                assert len(e["content"]) <= 4000
```

**Implement** in `feishu_card.py`:
```python
def build_progress_card_json(
    *,
    accumulated_text: str,
    tool_lines: list[str],
    status_line: str,
) -> dict:
```
- Assembles: accumulated_text body → tool chain line (joined by " → ") → italic status_line
- Calls `CardElementValidator.validate()` before returning
- Card config: `wide_screen_mode: True, update_multi: True`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 5: LiveCardManager — state machine basics

**Test**:
```python
class TestLiveCardManager:
    def test_initial_state_is_idle(self):
        mgr = LiveCardManager()
        assert mgr.state == LiveCardState.IDLE

    def test_start_sets_ack_sent(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        assert mgr.state == LiveCardState.ACK_SENT
        assert mgr.card_message_id == "msg_001"
        assert mgr.started_at == 100.0

    def test_accept_text_transitions_to_live(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("hello")
        assert mgr.state == LiveCardState.LIVE
        assert mgr.accumulated_text == "hello"

    def test_append_tool_line(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.append_tool("Read")
        assert mgr.state == LiveCardState.LIVE
        assert len(mgr.tool_lines) == 1
        assert "阅读文件" in mgr.tool_lines[0]
        assert mgr.last_tool == "Read"

    def test_reset_clears_all(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("text")
        mgr.append_tool("Bash")
        mgr.reset()
        assert mgr.state == LiveCardState.IDLE
        assert mgr.accumulated_text == ""
        assert mgr.tool_lines == []
        assert mgr.card_message_id is None
```

**Implement** in `feishu.py`:
- `LiveCardState` enum: `IDLE, ACK_SENT, LIVE, FINALIZING`
- `LiveCardManager` dataclass/class with fields: `state`, `card_message_id`, `accumulated_text`, `tool_lines`, `started_at`, `last_tool`, `last_patch_ts`, `heartbeat_task`, `degraded`
- Methods: `start()`, `update_text()`, `append_tool()`, `reset()`, `build_card()`, `should_throttle()`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 6: LiveCardManager — build_card assembly & throttling

**Test**:
```python
    def test_build_card_ack_state(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        card = mgr.build_card(now=105.0)
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "已思考 5s" in md

    def test_build_card_live_state_with_text(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=100.0)
        mgr.update_text("analysis result")
        mgr.append_tool("Read")
        card = mgr.build_card(now=112.0)
        md = "\n".join(e["content"] for e in card["elements"] if e["tag"] == "markdown")
        assert "analysis result" in md
        assert "阅读文件" in md
        assert "12s" in md

    def test_should_throttle_within_interval(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.last_patch_ts = 10.0
        assert mgr.should_throttle(now=10.5) is True   # 0.5s < 1.5s
        assert mgr.should_throttle(now=12.0) is False   # 2.0s > 1.5s

    def test_should_throttle_never_patched(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        assert mgr.should_throttle(now=0.1) is False  # first patch always OK
```

**Implement**:
- `build_card(now: float) -> dict` — calls `build_progress_card_json()` with current state
- `should_throttle(now: float) -> bool` — checks `now - last_patch_ts < MIN_PATCH_INTERVAL`
- Constants at module level: `MIN_PATCH_INTERVAL = 1.5`, `HEARTBEAT_INTERVAL = 5.0`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 7: LiveCardManager — degradation flag

**Test**:
```python
    def test_mark_degraded(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        assert mgr.degraded is False
        mgr.mark_degraded()
        assert mgr.degraded is True

    def test_build_card_still_works_when_degraded(self):
        mgr = LiveCardManager()
        mgr.start("msg_001", started_at=0.0)
        mgr.update_text("some text")
        mgr.mark_degraded()
        card = mgr.build_card(now=5.0)
        assert card is not None  # can still build for L3 farewell patch
```

**Implement**: `mark_degraded()` method, `degraded: bool` field (default False).

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 8: Adapter — on_processing_start creates LiveCardManager

**Test**:
```python
class TestFeishuLiveCardIntegration:
    @pytest.fixture
    def adapter(self):
        # Minimal FeishuAdapter with mocked _client
        ...

    @pytest.mark.asyncio
    async def test_on_processing_start_creates_live_card(self, adapter):
        event = make_event(chat_id="chat_001", message_id="msg_in_001")
        await adapter.on_processing_start(event)
        assert "chat_001" in adapter._live_cards
        live = adapter._live_cards["chat_001"]
        assert live.state == LiveCardState.ACK_SENT
        assert live.card_message_id is not None

    @pytest.mark.asyncio
    async def test_on_processing_start_without_card_mode(self, adapter):
        adapter._card_mode_enabled = False
        event = make_event(chat_id="chat_001", message_id="msg_in_001")
        await adapter.on_processing_start(event)
        assert "chat_001" not in adapter._live_cards
```

**Implement**:
- Add `self._live_cards: Dict[str, LiveCardManager] = {}` to `__init__`
- In `on_processing_start()`: after sending ACK card, create `LiveCardManager` and call `start(ack_message_id, started_at=time.monotonic())`
- Start heartbeat task

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 9: Adapter — send() interception (progress vs final)

**Test**:
```python
    @pytest.mark.asyncio
    async def test_send_progress_patches_card(self, adapter):
        # Set up live card in ACK_SENT state
        adapter._live_cards["chat_001"] = make_live_card(state=ACK_SENT, msg_id="ack_001")
        adapter._pending_ack_cards["chat_001"] = "ack_001"
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="ack_001"))
        # send() called with progress (pending_ack_cards peek, not pop)
        result = await adapter.send("chat_001", "Reading config.yaml...")
        assert result.success
        adapter._patch_card.assert_called_once()
        # ACK card still in pending_ack_cards (not consumed)
        assert "chat_001" in adapter._pending_ack_cards

    @pytest.mark.asyncio
    async def test_send_final_patches_and_cleans_up(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=LIVE, msg_id="ack_001")
        adapter._pending_ack_cards["chat_001"] = "ack_001"
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="ack_001"))
        # Final answer: pending_ack_cards will be popped (existing behavior)
        result = await adapter.send("chat_001", "Here is my full answer...",
                                     metadata={"footer_line": "📊 ↑1k | ↓2k"})
        assert result.success
        assert "chat_001" not in adapter._pending_ack_cards
        assert "chat_001" not in adapter._live_cards
```

**Implement** — modify `send()`:
1. Before the existing `ack_msg_id = self._pending_ack_cards.pop(...)` logic, check if live card exists
2. **Progress path**: if `chat_id` has live card AND `_pending_ack_cards[chat_id]` exists (peek without pop) AND no `footer_line`/`status_text` in metadata → treat as progress update → `live.update_text(content)` or `live.append_tool(...)`, throttled `_patch_card()`
3. **Final path**: if live card exists AND (`footer_line` in metadata OR the `pop()` succeeds) → build final card with footer, patch, clean up live card
4. Detection heuristic: the gateway sets `metadata["footer_line"]` on final answers; progress messages have no such key

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 10: Adapter — edit_message() interception

**Test**:
```python
    @pytest.mark.asyncio
    async def test_edit_message_updates_accumulated_text(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=LIVE, msg_id="ack_001")
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="ack_001"))
        result = await adapter.edit_message("chat_001", "progress_msg_001",
                                            "Updated streaming text...")
        assert result.success
        assert adapter._live_cards["chat_001"].accumulated_text == "Updated streaming text..."

    @pytest.mark.asyncio
    async def test_edit_message_no_live_card_falls_through(self, adapter):
        # No live card → original edit path
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="msg_001"))
        result = await adapter.edit_message("chat_001", "msg_001", "edited text")
        # Should use original edit_message logic (card mode edit via _patch_card)
        assert result.success
```

**Implement** — modify `edit_message()`:
- At the top: check if live card exists and is in `ACK_SENT` or `LIVE` state
- If yes: `live.update_text(content)`, throttled `_patch_card(live.card_message_id, live.build_card())`
- If no: fall through to existing edit_message logic

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 11: Adapter — on_processing_complete cleanup

**Test**:
```python
    @pytest.mark.asyncio
    async def test_on_processing_complete_cancels_heartbeat(self, adapter):
        live = make_live_card(state=LIVE, msg_id="ack_001")
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        live.heartbeat_task = mock_task
        adapter._live_cards["chat_001"] = live
        event = make_event(chat_id="chat_001", message_id="msg_in_001")
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert "chat_001" not in adapter._live_cards
        mock_task.cancel.assert_called_once()
```

**Implement** — modify `on_processing_complete()`:
- Before existing reaction cleanup: if `chat_id` has live card, cancel heartbeat, remove from `_live_cards`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 12: Heartbeat task lifecycle

**Test**:
```python
    @pytest.mark.asyncio
    async def test_heartbeat_patches_periodically(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=ACK_SENT, msg_id="ack_001")
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="ack_001"))
        # Run heartbeat for just over one interval
        task = asyncio.create_task(adapter._heartbeat_loop("chat_001"))
        await asyncio.sleep(0.1)  # Let first tick fire (interval mocked to 0.05s)
        adapter._live_cards["chat_001"].reset()  # Transitions to IDLE, stops loop
        await asyncio.sleep(0.1)
        assert task.done() or task.cancelled()

    @pytest.mark.asyncio
    async def test_heartbeat_failure_is_silent(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=ACK_SENT, msg_id="ack_001")
        adapter._patch_card = AsyncMock(return_value=SendResult(success=False, error="network"))
        task = asyncio.create_task(adapter._heartbeat_loop("chat_001"))
        await asyncio.sleep(0.1)
        adapter._live_cards["chat_001"].reset()
        await asyncio.sleep(0.1)
        # Should not raise, just skip the failed patch
```

**Implement**:
- `async def _heartbeat_loop(self, chat_id: str)` on FeishuAdapter
- Loop: `while live.state in (ACK_SENT, LIVE): sleep(HEARTBEAT_INTERVAL), build_card, _patch_card (single attempt, catch exceptions)`
- Started in `on_processing_start()`, reference stored in `live.heartbeat_task`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 13: Three-tier degradation in patch flow

**Test**:
```python
class TestLiveCardDegradation:
    @pytest.mark.asyncio
    async def test_retry_on_429(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=LIVE, msg_id="ack_001")
        # First call returns 429-like, second succeeds
        adapter._patch_card = AsyncMock(side_effect=[
            SendResult(success=False, error="429 rate limited"),
            SendResult(success=True, message_id="ack_001"),
        ])
        result = await adapter._live_card_patch("chat_001")
        assert result.success
        assert adapter._patch_card.call_count == 2

    @pytest.mark.asyncio
    async def test_degrade_to_text_on_4xx(self, adapter):
        adapter._live_cards["chat_001"] = make_live_card(state=LIVE, msg_id="ack_001")
        adapter._patch_card = AsyncMock(
            return_value=SendResult(success=False, error="403 forbidden"))
        await adapter._live_card_patch("chat_001")
        assert adapter._live_cards["chat_001"].degraded is True

    @pytest.mark.asyncio
    async def test_degraded_send_uses_text_path(self, adapter):
        live = make_live_card(state=LIVE, msg_id="ack_001")
        live.mark_degraded()
        adapter._live_cards["chat_001"] = live
        adapter._pending_ack_cards["chat_001"] = "ack_001"
        # send() should skip live card path and use original text path
        result = await adapter.send("chat_001", "final answer")
        # Verify it fell through to original send logic
```

**Implement**:
- `async def _live_card_patch(self, chat_id: str) -> SendResult` — wrapper around `_patch_card` with retry logic
- L1: retry on 429 / 5xx / network error, exponential backoff (1s, 3s, 9s), max 3 attempts
- L2: on non-retryable 4xx or retries exhausted → `live.mark_degraded()`
- L3: one last patch attempt with "⚠️ 卡片更新失败" message
- Error code 230020 (Feishu frequency limit) treated as retryable despite HTTP 400
- In `send()` / `edit_message()` interception: skip live card path if `live.degraded`

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 14: Deferred patch scheduling

**Test**:
```python
    @pytest.mark.asyncio
    async def test_throttled_patch_deferred(self, adapter):
        live = make_live_card(state=LIVE, msg_id="ack_001")
        live.last_patch_ts = time.monotonic()  # just patched
        adapter._live_cards["chat_001"] = live
        adapter._patch_card = AsyncMock(return_value=SendResult(success=True, message_id="ack_001"))
        # Immediate call should be throttled
        result = await adapter._try_patch_live_card("chat_001")
        # Patch should be deferred, not called immediately
        assert adapter._patch_card.call_count == 0
        # Wait for deferred fire
        await asyncio.sleep(MIN_PATCH_INTERVAL + 0.1)
        assert adapter._patch_card.call_count == 1
```

**Implement**:
- `async def _try_patch_live_card(self, chat_id: str) -> Optional[SendResult]` — checks `should_throttle()`, if throttled schedules `call_later()`, returns None
- Deferred callback: `_deferred_patch_tasks: Dict[str, asyncio.TimerHandle]` — at most one per chat_id, newer deferred replaces older

**Cycle**: write tests → run (fail) → implement → run (pass) → commit

---

### Task 15: End-to-end lifecycle test

**Test**:
```python
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, adapter):
        """ACK → progress edits → tool activity → final answer → cleanup"""
        event = make_event(chat_id="chat_001", message_id="msg_in_001")

        # 1. Processing start → ACK card sent, live card created
        await adapter.on_processing_start(event)
        assert adapter._live_cards["chat_001"].state == LiveCardState.ACK_SENT

        # 2. Streaming text → card patched with accumulated text
        await adapter.edit_message("chat_001", "progress_001", "Analyzing code...")
        assert adapter._live_cards["chat_001"].state == LiveCardState.LIVE

        # 3. Tool progress → card patched with tool line
        await adapter.send("chat_001", "Read · 阅读文件: config.yaml")
        assert len(adapter._live_cards["chat_001"].tool_lines) >= 1

        # 4. Final answer → card patched with answer + footer, live card removed
        await adapter.send("chat_001", "Here is the answer.",
                          metadata={"footer_line": "📊 stats"})
        assert "chat_001" not in adapter._live_cards

        # 5. Processing complete → no crash (live card already gone)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
```

**Implement**: No new code — this validates the integration of tasks 8-12.

**Cycle**: run test → must pass → commit

---

## TDD Test Matrix

| Test Class | Test Count | Covers |
|-----------|-----------|--------|
| `TestCardElementValidator` | 6 | markdown split, table limit, element merge, byte truncation |
| `TestBuildProgressCardJson` | 4 | thinking-only card, text+tools, tool chain separator, validation |
| `TestLiveCardManager` | 9 | state transitions, build_card assembly, throttling, degradation |
| `TestFeishuLiveCardIntegration` | 8 | adapter lifecycle, send/edit interception, heartbeat, cleanup |
| `TestLiveCardDegradation` | 3 | retry, degrade, degraded-send fallback |
| `TestDeferredPatch` | 1 | throttled patch scheduling |
| `TestFullLifecycle` | 1 | end-to-end ACK→progress→final |
| **Total** | **32** | |

## Log Coverage Matrix

| Event | Log Level | Message Pattern | Location |
|-------|----------|----------------|----------|
| Live card created | INFO | `[Feishu] LiveCard created: {chat_id} → {msg_id}` | `on_processing_start` |
| State transition | DEBUG | `[Feishu] LiveCard {chat_id}: {old_state} → {new_state}` | `LiveCardManager.start/update_text/append_tool/reset` |
| Card patched | DEBUG | `[Feishu] LiveCard patch: {chat_id} (elapsed={N}s)` | `_try_patch_live_card` |
| Patch throttled | DEBUG | `[Feishu] LiveCard patch throttled: {chat_id}, deferring {delta}s` | `_try_patch_live_card` |
| Heartbeat tick | DEBUG | `[Feishu] LiveCard heartbeat: {chat_id} {elapsed}s` | `_heartbeat_loop` |
| Heartbeat failed (silent) | DEBUG | `[Feishu] LiveCard heartbeat patch failed: {error}` | `_heartbeat_loop` |
| Retry (L1) | WARNING | `[Feishu] LiveCard patch retry {attempt}/3: {error}` | `_live_card_patch` |
| Degraded (L2) | WARNING | `[Feishu] LiveCard degraded to text: {chat_id} reason={error}` | `_live_card_patch` |
| Final card patched | INFO | `[Feishu] LiveCard finalized: {chat_id}` | `send()` final path |
| Live card cleanup | DEBUG | `[Feishu] LiveCard cleanup: {chat_id}` | `on_processing_complete` |
| Progress intercepted | DEBUG | `[Feishu] LiveCard intercepted progress send: {chat_id}` | `send()` |
| Streaming intercepted | DEBUG | `[Feishu] LiveCard intercepted edit: {chat_id}` | `edit_message()` |

## Self-Commitment

- [ ] `gateway/run.py` has exactly 0 lines changed
- [ ] Every task follows TDD: test written first, run to see failure, then implement
- [ ] All 32 tests pass before merging
- [ ] CardElementValidator enforces all 4 limits (markdown 4000, tables 5, elements 30, bytes 24000)
- [ ] Degradation path tested: if card patching fails, user still gets the answer as text
- [ ] Heartbeat fires silently — failures never propagate to the user or crash the adapter
- [ ] No new dependencies added — uses only stdlib + existing `lark_oapi`
- [ ] `_pending_ack_cards` pop-vs-peek logic correctly distinguishes progress from final

## Inducement Self-Answer

**Q: What's the riskiest part of this design?**

The `send()` interception is the riskiest: it must correctly distinguish progress messages from the final answer. The heuristic (peek vs pop of `_pending_ack_cards` + check for `footer_line` in metadata) has two failure modes:
1. **False positive** (progress treated as final): card gets "finalized" too early; remaining content arrives as separate messages. Mitigated by: only `footer_line` in metadata triggers finalization, and the gateway only sets this on the actual final send.
2. **False negative** (final treated as progress): card never shows the complete answer. Mitigated by: the `_pending_ack_cards.pop()` in the existing code path acts as a definitive final-answer signal; we preserve this semantic exactly.

The degradation path is the safety net: even if interception logic has bugs, `live.degraded = True` causes all subsequent messages to fall through to the original text path. The user always gets the answer.

**Q: Why not modify `run.py` to pass explicit "is_progress" flags?**

Modifying `run.py` would be cleaner semantically but violates the design constraint (gateway stays platform-agnostic). The adapter-layer detection is good enough because the gateway's calling conventions are stable and well-documented in the existing code. Adding coupling to `run.py` would mean every future gateway change risks breaking the live card system.
