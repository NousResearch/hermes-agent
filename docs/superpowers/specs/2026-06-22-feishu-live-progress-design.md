# Feishu Live Progress Card

## Product Goal

Replace the static "⏳ 正在思考..." placeholder with a single interactive card that updates in real-time throughout the entire agent turn, giving users continuous visibility into what the bot is doing.

### Core Experience (4 Layers)

| Layer | Timing | Card Shows |
|-------|--------|------------|
| Acknowledge | T=0s | "⏳ 正在思考..." |
| Thinking | T=0–first output | "⏳ 已思考 {N}s · {tool_semantic}" with timer |
| Progress | During tool calls + LLM streaming | Streaming text + tool status lines, accumulating |
| Final | Turn complete | Complete answer + runtime footer (tokens/cost/elapsed) |

One card, one message_id, patched in-place from start to finish.

### Acceptance Criteria

1. User sends a message → sees ACK card within 1s
2. During processing, card updates at least every 5s with elapsed time and current activity
3. Tool calls appear as semantic status lines ("📖 阅读文件", "💻 执行命令", etc.)
4. LLM streaming text appears progressively (flushed every ~2s)
5. Final answer replaces all progress content, with runtime footer
6. If card patching fails, content degrades to plain text — user never loses the answer
7. No duplicate messages — everything in one card per turn

---

## Architecture

### Approach: Adapter-Layer Interception

All changes are scoped to the Feishu adapter layer (`feishu.py` + `feishu_card.py`). The gateway (`run.py`) is not modified. The adapter intercepts `send()` and `edit_message()` calls from the gateway's existing progress consumer and redirects them to patch the ACK card instead of creating separate messages.

### Why This Approach

- Gateway's progress queue, tool_progress system, and streaming consumer already produce the right events at the right time — we just need to route their output to the card
- Keeps gateway platform-agnostic; all Feishu-specific logic stays in the adapter
- Matches NanoClaw's architecture where the channel layer owns the card lifecycle
- Minimizes risk: if the live card system fails, the existing text fallback path is untouched

---

## Detailed Design

### 1. LiveCardManager

A per-chat state machine managing the card lifecycle. One instance per `chat_id`, stored in `FeishuAdapter._live_cards: Dict[str, LiveCardManager]`.

**State Enum:**

```
IDLE → ACK_SENT → LIVE → FINALIZING → IDLE
```

| State | Trigger In | Trigger Out |
|-------|-----------|-------------|
| `IDLE` | Initial / after cleanup | `on_processing_start()` → `ACK_SENT` |
| `ACK_SENT` | ACK card sent | First `send()` or `edit_message()` → `LIVE` |
| `LIVE` | Receiving updates | `send()` with final content → `FINALIZING` |
| `FINALIZING` | Final patch in progress | Patch complete → `IDLE` |

**Held State:**

| Field | Type | Purpose |
|-------|------|---------|
| `card_message_id` | `str` | Feishu message_id of the ACK card (patch target) |
| `state` | `LiveCardState` | Current state enum |
| `accumulated_text` | `str` | Buffered streaming text (grows over time) |
| `tool_lines` | `list[str]` | Tool progress lines, e.g. "📖 阅读文件" |
| `started_at` | `float` | `time.monotonic()` at processing start |
| `last_tool` | `str` | Most recent tool name (for heartbeat display) |
| `heartbeat_task` | `asyncio.Task \| None` | Reference to cancel on completion |
| `last_patch_ts` | `float` | Monotonic timestamp of last successful patch |

**Concurrency:** Hermes Feishu adapter has per-chat serial locks (`_chat_locks`), so a single `chat_id` processes one message at a time. LiveCardManager does not need its own lock — the chat lock guarantees no concurrent turns on the same chat_id.

### 2. Card Content Assembly

LiveCardManager assembles card JSON on each patch by combining its held state:

**Progress State (LIVE):**

```markdown
{accumulated_text}

{tool_lines joined by " → "}

*⏳ 已思考 {elapsed}s · {last_tool_semantic}*
```

- `accumulated_text` is the primary body (streaming LLM output)
- Tool lines render as a chain: "📖 阅读文件 → 💻 执行命令 → 🔍 搜索代码"
- Heartbeat status is always the last line, italic, recalculated on each patch
- If no text yet (pure thinking), card shows only the heartbeat line

**Final State (FINALIZING):**

```markdown
{complete_answer_markdown}

---
📊 ↑{in} | ↓{out} | cache:{cache} | ${cost} | ⏳{elapsed} | 🧠{model}
```

- All progress lines and heartbeat removed
- Full answer + `build_card_footer_line()` appended via `<hr>` + `<note>` element
- Uses existing `runtime_footer` config fields

### 3. Three Update Sources

#### 3a. Heartbeat Timer (new)

Started by `on_processing_start()`, fires every 5 seconds:

```python
async def _heartbeat_loop(self, chat_id: str):
    live = self._live_cards[chat_id]
    while live.state in (ACK_SENT, LIVE):
        await asyncio.sleep(HEARTBEAT_INTERVAL)  # 5.0s
        if live.state not in (ACK_SENT, LIVE):
            break
        elapsed = time.monotonic() - live.started_at
        semantic = TOOL_SEMANTICS.get(live.last_tool, ("", ""))[1] if live.last_tool else ""
        status = f"⏳ 已思考 {int(elapsed)}s" + (f" · {semantic}" if semantic else "")
        await live.patch_heartbeat(status, ephemeral=True)
```

- Ephemeral: single attempt, silent failure, next tick retries
- Cancelled by `on_processing_complete()` or when state leaves LIVE

#### 3b. Tool Progress (intercepted from gateway)

Gateway's progress consumer calls `adapter.send()` with tool progress text. The interception in `send()`:

```python
async def send(self, chat_id, content, ...):
    live = self._live_cards.get(chat_id)
    if live and live.state in (ACK_SENT, LIVE):
        # Detect if this is a progress message vs final answer
        if not self._is_final_content(content, metadata):
            live.append_tool_line(content)
            live.state = LiveCardState.LIVE
            return await live.patch_card()
        else:
            # Final answer
            return await live.finalize(content, metadata)
    # No live card — original path
    ...
```

**Detecting final vs progress:** The gateway's progress consumer sends progress text through `send()`, and the final answer also comes through `send()`. Distinguishing them:
- Progress messages are sent while the gateway's progress consumer is active; these arrive as short status lines
- The final answer arrives after the agent turn completes; this is the `send()` call that, in the original code, would pop `_pending_ack_cards[chat_id]` and patch the ACK card
- Implementation: the existing `send()` already does `ack_msg_id = self._pending_ack_cards.pop(chat_id, None)` — when this pop succeeds, it's the final answer; when `_pending_ack_cards[chat_id]` still exists (peek, not pop), prior `send()` calls are progress
- As a safety net: if `metadata` contains `footer_line` or `status_text` keys (set by the gateway on final content), treat as final regardless

#### 3c. Streaming Text (intercepted from gateway)

Gateway's streaming consumer calls `adapter.edit_message()` to update text progressively. The interception:

```python
async def edit_message(self, chat_id, message_id, content, ...):
    live = self._live_cards.get(chat_id)
    if live and live.state in (ACK_SENT, LIVE):
        live.accumulated_text = content  # Replace with latest accumulated text
        live.state = LiveCardState.LIVE
        return await live.patch_card()
    # No live card — original path
    ...
```

### 4. Throttling

| Constant | Value | Purpose |
|----------|-------|---------|
| `HEARTBEAT_INTERVAL` | 5.0s | Time between heartbeat patches |
| `MIN_PATCH_INTERVAL` | 1.5s | Minimum gap between any two patches |
| `TEXT_FLUSH_DELAY` | 2.0s | How long to buffer streaming text before patching |

**Collision resolution:** If a text flush and heartbeat both want to patch within `MIN_PATCH_INTERVAL`:
- Text flush wins (carries real content)
- Heartbeat skips this tick — next tick picks up the latest state anyway

**Implementation:** `LiveCardManager.patch_card()` checks `time.monotonic() - last_patch_ts < MIN_PATCH_INTERVAL`. If too soon, it schedules a deferred patch via `asyncio.get_event_loop().call_later()` instead of dropping the update entirely.

### 5. Three-Tier Degradation

Reference: NanoClaw's `card-renderer.ts` retry + fallback strategy.

| Tier | Trigger | Behavior |
|------|---------|----------|
| L1: Retry | Patch returns 429 / 5xx / network timeout | Exponential backoff retry, 3 attempts (1s, 3s, 9s) |
| L2: Degrade | 4xx (non-429), or retries exhausted | Set `live.degraded = True`; subsequent content goes through original `send()` as text messages |
| L3: Mark | L2 triggered | Patch the ACK card one last time: "⚠️ 卡片更新失败，内容已转为普通消息发出" |

**Ephemeral updates** (heartbeat, tool activity): Single attempt, no retry, no degradation trigger. If it fails, the next heartbeat will retry with fresh state.

**Feishu-specific:** Feishu error code 230020 (frequency limit) is treated as transient (L1 retry) despite returning HTTP 400, matching NanoClaw's handling.

### 6. Card Element Validation

Pre-flight checks before every `_patch_card()` call, preventing Feishu API rejections:

| Check | Limit | Degradation |
|-------|-------|-------------|
| Single markdown block length | 4000 chars | Split by paragraphs into multiple markdown elements |
| Table count | 5 per card | Excess tables converted to markdown-formatted text tables |
| Element count | 30 per card | Merge adjacent markdown elements |
| Total card bytes | 24,000 bytes | Truncate from end, append "...(内容过长已截断)" |

Implemented as `CardElementValidator` class in `feishu_card.py`. Called by `build_progress_card_json()` and `build_card_json()` before returning the card dict.

### 7. Lifecycle Integration Points

**`on_processing_start(event)`** — entry:
```
1. Add reaction (existing)
2. Send ACK card (existing) → capture message_id
3. Create LiveCardManager(chat_id, message_id)
4. Start heartbeat task
5. State = ACK_SENT
```

**`send(chat_id, content, ...)`** — interception:
```
if live card exists and not degraded:
    if is_final_content:
        cancel heartbeat
        patch card with final answer + footer
        state = FINALIZING → IDLE
        remove from _live_cards
    else:
        append to tool_lines or accumulated_text
        state = LIVE
        patch card (throttled)
    return SendResult
else:
    original send() path
```

**`edit_message(chat_id, message_id, content, ...)`** — interception:
```
if live card exists and not degraded:
    update accumulated_text
    state = LIVE
    patch card (throttled)
    return SendResult
else:
    original edit_message() path
```

**`on_processing_complete(event, outcome)`** — cleanup:
```
1. Cancel heartbeat task
2. Remove reaction (existing)
3. Remove LiveCardManager from _live_cards
```

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `gateway/platforms/feishu.py` | Add `LiveCardManager` class; modify `on_processing_start`, `on_processing_complete`, `send()`, `edit_message()` | +250, ~30 modified |
| `gateway/platforms/feishu_card.py` | Add `build_progress_card_json()`, `CardElementValidator` class | +150 |
| `tests/gateway/test_feishu_live_card.py` | New test file: state machine, interception, throttling, degradation, validation | +300 |
| `gateway/run.py` | **No changes** | 0 |

## Not In Scope

- Other platforms (Telegram, Discord, etc.)
- Config schema changes — uses existing `_card_mode_enabled` flag
- Gateway-level refactoring
- Backward-incompatible changes to card footer format
