# WeCom Native Streaming 改造记录

> **分支**：`feat/wecom-native-streaming`  
> **日期**：2026-06-04 ~ 2026-06-07  
> **基于**：hermes-agent main (`40420a619`)  
> **最新 commit**：`c49384360`

## 背景

Hermes Agent 的 WeCom 适配器原本不支持流式输出（`SUPPORTS_MESSAGE_EDITING = False`），用户发消息后要等 LLM 完整生成才能看到回复。

本改造对齐腾讯官方 `@wecom/wecom-openclaw-plugin` 行为，使用 WeCom AI Bot 的 `msgtype: "stream"` 协议实现原生流式输出。

## 核心架构

### Per-Turn 模型

每个 `GatewayStreamConsumer` 实例持有独立的 `turn_id`（UUID），WeCom adapter 以 `(chat_id, turn_id)` 为 key 管理 `StreamTurn` 状态。多个并发 consumer（不同用户、不同 chat、同 chat 多任务）完全隔离。

```python
class StreamTurn:
    chat_id: str
    req_id: str        # 绑定的 WeCom req_id（创建时锁定）
    stream_id: str     # 生成的 stream_id
    seeded: bool       # seed frame 是否已发送
    finalized: bool
    expired: bool
```

### send() 和 streaming 完全独立

`send()` 只发送独立 markdown 消息，**从不触碰任何 active stream**。Stream 生命周期完全由 `GatewayStreamConsumer` 通过 `send_stream_frame(turn_id=...)` 管理。

### 双通道 Priority Queue

```
Control Lane (6 reserved tokens):  approval prompts, finalize frames
Normal Lane  (24 tokens):          普通消息
Fire-and-forget (背压可跳过):      stream 中间帧
```

Token bucket: 30 tokens/minute/chat，按分钟窗口重置。

## 流式协议

### 生命周期

```
1. Consumer run() → send_stream_frame("", turn_id=X)
   → Adapter: 创建 StreamTurn, 发 seed frame "<think></think>", 设 turn.seeded=True
   → WeCom 客户端显示 thinking bubble（对齐官方插件 THINKING_MESSAGE）

2. Agent 生成内容 → on_delta("累积文本...")
   → Consumer: send_stream_frame("累积文本...", turn_id=X)
   → Adapter: 背压检查 → 跳过或 fire-and-forget 发送（不等 ack）

3. Agent 完成 → finish()
   → Consumer: send_stream_frame("完整内容", finalize=True, turn_id=X)
   → Adapter: await ack, 检测 846608
   → WeCom 客户端替换 thinking bubble 为最终内容
```

### Seed Frame 规则

- **单一 owner**：只有 adapter 在 `_send_stream_frame_inner` 中发 seed
- **Seed 内容**：`"<think></think>"` — 对齐官方 OpenClaw 插件的 `THINKING_MESSAGE` 常量，触发 WeCom thinking bubble 而非空 typing dots
- **`turn.seeded` 标志**：防止 double seed（会触发 WeCom errcode 6000）
- **Consumer 发空帧**：adapter 检测到空 text + 未 seeded → 发 `<think></think>` seed 并返回
- **_native_stream_opened 跟踪**：用于 fallback finalize 判断

### 中间帧背压（Backpressure）

对齐官方 OpenClaw 插件的 `replyStreamNonBlocking` 语义：

- **中间帧可跳过**：如果上一帧发送距今 < 150ms（`STREAM_FRAME_SKIP_WINDOW`），当前帧被丢弃
- **累积文本保证不丢信息**：跳过的帧内容会更新 `turn.accumulated_text`，下一次未跳过的帧或 finalize 会携带完整内容
- **Finalize 帧永远发送**：不受背压影响，确保 stream 可靠关闭
- **自适应**：慢网络 → 帧间隔自然拉长 → 几乎不跳过；快 LLM + 快网络 → 高频输出 → 自动降帧

```
官方插件:  replyStreamNonBlocking — 上一帧未 ack 则跳过
Hermes:    _last_frame_sent_at + 150ms skip window — 等价语义
```

### 边界处理

| 边界类型 | finalize | reset | 说明 |
|---------|----------|-------|------|
| **Segment break（工具边界）** | ❌ | ❌ | 保持 cumulative text，一条消息 |
| **Approval boundary** | ✅ | ✅ | 关闭旧 stream，新 turn_id，清 reply_to |
| **Turn done** | ✅ | ✅ | 正常关闭 |

### Approval Boundary 机制

通过 `GatewayStreamConsumer` 的队列信号串行处理（不越级操作）：

```python
# approval callback (agent thread):
consumer.close_for_approval_prompt()
  → puts (_APPROVAL_BOUNDARY, future, cancelled_flag) in queue
  → returns (future, cancelled_flag)

# consumer run() (async task, serial processing):
_handle_approval_boundary():
  1. Flush accumulated text
  2. Finalize stream (visible placeholder or invisible if cancelled)
  3. Reset: _native_stream_opened=False, new turn_id, clear _initial_reply_to_id
  4. If finalize failed: disable native streaming → fallback to send()
  5. Resolve future with boundary_ok status
```

**Post-approval**：`_initial_reply_to_id = None` 让 adapter 使用 `_last_chat_req_ids[chat]`（由 `/approve` 消息更新），避免绑定过期 req_id。

### 错误处理

| 错误 | 处理 |
|------|------|
| 846608 (stream expired >6min) | 标记 turn.expired, `_stream_expired_chats.add(chat)` |
| 846609 (WS session lost) | 清空 `_last_chat_req_ids`，不关 WS |
| errcode 6000 (version conflict) | 清理 turn, 返回 False → fallback |
| WebSocket 断线 | 重连，passive reply 失败自动 fallback proactive send |
| Passive reply timeout | 自动 fallback 到 proactive `aibot_send_msg` |

### Stream Expired 逻辑

- `_stream_expired_chats` 只阻止**新 turn 创建**
- 已存在的 turn（有 turn_id 且在 `_stream_turns` 中）可以继续 finalize
- 新入站消息通过 `_remember_chat_req_id` 清除 expired 标记

## Fallback 策略

### Native streaming 失败

```
stream frame 失败 → _use_native_streaming=False
  → best-effort finalize (如果 _native_stream_opened)
    → 不标记 content_delivered（不信任 best-effort）
  → fall through 到 send() 路径
```

### Passive reply 失败

```python
# _send_inner():
try:
    response = await self._send_reply_markdown(reply_req_id, content)
except (asyncio.TimeoutError, RuntimeError):
    # Passive failed (stale req_id) → proactive send
    response = await self._send_request(APP_CMD_SEND, {...})
```

## 改动文件

| 文件 | 说明 |
|------|------|
| `gateway/platforms/wecom.py` | Per-turn model, priority queue, stream protocol, fallback |
| `gateway/stream_consumer.py` | Approval boundary, segment break, seed lifecycle, re-seed |
| `gateway/run.py` | Approval callback, /approve /deny handlers |
| `cron/scheduler.py` | Delivery diagnostic logging |
| `tests/gateway/test_stream_consumer_wecom_native.py` | Native streaming + segment break tests |
| `tests/gateway/test_wecom_per_turn.py` | Per-turn isolation + multi-user tests |
| `tests/gateway/test_approval_boundary.py` | Approval boundary regression tests |

## 测试覆盖

**总计：30 tests passed**

### test_wecom.py::TestSendStreamFrame (4 tests)
- Seed frame sends `<think></think>` ✓
- Stream ID shared across frames ✓
- Backpressure skip within 150ms window ✓
- Finalize sends finish=true and resets state ✓

### test_stream_consumer_wecom_native.py (13 tests)
- Seed frame, full run, throttling, fallback
- **Segment break preserves cumulative text** ✓
- **Segment break no extra finalize** ✓

### test_wecom_per_turn.py (8 tests)
- Multiple users concurrent streaming
- One user expired, others unaffected
- Same chat concurrent turns isolated
- Native fallback closes stream

### test_approval_boundary.py (5 tests)
- Cancelled sends invisible finalize
- Finalize failure returns False
- Clears initial_reply_to for fresh req_id
- Re-seed uses new turn_id
- Success path sends visible placeholder

## 回退指南

```bash
# 完全回退
git checkout main

# 只关闭 streaming（保留代码）
# config.yaml:
display:
  platforms:
    wecom:
      streaming: false
```
