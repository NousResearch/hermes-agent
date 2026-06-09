# WeCom Native Streaming 改造记录

> **分支**：`feat/wecom-native-streaming`  
> **日期**：2026-06-04 ~ 2026-06-09  
> **基于**：hermes-agent main (`40420a619`)  
> **最新 commit**：`c747fb17b`

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

### 中间帧节流（Throttle）

中间帧不进 token bucket 队列，不阻塞 finalize，但有两层保护防止帧积压：

| 保护层 | 参数 | 说明 |
|--------|------|------|
| **时间节流** | `STREAM_FRAME_SKIP_WINDOW = 200ms` | 上一帧发送距今 < 200ms 时丢弃当前帧 |
| **帧数上限** | `MAX_INTERMEDIATE_FRAMES = 85` | 达到后所有中间帧丢弃，预留 finalize 空间 |

关键语义：
- **中间帧可跳过**：累积文本保证不丢信息，下一帧或 finalize 携带完整内容
- **Finalize 帧永远发送**：不受节流影响，确保 stream 可靠关闭
- **与官方的差异**：官方 `replyStreamNonBlocking` 基于 in-flight ack 状态跳过；我们的 fire-and-forget 没有 ack 信号，改用时间窗口 + 帧数上限模拟等效效果

```
官方插件:  replyStreamNonBlocking — 有 pending ack 则 skip
openclaw:  800ms throttle + 85 帧上限
Hermes:    200ms throttle + 85 帧上限（无 ack，靠时间模拟）
```

### 边界处理

| 边界类型 | finalize | reset | 说明 |
|---------|----------|-------|------|
| **Segment break（工具边界）** | ❌ | ❌ | 保持 cumulative text，一条消息 |
| **Approval boundary** | ✅ | ✅ | Finalize 当前 stream，禁用 native，post-approval 走 send() |
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
  1. Finalize stream with accumulated text (stable pre-approval message)
     - ok = await send_stream_frame(accumulated or "⏸ 等待审批中...", finalize=True)
     - 如果 finalize 失败（返回 False 或异常）: fallback 到 send() 投递 pre-approval 文本
  2. Disable native streaming (_use_native_streaming=False)
  3. Set cfg.buffer_only=True (post-approval 内容积累到 got_done 一次性投递)
  4. Reset segment state
  5. Resolve future with True
```

**设计决策：不跨 approval 保持 stream**

- WeCom stream finalize ack 只代表服务端收到 frame，不保证客户端渲染
- Approval 等待期间 stream 处于半开状态，客户端可能停止跟踪
- 如果 `content_delivered=True` 但客户端未渲染，normal final send 被 suppress → 用户什么都看不到
- Approval 本身是交互边界，"审批前状态 + 审批后结果" 分为两条消息是可接受的 UX

**Post-approval 输出走 buffer_only send()**：native streaming 在 boundary 后被禁用，`buffer_only=True` 让 consumer 积累所有 post-approval 文本，在 `got_done` 时一次性通过 `send()` 投递。

**Post-approval req_id 分离**：

确认消息（"✅ Approved..."）通过 `force_proactive_send` metadata 强制走 `APP_CMD_SEND`（DM）或绑定 `reply_to=event.message_id` 精确定位 req_id（群聊 fallback）。

```python
# slash_commands.py _handle_approve_command:
await _adapter.send(
    source.chat_id,
    confirmation_text,
    reply_to=event.message_id,  # 绑定 /approve 入站消息的 req_id
    metadata={
        "is_approval_prompt": True,       # → control lane (高优先级)
        "force_proactive_send": True,     # → DM 走 APP_CMD_SEND
    },
)
```

**群聊安全**：`force_proactive_send` 对群聊不生效（`_group_chat_ids` 检测），群聊仍走 passive reply。群聊若无可用 req_id 则 fail-early 并记 warning。

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
    # Passive failed (stale req_id) → proactive send (DM only)
    response = await self._send_request(APP_CMD_SEND, {...})

# Group chats: no req_id → fail early (APP_CMD_SEND is blocked)
if chat_id in self._group_chat_ids:
    return SendResult(success=False, error="No req_id for group")
```

## send_message 工具跨 loop 死锁修复（2026-06-07）

### 问题

Agent 通过 `send_message` 工具调用 WeCom adapter 的 `send()` 时，adapter 的
`_enqueue_chat_send()` 内部使用 `asyncio.Queue` + worker task 做 per-chat FIFO
发送。这些 queue/worker 绑定在 gateway 主事件循环上。

但 agent 的 tool handler 运行在线程池 worker 线程中，通过 `_run_async()` →
`_get_worker_loop()` 获取一个独立的 per-thread event loop。跨 loop 调用
`adapter.send()` 导致：

1. `_enqueue_chat_send` 在 worker loop 上创建 Future
2. Gateway 主循环的 `_chat_send_worker` 处理请求后调用 `future.set_result()`
3. `set_result()` 用非线程安全的 `loop.call_soon()` 通知 worker loop
4. Worker loop 在 `selector.select(None)` 中无限等待，永远不会被唤醒 → **死锁**

### 修复

在 `tools/send_message_tool.py` 的 `_send_via_adapter()` 中：

| 场景 | 处理 |
|------|------|
| `current_loop is gateway_loop` | 直接 `await adapter.send()` |
| `current_loop is not gateway_loop` 且 gateway loop running | `safe_schedule_threadsafe()` → `asyncio.shield(asyncio.wrap_future(fut))` |
| `gateway_loop` 存在但已停止 | 返回明确错误，不尝试 direct await |
| 无 `gateway_loop`（CLI/测试） | 直接 `await adapter.send()` |

关键设计：
- **`asyncio.shield`**：保护已入队的发送不被 caller cancel 取消，避免"工具报错但消息稍后发出 → agent 重试 → 重复发送"
- **无外层 timeout**：依赖 adapter 内部 `REQUEST_TIMEOUT_SECONDS=15s` 和上层 `_run_async` 300s timeout，不引入额外取消语义

### 改动文件

| 文件 | 说明 |
|------|------|
| `tools/send_message_tool.py` | `_send_via_adapter()` 跨 loop dispatch 逻辑 |
| `tests/tools/test_send_message_cross_loop.py` | 4 个 regression tests |

### 测试覆盖

```
tests/tools/test_send_message_cross_loop.py — 4 passed
  ✓ 跨 loop 时 send 在 gateway loop 执行
  ✓ 同 loop 时直接 await（防自锁）
  ✓ Gateway loop 已停止时返回错误
  ✓ Shield 防止 cancel 取消已入队发送
```

## 改动文件

| 文件 | 说明 |
|------|------|
| `gateway/platforms/wecom.py` | Per-turn model, priority queue, stream protocol, fallback, `force_proactive_send` + group safety |
| `gateway/stream_consumer.py` | Approval boundary, segment break, seed lifecycle, re-seed, cancelled finalize fix |
| `gateway/slash_commands.py` | /approve /deny confirmation: `reply_to=event.message_id` + `force_proactive_send` |
| `gateway/run.py` | Approval callback, native stream gate, approval boundary integration |
| `tools/send_message_tool.py` | 跨 loop dispatch 修复（`_send_via_adapter`） |
| `cron/scheduler.py` | Delivery diagnostic logging |
| `tests/gateway/test_stream_consumer_wecom_native.py` | Native streaming + segment break tests |
| `tests/gateway/test_wecom.py` | Proactive send regression, group safety tests |
| `tests/gateway/test_wecom_per_turn.py` | Per-turn isolation + multi-user tests |
| `tests/gateway/test_approval_boundary.py` | Approval boundary: cancelled finalize, failure handling |
| `tests/tools/test_send_message_cross_loop.py` | 跨 loop dispatch regression tests |

## 测试覆盖

**总计：102 tests passed**

### test_wecom.py::TestSendStreamFrame (5 tests)
- Seed frame sends `<think></think>` ✓
- Stream ID shared across frames ✓
- Throttle skip within 200ms window ✓
- Frame cap drops excess intermediate frames ✓
- Finalize sends finish=true and resets state ✓

### test_wecom.py::TestSend (5 new tests)
- Approval confirmation uses proactive send (DM) ✓
- Approval request prompt keeps passive reply ✓
- Active stream doesn't force all send() to proactive ✓
- force_proactive falls back to passive for groups ✓
- Group send fails early without req_id ✓

### test_stream_consumer_wecom_native.py (13 tests)
- Seed frame, full run, throttling, fallback
- **Segment break preserves cumulative text** ✓
- **Segment break no extra finalize** ✓

### test_wecom_per_turn.py (8 tests)
- Multiple users concurrent streaming
- One user expired, others unaffected
- Same chat concurrent turns isolated
- Native fallback closes stream

### test_approval_boundary.py (4 tests)
- **Boundary finalizes stream and disables native** ✓
- **Empty accumulated uses placeholder** ✓
- **Post-approval content uses send() not stream** ✓
- **Stream-not-opened boundary skips finalize** ✓

### test_send_message_cross_loop.py (4 tests)
- Cross-loop dispatches to gateway loop ✓
- Same loop uses direct await (防自锁) ✓
- Gateway loop not running returns error ✓
- Shield prevents cancel of enqueued send ✓

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
