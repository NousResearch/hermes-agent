# WeCom Native Streaming 改造记录

> **分支**：`feat/wecom-native-streaming`
> **日期**：2026-06-04 ~ 2026-06-05
> **基于**：hermes-agent main (`40420a619`)
> **最终版本**：Round 5 完成 (`281f31b68`)

## 背景

Hermes Agent 的 WeCom 适配器原本不支持流式输出（`SUPPORTS_MESSAGE_EDITING = False` 导致 gateway 跳过 streaming），用户发消息后要等 LLM 完整生成才能看到回复，没有"输入中"反馈。

本改造对齐了腾讯官方 `@wecom/wecom-openclaw-plugin` 的行为，使用 WeCom AI Bot 的 `msgtype: "stream"` 协议实现：
- 收到消息后，stream consumer run() 开始时立即发 seed frame → 客户端显示 thinking bubble
- 按内容变化推送累积内容（通过 priority queue 控制频率）
- `finish=true` 关闭流，替换 thinking 动画为最终内容

## 改动文件清单

| 文件 | 说明 |
|------|------|
| `gateway/platforms/wecom.py` | Per-turn 流式协议、priority queue、846608/846609 处理、per-turn isolation |
| `gateway/stream_consumer.py` | Native streaming transport、seed frame lifecycle、fallback finalize |
| `gateway/run.py` | Streaming gate 放行 native 适配器 |
| `gateway/display_config.py` | WeCom 默认 `streaming: True` |
| `hermes_cli/config.py` | DEFAULT_CONFIG 暴露 wecom streaming toggle |
| `tools/send_message_tool.py` | WeCom 复用 gateway adapter、识别 WeCom chat_id 格式 |
| `tests/gateway/test_wecom.py` | 适配器单测（流式协议、expired、per-turn） |
| `tests/gateway/test_stream_consumer_wecom_native.py` | Consumer ↔ adapter 集成测试 |
| `tests/gateway/test_wecom_per_turn.py` | Per-turn 隔离、多用户并发测试（新文件） |

## 关键架构决策

### 1. Per-Turn 模型（最终架构）

**问题背景：**
- 初版使用单 chat 单 stream 模型（`_active_stream_id`）
- 多个 consumer 并发时互相干扰
- 一个 consumer 遇到 `/approve` 等操作会阻塞其他 consumer

**最终方案：Per-Turn Isolation**
```python
# 每个 stream consumer 分配唯一 turn_id (UUID)
# 状态完全隔离
_stream_turns: Dict[str, StreamTurn]  # key = f"{chat_id}:{turn_id}"

# StreamTurn 包含
class StreamTurn:
    chat_id: str
    req_id: str       # 绑定的 WeCom req_id
    stream_id: str    # 生成的 stream_id
    expired: bool     # 这个 turn 是否 expired
```

**隔离级别：**
- Chat 级别：不同用户（不同 chat_id）完全独立
- Turn 级别：同一用户的不同 consumer（不同 turn_id）完全独立
- Expired 传播：只标记特定 turn，不影响其他 turn

### 2. send() 和 streaming 完全独立

**错误设计（Round 1-2）：**
```python
# send() 尝试关闭 active stream
if self._active_stream_id is not None:
    await finalize_stream(self._active_stream_id)
```
问题：send() 可能关闭错误的 stream，导致其他 consumer 的 stream 被意外关闭。

**正确设计（Round 3+）：**
```python
# send() 只发送独立消息，不触碰任何 stream
async def send(self, chat_id, content, ...):
    # 直接发送 proactive markdown 消息
    await self._send_reply_request(req_id, body)
    # 不关闭、不检查、不修改任何 stream 状态
```

**Stream 生命周期由创建者管理：**
- GatewayStreamConsumer 创建 stream → 负责 finalize
- send() 创建消息 → 不涉及 stream

### 3. Priority Queue 频率控制

**WeCom 限制：30 帧/分钟/会话**

**实现：**
```python
# 所有出站消息（包括 stream 帧）进入优先级队列
_control_send_worker 消费队列，按优先级和时间戳排序
- 高优先级：finish=true 帧、proactive send、approve/deny 响应
- 中优先级：用户回复的首帧（req_id 已消费）
- 低优先级：中间内容帧（fire-and-forget）

# 频率限制
per_chat_window = 60 秒
per_chat_limit = 30 条
```

**关键特性：**
- 同一 chat 的消息串行发送（避免乱序）
- 不同 chat 并发发送（不同用户不互相阻塞）
- approve/deny 等用户操作高优先级（即时响应）

### 4. Stream Expired 逻辑（Round 4 修复）

**错误设计（Round 3）：**
```python
# 入口就检查 chat 级别 expired
if chat in self._stream_expired_chats:
    return False  # ❌ 阻断所有帧，包括已有 turn 的后续帧
```

**正确设计（Round 4+）：**
```python
# 步骤 1: 只在无 turn_id 时检查 chat expired
if not turn_id and chat in self._stream_expired_chats:
    return False  # 阻止新 turn 创建

# 步骤 2: 查找或创建 turn
if turn_id:
    turn = self._stream_turns.get(f"{chat}:{turn_id}")
    if not turn:
        # 创建新 turn 前检查 chat expired
        if chat in self._stream_expired_chats:
            return False
        turn = StreamTurn(chat, req_id)
else:
    turn = self._find_active_turn_for_chat(chat)
    if not turn:
        # 创建新 turn 前检查 chat expired
        if chat in self._stream_expired_chats:
            return False
        turn = StreamTurn(chat, req_id)

# 步骤 3: 检查特定 turn 是否 expired
if turn.expired:
    return False  # 只阻断这个 turn
```

**效果：**
- Consumer 1 expired → 只标记 turn-1 和 chat
- Consumer 2（已存在的 turn-2）可以继续 finalize ✅
- Consumer 3（新 turn）无法创建 ✅

### 5. Seed Frame Lifecycle（Round 5 修复）

**问题：Seed frame 长度为 0**
```python
# Seed frame 成功打开 bubble
await send_stream_frame("", chat_id=chat)  # len=0

# 但 fallback 检查
if self._native_last_pushed_len > 0:  # ❌ False
    await finalize_stream()
# 首帧失败时跳过 finalize → bubble 残留
```

**修复：显式跟踪 stream 打开状态**
```python
# 新增标志
self._native_stream_opened = False

# Seed 成功后
if seed_ok:
    self._native_stream_opened = True  # ✅

# Fallback 检查
if self._native_stream_opened:  # ✅ 基于 "stream 是否打开"
    await finalize_stream()
```

---

## 核心机制

### 1. 流式协议（wecom.py）

```python
# 类属性
SUPPORTS_NATIVE_STREAMING = True
MAX_STREAM_CONTENT_LENGTH = 20480     # WeCom server 字节上限
STREAM_EXPIRED_ERRCODE = 846608       # >6min 无更新，流过期
STREAM_NOT_SUBSCRIBED_ERRCODE = 846609 # WS session 失效

# 核心方法
send_stream_frame(text, *, finalize, chat_id, reply_to, turn_id) -> bool
_send_stream_reply(reply_req_id, stream_id, content, finish) -> Dict
_resolve_stream_req_id(chat_id, reply_to) -> Optional[str]
supports_native_streaming(chat_type, metadata) -> bool
```

**协议流程**：
```
1. Consumer run() 开始
   → send_stream_frame("", chat_id=X, turn_id=UUID)
   → Seed 空帧（高优先级）
   → 客户端显示 thinking bubble
   → _native_stream_opened = True

2. LLM 生成内容
   → send_stream_frame("累积内容...", chat_id=X, turn_id=UUID)
   → 进入 priority queue（低优先级）
   → Fire-and-forget 发送

3. LLM 完成
   → send_stream_frame("完整回复", finalize=True, chat_id=X, turn_id=UUID)
   → 进入 priority queue（高优先级）
   → 等待 ack，检测 846608
   → 客户端替换为最终内容

4. 清理
   → turn_key = f"{chat_id}:{turn_id}"
   → del _stream_turns[turn_key]
```

### 2. Fire-and-forget vs Await Ack

| Frame 类型 | 方法 | 原因 |
|-----------|------|------|
| Seed frame | Fire-and-forget (`_send_json`) | 快速打开 bubble |
| 中间内容帧 | Fire-and-forget (`_send_json`) | WeCom 不回 ack，等待会 timeout |
| Finalize frame | Await ack (`_send_reply_request`) | 检测 846608 expired |

### 3. send_typing 设计：no-op

```python
async def send_typing(self, chat_id: str, metadata=None) -> None:
    """No-op: WeCom typing 由 stream consumer seed frame 负责。"""
    pass
```

**理由：**
- `_keep_typing` 是为 Telegram/Discord 设计（typing 5 秒过期需刷新）
- WeCom stream 不过期，seed frame 足够
- 避免 `_keep_typing` 竞态开出孤儿 stream

### 4. 错误处理

| errcode | 含义 | Per-Turn 处理 |
|---------|------|------------|
| 846608 | Stream >6min 未更新，已过期 | 标记 `turn.expired = True` + `_stream_expired_chats.add(chat)` |
| 846609 | WS session 失效（被踢或连接断） | 清空 `_last_chat_req_ids`，不主动关 WS |

**Why 846609 不主动关 WS：**
- WeCom 一个 bot 只允许一条长链接
- 主动关闭会触发重连 → 建立副连接 → 被踢 → 无限循环

### 5. Native Fallback（Round 4-5）

**场景：Native streaming 失败 mid-stream**

```python
# Stream consumer 检测到 send_stream_frame 失败
try:
    await adapter.send_stream_frame(text, ...)
except Exception:
    ok = False

if not ok:
    self._use_native_streaming = False
    
    # Best-effort finalize（如果 stream 已打开）
    if self._native_stream_opened:
        try:
            await adapter.send_stream_frame(
                text, finalize=True, turn_id=self._turn_id
            )
            # 成功关闭 → 用户看到完整内容 ✅
            return True
        except:
            # 失败 → fall through 到 send()
            pass
    
    # Fallback: 独立 proactive send
    await adapter.send(chat_id, text)
```

**对齐官方：** OpenClaw 的 `finishThinkingStream()` 也做了这个处理。

### 6. send_message 工具复用 gateway adapter

```python
# tools/send_message_tool.py
# ❌ 错误（每次新建连接）
result = await _send_wecom(pconfig.extra, chat_id, chunk)

# ✅ 正确（复用 gateway adapter）
result = await _send_via_adapter(platform, pconfig, chat_id, chunk, ...)
```

**原因：**
- `_send_wecom()` 每次 `new WeComAdapter().connect()`
- 建立第二条 WS 长链接
- WeCom 服务端踢掉副连接时，主连接 session 也失效
- 所有 stream 帧 → 846609 → typing 永远残留

---

## 频率限制分析（30条/分钟）

**WeCom 限制：30 帧/分钟/会话，1000 条/小时**

### Priority Queue 策略

```python
# 优先级定义
PRIORITY_HIGH = 1    # approve/deny、finish frame、proactive send
PRIORITY_MID = 5     # 首帧（req_id 已消费）
PRIORITY_LOW = 10    # 中间内容帧（fire-and-forget）

# 排序
1. 按优先级（数字越小越优先）
2. 同优先级按时间戳（先入先出）

# 限流
per_chat_window = 60 秒
per_chat_limit = 30 条
超过限制 → 延迟发送（等待窗口滚动）
```

### 并发场景

| 场景 | 行为 |
|------|------|
| 3 个用户同时问问题 | 各自独立 queue，并发发送 |
| 同一用户 2 个并发 consumer | 共享 chat queue，串行发送 |
| User A 30 帧用完 | 延迟后续帧，不影响 User B |
| `/approve` 在 queue 中 | 高优先级，插队发送 |

---

## Commit 历程

### Round 1-3: 基础实现 + 初步修复
```
ab187194b fix(wecom): reuse live gateway adapter
383ef6f55 fix(wecom): stop reconnecting on server-kick
c92b9bb93 fix(stream_consumer): always finalize native stream
...
```

### Round 4: Per-Turn 模型完成
```
90b20d0f7 fix(wecom): codex round 4 - true per-turn isolation complete
```
- ✅ Per-turn 隔离
- ✅ Chat expired 只阻止新 turn
- ✅ send() 不关闭 stream
- ✅ Priority queue

### Round 5: 边界情况修复
```
d08e4d15c fix(wecom): codex round 5 - seed frame edge case
281f31b68 test(wecom): add multi-user concurrent streaming tests
```
- ✅ Seed frame lifecycle 跟踪
- ✅ Test cleanup (no pending workers)
- ✅ 多用户并发测试

---

## 测试覆盖

### 单元测试（test_wecom.py）
- send_stream_frame 基础行为
- 846608 expired 处理
- 846609 WS session 失效
- Per-turn state 管理

### 集成测试（test_stream_consumer_wecom_native.py）
- Stream consumer ↔ adapter 交互
- Seed → content → finalize 完整流程
- Native fallback 路径

### 并发测试（test_wecom_per_turn.py）
- ✅ 单用户单 turn
- ✅ 单用户多 turn（同 chat）
- ✅ 多用户并发（不同 chat）
- ✅ 一个 turn expired，另一个继续
- ✅ 一个用户 expired，其他不受影响
- ✅ Seed 成功首帧失败
- ✅ Native fallback 关闭 stream

**总计：50/50 测试通过**

---

## 对齐官方行为

| 行为 | OpenClaw | Hermes (本方案) |
|------|----------|----------------|
| 流式协议 | `msgtype: "stream"` | ✅ 相同 |
| 单 WS 连接 | ✅ | ✅ |
| Per-turn 隔离 | ✅ (session 机制) | ✅ (turn_id) |
| send() 独立 | ✅ | ✅ |
| Seed frame | ✅ | ✅ |
| Fire-and-forget 中间帧 | ✅ | ✅ |
| Finalize await ack | ✅ | ✅ |
| 频率控制 | Queue | ✅ Priority Queue |
| 846608 处理 | Fallback to send | ✅ 相同 |
| 846609 处理 | 不关 WS | ✅ 相同 |
| 空内容兜底 | 占位符 | ✅ 相同 |

---

## 生产就绪清单

- ✅ Per-turn 完全隔离
- ✅ 多用户并发无干扰
- ✅ send() 和 streaming 独立
- ✅ Priority queue 频率控制
- ✅ 846608/846609 正确处理
- ✅ Seed frame lifecycle 跟踪
- ✅ Native fallback 关闭 stream
- ✅ 50/50 测试通过
- ✅ 0 pending worker 警告
- ✅ 100% 对齐官方模型

**Status: Production Ready 🚀**
