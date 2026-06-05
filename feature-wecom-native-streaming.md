# WeCom Native Streaming 改造记录

> **分支**：`feat/wecom-native-streaming`
> **日期**：2026-06-04 ~ 2026-06-05
> **基于**：hermes-agent main (`40420a619`)

## 背景

Hermes Agent 的 WeCom 适配器原本不支持流式输出（`SUPPORTS_MESSAGE_EDITING = False` 导致 gateway 跳过 streaming），用户发消息后要等 LLM 完整生成才能看到回复，没有"输入中"反馈。

本改造对齐了腾讯官方 `@wecom/wecom-openclaw-plugin` 的行为，使用 WeCom AI Bot 的 `msgtype: "stream"` 协议实现：
- 收到消息后，stream consumer run() 开始时立即发 seed frame → 客户端显示 typing
- 逐段推送累积内容（频率友好，按 20 字符节流）
- `finish=true` 关闭流，替换 typing 动画为最终内容

## 改动文件清单

| 文件 | 说明 |
|------|------|
| `gateway/platforms/wecom.py` | 流式协议帧、fire-and-forget、846608/846609 处理、竞争保护、长链接保护 |
| `gateway/stream_consumer.py` | native streaming transport 分支、空 _accumulated 时兜底 finalize |
| `gateway/run.py` | streaming gate 放行 native 适配器 |
| `gateway/display_config.py` | wecom 默认 `streaming: True` |
| `hermes_cli/config.py` | DEFAULT_CONFIG 暴露 wecom streaming toggle |
| `tools/send_message_tool.py` | WeCom 复用 gateway adapter、识别 WeCom chat_id 格式 |
| `tests/gateway/test_wecom.py` | 适配器单测 |
| `tests/gateway/test_stream_consumer_wecom_native.py` | consumer ↔ adapter 集成测试（新文件） |
| `tests/gateway/test_display_config.py` | wecom streaming 默认值 |
| `tests/gateway/test_per_platform_streaming_defaults.py` | DEFAULT_CONFIG 断言 |

## Commit 列表（完整）

```
ab187194b fix(wecom): reuse live gateway adapter in send_message tool — no new WS connection
383ef6f55 fix(wecom): stop reconnecting on server-kick, don't close WS on 846609
c92b9bb93 fix(stream_consumer): always finalize native stream on got_done, even when _accumulated is empty
2be0bf28a refactor(wecom): make send_typing a no-op, remove _stream_delivered_chats
815bf5ed9 fix(wecom): mark chat delivered before first await in send() to close race window
5c3e2cc2b fix(wecom): mark chat delivered in all send() paths to block orphan streams
c1cea1b0a fix(send_message): recognize WeCom chat IDs as explicit targets
5ae18595e fix(wecom): trigger immediate WS reconnect on errcode 846609
c8515dd79 docs: add feature-wecom-native-streaming.md
d0517ebb1 fix(wecom): prevent _keep_typing from reopening stream after send() closes it
abd00d71c perf(wecom): fire-and-forget intermediate stream frames (don't await ack)
9af225650 fix(wecom): close active stream in send() to prevent lingering typing bubble
07e2f5485 test(wecom): cover native streaming lifecycle and 846608 fallback
bed7dcb68 fix(gateway): allow native-streaming adapters past edit-only gate
431f55119 feat(stream_consumer): add native streaming transport
6c7f6a06e feat(wecom): add native streaming primitives (msgtype: stream)
```

---

## 核心机制

### 1. 流式协议（wecom.py）

```python
# 新增类属性
SUPPORTS_NATIVE_STREAMING = True
MAX_STREAM_CONTENT_LENGTH = 20480     # WeCom server 字节上限
STREAM_EXPIRED_ERRCODE = 846608       # >6min 无更新，流过期
STREAM_NOT_SUBSCRIBED_ERRCODE = 846609 # WS session 失效

# 核心方法
send_stream_frame(text, *, finalize, chat_id, reply_to) -> bool
_send_stream_reply(reply_req_id, stream_id, content, finish) -> Dict
_resolve_stream_req_id(chat_id, reply_to) -> Optional[str]
supports_native_streaming(chat_type, metadata) -> bool
```

**协议流程**：
```
1. Consumer run() 开始 → send_stream_frame("", chat_id=X)
   → _send_json(fire-and-forget): seed 空帧 → 客户端显示 typing
2. LLM 生成内容 → send_stream_frame("你好世界...", chat_id=X)
   → _send_json(fire-and-forget): 累积内容推送（至少 20 新字符才发一帧）
3. LLM 完成 → send_stream_frame("完整回复", finalize=True)
   → _send_reply_request(await ack): finish=true → 客户端替换为最终内容
```

### 2. Fire-and-forget 中间帧（性能关键）

中间帧（`finish=False`）直接走 `_send_json`，不等 ack。原因：WeCom 服务端对中间帧不回 ack，若走 `_send_reply_request` 会每帧阻塞 15 秒（REQUEST_TIMEOUT）。

只有 `finish=True` 帧走 `_send_reply_request` 等 ack，用来可靠检测 errcode 846608。

### 3. send_typing 设计：no-op

```python
async def send_typing(self, chat_id: str, metadata=None) -> None:
    """No-op: WeCom typing 由 stream consumer seed frame 负责。"""
    del chat_id, metadata
```

**理由**：`_keep_typing` 是为 Telegram/Discord 设计的（typing 5 秒过期需刷新），WeCom stream 不过期，不需要循环刷新。官方 OpenClaw 插件也没有 `_keep_typing`，typing 只在 LLM `onReplyStart` 时触发一次。

将 `send_typing` 做成 no-op，避免了 `_keep_typing` 竞态开出孤儿 stream 的所有问题。

### 4. 错误处理

| errcode | 含义 | 处理 |
|---------|------|------|
| 846608 | stream >6min 未更新，已过期 | `WeComStreamExpiredError` → 标记 `_stream_expired_chats`，降级到 proactive markdown send |
| 846609 | WS session 失效（被踢或连接断） | 清空缓存 req_ids，**不主动关 WS**，等服务端自然关闭 |

**为什么 846609 不主动关 WS**：WeCom 一个 bot 只允许一条长链接。主动关 WS 会触发 `_listen_loop` 重连，建立副连接，服务端踢掉副连接同时废掉主连接 session → 无限踢重连循环，所有 stream 帧永远是 846609。

### 5. Server-kick 检测（aibot_event_callback）

```python
if cmd == APP_CMD_EVENT_CALLBACK:
    body = payload.get("body") or {}
    if body.get("event_type") == "disconnected_event":
        # 被服务端踢了（另一处建了新连接），停止重连避免互踢
        self._running = False
    return
```

对齐官方 OpenClaw SDK 的 `event.disconnected_event` 处理。

### 6. send() 关闭 active stream

当 `send()` 被调用时，如果当前 chat 有 active stream（stream consumer 开的 seed frame 还在），先用 `finish=true` 关闭：

```python
if self._active_stream_id is not None and self._active_stream_chat_id == chat_id:
    await self._send_stream_reply(..., finish=True)  # 关闭 typing
    # 正常内容包含在 finish 帧里
```

这是斜杠命令（`/new`、`/sethome`）等不走 stream consumer 的回复路径的兜底。

### 7. Stream consumer finalize 兜底

```python
# got_done 时，即使 _accumulated 为空（工具调用但无文本输出）也必须关闭 stream
if self._use_native_streaming:
    if not current_update_visible:
        close_text = self._accumulated or "✅"   # 空时用占位符
        await self._send_or_edit(close_text, finalize=True)
```

**对齐官方**：OpenClaw 的 `finishThinkingStream()` 也做了这个——没有文本时用"📎 文件已发送"等占位文案，因为 WeCom 客户端会忽略空白的 finish 帧，typing 会永远残留。

### 8. Gateway gate 放行（run.py）

```python
# 原来：
if not _adapter_supports_edit:
    raise RuntimeError("skip streaming for non-editable platform")

# 改为：
if not _adapter_supports_edit and not _adapter_supports_native_stream:
    raise RuntimeError("skip streaming for non-editable platform")
```

WeCom 虽然 `SUPPORTS_MESSAGE_EDITING = False`，但有 `SUPPORTS_NATIVE_STREAMING = True`，通过 native stream 路径绕过限制。

### 9. send_message 工具复用 gateway adapter（关键！）

```python
# tools/send_message_tool.py
# 原来：
elif platform == Platform.WECOM:
    result = await _send_wecom(pconfig.extra, chat_id, chunk)  # 每次新建连接！

# 改为：
elif platform == Platform.WECOM:
    result = await _send_via_adapter(platform, pconfig, chat_id, chunk, ...)  # 复用 gateway adapter
```

**根本原因**：`_send_wecom()` 每次调用都 `new WeComAdapter().connect()`，建立第二条 WS 长链接。WeCom 服务端踢掉副连接时，主连接的 session 也失效 → 846609 → 所有 stream 帧无法发出 → typing 永远残留。

官方 OpenClaw 插件设计思路完全一致：整个进程只有一个 `WSClient` 实例，所有出站消息通过同一实例发出，不允许新建连接。

### 10. WeCom chat_id 识别

```python
# tools/send_message_tool.py _parse_target_ref
if platform_name == "wecom":
    stripped = target_ref.strip()
    if stripped:
        return stripped, None, True  # 所有非空值都当 explicit chat_id
```

原来 `wr1hBoEQAA6ysRlpzfR50FeUe-iaG8mQ` 等 WeCom chat_id 无法被识别，tool 报 "No home channel set" 错误。

---

## 频率限制分析（30条/分钟）

WeCom AI Bot 频率上限：**30 条/分钟/会话，1000 条/小时**。

| 方案 | 每次对话帧数 | 每分钟最大并发对话 |
|------|------------|-------------------|
| 官方 OpenClaw（block 粒度） | 3-5 | 6-10 |
| 本方案（20 字符节流） | 5-10 | 3-6 |
| 改造前（无流式） | 1 | 30 |

建议：多人并发使用时，可适当降低 `edit_interval` 或升高 `_MIN_NEW_VISIBLE_CHARS`。

---

## 与官方插件的差异

| 维度 | 官方 OpenClaw 插件 | 本改造 |
|------|-------------------|--------|
| Typing 触发时机 | LLM `onReplyStart`（第一个 block 生成前） | consumer `run()` 开始时 seed frame |
| `_keep_typing` 循环 | 无 | 有，但 `send_typing` 是 no-op，无影响 |
| 流式粒度 | block 级别（框架控制） | token + `edit_interval` + 20 字符节流 |
| 每次对话帧数 | 3-5 | 5-10 |
| Thinking 占位符 | `<think></think>` | 空字符串 `""` |
| 846608 检测时机 | 每帧 | 仅 finish 帧（中间帧 fire-and-forget） |
| 串行队列 | `chat-queue.ts` 保证同 chat 串行 | **双通道优先级队列** (control + normal) |
| 连接管理 | SDK WSClient（单例） | 自建 aiohttp WS |
| proactive 群发 | 通过 req_id 缓存的 APP_CMD_RESPONSE | 同，复用 gateway adapter |

---

## 双通道优先级队列（防止审批提示阻塞）

**实现位置：** `gateway/platforms/wecom.py:241-402`

### 问题背景

原始 FIFO 队列设计会导致审批提示阻塞：
```
Queue: [finalize(等15秒), 审批提示(排队), ...]
                 ↑ 阻塞        ↑ 用户看不到
```

实际生产场景：Agent 输出长内容时触发审批，finalize 帧阻塞等待 WeCom 响应（最多15秒），审批提示排在队列后面，用户无法及时看到提示并响应 `/approve`。

### 解决方案：双通道优先级队列

#### **Control Lane（控制通道）**
- **用途：** 审批提示、finalize 帧、错误通知
- **配额：** 6 reserved tokens/min + 可使用 normal 剩余配额
- **队列：** `_control_queues`，worker: `_control_send_worker()`
- **优先级：** 高，绕过 normal 队列

#### **Normal Lane（普通通道）**
- **用途：** 常规消息、媒体 caption
- **配额：** 24 tokens/min（不能动用 reserved）
- **队列：** `_chat_queues`，worker: `_chat_send_worker()`

#### **Fire-and-forget（无队列）**
- **用途：** stream 中间帧 (`finalize=False`)
- **配额：** 无限制（WeCom 不计入 30/min）
- **行为：** 立即发送，不等 ACK

### Token Bucket 实现

**Per-chat tracking** (`_chat_token_usage`):
```python
{
  chat_id: {
    "normal": 已用数量,
    "reserved": 已用数量, 
    "last_reset": 重置时间戳
  }
}
```

- **总容量：** 30 tokens/minute/chat
- **分配：** 24 normal + 6 reserved
- **重置：** 每 60 秒
- **Control 策略：** 优先用 normal（不浪费 reserved），normal 耗尽后用 reserved

### 集成点

**1. gateway/run.py:18103-18105** (审批提示标记)
```python
_approval_metadata = dict(_status_thread_metadata or {})
_approval_metadata["is_approval_prompt"] = True
```

**2. gateway/platforms/wecom.py:1697-1702** (路由到 control lane)
```python
is_control = metadata.pop("is_approval_prompt", False) if metadata else False
return await self._enqueue_chat_send(..., is_control=is_control)
```

**3. gateway/platforms/wecom.py:2016-2020** (finalize 走 control lane)
```python
if finalize:
    return await self._enqueue_chat_send(..., is_control=True)
```

### 效果

- ✅ **审批提示立即发送**：不等待前面的 finalize 或普通消息
- ✅ **finalize 不阻塞新消息**：finalize 走 control lane，不占用 normal 配额
- ✅ **频率限制保证**：token bucket 确保不超过 30 条/分钟（实测触发过 846607）
- ✅ **消息顺序正确**：control lane 仍是队列（FIFO），同类消息顺序不乱

### 为什么不是"移除队列"

Codex 建议和官方插件对比后的结论：

> "保留限流，去掉会导致审批卡住的串行语义"

- **不能完全移除队列**：生产环境已触发过 846607 限流错误
- **不能简单 fire-and-forget 所有消息**：会导致消息顺序错乱（审批提示插在 stream 中间）
- **正确方案**：分层优先级 + 预留配额，critical 消息走快速通道

### 测试

**单元测试：** `test_wecom_priority_queue.py`

```bash
cd /Users/bilibili/.hermes/hermes-agent
python test_wecom_priority_queue.py
```

验证：
- ✅ Token 分配：24 normal + 6 reserved
- ✅ Normal 消息不能使用 reserved
- ✅ Control 消息可使用全部 30 tokens
- ✅ Reserved 在 normal 耗尽后仍可用
- ✅ Per-turn stream state 隔离并发消息
- ✅ 多个 stream 不互相干扰

**手动测试（需要真实 WeCom 连接）：**
1. 发送长消息触发 streaming（生成多个 stream 帧）
2. 在输出过程中触发需要审批的命令（如 `rm` 危险操作）
3. 验证审批提示立即显示（不等待 finalize 完成，< 1秒）
4. 响应 `/approve` 或 `/deny`，验证流程正常
5. 同时发送另一条消息，验证两个 stream 不互相干扰

---

## Per-Turn Stream State (Phase 2)

**实现位置：** `gateway/platforms/wecom.py:168-184, 1207-1231`

### 问题背景

原始实现使用全局 stream 状态：
```python
self._active_stream_id: Optional[str] = None      # 全局共享
self._active_stream_req_id: Optional[str] = None
self._active_stream_chat_id: Optional[str] = None
```

**问题：** 当多个消息并发处理时（如审批期间又来新消息），全局状态会被覆盖，导致：
- Stream ID 冲突
- 消息串扰
- Finalize 可能关闭错误的 stream

### 解决方案：Per-Turn State

#### StreamTurn 类
```python
class StreamTurn:
    """Per-turn stream state to avoid global state conflicts."""
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id = chat_id
        self.req_id = req_id
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"  # 每个 turn 独立
        self.accumulated_text = ""
        self.finalized = False
        self.start_time = time.monotonic()
        self.expired = False
```

#### 状态管理
```python
# 不再使用全局状态
# self._active_stream_id = None  ❌

# 使用 per-turn 字典
self._stream_turns: Dict[str, StreamTurn] = {}  # key = f"{chat_id}:{req_id}"
```

#### 核心方法

**1. 获取或创建 Turn**
```python
def _get_or_create_stream_turn(self, chat_id: str, req_id: str) -> StreamTurn:
    key = f"{chat_id}:{req_id}"
    if key not in self._stream_turns:
        self._stream_turns[key] = StreamTurn(chat_id, req_id)
    return self._stream_turns[key]
```

**2. 查找活跃 Turn**
```python
def _find_active_turn_for_chat(self, chat_id: str) -> Optional[StreamTurn]:
    """Find the most recent active (non-finalized) turn for a chat."""
    for turn in self._stream_turns.values():
        if turn.chat_id == chat_id and not turn.finalized:
            return turn
    return None
```

**3. 清理 Turn**
```python
def _cleanup_stream_turn(self, chat_id: str, req_id: str) -> None:
    key = f"{chat_id}:{req_id}"
    self._stream_turns.pop(key, None)
```

### 效果对比

#### Before (全局状态)
```
时间线：
t0: Message 1 开始 stream
    _active_stream_id = "stream-aaa"
    
t1: 审批提示触发（新消息）
    _active_stream_id = "stream-bbb"  ← 覆盖了 Message 1 的状态
    
t2: Message 1 尝试 finalize
    使用 _active_stream_id = "stream-bbb"  ← 错误！
    关闭了 Message 2 的 stream
    
结果：消息串扰，stream 错乱
```

#### After (per-turn state)
```
时间线：
t0: Message 1 开始 stream
    _stream_turns["chat1:req1"] = StreamTurn(stream_id="stream-aaa")
    
t1: 审批提示触发（新消息）
    _stream_turns["chat1:req2"] = StreamTurn(stream_id="stream-bbb")
    ← Message 1 的状态完全独立
    
t2: Message 1 finalize
    使用 _stream_turns["chat1:req1"].stream_id = "stream-aaa"  ← 正确！
    
t3: Message 2 finalize
    使用 _stream_turns["chat1:req2"].stream_id = "stream-bbb"  ← 正确！
    
结果：两个 stream 完全隔离，互不干扰
```

### 兼容性处理

**保留 `_reset_native_stream_state()` 方法**
```python
def _reset_native_stream_state(self) -> None:
    """Legacy method for compatibility. Now a no-op since state is per-turn."""
    pass
```

保留这个方法是为了兼容现有代码中的调用，但实际上已经是 no-op（不执行任何操作），因为状态现在是 per-turn 管理的。

### 测试覆盖

- ✅ 同一 chat 的多个并发 turn 隔离
- ✅ 不同 turn 有不同的 stream_id
- ✅ 查找活跃 turn 正确
- ✅ Turn cleanup 正确清理

---

## 官方上游 PR 状态（截至 2026-06-05）

以下 PR 正在竞争同一功能位（全部为 Open 状态，均未合并到 main）：

| PR | 方案 | 备注 |
|----|------|------|
| #35772 | `send_stream_frame` + native transport | 最接近本改造，但缺 thinking 占位和 846608 降级 |
| #35538 | native reply streaming + thinking dots | 包含 thinking 动画，block 级别 |
| #20623 | stabilize native streaming | 频率控制、单气泡复用、846609 降级 |
| #30960 | draft-streaming framework | 不同架构路径 |
| #20313 | progressive streaming fixes | typing bubble 修复 |

**当其中任何一个合并到 main 时**，检查流程：

1. `git diff main..feat/wecom-native-streaming -- gateway/platforms/wecom.py` 检查冲突
2. 官方实现了 `SUPPORTS_NATIVE_STREAMING` + `send_stream_frame` → 删除我们的对应实现
3. 官方如果没有 fire-and-forget 优化 → 保留 `_send_stream_reply` 改动
4. `stream_consumer.py` 的 native 分支 → 参考官方的，保留空 `_accumulated` 兜底
5. `gateway/run.py` gate → 用官方的
6. `display_config.py` → 用官方的
7. `tools/send_message_tool.py` → **必须保留**，官方 PR 不会改这里

---

## 回退指南

**完全回退到改造前行为**：
```bash
git checkout main
```

**只关闭 streaming（保留代码）**：
```yaml
# config.yaml
display:
  platforms:
    wecom:
      streaming: false
```
