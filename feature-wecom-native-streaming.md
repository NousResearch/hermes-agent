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
| 串行队列 | `chat-queue.ts` 保证同 chat 串行 | 依赖 gateway session lock |
| 连接管理 | SDK WSClient（单例） | 自建 aiohttp WS |
| proactive 群发 | 通过 req_id 缓存的 APP_CMD_RESPONSE | 同，复用 gateway adapter |

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
