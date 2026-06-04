# WeCom Native Streaming 改造记录

> **分支**：`feat/wecom-native-streaming`
> **日期**：2026-06-04
> **基于**：hermes-agent main (`40420a619`)

## 背景

Hermes Agent 的 WeCom 适配器原本不支持流式输出（`SUPPORTS_MESSAGE_EDITING = False` 导致 gateway 跳过 streaming），用户发消息后要等 LLM 完整生成才能看到回复，没有"输入中"反馈。

本改造对齐了腾讯官方 `@wecom/wecom-openclaw-plugin` 的行为，使用 WeCom AI Bot 的 `msgtype: "stream"` 协议实现：
- 收到消息后立即显示"输入中"动画
- 逐段推送累积内容（频率友好）
- `finish=true` 关闭流

## 改动文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `gateway/platforms/wecom.py` | +290 | 流式协议帧、thinking 占位、fire-and-forget、846608 降级、竞争保护 |
| `gateway/stream_consumer.py` | +173/-14 | native streaming transport 分支 |
| `gateway/run.py` | +16/-2 | streaming gate 放行 native 适配器 |
| `gateway/display_config.py` | +7/-1 | wecom 默认 streaming=True |
| `hermes_cli/config.py` | +4 | DEFAULT_CONFIG 暴露 wecom streaming toggle |
| `tests/gateway/test_wecom.py` | +348 | 适配器单测 |
| `tests/gateway/test_stream_consumer_wecom_native.py` | +323（新文件） | consumer ↔ adapter 集成测试 |
| `tests/gateway/test_display_config.py` | +25 | wecom streaming 默认值 |
| `tests/gateway/test_per_platform_streaming_defaults.py` | +2 | DEFAULT_CONFIG 断言 |

## Commit 列表

```
d0517ebb1 fix(wecom): prevent _keep_typing from reopening stream after send() closes it
abd00d71c perf(wecom): fire-and-forget intermediate stream frames (don't await ack)
9af225650 fix(wecom): close active stream in send() to prevent lingering typing bubble
07e2f5485 test(wecom): cover native streaming lifecycle and 846608 fallback
bed7dcb68 fix(gateway): allow native-streaming adapters past edit-only gate
431f55119 feat(stream_consumer): add native streaming transport
6c7f6a06e feat(wecom): add native streaming primitives (msgtype: stream)
```

## 核心机制

### 1. 流式协议（wecom.py）

```python
# 新增类属性
SUPPORTS_NATIVE_STREAMING = True
MAX_STREAM_CONTENT_LENGTH = 20480

# 核心方法
send_stream_frame(text, *, finalize, chat_id, reply_to) -> bool
_send_stream_reply(reply_req_id, stream_id, content, finish) -> Dict
_resolve_stream_req_id(chat_id, reply_to) -> Optional[str]
supports_native_streaming(chat_type, metadata) -> bool
```

**协议流程**：
```
1. send_stream_frame("", chat_id=X)
   → _send_json(fire-and-forget): seed 空帧 → 客户端显示 typing
2. send_stream_frame("你好世界", chat_id=X)
   → _send_json(fire-and-forget): 累积内容推送
3. send_stream_frame("完整回复", chat_id=X, finalize=True)
   → _send_reply_request(await ack): finish=true → 客户端替换为最终内容
```

### 2. Fire-and-forget 中间帧（性能关键）

中间帧（`finish=False`）不走 `_send_reply_request`（等 ack），直接走 `_send_json`。原因：WeCom 服务端对中间帧不回 ack，走 `_send_reply_request` 会每帧阻塞 15 秒（REQUEST_TIMEOUT）。

只有 `finish=True` 帧走 `_send_reply_request` 等 ack，用来可靠检测 `errcode 846608`。

### 3. 频率控制（stream_consumer.py）

```python
_MIN_NEW_VISIBLE_CHARS = 20  # 至少新增 20 字符才发一帧
```

配合 `edit_interval`（1.5s），一条 30s 回复约产生 5-10 帧，远低于 30/min 限制。

### 4. 竞争保护

| 问题 | 保护机制 |
|------|---------|
| `_keep_typing` 在 `send()` 关闭 stream 后又开新 stream | `_stream_delivered_chats` 标记 |
| stream 过期 (>6min) | `_stream_expired_chats` + 846608 检测 |
| `send()` 调用时有 active stream | 自动用 `finish=true` 关闭后再送内容 |
| content 为空的 finish 帧（WeCom 忽略空白） | 兜底用 `"✅"` 可见字符 |

### 5. Gateway gate 放行（run.py）

```python
# 原来：
if not _adapter_supports_edit:
    raise RuntimeError("skip streaming for non-editable platform")

# 改为：
if not _adapter_supports_edit and not _adapter_supports_native_stream:
    raise RuntimeError("skip streaming for non-editable platform")
```

### 6. Display config（display_config.py）

```python
"wecom": {**_TIER_LOW, "streaming": True},  # 原来是 _TIER_LOW（streaming=False）
```

## 官方可能的合并路径

以下 PR 正在竞争同一功能位（截至 2026-06-04）：

| PR | 状态 | 方案 |
|----|------|------|
| #35772 | Open | `send_stream_frame` + native transport（最接近我们的实现） |
| #35538 | Open | native reply streaming + markdown chunking + thinking dots |
| #20623 | Open | stabilize native streaming（频率控制、单气泡复用） |
| #30960 | Open | draft-streaming framework |
| #34012 | Open | REQUIRES_EDIT_FINALIZE framework |

**当其中任何一个被合并到 main 时**，我们应该：

1. `git diff main..feat/wecom-native-streaming -- gateway/platforms/wecom.py` 检查冲突
2. 官方如果实现了 `SUPPORTS_NATIVE_STREAMING` + `send_stream_frame` → 删除我们的对应代码
3. 官方如果没有 fire-and-forget 优化 → 保留我们的 `_send_stream_reply` 改动
4. 官方如果没有 `_stream_delivered_chats` 竞争保护 → 保留
5. `stream_consumer.py` 中的 `_resolve_native_streaming` 分支 → 如果官方合了类似代码，回退我们的
6. `gateway/run.py` gate 改动 → 官方一定会改这里，直接用他们的
7. `display_config.py` → 官方会自己开 wecom streaming，用他们的

## 回退指南

如果需要**完全回退**到改造前行为：

```bash
# 方式1：回退到 main
git checkout main

# 方式2：只关闭 streaming（保留代码但不生效）
# 在 config.yaml 中：
display:
  platforms:
    wecom:
      streaming: false
```

如果只需要**关闭 typing 但保留流式**，在 `send_typing` 中 early return：

```python
async def send_typing(self, chat_id: str, metadata=None) -> None:
    return  # 禁用 typing，流式仍然生效
```

## 与官方插件的已知差异

| 维度 | 官方 OpenClaw 插件 | 本改造 |
|------|-------------------|--------|
| 流式粒度 | block 级别（OpenClaw 核心控制） | token + edit_interval + 20 chars 节流 |
| 每次对话帧数 | 3-5 | 5-10 |
| Thinking 占位 | `<think></think>` | 空字符串 `""` |
| NonBlocking 帧跳过 | SDK 提供但实际未使用 | 未实现（fire-and-forget 等效） |
| 846608 检测时机 | 每帧都检测 | 仅 finish 帧（中间帧 fire-and-forget） |
| 串行队列 | `chat-queue.ts` 保证同 chat 串行 | 无（依赖 gateway 的 session lock） |
| 连接模式 | SDK WSClient | 自建 aiohttp WS |
