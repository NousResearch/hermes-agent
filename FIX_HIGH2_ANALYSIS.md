# Fix High #2 分析：send() 不应该关闭 stream

## 问题

当前 `send()` 会按 chat 找第一个 active turn 并关闭它：

```python
# gateway/platforms/wecom.py:1861
active_turn = self._find_active_turn_for_chat(chat_id.strip())
if active_turn and not active_turn.finalized:
    # 关闭这个 turn
    await self._send_stream_reply(..., finish=True)
    active_turn.finalized = True
```

这有三个问题：

1. **会关错 turn**：如果有两个并发 consumer，`send()` 可能关闭错误的 turn
2. **职责不清**：GatewayStreamConsumer 会调用 `send_stream_frame(finalize=True)` 来关闭自己的 turn，不需要 `send()` 帮忙
3. **不对齐官方模型**：官方 wecom-openclaw-plugin 中，send() 和 stream 是独立的

## 正确的设计

### 官方模型

```typescript
// wecom-openclaw-plugin
async function handleMessage(msg) {
  const state = { streamId: generateStreamId(), reqId: msg.req_id };
  
  // Stream 由 message handler 自己管理
  await deliver(state.streamId, content);           // 发送帧
  await finishThinkingStream(state.streamId);       // 关闭 stream
  
  // send() 是独立的，不关闭 stream
  await sendMessage(chatId, "some text");
}
```

### 我们的实现应该是

```python
# GatewayStreamConsumer 负责自己的 stream
async def run(self):
    # 发送帧
    await self.adapter.send_stream_frame(text, turn_id=self._turn_id)
    
    # 关闭自己的 stream
    await self.adapter.send_stream_frame(text, finalize=True, turn_id=self._turn_id)

# send() 不关闭任何 stream，只发送独立消息
await adapter.send(chat_id, "some text")
```

## 修复方案

### 方案 A：完全删除 send() 中关闭 stream 的逻辑 ✅ **推荐**

```python
async def _send_inner(..., skip_stream_finalize=False):
    # 删除这段逻辑：
    # if not skip_stream_finalize:
    #     active_turn = self._find_active_turn_for_chat(...)
    #     if active_turn:
    #         # 关闭 turn
    
    # 直接发送消息
    reply_req_id = self._reply_req_id_for_message(reply_to)
    ...
```

**优点：**
- ✅ 职责清晰：stream 由创建者管理
- ✅ 不会关错 turn
- ✅ 对齐官方模型

**缺点：**
- ⚠️ 需要确保 GatewayStreamConsumer 总是 finalize（已经做到了）

### 方案 B：只关闭没有 turn_id 的 turn ⚠️ **不推荐**

保留 `_find_active_turn_for_chat()` 逻辑，但只关闭那些没有 turn_id 的 turn（旧代码路径）。

**问题：**
- 还是会按 chat 猜测，不够精确
- 逻辑复杂，难以维护

## 推荐方案

**采用方案 A**：
1. 删除 `send()` 中关闭 active turn 的逻辑
2. `skip_stream_finalize` 参数变得多余，可以删除
3. GatewayStreamConsumer 负责自己的 stream 生命周期

## Stream Expired 的修复

类似地，stream expired 也不应该把整个 chat 的 turn 都标死：

```python
# 当前（问题）
except WeComStreamExpiredError:
    self._stream_expired_chats.add(chat)
    # 把整个 chat 的所有 turn 都标死 ❌
    for turn in self._stream_turns.values():
        if turn.chat_id == chat:
            turn.expired = True

# 正确的做法
except WeComStreamExpiredError:
    # 只标记这个 turn expired ✅
    turn.expired = True
    if turn_id:
        self._stream_turns.pop(f"{chat}:{turn_id}", None)
    else:
        self._cleanup_stream_turn(chat, turn.req_id)
    
    # 只在全局标记 chat expired（阻止新 stream）
    self._stream_expired_chats.add(chat)
```

**原因：**
- Stream expired 只影响当前 turn
- 其他并发 turn 可能用不同的 req_id，可以继续
- 只在 chat 级别记录"expired"，阻止新 stream 创建即可
