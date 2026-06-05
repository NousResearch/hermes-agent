# Codex Review 问题分析

## Issue 1: StreamTurn 切换到新 req_id ✅ **这是真实问题**

### 问题描述
当用户发 `/approve` 时：
1. WeCom 入站消息刷新 `_last_chat_req_ids[chat]` = approve 消息的 req_id
2. Agent 恢复后继续 streaming
3. `send_stream_frame()` 每次都调用 `_resolve_stream_req_id(chat, reply_to)`
4. **问题：** `reply_to` 来自 `initial_reply_to_id`（原始触发消息），但代码逻辑会先查 `_reply_req_id_for_message(reply_to)`，这个可能返回 None（如果 reply_to 是很久前的消息）
5. 然后 fallback 到 `_last_chat_req_ids.get(chat)`，这时获取到 approve 的 req_id
6. 创建新的 StreamTurn，原来的 stream 关不掉

### 实际代码流程

**Stream Consumer 初始化：**
```python
# gateway/run.py
_stream_consumer = GatewayStreamConsumer(
    initial_reply_to_id=event_message_id,  # 原始触发消息的 ID
)
```

**每次发送帧时：**
```python
# gateway/stream_consumer.py
await self.adapter.send_stream_frame(
    text,
    reply_to=self._initial_reply_to_id,  # 传递原始消息 ID
)

# gateway/platforms/wecom.py:_send_stream_frame_inner()
req_id = self._resolve_stream_req_id(chat, reply_to)  # reply_to = 原始消息 ID

# _resolve_stream_req_id()
req_id = self._reply_req_id_for_message(reply_to)  # 尝试从缓存查找
if req_id:
    return req_id
return self._last_chat_req_ids.get(chat)  # ← 问题在这里！
```

**`_reply_req_id_for_message()` 的问题：**
```python
# gateway/platforms/wecom.py
def _reply_req_id_for_message(self, message_id: Optional[str]) -> Optional[str]:
    if not message_id:
        return None
    return self._reply_req_ids.get(str(message_id).strip()) or None
```

这个缓存可能：
1. 没有保存 event_message_id 对应的 req_id
2. 已经过期被清理
3. event_message_id 本身就不是 WeCom 消息 ID

**结果：** Fallback 到 `_last_chat_req_ids[chat]`，拿到 approve 消息的 req_id！

### 解决方案

**方案 1：StreamTurn 在第一次创建时锁定 req_id**

StreamTurn 应该在创建时就固定 req_id，后续不再变化：

```python
class StreamTurn:
    def __init__(self, chat_id: str, req_id: str):
        self.chat_id = chat_id
        self.req_id = req_id  # 固定，不再变化
        self.stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        self.locked_req_id = req_id  # 明确标记已锁定
        # ...

# 在 _send_stream_frame_inner() 中
# 如果已有 turn，使用它的 req_id，而不是重新 resolve
active_turn = self._find_active_turn_for_chat(chat)
if active_turn and not active_turn.finalized:
    # 复用现有 turn，不创建新的
    turn = active_turn
else:
    # 创建新 turn
    req_id = self._resolve_stream_req_id(chat, reply_to)
    if not req_id:
        return False
    turn = self._get_or_create_stream_turn(chat, req_id)
```

**方案 2：Stream Consumer 传递 turn 标识**

让 stream consumer 传递一个 turn_id，作为 metadata：

```python
# gateway/stream_consumer.py
self._turn_id = str(uuid.uuid4())  # 创建时生成

await self.adapter.send_stream_frame(
    text,
    reply_to=self._initial_reply_to_id,
    metadata={"turn_id": self._turn_id},  # 传递 turn ID
)

# gateway/platforms/wecom.py
# 使用 turn_id 作为 StreamTurn 的 key，而不是 chat:req_id
```

### 验证测试

需要测试：
```python
def test_approve_doesnt_switch_req_id():
    """Test that /approve doesn't switch stream to new req_id"""
    # 1. 开始 stream (req_id = "req1")
    # 2. 模拟 /approve 刷新 _last_chat_req_ids["chat1"] = "req2"
    # 3. 继续发送 stream frame
    # 4. 验证仍然使用 req1，而不是 req2
```

---

## Issue 2: 审批提示关闭错误的 turn ✅ **这是真实问题**

### 问题描述

`_send_inner()` 调用 `_find_active_turn_for_chat()`，这个方法只按 chat_id 找第一个未 finalized 的 turn：

```python
def _find_active_turn_for_chat(self, chat_id: str) -> Optional[StreamTurn]:
    for turn in self._stream_turns.values():
        if turn.chat_id == chat_id and not turn.finalized:
            return turn  # ← 返回第一个匹配的
    return None
```

**问题场景：**
1. Agent 正在 streaming（turn1, req_id="req1"）
2. 触发审批，发送审批提示（这是新消息，创建 turn2, req_id="req2"）
3. 审批提示走 `send()`，里面调用 `_find_active_turn_for_chat()`
4. 可能找到 turn1，然后用审批文本 finalize turn1
5. turn1 被错误关闭，turn2 也可能受影响

### 解决方案

**审批提示不应该关闭任何 stream**，应该是独立的消息：

```python
async def _send_inner(
    self,
    chat_id: str,
    content: str,
    reply_to: Optional[str] = None,
    is_approval: bool = False,  # 新增参数
) -> SendResult:
    try:
        # 如果是审批提示，跳过 stream finalize 逻辑
        if not is_approval:
            active_turn = self._find_active_turn_for_chat(chat_id.strip())
            if active_turn and not active_turn.finalized:
                # 关闭 active stream
                # ...
```

或者更简单：审批提示应该指定 reply_to，确保不会误匹配。

---

## Issue 3: /approve 确认回包误关新 stream ⚠️ **需要验证**

### 问题描述

`_handle_approve_command()` 流程：
1. 先调用 `resolve_gateway_approval()` → agent 线程立刻恢复
2. 然后 base adapter 发送 handler 返回的文本（"approved/resuming"）
3. 如果 agent 已经开了新 stream，这条确认消息可能会关闭新 stream

### 需要检查

1. `/approve` handler 是否返回文本？
2. 返回的文本是否会触发 `send()`？
3. 是否会进入 `_send_inner()` 的 stream finalize 逻辑？

---

## Issue 4: Control lane 配额耗尽等 60 秒 ⚠️ **理论问题，但概率极低**

### 问题描述

`_bucket_try_consume_control()` 在 30 token 全部用完后等待 60 秒：

```python
def _bucket_try_consume_control(self, chat_id: str) -> float:
    # ...
    # Both exhausted, wait until next minute
    return 60.0 - (now - usage["last_reset"])
```

approval notify 只等 15 秒：
```python
_approval_send_fut.result(timeout=15)
```

**理论上：** 如果 chat 已经打满 30/min，审批提示会等待，超过 15 秒超时。

**实际上：** 
- 我们预留了 6 个 reserved token
- 正常场景下不会在 1 分钟内发 30 条消息
- 即使 normal 用完，reserved 还有 6 个

**是否需要修复：** 可以改进，但不是高优先级。

---

## Issue 5: 测试覆盖不足 ✅ **这是真实问题**

测试没有覆盖：
- /approve 刷新 req_id 后旧 stream 是否能 finalize
- /approve ack 是否会误关恢复后的 stream
- 多 active turn 同 chat 时是否关错 turn

需要补充集成测试。

---

## 总结

| Issue | 是否真实 | 优先级 | 是否需要修复 |
|-------|---------|--------|------------|
| 1. StreamTurn 切换 req_id | ✅ 是 | High | **是** |
| 2. 审批提示关错 turn | ✅ 是 | High | **是** |
| 3. /approve ack 误关 stream | ⚠️ 需验证 | High | **需要先验证** |
| 4. Control lane 等 60 秒 | ⚠️ 概率低 | Medium | 可选 |
| 5. 测试覆盖不足 | ✅ 是 | Medium | **是** |

**建议修复顺序：**
1. Issue 1: StreamTurn 锁定 req_id
2. Issue 2: 审批提示不关闭 stream
3. Issue 3: 验证 /approve ack 行为，如需要则修复
4. Issue 5: 补充测试
5. Issue 4: 可选改进

---

**下一步：** 你确认后，我立即修复 Issue 1 和 2。
