# Codex Review Round 2 - 深度分析

## 问题根源：架构不匹配

### 官方模型 vs 当前实现

**官方 wecom-openclaw-plugin 模型：**
```typescript
// monitor.ts:831
function handleInboundMessage(msg) {
  const streamId = generateStreamId();  // 每条入站创建
  const state = { streamId, reqId: msg.req_id };  // 闭包绑定
  
  deliver(state.streamId, content);      // 使用闭包 streamId
  finishThinkingStream(state.streamId);  // 使用闭包 streamId
}
```

**当前实现（问题）：**
```python
# 全局查找 active turn
def _send_stream_frame_inner():
    turn = self._find_active_turn_for_chat(chat)  # ← 按 chat 猜测
    if turn:
        reuse turn  # 可能复用错误的 turn
```

**核心问题：**
1. **没有 turn handle 传递** - GatewayStreamConsumer 不持有自己的 turn
2. **按 chat 猜测 active turn** - 多个 consumer 会互相干扰
3. **req_id 通过 _last_chat_req_ids 回退** - /approve 刷新后会切换

---

## Finding 1: /approve ack 关闭 stream

### 检查我的修复

让我检查 commit `7a420f6a6` 是否真的修复了...

**我的修复：**
```python
# gateway/run.py
async def _handle_approve_command(...):
    count = resolve_gateway_approval(...)
    
    if _adapter:
        asyncio.create_task(_adapter.send(
            source.chat_id,
            confirmation_text,
            metadata={"skip_stream_finalize": True}
        ))
    
    return None
```

**Review 说的问题路径：**
- `run.py:14825` → `resolve_gateway_approval()`
- `base.py:3865` → handler 返回文本 → `send()`
- 没有 `is_approval_prompt` → 进入普通 finalize 逻辑

**我的修复是否覆盖？**
- ✅ 我让 handler 返回 `None`
- ✅ 我直接调用 `send()` 并传 `skip_stream_finalize=True`
- ✅ 不会进入 `base.py:3865` 的默认流程

**但等等...** 让我检查 `skip_stream_finalize` 是否等同于 `is_approval_prompt`？

```python
# wecom.py send() 方法
if metadata:
    is_control = metadata.pop("is_approval_prompt", False)
    if is_control:
        skip_stream_finalize = True  # ← 只有 is_approval_prompt 设置这个
```

**问题！** `skip_stream_finalize` 不是 metadata key，只是内部变量！

我应该传 `{"is_approval_prompt": True}` 而不是 `{"skip_stream_finalize": True}`！

---

## Finding 2: StreamTurn 不是真正的 per-turn ✅ **关键问题**

这个分析是正确的。需要实施方案 A：Consumer 持有 turn_id。

---

## 正确的修复方案

### 修复 1: /approve ack metadata 错误

```python
# gateway/run.py
metadata={"is_approval_prompt": True}  # ← 不是 skip_stream_finalize
```

### 修复 2: 真正的 per-turn 模型

实施方案 A：
1. GatewayStreamConsumer 生成 turn_id
2. send_stream_frame() 传递 turn_id  
3. WeComAdapter 按 turn_id 查找 turn

---

**结论：Review 是对的，我的修复有缺陷**
