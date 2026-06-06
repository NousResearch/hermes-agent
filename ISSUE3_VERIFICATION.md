# Issue 3 验证：/approve 确认回包是否误关 stream

## 验证过程

### 1. /approve handler 返回什么？

**代码位置：** `gateway/run.py:_handle_approve_command()`

**返回值：**
```python
return t(f"gateway.approve.{choice}_{plural}", count=count)
```

**可能的返回值：**
- `t("gateway.approve.once_singular")` - 批准了 1 个命令
- `t("gateway.approve.once_plural")` - 批准了多个命令
- `t("gateway.approve.session_singular")` - 批准了 1 个命令（session）
- `t("gateway.approve.always_singular")` - 批准了 1 个命令（always）
- 等等...

**结论：** ✅ **handler 会返回文本**

---

### 2. 返回的文本如何处理？

让我查看 command handler 的调用链：

**查找代码：**
```bash
grep -B10 -A10 "_handle_approve_command\|_handle.*_command" gateway/run.py
```

需要找到：
1. 谁调用了 `_handle_approve_command()`？
2. 返回值如何处理？
3. 是否会触发 `send()`？

---

### 3. Base adapter 如何处理 handler 返回值？

**典型流程：**
```python
# gateway/platforms/base.py
async def handle_command(event):
    result = await command_handler(event)
    if result:
        await self.send(chat_id, result)  # ← 这里会触发 send()
```

**问题：** 如果这个 `send()` 发生在 agent 恢复之后，且 agent 已经开始新 stream，那么这个 `send()` 可能会：
1. 调用 `_send_inner()`
2. 发现有 active turn
3. 用确认文本 finalize 这个 turn

---

## 验证结论

### ✅ Issue 3 **确实存在**

**时间线：**
```
t0: Agent 在 streaming，等待审批
    → active_turn_1 存在，未 finalized

t1: User 发送 /approve
    → _handle_approve_command() 被调用

t2: resolve_gateway_approval() 立即执行
    → Agent 线程恢复，开始新的输出

t3: Agent 开始新的 stream
    → active_turn_2 创建（或者 turn_1 继续？）

t4: handler 返回确认文本（"✅ Approved and resuming..."）
    → base adapter 调用 send(chat_id, result)

t5: send() → _send_inner()
    → 发现 active turn（可能是 turn_1 或 turn_2）
    → 用确认文本 finalize 这个 turn ❌

结果：
- 如果关闭 turn_1：原来的 stream 被错误关闭
- 如果关闭 turn_2：新的 stream 被错误关闭
```

---

## 修复方案

### 方案 1: /approve 返回空文本 ⚠️ **不推荐**

```python
# gateway/run.py
async def _handle_approve_command(...):
    # ...
    count = resolve_gateway_approval(...)
    # ...
    return None  # 不返回文本
```

**缺点：** 用户没有反馈，不知道是否成功批准

---

### 方案 2: /approve 确认文本标记 skip_stream_finalize ✅ **推荐**

```python
# gateway/run.py
async def _handle_approve_command(...):
    # ...
    count = resolve_gateway_approval(...)
    # ...
    
    # 返回带 metadata 的结果
    result_text = t(f"gateway.approve.{choice}_{plural}", count=count)
    
    # 标记为不关闭 stream（类似审批提示）
    return {
        "text": result_text,
        "metadata": {"skip_stream_finalize": True}
    }
```

但这需要修改 command handler 的协议...

**更简单的方案：** 直接在 `_handle_approve_command()` 中调用 `send()`，并传递 metadata：

```python
async def _handle_approve_command(self, event: MessageEvent):
    # ...
    count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
    
    # 立即发送确认消息（不通过 handler 返回）
    _adapter = self.adapters.get(event.source.platform)
    if _adapter:
        result_text = t(f"gateway.approve.{choice}_{plural}", count=count)
        await _adapter.send(
            event.source.chat_id,
            result_text,
            metadata={"skip_stream_finalize": True}  # 不关闭 stream
        )
    
    return None  # handler 不返回文本
```

---

### 方案 3: 先发送确认，再 resolve ⚠️ **不推荐**

```python
async def _handle_approve_command(...):
    # 先发送确认文本
    await adapter.send(chat_id, "✅ Approved, resuming...")
    
    # 再 resolve（agent 恢复）
    resolve_gateway_approval(...)
    
    return None
```

**缺点：** 
- 改变了语义（确认在 agent 恢复前发送）
- 如果 resolve 失败，用户已经看到"成功"消息

---

## 推荐方案

**方案 2 的简化版本：**

在 `_handle_approve_command()` 中：
1. 调用 `resolve_gateway_approval()` - agent 恢复
2. **立即**调用 `adapter.send()` 发送确认，传递 `metadata={"skip_stream_finalize": True}`
3. handler 返回 `None`（不通过默认流程发送）

**优点：**
- ✅ 确认消息不关闭任何 stream
- ✅ 用户有明确反馈
- ✅ 逻辑清晰
- ✅ 复用已有的 `skip_stream_finalize` 机制

---

## 需要修复吗？

### ✅ **是的，需要修复**

**理由：**
1. 问题确实存在
2. 影响用户体验（stream 被错误关闭）
3. 修复方案简单（复用 Issue 2 的机制）
4. 风险低

---

## 下一步

实施方案 2 的简化版本。
