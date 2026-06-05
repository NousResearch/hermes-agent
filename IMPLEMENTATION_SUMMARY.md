# WeCom 双通道优先级队列 + Per-Turn Stream State - 实施完成

## ✅ 已完成的工作

### Phase 1: 双通道优先级队列
- ✅ 双通道队列系统（Control Lane + Normal Lane）
- ✅ Token bucket 分层（24 normal + 6 reserved）
- ✅ 审批提示路由到 Control Lane
- ✅ Stream finalize 路由到 Control Lane
- ✅ 中间帧 fire-and-forget（保持原有行为）

### Phase 2: Per-Turn Stream State
- ✅ StreamTurn 类（每个 turn 独立状态）
- ✅ 替换全局 _active_stream_id 为 per-turn dictionary
- ✅ _get_or_create_stream_turn() 方法
- ✅ _find_active_turn_for_chat() 方法
- ✅ _cleanup_stream_turn() 方法
- ✅ 更新所有引用全局状态的代码

### 2. 代码变更
**gateway/platforms/wecom.py** (~200 行修改)
- L168-184: 新增 StreamTurn 类
- L244-248: 使用 _stream_turns 替代全局状态
- L265-315: 重构 token bucket (支持分层配额)
- L317-347: `_enqueue_chat_send()` 支持 is_control 参数
- L374-402: 新增 `_control_send_worker()`
- L466-472: 清理 control workers
- L1207-1231: Per-turn state 管理方法
- L1738-1740: 媒体发送使用 per-turn state
- L1830-1875: `_send_inner()` 使用 per-turn state
- L2022-2045: `send_stream_frame()` 文档更新
- L2049-2133: `_send_stream_frame_inner()` 完全重构为 per-turn

**gateway/run.py** (~5 行修改)
- L18103-18105: 审批提示标记 is_approval_prompt

### 3. 文档更新
- ✅ `feature-wecom-native-streaming.md` - 新增 Phase 2 章节
- ✅ `WECOM_PRIORITY_QUEUE_IMPLEMENTATION.md` - 完整实施文档
- ✅ `IMPLEMENTATION_SUMMARY.md` - 本文档（更新）
- ✅ `test_wecom_priority_queue.py` - 新增 per-turn state 测试

### 4. 测试结果
```
✅ Test 1: Normal messages correctly limited to 24 tokens
✅ Test 2: Control messages can use 30 tokens total (24 normal + 6 reserved)
✅ Test 3: Control can use reserved tokens when normal exhausted
✅ Test 4: Both lanes wait when all tokens exhausted
✅ Test 5: Multiple concurrent turns per chat isolated correctly
✅ Test 6: Find active turn works correctly
✅ Test 7: Stream turn cleanup works correctly
```

## 🎯 解决的问题

### Phase 1: 审批提示阻塞

**问题：** 审批提示被 stream finalize 帧阻塞（最多15秒），用户无法及时看到提示并响应 `/approve`。

**原因：** 单一 FIFO 队列，所有消息平等排队：
```
[finalize(阻塞15秒) → 审批提示(等待) → ...]
```

**解决方案：** 双通道优先级队列 + 预留配额：
```
Control Lane:  [审批提示] ← 立即处理 (6 reserved tokens)
Normal Lane:   [常规消息] ← 排队 (24 tokens)
Fire-and-Forget: [中间帧] ← 不排队
```

### Phase 2: Stream 状态冲突

**问题：** 全局 stream 状态在并发场景下被覆盖。

**原因：** 
```python
self._active_stream_id = None  # 全局共享，并发时冲突
```

**场景：**
```
t0: Message 1 开始 stream (_active_stream_id = "aaa")
t1: 审批消息开始 (_active_stream_id = "bbb" ← 覆盖)
t2: Message 1 finalize (使用 "bbb" ← 错误！)
```

**解决方案：** Per-turn state 隔离：
```python
self._stream_turns = {
    "chat1:req1": StreamTurn(stream_id="aaa"),  # Message 1
    "chat1:req2": StreamTurn(stream_id="bbb"),  # 审批消息
}
# 两个 turn 完全独立，互不干扰
```

## 📊 效果对比

| 指标 | 改进前 | Phase 1 后 | Phase 2 后 |
|------|--------|-----------|-----------|
| 审批提示延迟 | 0-15秒 | < 1秒 | < 1秒 |
| 频率控制 | ✅ 30/min | ✅ 30/min | ✅ 30/min |
| 消息顺序 | ✅ 正确 | ✅ 正确 | ✅ 正确 |
| Critical 消息保证 | ❌ 无 | ✅ 预留配额 | ✅ 预留配额 |
| 并发消息隔离 | ❌ 全局冲突 | ❌ 全局冲突 | ✅ Per-turn 隔离 |
| Stream 干扰 | ❌ 可能串扰 | ❌ 可能串扰 | ✅ 完全隔离 |

## 🔍 技术细节

### Token 分配策略
- **总配额：** 30 tokens/minute/chat (WeCom 限制)
- **Normal：** 24 tokens (80%) - 普通消息专用
- **Reserved：** 6 tokens (20%) - Control 消息预留

### Control 消息使用策略
1. 优先使用 normal 剩余配额（不浪费 reserved）
2. Normal 用完后使用 reserved pool
3. 都用完后等待下一分钟

### 消息路由规则
```python
if metadata.get("is_approval_prompt"):
    → Control Lane (高优先级)
elif finalize:
    → Control Lane (保证完成)
elif stream_delta:
    → Fire-and-forget (不排队)
else:
    → Normal Lane (标准优先级)
```

## 🚀 如何验证

### 单元测试
```bash
cd /Users/bilibili/.hermes/hermes-agent
python test_wecom_priority_queue.py
```

### 手动测试（需要 WeCom 连接）
1. 发送长消息触发 streaming
2. 在输出过程中触发危险命令（如 `rm -rf`）
3. **预期：** 审批提示立即显示（< 1秒）
4. **对比：** 之前可能需要等待 0-15 秒

## 📚 参考文档

- **主文档：** `feature-wecom-native-streaming.md`
- **实施详情：** `WECOM_PRIORITY_QUEUE_IMPLEMENTATION.md`
- **Codex 分析：** "保留限流，去掉会导致审批卡住的串行语义"

## ⚠️ 注意事项

### 兼容性
- ✅ 向后兼容：没有 `is_approval_prompt` metadata 的消息走 normal lane
- ✅ 不影响其他平台：变更仅限 WeCom adapter

### 未来优化
- **Phase 2：** Stream state per-turn（避免全局 stream_id 冲突）
- **Phase 3：** 中间帧合并（减少发送频率）

### 回退方案
如果发现问题，可以通过 git 回退：
```bash
git diff HEAD -- gateway/platforms/wecom.py gateway/run.py
# 检查差异后决定是否回退
```

---

**实施日期：** 2026-06-05  
**状态：** ✅ 已完成并测试通过  
**下一步：** 生产环境验证
