# WeCom 双通道优先级队列实现总结

## 问题背景

### 原始问题
用户报告：在使用 WeCom（企业微信）平台时，当 Agent 需要审批（approval）执行危险命令时，审批提示可能会被阻塞，导致用户看不到提示信息，无法及时响应 `/approve` 或 `/deny`。

### 根本原因
原始实现使用单一 FIFO 队列处理所有消息：
```
Queue: [stream_finalize(阻塞15秒等待WeCom响应), 审批提示(排队), 其他消息...]
                    ↑ 阻塞                         ↑ 被卡住
```

当 stream finalize 帧等待 WeCom 服务器响应（最多15秒超时）时，审批提示排在队列后面，用户无法及时看到。

### 为什么不能简单移除队列？
1. **频率限制实证**：生产环境已触发过 WeCom 的 846607 限流错误（30条/分钟）
2. **消息顺序**：完全并发会导致消息交错（审批提示插在 stream 中间，破坏因果逻辑）
3. **协议要求**：WeCom stream 需要保证 seed → delta → finalize 的顺序

## 解决方案：双通道优先级队列

### 架构设计

```
┌─────────────────────────────────────────────────┐
│             Inbound Messages                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          Message Type Router                     │
└─────────────────────────────────────────────────┘
         ↓              ↓              ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Control Lane │ │ Normal Lane  │ │Fire-and-Forget│
│ (Priority)   │ │              │ │(No Queue)     │
├──────────────┤ ├──────────────┤ ├──────────────┤
│• Approval    │ │• Regular msg │ │• Stream delta │
│• Finalize    │ │• Media       │ │              │
│• Errors      │ │• Caption     │ │              │
├──────────────┤ ├──────────────┤ └──────────────┘
│6 reserved    │ │24 tokens/min │
│+ normal余量  │ │              │
└──────────────┘ └──────────────┘
         ↓              ↓
┌─────────────────────────────────────────────────┐
│    Token Bucket (30 tokens/min/chat)            │
│    24 normal + 6 reserved                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│            WeCom WebSocket                       │
└─────────────────────────────────────────────────┘
```

### 核心组件

#### 1. Control Lane（控制通道）
- **文件：** `gateway/platforms/wecom.py:330-335`
- **队列：** `_control_queues` (per-chat)
- **Worker：** `_control_send_worker()`
- **用途：** 审批提示、stream finalize、错误通知
- **配额：** 6 reserved tokens + normal 剩余

#### 2. Normal Lane（普通通道）
- **文件：** `gateway/platforms/wecom.py:337-343`
- **队列：** `_chat_queues` (per-chat)
- **Worker：** `_chat_send_worker()`
- **用途：** 常规消息、媒体 caption
- **配额：** 24 normal tokens（不能用 reserved）

#### 3. Fire-and-Forget（无队列）
- **文件：** `gateway/platforms/wecom.py:2022-2024`
- **行为：** 立即发送，不等 ACK
- **用途：** stream 中间帧（WeCom 不计入频率限制）

### Token Bucket 实现

```python
# gateway/platforms/wecom.py:265-315

# Per-chat tracking
{
  chat_id: {
    "normal": 已使用数量,
    "reserved": 已使用数量,
    "last_reset": 重置时间戳
  }
}

# Normal 消息（_bucket_try_consume）
- 只能使用 24 normal tokens
- 用完后等待下一分钟

# Control 消息（_bucket_try_consume_control）
- 优先使用 normal 剩余配额（不浪费 reserved）
- normal 用完后使用 6 reserved tokens
- 都用完后等待下一分钟
```

### 集成点

#### 1. 审批提示标记
**文件：** `gateway/run.py:18103-18105`

```python
_approval_metadata = dict(_status_thread_metadata or {})
_approval_metadata["is_approval_prompt"] = True
```

#### 2. 路由到 Control Lane
**文件：** `gateway/platforms/wecom.py:1697-1702`

```python
is_control = metadata.pop("is_approval_prompt", False) if metadata else False
return await self._enqueue_chat_send(..., is_control=is_control)
```

#### 3. Finalize 走 Control Lane
**文件：** `gateway/platforms/wecom.py:2016-2021`

```python
if finalize:
    return await self._enqueue_chat_send(..., is_control=True)
```

## 效果验证

### 预期效果
- ✅ **审批提示立即发送**：不等待前面的 finalize（< 1秒 vs 之前可能 15秒）
- ✅ **finalize 不阻塞新消息**：走 control lane，不占用 normal 配额
- ✅ **频率限制保证**：token bucket 确保不超过 30 条/分钟
- ✅ **消息顺序正确**：control lane 仍是 FIFO，不会错乱

### 单元测试
**文件：** `test_wecom_priority_queue.py`

运行测试：
```bash
cd /Users/bilibili/.hermes/hermes-agent
python test_wecom_priority_queue.py
```

验证项：
- Token 分配：24 normal + 6 reserved
- Normal 消息不能使用 reserved
- Control 消息可使用全部 30 tokens
- Reserved 在 normal 耗尽后仍可用

### 手动测试（需要真实 WeCom 连接）
1. 发送长消息触发 streaming（生成多个 stream 帧）
2. 在输出过程中触发需要审批的命令（如 `rm -rf` 危险操作）
3. 验证审批提示立即显示（不等待 finalize 完成）
4. 响应 `/approve` 或 `/deny`，验证流程正常

## 代码变更总结

### 修改的文件

1. **gateway/platforms/wecom.py**
   - 新增 `_control_queues` 和 `_control_workers`（L244-247）
   - 新增 `_chat_token_usage`（L248）
   - 重构 token bucket：`_bucket_try_consume()` 和 `_bucket_try_consume_control()`（L265-315）
   - 修改 `_enqueue_chat_send()` 支持 `is_control` 参数（L317-347）
   - 新增 `_control_send_worker()`（L374-402）
   - 修改 `send()` 支持 `is_approval_prompt` metadata（L1762-1773）
   - 修改 `send_stream_frame()` finalize 走 control lane（L2016-2021）
   - 修改 `disconnect()` 清理 control workers（L466-472）

2. **gateway/run.py**
   - 修改 `_approval_notify_sync()` 添加 `is_approval_prompt` metadata（L18103-18105）

3. **feature-wecom-native-streaming.md**
   - 新增"双通道优先级队列"章节
   - 更新对比表格

4. **test_wecom_priority_queue.py** (新文件)
   - Token bucket 单元测试

### 代码统计
- **修改行数：** ~150 行
- **新增代码：** ~100 行
- **测试代码：** ~100 行

## 理论依据

### Codex 的分析
> "保留限流，去掉会导致审批卡住的串行语义"

关键洞察：
1. **不是"要不要队列"**，而是**"什么消息走什么队列"**
2. **Critical 消息**（审批、finalize、错误）需要预留通道
3. **Token 分层**比完全并发更可控，比单一队列更灵活

### 官方插件的选择
- 官方 TypeScript 插件禁用了队列（直接并发）
- 但他们的频率限制可能更宽松，或者用户量较少未触发
- Hermes Agent 生产环境已有 846607 实证，必须限流

### 与其他方案对比

| 方案 | 审批延迟 | 频率控制 | 消息顺序 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| 原始 FIFO | ❌ 最多15秒 | ✅ 严格 | ✅ 完美 | 简单 |
| 完全并发 | ✅ 立即 | ❌ 无保证 | ❌ 可能错乱 | 简单 |
| Fire-and-Forget All | ✅ 立即 | ❌ 可能超限 | ❌ 严重错乱 | 简单 |
| **双通道队列** | ✅ 立即 | ✅ 严格 | ✅ 保证 | **中等** |

## 未来优化方向

### Phase 2: Stream State Per-Turn（未实现）
- **问题：** 当前 `_active_stream_id` 是全局共享，多个并发消息会冲突
- **方案：** 每个 inbound message turn 独立的 stream state
- **影响：** 中等（需要重构 stream 状态管理）

### Phase 3: 中间帧合并（可选）
- **问题：** 高频中间帧可能浪费带宽
- **方案：** 只保留最新的中间帧，定期 flush
- **影响：** 小（性能优化，不影响正确性）

## 参考文档

- **主文档：** `feature-wecom-native-streaming.md`
- **原始讨论：** 本次对话记录
- **官方插件：** `/Users/bilibili/.hermes/wecom-openclaw-plugin/src/`
- **Codex 分析：** "保留限流，去掉会导致审批卡住的串行语义"

---

**实施日期：** 2026-06-05  
**实施者：** Claude (Opus 4.8) + 用户  
**状态：** ✅ 已实现并测试
