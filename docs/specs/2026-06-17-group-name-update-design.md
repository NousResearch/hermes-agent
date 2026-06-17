# 飞书群聊群名自动更新

## 核心功能 + 核心目标

多群并行任务场景下，用户在飞书群列表中一眼看出每个群当前任务的主题，不需要点进群才能知道在干什么。Agent 在首次回复时自动提炼任务主题并更新群名，任务结束时恢复为待命状态。

## 设计方案

采用 NanoClaw 已验证的 `<group-name>` XML 标签模式：

1. **Agent 输出标签**：通过 system prompt 指令，让 agent 在群聊首次回复时输出 `<group-name>任务主题</group-name>` 标签
2. **Adapter 层提取**：在 agent 输出发送到飞书之前，解析并移除 `<group-name>` 标签，提取群名
3. **调用飞书 API**：用 `im.v1.chat.update` 更新群名，最长 20 字符
4. **任务结束重置**：`/new` 命令和 session 过期时，群名改回 `🤖|待命`

## 详细规格

### 1. 群名提取（group_name.py）

新建 `gateway/platforms/group_name.py`：

```python
def extract_group_name(text: str) -> tuple[str, str | None]:
    """从 agent 输出中提取 <group-name> 标签。
    
    Returns: (clean_text, group_name)
    - clean_text: 移除标签后的文本
    - group_name: 提取的群名（截断到 20 字符），无标签时为 None
    """
```

- 正则匹配 `<group-name>(.*?)</group-name>`
- 移除所有匹配的标签（可能有多个，取第一个非空的）
- 群名截断到 20 字符（飞书限制）
- 空标签 `<group-name></group-name>` 返回 None

### 2. 限频器（group_name.py）

```python
class GroupNameRateLimiter:
    """每个 chat_id 最多 5 分钟更新一次群名。"""
```

- `should_update(chat_id) -> bool`
- `record_update(chat_id)`
- 间隔：300 秒（5 分钟）
- 不同 chat_id 独立计时

### 3. 飞书 API：update_chat_name（feishu.py）

在 FeishuAdapter 中新增方法：

```python
async def update_chat_name(self, chat_id: str, name: str) -> bool:
    """更新群聊名称。非致命——失败只 warn 不阻断消息流。"""
```

- 调用 `self._client.im.v1.chat.update`
- 需要导入 `UpdateChatRequest`（或用 lark SDK 的请求构建器）
- `name` 截断到 20 字符
- 异常只 warning 日志，不影响消息发送

### 4. 集成点：agent 输出处理（gateway/run.py）

在 agent 响应文本发送到 adapter 之前（`_sanitize_gateway_final_response` 附近）：

1. 调用 `extract_group_name(response)` 提取群名
2. 用 clean_text 替换原始 response
3. 如果提取到群名 + 是群聊 + 限频器允许 → 调用 `adapter.update_chat_name()`

### 5. 集成点：session 结束重置

触发时机：
- `/new` 命令：在 `slash_commands.py` 的 `_handle_reset_command()` 中，session:end hook 触发后
- session 过期：在 `_session_expiry_watcher()` 的 finalize 流程中

两处都调用 `adapter.update_chat_name(chat_id, "🤖|待命")`，仅对群聊 session（chat_type != "p2p"）生效。

### 6. System prompt 指令

在 `build_session_context_prompt()` 中，当 session 是群聊时，追加指令：

```
When replying in a group chat, include a <group-name> tag in your FIRST response 
to summarize the task in ≤5 Chinese characters. Example:
<group-name>修复群聊功能</group-name>
Only include this tag once per conversation, in the first reply.
```

## 边界 Case

- **DM 场景**：不触发任何群名逻辑，`<group-name>` 标签即使出现也只做移除不调 API
- **Agent 忘记输出标签**：不影响功能，群名保持上次状态
- **标签出现在非首次回复**：正常处理，受限频器控制
- **群名包含 emoji**：正常传递，飞书支持 emoji 群名
- **Bot 没有改群名权限**：API 报错，warning 日志，不影响消息
- **并发更新**：限频器用 dict + 时间戳，单进程无需加锁

## 不做的事

- 不做 DM 场景的标题更新
- 不做心跳期间的群名动态更新（简单起见，只在 agent 回复时更新）
- 不做群名历史记录/回滚
- 不做轻量 LLM 旁路调用自动总结（依赖 agent 自身输出）

## 验收标准

1. 群聊中 agent 首次回复时，群名被更新为 agent 提炼的任务主题
2. `<group-name>` 标签从实际发送的消息中被移除，用户看不到
3. 5 分钟内同一群的多次更新只生效第一次
4. 发 `/new` 后群名恢复为 `🤖|待命`
5. DM 场景不受影响
6. API 失败不阻断消息发送
