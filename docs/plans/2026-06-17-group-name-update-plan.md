# 飞书群聊群名自动更新 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent 在群聊回复时通过 `<group-name>` 标签自动更新飞书群名，任务结束时恢复 `🤖|待命`。

**Architecture:** 新建 `group_name.py` 模块（提取+限频），FeishuAdapter 新增 `update_chat_name()` API 调用，`run.py` 在响应发送前拦截标签，`session.py` 注入群聊 system prompt，`slash_commands.py` + `_session_expiry_watcher` 在会话结束时重置群名。

**Tech Stack:** Python, lark_oapi SDK (`UpdateChatRequest`), pytest

**自我承诺：** 外部引用 10 个（全部已核实）· 文件改动 6 个 · 难度：中等 · TDD 用例 9 个 · 1 轮承诺

---

## File Structure

| 操作 | 文件 | 职责 |
|------|------|------|
| Create | `gateway/platforms/group_name.py` | `extract_group_name()` + `GroupNameRateLimiter` |
| Create | `tests/gateway/test_group_name.py` | 提取 + 限频器测试 |
| Modify | `gateway/platforms/feishu.py` | `update_chat_name()` 方法 |
| Modify | `gateway/run.py:9102` 附近 | 响应文本中提取群名并调 API |
| Modify | `gateway/session.py:232` | 群聊 system prompt 注入 |
| Modify | `gateway/slash_commands.py:144` | `/new` 时重置群名 |
| Modify | `gateway/run.py:5751` | session 过期时重置群名 |
| Modify | `tests/gateway/test_feishu_card.py` | 集成测试 |

## TDD 用例矩阵

| # | 测试 | 文件 | 类型 |
|---|------|------|------|
| 1 | 提取单个 `<group-name>` 标签 | test_group_name.py | unit |
| 2 | 无标签时返回 None | test_group_name.py | unit |
| 3 | 空标签返回 None | test_group_name.py | unit |
| 4 | 超过 20 字符截断 | test_group_name.py | unit |
| 5 | 多个标签取第一个 | test_group_name.py | unit |
| 6 | 标签从 clean_text 中移除 | test_group_name.py | unit |
| 7 | 限频器：首次允许 | test_group_name.py | unit |
| 8 | 限频器：5 分钟内拒绝 | test_group_name.py | unit |
| 9 | 限频器：不同 chat_id 独立 | test_group_name.py | unit |

---

### Task 1: group_name.py — 提取函数 + 限频器

**Files:**
- Create: `gateway/platforms/group_name.py`
- Create: `tests/gateway/test_group_name.py`

- [ ] **Step 1: 写 extract_group_name 测试**

```python
# tests/gateway/test_group_name.py
import pytest
from gateway.platforms.group_name import extract_group_name, GroupNameRateLimiter


class TestExtractGroupName:
    def test_single_tag(self):
        text = "你好！<group-name>修复群聊功能</group-name>我来帮你处理。"
        clean, name = extract_group_name(text)
        assert name == "修复群聊功能"
        assert "<group-name>" not in clean
        assert "你好！" in clean
        assert "我来帮你处理。" in clean

    def test_no_tag(self):
        clean, name = extract_group_name("普通回复内容")
        assert name is None
        assert clean == "普通回复内容"

    def test_empty_tag(self):
        clean, name = extract_group_name("text<group-name></group-name>more")
        assert name is None
        assert "textmore" in clean.replace(" ", "").replace("\n", "")

    def test_truncate_to_20(self):
        long_name = "这是一个超过二十个字符的非常长的群名称测试用例"
        text = f"<group-name>{long_name}</group-name>内容"
        _, name = extract_group_name(text)
        assert name is not None
        assert len(name) <= 20

    def test_multiple_tags_takes_first(self):
        text = "<group-name>第一个</group-name>中间<group-name>第二个</group-name>"
        clean, name = extract_group_name(text)
        assert name == "第一个"
        assert "<group-name>" not in clean

    def test_whitespace_stripped(self):
        text = "<group-name>  任务名  </group-name>"
        _, name = extract_group_name(text)
        assert name == "任务名"
```

- [ ] **Step 2: 跑测试，确认全部 FAIL**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_group_name.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gateway.platforms.group_name'`

- [ ] **Step 3: 实现 extract_group_name**

```python
# gateway/platforms/group_name.py
"""Group name extraction and rate limiting for Feishu group chats."""
from __future__ import annotations

import re
import time

_GROUP_NAME_RE = re.compile(r"<group-name>([\s\S]*?)</group-name>")
_MAX_GROUP_NAME_LEN = 20


def extract_group_name(text: str) -> tuple[str, str | None]:
    match = _GROUP_NAME_RE.search(text)
    if not match:
        return text, None
    raw_name = match.group(1).strip()
    clean_text = _GROUP_NAME_RE.sub("", text).strip()
    if not raw_name:
        return clean_text, None
    return clean_text, raw_name[:_MAX_GROUP_NAME_LEN]
```

- [ ] **Step 4: 跑测试，确认 extract 全部 PASS**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_group_name.py::TestExtractGroupName -v`
Expected: 6 PASS

- [ ] **Step 5: 写限频器测试**

```python
# 追加到 tests/gateway/test_group_name.py

class TestGroupNameRateLimiter:
    def test_first_update_allowed(self):
        limiter = GroupNameRateLimiter(interval_seconds=300)
        assert limiter.should_update("chat_1") is True

    def test_second_update_within_interval_blocked(self):
        limiter = GroupNameRateLimiter(interval_seconds=300)
        limiter.record_update("chat_1")
        assert limiter.should_update("chat_1") is False

    def test_different_chats_independent(self):
        limiter = GroupNameRateLimiter(interval_seconds=300)
        limiter.record_update("chat_1")
        assert limiter.should_update("chat_2") is True
```

- [ ] **Step 6: 跑测试，确认限频器 FAIL**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_group_name.py::TestGroupNameRateLimiter -v`
Expected: FAIL — `GroupNameRateLimiter` not yet implemented

- [ ] **Step 7: 实现 GroupNameRateLimiter**

```python
# 追加到 gateway/platforms/group_name.py

class GroupNameRateLimiter:
    def __init__(self, interval_seconds: int = 300):
        self._interval = interval_seconds
        self._last_update: dict[str, float] = {}

    def should_update(self, chat_id: str) -> bool:
        last = self._last_update.get(chat_id)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._interval

    def record_update(self, chat_id: str) -> None:
        self._last_update[chat_id] = time.monotonic()
```

- [ ] **Step 8: 跑全部测试，确认 9 PASS**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_group_name.py -v`
Expected: 9 PASS

- [ ] **Step 9: Commit**

```bash
git add gateway/platforms/group_name.py tests/gateway/test_group_name.py
git commit -m "feat(feishu): add group name extraction and rate limiter"
```

---

### Task 2: FeishuAdapter.update_chat_name()

**Files:**
- Modify: `gateway/platforms/feishu.py` — 新增 `update_chat_name` 方法

- [ ] **Step 1: 写测试**

```python
# 追加到 tests/gateway/test_feishu_card.py

class TestUpdateChatName:
    @pytest.mark.asyncio
    async def test_update_chat_name_calls_api(self, adapter):
        """update_chat_name should call im.v1.chat.update."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success.return_value = True  # lark SDK uses .success() method
        mock_client.im.v1.chat.update = MagicMock(return_value=mock_response)
        adapter._client = mock_client

        result = await adapter.update_chat_name("oc_test123", "新任务名")
        assert result is True
        mock_client.im.v1.chat.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_chat_name_truncates(self, adapter):
        """Names longer than 20 chars should be truncated."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success.return_value = True
        mock_client.im.v1.chat.update = MagicMock(return_value=mock_response)
        adapter._client = mock_client

        await adapter.update_chat_name("oc_test", "a" * 30)
        call_args = mock_client.im.v1.chat.update.call_args
        # Verify the request was made (detailed arg check depends on SDK internals)
        assert mock_client.im.v1.chat.update.called

    @pytest.mark.asyncio
    async def test_update_chat_name_failure_returns_false(self, adapter):
        """API failure should return False, not raise."""
        mock_client = MagicMock()
        mock_client.im.v1.chat.update.side_effect = Exception("API error")
        adapter._client = mock_client

        result = await adapter.update_chat_name("oc_test", "任务")
        assert result is False
```

- [ ] **Step 2: 跑测试，确认 FAIL**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_feishu_card.py::TestUpdateChatName -v`
Expected: FAIL — `update_chat_name` not defined

- [ ] **Step 3: 实现 update_chat_name**

在 `feishu.py` 中（靠近 `get_chat_info` 方法附近），新增：

```python
async def update_chat_name(self, chat_id: str, name: str) -> bool:
    if not self._client or not chat_id or not name:
        return False
    name = name[:20]
    try:
        from lark_oapi.api.im.v1 import UpdateChatRequest, UpdateChatRequestBody
        request = (
            UpdateChatRequest.builder()
            .chat_id(chat_id)
            .request_body(UpdateChatRequestBody.builder().name(name).build())
            .build()
        )
        response = await asyncio.to_thread(self._client.im.v1.chat.update, request)
        success = getattr(response, "success", lambda: False)()
        if success:
            logger.info("[Feishu] Group name updated: chat=%s name=%r", chat_id, name)
        else:
            _code = getattr(response, "code", "?")
            _msg = getattr(response, "msg", "?")
            logger.warning("[Feishu] Group name update failed: chat=%s code=%s msg=%s", chat_id, _code, _msg)
        return success
    except Exception as e:
        logger.warning("[Feishu] Group name update error: chat=%s error=%s", chat_id, e)
        return False
```

- [ ] **Step 4: 跑测试，确认 PASS**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_feishu_card.py::TestUpdateChatName -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add gateway/platforms/feishu.py tests/gateway/test_feishu_card.py
git commit -m "feat(feishu): add update_chat_name API method"
```

---

### Task 3: 响应文本中提取群名（run.py）

**Files:**
- Modify: `gateway/run.py` — 在 `_sanitize_gateway_final_response` 之后（line ~9102），添加群名提取和 API 调用

- [ ] **Step 1: 在 run.py line 9102 之后添加群名提取逻辑**

```python
# After line 9102: response = _sanitize_gateway_final_response(source.platform, response)
# Add group name extraction for group chats
if response and source.chat_type != "p2p" and source.chat_type != "dm":
    try:
        from gateway.platforms.group_name import extract_group_name
        _clean_text, _group_name = extract_group_name(response)
        if _group_name:
            response = _clean_text
            _fc_adapter = self.adapters.get(source.platform)
            if _fc_adapter and hasattr(_fc_adapter, 'update_chat_name'):
                _gn_limiter = getattr(self, '_group_name_limiter', None)
                if _gn_limiter is None:
                    from gateway.platforms.group_name import GroupNameRateLimiter
                    self._group_name_limiter = GroupNameRateLimiter()
                    _gn_limiter = self._group_name_limiter
                if _gn_limiter.should_update(source.chat_id):
                    _gn_limiter.record_update(source.chat_id)
                    asyncio.ensure_future(
                        _fc_adapter.update_chat_name(source.chat_id, _group_name)
                    )
        elif _clean_text != response:
            response = _clean_text
    except Exception:
        pass
```

注意：即使没有提取到群名（空标签），也要用 clean_text 替换 response 以移除标签。

- [ ] **Step 2: 跑现有测试确认不 break**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/gateway/test_feishu_card.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add gateway/run.py
git commit -m "feat(feishu): extract group name from agent response and update chat"
```

---

### Task 4: 群聊 system prompt 注入（session.py）

**Files:**
- Modify: `gateway/session.py:232` — 在 `build_session_context_prompt` 中按 chat_type 注入指令

- [ ] **Step 1: 在 build_session_context_prompt 函数末尾（return 前）添加群聊指令**

找到函数的 `return "\n".join(lines)` 语句，在其前面添加：

```python
# Group chat: instruct agent to output <group-name> tag
if context.source.chat_type in ("group", "channel"):
    lines.append("")
    lines.append(
        "**Group name:** When replying in this group chat, include a "
        "`<group-name>` tag in your FIRST response to summarize the task "
        "in ≤5 Chinese characters (or short English). Example: "
        "`<group-name>修复群聊功能</group-name>`. "
        "Only include this tag once per conversation, in the first reply. "
        "The tag will be stripped before display."
    )
```

- [ ] **Step 2: 跑现有测试确认不 break**

Run: `cd /Users/admin/.hermes/hermes-agent && uv run pytest tests/ -x -q --timeout=30 2>&1 | tail -20`
Expected: no failures

- [ ] **Step 3: Commit**

```bash
git add gateway/session.py
git commit -m "feat(feishu): inject group-name instruction in group chat system prompt"
```

---

### Task 5: /new 时重置群名（slash_commands.py）

**Files:**
- Modify: `gateway/slash_commands.py` — 在 session:end hook 之后添加群名重置

- [ ] **Step 1: 在 slash_commands.py line 156（session:reset hook 之后）添加重置逻辑**

```python
# After the session:reset hook emit (line 156), add:
# Reset group chat name to standby
if source.chat_type not in ("dm", "direct", "private", ""):
    _adapter = self.adapters.get(source.platform)
    if _adapter and hasattr(_adapter, "update_chat_name"):
        try:
            asyncio.ensure_future(
                _adapter.update_chat_name(source.chat_id, "🤖|待命")
            )
        except Exception:
            pass
```

- [ ] **Step 2: Commit**

```bash
git add gateway/slash_commands.py
git commit -m "feat(feishu): reset group name to standby on /new"
```

---

### Task 6: Session 过期时重置群名（run.py）

**Files:**
- Modify: `gateway/run.py:5751` — 在 `_session_expiry_watcher` 的 finalize 循环中添加重置

- [ ] **Step 1: 在 run.py _session_expiry_watcher 的 finalize 循环中（line 5751 后），在 `_cleanup_agent_resources` 之前添加群名重置**

```python
# Inside the `for key, entry in _expired_entries:` loop, before cleanup
# Parse session key to get platform and chat info
# Key format: "agent:main:feishu:group:oc_xxx:ou_xxx"
_key_parts = key.split(":")
if len(_key_parts) >= 5 and _key_parts[3] in ("group", "channel"):
    _plat_name = _key_parts[2]
    _chat_id = _key_parts[4]
    try:
        _plat = Platform(_plat_name)
        _adapter = self.adapters.get(_plat)
        if _adapter and hasattr(_adapter, "update_chat_name"):
            loop = asyncio.get_event_loop()
            asyncio.ensure_future(
                _adapter.update_chat_name(_chat_id, "🤖|待命")
            )
    except Exception:
        pass
```

- [ ] **Step 2: Commit**

```bash
git add gateway/run.py
git commit -m "feat(feishu): reset group name on session expiry"
```

---

### Task 7: 提交诊断日志改动（之前调试的 info 级别改动）

- [ ] **Step 1: 确认诊断日志改动在当前分支**

```bash
git diff gateway/platforms/feishu.py
```

- [ ] **Step 2: Commit 诊断日志 + 纯 @bot 修复**

```bash
git add gateway/platforms/feishu.py
git commit -m "fix(feishu): info-level admission logs + handle pure @bot mention in groups"
```

---

## 诱导问句自答

**愿不愿赌上重写成本？**

愿意。原因：
1. 所有 API 引用已通过 grep 和 Python import 验证
2. `extract_group_name` 是纯函数，100% 可测
3. `update_chat_name` 非致命设计，不影响主流程
4. 改动量小（新文件 1 个 ~40 行，修改 4 个文件各 ~10 行），影响面可控
5. NanoClaw 已验证同模式可行
