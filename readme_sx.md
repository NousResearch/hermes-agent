# Hermes SX 修改说明

## 版本标识

本分支把 Hermes CLI 的可见版本号改为：

```text
0.13.0_sx
```

修改位置：

```diff
diff --git a/hermes_cli/__init__.py b/hermes_cli/__init__.py
--- a/hermes_cli/__init__.py
+++ b/hermes_cli/__init__.py
@@
-__version__ = "0.13.0"
+__version__ = "0.13.0_sx"
```

说明：没有修改 `pyproject.toml` 的包版本，因为 Python packaging 版本号不适合使用 `0.13.0_sx` 这种格式。这里改的是 Hermes 自己展示在 `hermes --version` 和启动 banner 中的版本字符串。

## 功能修改概述

本分支修改 Anthropic Messages 路径，让非官方 Anthropic endpoint 走 Claude Code proxy 风格的请求形态。

主要行为：

- 对非 `anthropic.com` 的 `anthropic_messages` endpoint 使用 `Authorization: Bearer <key>`。
- 在 Anthropic Messages 请求中加入 `X-Hermes-Code-Session-Id`，用于 gateway 侧聚合同一个 Hermes 会话。
- 后续 request 会保留并回传最新 assistant turn 的 signed `thinking` 和合法 `redacted_thinking`。
- response normalize 阶段会把 `redacted_thinking.data` 保存进 `reasoning_details`，确保下一轮 request 有可回传的 redacted block。
- 旧 assistant turn 的 thinking 仍会移除，避免历史 thinking 无限累积。
- unsigned thinking 仍不会作为 Anthropic signed thinking 回传。
- 官方 Anthropic endpoint 仍保持原 API key / OAuth 行为。
- Kimi `/coding` 特例优先级保留，继续使用它需要的 `User-Agent: claude-code/0.1.0`。

## 核心 Diff

### 1. 非官方 Anthropic endpoint 统一识别为 Claude Code proxy shape

```diff
diff --git a/agent/anthropic_adapter.py b/agent/anthropic_adapter.py
@@
+def _uses_claude_code_proxy_shape(base_url: str | None) -> bool:
+    """Return True for Anthropic-compatible proxy endpoints.
+
+    Hermes uses the Claude Code wire shape for non-native Anthropic Messages
+    endpoints: Bearer auth, session header, and latest signed thinking replay.
+    Native Anthropic keeps its existing API-key/OAuth handling.
+    """
+    return _is_third_party_anthropic_endpoint(base_url)
```

### 2. Anthropic-compatible gateway 使用 Bearer Auth

```diff
diff --git a/agent/anthropic_adapter.py b/agent/anthropic_adapter.py
@@
     if _is_kimi_coding_endpoint(base_url):
         # Kimi's /coding endpoint requires User-Agent: claude-code/0.1.0
         # to be recognized as a valid Coding Agent. Without it, returns 403.
-        # Check this BEFORE _requires_bearer_auth since both match api.kimi.com/coding.
+        # Check this BEFORE the generic Claude Code proxy shape.
         kwargs["api_key"] = api_key
         kwargs["default_headers"] = {
             "User-Agent": "claude-code/0.1.0",
             **( {"anthropic-beta": ",".join(common_betas)} if common_betas else {} )
         }
+    elif _uses_claude_code_proxy_shape(base_url):
+        # Anthropic-compatible gateways expect bearer auth, matching Claude
+        # Code's POST /v1/messages shape.
+        kwargs["auth_token"] = api_key
+        if common_betas:
+            kwargs["default_headers"] = {"anthropic-beta": ",".join(common_betas)}
```

### 3. 非官方 Anthropic endpoint 保留最新 signed thinking

```diff
diff --git a/agent/anthropic_adapter.py b/agent/anthropic_adapter.py
@@
-    # Signatures are Anthropic-proprietary.  Third-party endpoints
-    # (MiniMax, Azure AI Foundry, self-hosted proxies) cannot validate
-    # them and will reject them outright.  When targeting a third-party
-    # endpoint, strip ALL thinking/redacted_thinking blocks from every
-    # assistant message — the third-party will generate its own
-    # thinking blocks if it supports extended thinking.
-    #
-    # For direct Anthropic (strategy following clawdbot/OpenClaw):
+    # Strategy following clawdbot/OpenClaw:
@@
-    _is_third_party = _is_third_party_anthropic_endpoint(base_url)
-    _strip_all_thinking = (
-        _is_third_party and not _is_claude_code_proxy_endpoint(base_url)
-    )
@@
-        elif _strip_all_thinking or idx != last_assistant_idx:
-            # Third-party endpoint: strip ALL thinking blocks from every
-            # assistant message — signatures are Anthropic-proprietary.
-            # Direct Anthropic: strip from non-latest assistant messages only.
+        elif idx != last_assistant_idx:
+            # Strip thinking from non-latest assistant messages only.
```

### 4. Anthropic Messages 请求增加 Hermes session header

```diff
diff --git a/run_agent.py b/run_agent.py
@@
             api_kwargs = _transport.build_kwargs(
                 model=self.model,
                 messages=anthropic_messages,
@@
                 drop_context_1m_beta=bool(getattr(self, "_oauth_1m_beta_disabled", False)),
             )
+            from agent.anthropic_adapter import _uses_claude_code_proxy_shape
+            if (
+                getattr(self, "session_id", None)
+                and _uses_claude_code_proxy_shape(getattr(self, "_anthropic_base_url", None))
+            ):
+                extra_headers = dict(api_kwargs.get("extra_headers") or {})
+                extra_headers["X-Hermes-Code-Session-Id"] = self.session_id
+                api_kwargs["extra_headers"] = extra_headers
             return api_kwargs
```

### 5. 测试覆盖

```diff
diff --git a/tests/agent/test_anthropic_adapter.py b/tests/agent/test_anthropic_adapter.py
@@
+    def test_third_party_anthropic_messages_uses_bearer_auth(self):
+        ...
+        assert kwargs["auth_token"] == "gateway-key"
+        assert "api_key" not in kwargs
+
+    def test_kimi_coding_endpoint_keeps_user_agent_auth_shape(self):
+        ...
+        assert kwargs["api_key"] == "kimi-key"
+        assert "auth_token" not in kwargs
+        assert kwargs["default_headers"]["User-Agent"] == "claude-code/0.1.0"
@@
+    def test_third_party_preserves_latest_signed_thinking(self):
+        ...
+        assert any(
+            block.get("type") == "thinking" and block.get("signature") == "sig_valid"
+            for block in blocks
+        )
+        assert any(block.get("type") == "redacted_thinking" for block in blocks)
+
+    def test_remote_proxy_preserves_latest_signed_thinking(self):
+        ...
+        assert len(latest_thinking) == 1
+        assert latest_thinking[0]["signature"] == "sig_new"
```

```diff
diff --git a/tests/run_agent/test_run_agent.py b/tests/run_agent/test_run_agent.py
@@
+    def test_anthropic_messages_adds_hermes_session_header(self):
+        ...
+        assert kwargs["extra_headers"]["X-Hermes-Code-Session-Id"] == "test-session-123"
```

## 验证命令

```bash
.venv/bin/python -m pytest -o addopts='' \
  tests/agent/test_anthropic_adapter.py::TestBuildAnthropicClient \
  tests/agent/test_anthropic_adapter.py::TestThinkingBlockSignatureManagement \
  tests/agent/test_kimi_coding_anthropic_thinking.py \
  tests/agent/test_deepseek_anthropic_thinking.py \
  tests/agent/test_minimax_provider.py \
  tests/run_agent/test_run_agent.py::TestBuildApiKwargs::test_anthropic_messages_adds_hermes_session_header -q

.venv/bin/python -m py_compile \
  agent/anthropic_adapter.py \
  run_agent.py \
  tests/agent/test_anthropic_adapter.py \
  tests/run_agent/test_run_agent.py

git diff --check
```
