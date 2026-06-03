# Hermes 架构升级 PR 提案

基于 Claude Code 和 Codex 底层逻辑分析，为 Hermes Agent 上游项目提出的改进。

## 概要

Hermes 已有良好的基础设施（tool_guardrails、fallback chain、retry counters），但缺少 Claude Code 的几项关键架构设计。

**2026-06-03 已本地实施**：
- ✅ SOUL.md: 7个架构模块注入
- ✅ 配置: hard_stop + abort_on_summary_failure
- ✅ 3个新Skill: compression-strategy / error-escalation / tool-pipeline
- ✅ `agent/circuit_breaker.py`: 独立断路器模块
- ✅ **深度融合1**: 断路器集成到 Agent Loop（4个注入点）

---

## PR 1: 启用默认断路器 hard_stop

**改动**：`config.yaml` 默认值

```yaml
tool_loop_guardrails:
  hard_stop_enabled: false  # → true
```

**理由**：
- 代码已完整实现 hard_stop 逻辑（`agent/tool_guardrails.py`），默认关闭
- Claude Code 的断路器是核心兜底机制，应默认开启
- 阈值已有合理的默认值（exact_failure:5, same_tool:8）

**影响**：极小。硬停止只作用于工具循环内的死循环场景，正常使用不受影响。

**相关文件**：
- `agent/tool_guardrails.py` (已有实现)
- `hermes_cli/config.py: DEFAULT_CONFIG` (改默认值)

---

## PR 2: 类型化终止/继续状态

**当前状态**：`run_conversation()` 返回 `{"final_response": "...", "messages": [...]}`

**目标**：返回类型化状态，参考 Claude Code 的 10 种终止 + 7 种继续

```python
# 终止状态
class TerminalReason(Enum):
    COMPLETED = "completed"
    ABORTED_STREAMING = "aborted_streaming"
    ABORTED_TOOLS = "aborted_tools"
    PROMPT_TOO_LONG = "prompt_too_long"
    MODEL_ERROR = "model_error"
    IMAGE_ERROR = "image_error"
    MAX_TURNS = "max_turns"
    HOOK_PREVENTED = "hook_prevented"
    HOOK_STOPPED = "hook_stopped"
    BLOCKING_LIMIT = "blocking_limit"

# run_conversation 返回值增强
def run_conversation(...) -> dict:
    return {
        "final_response": "...",
        "messages": [...],
        "terminal_reason": TerminalReason.COMPLETED,
        "terminal_metadata": {},  # extra diagnostics
    }
```

**理由**：
- 调用方（gateway、CLI、batch_runner）可根据终止原因做差异化处理
- 当前只能通过检查 `messages` 内容推断原因，不可靠

**改动文件**：
- `agent/conversation_loop.py` (主改动)
- 所有 `run_conversation()` 调用方

---

## PR 3: 输入依赖的工具安全判定

**当前状态**：`tool_guardrails.py` 有 `IDEMPOTENT_TOOL_NAMES` 和 `MUTATING_TOOL_NAMES` 的静态分类。

**问题**：`terminal("ls")` 和 `terminal("rm -rf /")` 被同等对待（都在 MUTATING_TOOL_NAMES 中）。

**方案**：参考 Claude Code 的 `isConcurrencySafe(input)` 模式

```python
def classify_tool_risk(tool_name: str, args: dict) -> RiskLevel:
    """Input-dependent safety classification."""
    if tool_name == "terminal":
        cmd = args.get("command", "")
        if _is_readonly_shell(cmd):
            return RiskLevel.SAFE
        if _is_destructive_shell(cmd):
            return RiskLevel.DESTRUCTIVE
        return RiskLevel.MEDIUM

    if tool_name in IDEMPOTENT_TOOL_NAMES:
        return RiskLevel.SAFE

    # ... per-tool input analysis
```

**改动文件**：
- 新增 `agent/tool_risk.py`
- 在 `model_tools.py` 的 `handle_function_call()` 中集成

---

## PR 4: 独立电路断路器模块（已实现）

**状态**：`agent/circuit_breaker.py` 已在本仓库实现，待 PR 上游。

**功能**：
- Per-feature 独立断路器（compression / delegation / fallback / etc.）
- 3种状态：CLOSED → OPEN → HALF_OPEN
- 自动和手动重置
- Preset 配置匹配 Claude Code 模式

**集成点**：
- `agent/conversation_loop.py`: 在 API 调用循环中集成
- `agent/context_compressor.py`: 压缩失败时触发
- `model_tools.py`: delegate_task 失败时触发

---

## PR 5: 上下文压缩分层策略

**当前状态**：Hermes 有单一压缩策略（`context_compressor.py`）

**目标**：实现 Claude Code 的 4 层递进压缩

```
Layer 1: 工具结果预算 (per-message size cap)
Layer 2: Snip Compact (物理删除最旧消息)
Layer 3: Microcompact (按 tool_use_id 清理)
Layer 4: Auto-Compact (子 agent 总结)
```

**方案**：
- 在 `agent/context_compressor.py` 中添加分层策略
- 每层压缩后检查token占用，够了就停
- 断路器：L4 失败3次 → 停止压缩，保留原始上下文

**改动文件**：
- `agent/context_compressor.py` (主改动)

---

## 优先级建议

| PR | 优先级 | 复杂度 | 收益 |
|----|--------|--------|------|
| PR 1: 启用 hard_stop | P0 | 极低（1行配置） | 高（兜底保护） |
| PR 4: 断路器模块 | P0 | 低（新增文件） | 高（系统性防护） |
| PR 3: 输入依赖安全 | P1 | 中（新增文件+集成） | 中（减少误报） |
| PR 2: 类型化状态 | P2 | 高（跨模块重构） | 中（调用方受益） |
| PR 5: 分层压缩 | P2 | 高（重构压缩器） | 中（渐进式优化） |

---

## 安全考量

所有改动均为**增强安全**：
- 断路器 = 防止死循环和资源浪费
- 输入依赖安全 = 减少危险操作
- 类型化状态 = 更好的错误处理

无破坏性变更（向后兼容）。

---

## 深度融合1: 断路器集成详情（已实施）

**注入点**：

| 位置 | 文件 | 行号 | 触发条件 |
|------|------|------|----------|
| 初始化 | `agent/agent_init.py` | ~410 | Agent 创建时初始化 `CircuitBreakerPanel` |
| 每轮开始 | `agent/conversation_loop.py` | ~454 | `advance_turn()` — 推进所有断路器状态 |
| API 响应无效 | `agent/conversation_loop.py` | ~1512 | 重试+fallback 耗尽 → `record_failure("api_retry")` |
| API 异常 | `agent/conversation_loop.py` | ~3336 | 模型错误重试耗尽 → `record_failure("model_error")` |
| 压缩失败 | `agent/conversation_loop.py` | ~2930, ~3010 | 压缩3次失败 → `record_failure("compression")` |

**断路器 Preset**：

```
compression:      3次 → 停止自动压缩
delegation:       3次 → 停止委托
provider_fallback: 3次 → 停止切换
same_solution:    2次 → 强制换方案
model_error:      3次 → 建议切换模型
no_progress:      5次 → 检查忙碌假象
api_retry:        3次 → 停止该轮API调用
```

**安全**：所有注入点使用 `hasattr(agent, '_circuit_breaker_panel')` 守卫，断路器模块不存在时完全透明（向后兼容）。
