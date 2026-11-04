# Hermes Cron 退化输出（degenerate repetition）投递防护 —— 修复计划

> 日期：2026-11-02
> 上游仓库：`https://github.com/NousResearch/hermes-agent`
> 基线 commit：`70db671fac918ee62203f9ce16d5545f7fd3a760`（main, 2026-08-04）
> 触发事件：一个轮询型 watchdog cron job（job id、profile 名与群 ID 均已隐去）于 2026-11-02 15:45 与 19:49 两次运行产生退化重复输出，全量投递到一个分片型 IM 群。

---

## 1. 事件事实（已由证据确认，非推测）

### 1.1 垃圾消息来源

用户收到的垃圾文本与本地保存的 cron 输出**逐字一致**：

| 证据 | 路径 | 事实 |
|---|---|---|
| 投递源文件 | `<HERMES_HOME>/profiles/<profile>/cron/output/<job-id>/2026-11-02_19-49-56.md` | `## Response` 段落共 **213,027 字符**（正常轮次 5–6 KB） |
| 垃圾定位 | 同上，第 **17045** 行 | 与用户引用的 "NOW FIN. NOW FIN. … STILL FIN." **逐字匹配** |
| 垃圾 token 密度 | 同上 | `NOW FIN` / `THIS FIN` / `GO NOW` / `EMIT NOW` 合计 **2177 次** |
| 退化证据 | 同上，第 16732 行起 | `DONE. THIS IS THE FINAL. EMIT NOW. GO. FINAL EMIT NOW. … NOW.` 数百次重复 |
| 模型自述 | 同上，第 16730 行后 | `Okay, I'm sorry for the internal noise. Let me just do it.` |
| 第二次同类事件 | `2026-11-02_15-45-48.md` | 220,151 字符；`Executing the patrol` 重复 537 次（退化但未达到 FIN 垃圾形态） |

### 1.2 投递路径（为什么 213KB 能整条进飞书）

`cron/scheduler.py::_deliver_result()` → `gateway/delivery.py::deliver()`：

- `gateway/delivery.py:488` — `if len(content) > MAX_PLATFORM_OUTPUT:` → 落盘审计副本。
- `gateway/delivery.py:503` — `if getattr(adapter, "splits_long_messages", False):` → **走"完整投递"分支，不截断**。
- `plugins/platforms/feishu/adapter.py:1468` — `splits_long_messages = True`（send() 原生按 `MAX_MESSAGE_LENGTH` 分片）。

**结论：飞书是分片型 adapter，退化输出被完整保留，并由 adapter 切成多条消息逐条发出。**
`gateway.log` 中的 `Cron output preserved for chunking adapter (N chars)` 日志印证了这条路径（如 `291010 chars`、`217290 chars`）。

### 1.3 触发条件

| 因素 | 事实 | 位置 |
|---|---|---|
| 模型路由不健康 | `route_health_probe.py` 实测：`auto_free` **FLAKY 1/2**；`auto`/`medium`/`high` DEAD；`low:free` HEALTHY 2/2 | `skills/devops/kanban-watchdog-recovery/scripts/route_health_probe.py` |
| 迭代预算过大 | 全局 `agent.max_turns: 90`，cron 读取 `_cfg.get("agent",{}).get("max_turns")` | `config.yaml:44`、`cron/scheduler.py:3276` |
| 单 job 无上限 | cron job 记录里 `max_turns: null`、`max_tokens: null`，且**运行期根本不读取 job 级字段**（只有全局 `_cfg`） | `cron/jobs.json`（job `<job-id>`）、`cron/scheduler.py:3276` |
| 无输出体量上限 | 退化时单条 assistant 消息可达数 KB，累积到 213KB，无任何截断/熔断 | `cron/scheduler.py`（`final_response` 直接进 `_deliver_result`） |
| 无退化检测 | 全仓检索 `repetition|degenerate|loop_detect|repetitive` 在 cron/gateway 主链路上**无任何命中** | `cron/scheduler.py`、`gateway/delivery.py` |

### 1.4 影响范围

- 在该机器的多 profile 环境中，仅该 profile 受影响：8 个 profile 的 cron output 目录中只有它命中垃圾模式（2 个文件）。
- 轮询式定时任务无人值守，垃圾消息会持续复现 → 已紧急 `pause` 止血（job `<job-id>`，`paused_at=2026-11-02T20:54:30`）。
- 该 job 本身带有静默契约：无进行中任务时不发送通知。

---

## 2. 修复目标

在 Hermes 核心加两道**通用**防线（不是针对本事件的特判），使任一 profile 的任一 cron job 在模型退化时都不会把垃圾内容投递给用户：

- **A. 退化检测 + 熔断（源头）**：识别 assistant 最终响应中的退化重复特征，命中则截断为可读摘要并抑制投递。
- **B. 投递体量上限（兜底）**：任何 cron 输出超过阈值即截断，避免单次几百 KB 灌进聊天。

同时补上**job 级配置缺口**：让 `max_turns` / `max_tokens` 真正可按 job 生效（当前只有全局值，且 job 字段被存为 `null` 后完全无人读取）。

---

## 3. 实现方案

### 3.1 新增：退化重复检测模块

**新建文件：`gateway/degenerate_output.py`**（放在 gateway 层，因为 webhook lane 也需要复用；与 `gateway/response_filters.py` 并列）

```python
"""Detect degenerate repetition in model output.

A model that has lost the plot often emits the same short phrase over and
over ("NOW FIN. NOW FIN. ...") instead of a report. Such output must never
reach a chat channel verbatim.
"""

MAX_SEGMENT_REPEATS = 8     # same segment repeated this many times
MIN_SEGMENT_LEN = 8         # chars; below this, coincidence is plausible
MAX_RESPONSE_CHARS = 20_000 # hard cap for a single cron delivery
REPEAT_WINDOW = 600         # chars window scanned for near-duplicate segments


def is_degenerate(text: str) -> bool:
    """True when `text` shows runaway repetition of a short phrase."""
    ...

def collapse_degenerate(text: str) -> str:
    """Return a readable prefix with the repetitive tail collapsed."""
    ...

def truncate_for_delivery(text: str, limit: int = MAX_RESPONSE_CHARS) -> str:
    """Cap delivery size, keeping a pointer to the saved audit copy."""
    ...
```

**检测算法要求（必须真实检测，不能只做长度截断）：**

1. **分段重复**：把文本切成 token 序列，滑窗寻找长度 ≥ `MIN_SEGMENT_LEN` 的连续片段；若同一片段在 `REPEAT_WINDOW` 字符窗口内出现 ≥ `MAX_SEGMENT_REPEATS` 次 → 判定退化。
2. **n-gram 退化率**：计算 4-gram 的重复占比；占比超阈值（例如 0.35）同样判定退化 —— 覆盖"NOW FIN. NOW FIN."这种逐段推进、片段本身不完全相同的变体。
3. **尾部熔断**：命中后，只保留**首个退化点之前**的正文（通常是真实内容），并追加明确标记：
   `[输出因模型退化重复已截断，前 N 字符有效内容见附件]`。

**不得**用"看起来像垃圾的黑名单词"做判定 —— 必须基于统计特征，否则会误伤正常的中文/英文报告。

### 3.2 接入点 1：cron 投递前熔断（主防线）

**文件：`cron/scheduler.py`**，在 `run_one_job()` 的投递分支（约 4003–4025 行）：

当前：
```python
deliver_content = final_response if success else _summarize_cron_failure_for_delivery(job, error)
should_deliver = bool(deliver_content.strip())
```

改为：**在 `should_deliver` 判定之前**插入退化检测：

```python
if success and deliver_content.strip():
    if is_degenerate(deliver_content):
        # 退化输出：绝不整条投递
        logger.warning("Job '%s': degenerate model output detected (%d chars) — collapsing",
                       job["id"], len(deliver_content))
        deliver_content = collapse_degenerate(deliver_content)
        # 退化后若没有有效正文 → 按空响应处理，不投递
```

要求：
- 退化检测必须在**落盘审计副本之后**执行（`save_job_output` 已在 3982 行先执行），保证原始证据不丢。
- 退化且截断后正文为空 → `should_deliver = False`，并让 `last_status` 反映异常（走 4039 行已有的空响应软失败分支），**不能**报 `ok`。
- 正常的、含重复但有实际信息的长报告**不得**被误杀 —— 必须有回归测试证明。

### 3.3 接入点 2：投递层体量上限（兜底）

**文件：`gateway/delivery.py`**，在 `deliver()` 的 `MAX_PLATFORM_OUTPUT` 分支（约 488–523 行）**之前**插入：

```python
from gateway.degenerate_output import truncate_for_delivery
content = truncate_for_delivery(content)   # 独立于 adapter 分片行为
```

要求：
- 该上限**优先于** `splits_long_messages` 分支 —— 分片 adapter 目前的"完整投递"正是本次 213KB 全量灌入飞书的直接原因。
- 截断时必须保留"完整输出已保存到 `<path>`"的指针（沿用现有 `_save_full_output` 审计副本机制）。
- 上限应可通过 `config.yaml` 调整，例如 `cron.max_delivery_chars`。

### 3.4 接入点 3：job 级 `max_turns` / `max_tokens` 生效（配置缺口）

**文件：`cron/scheduler.py:3276`**

当前：
```python
max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 500
```

改为**优先读取 job 级覆盖**：
```python
max_iterations = (
    job.get("max_turns")
    or _cfg.get("agent", {}).get("max_turns")
    or _cfg.get("max_turns")
    or 500
)
```

同理，`max_tokens` 需在创建 cron agent 时透传到 agent（`agent/agent_init.py:2079` 已有 `model.max_tokens` 的全局支持，需确认 cron agent 走同一路径；若 cron agent 绕过了该逻辑，需在 `cron/scheduler.py` 创建 agent 时显式传入 `max_tokens`）。

**同时**：`cron/jobs.py` 的 job 规范化 / `tools/cronjob_tools.py` 的 create/update 参数需要接受并持久化 `max_turns` / `max_tokens`（当前 job 记录里这两个字段恒为 `null`，说明 schema 侧已预留但未接通）。

### 3.5 可观测性

退化命中时 `logger.warning` 必须包含：job id、原始字符数、判定依据（重复片段样例或 n-gram 比例）、截断后字符数。便于事后审计，且不泄漏用户内容到日志（只记片段长度与 hash 前缀即可）。

---

## 4. 验收条件（机器可验证）

| # | 验收项 | 验证方式 |
|---|---|---|
| 1 | 退化检测能识别真实样本 | 用本次真实垃圾文本（`2026-11-02_19-49-56.md` 第 17045 行起）作 fixture → `is_degenerate()` 必须返回 `True` |
| 2 | 正常报告不误杀 | 用本次**正常**轮次的 `## Response`（如 `2026-11-02_20-44-49.md`、`2026-11-02_20-27-48.md`）作 fixture → 必须返回 `False` |
| 3 | 边界样本 | 含重复但有信息量的真实报告（如重复列举 8 个 board 名）→ `False`（若实现判定为 `True`，必须调阈值并说明理由） |
| 4 | 截断保留有效正文 | `collapse_degenerate()` 保留退化点之前的正文，并追加截断标记 |
| 5 | 投递上限生效 | 构造 100 KB 输出 → 投递内容 ≤ `max_delivery_chars`（默认 20000），且含审计副本路径指针 |
| 6 | 投递上限**优先于**分片 | 用 `splits_long_messages=True` 的 adapter（feishu）走一遍，确认不再全量分片灌入 |
| 7 | 退化命中时不投递 | 集成测试：退化响应 → `should_deliver` 为 False（或投递内容为截断摘要），`last_status` 不为 `ok` |
| 8 | job 级 `max_turns` 生效 | 创建带 `max_turns: 5` 的 job → 运行时实际迭代上限为 5；不带则回落到全局 90 |
| 9 | job 级 `max_tokens` 生效 | 创建带 `max_tokens: 2000` 的 job → 请求实际带上该上限 |
| 10 | 审计副本不丢 | 退化场景下 `save_job_output` 仍写出完整原始响应（便于取证） |
| 11 | 现有测试不回归 | `pytest tests/cron/ tests/test_gateway_delivery*.py`（按实际路径调整）全绿；新增测试覆盖上述 1–10 |
| 12 | 配置可调 | `config.yaml` 的 `cron.max_delivery_chars` 生效，且有默认值兜底 |

---

## 5. 范围外（明确不做）

- 不针对 "NOW FIN" 这类具体字符串做黑名单（会误伤，且不是根因）。
- 不改动模型路由策略本身（`auto_free` 路由健康度属于 LiteLLM 侧问题，单独处理）。
- 不重写 cron 调度架构，不引入 CrewAI。
- 不修改任何业务仓库源码（本次只动 Hermes 核心）。

---

## 6. 并行可行的额外缓解（非本 PR，可独立先行）

在核心修复落地前，可用配置层手段降低复发概率：

1. 把 watchdog job 的模型从 `auto_free`（实测 FLAKY 1/2）切到实测健康的 `low:free`（2/2 HEALTHY）。
2. 把全局 `agent.max_turns` 从 90 下调（本 profile 的 watchdog 场景不需要 90 轮）。
3. 保持 watchdog 的 `[SILENT]` 静默契约，避免空闲时发送通知。

这些属于**运行配置**，不进入代码 PR；但必须在 PR 合并后**重新验证**一次真实轮次，确认不再产生退化输出。
