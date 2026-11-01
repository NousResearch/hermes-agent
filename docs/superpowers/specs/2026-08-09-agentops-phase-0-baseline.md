# AgentOps Phase 0 基线盘点

| 字段 | 内容 |
|---|---|
| 日期 | 2026-08-09 |
| 盘点方式 | 只读：Git、launchd plist、进程表、Hermes profile/cron CLI、配置路径与脚本静态检查 |
| 实施分支 | `codex/agentops-phase-1-foundation` |
| 控制权限 | 本文不授予任何 Target 写权限；所有 AgentOps Target 初始为 `observe_only` |

## 1. 受保护资产与部署基线

实时工作目录为 `/Users/molly/Desktop/Hermes`，当前 SHA 为 `39e8b2b2bb4ca3e196dbc020bee1327f603bc8d2`，相对 `origin/main` 为 ahead 2、behind 5656。该目录存在用户未提交的 Python、Gateway、TUI、测试、脚本、应用和业务资料修改；这些文件均为受保护资产，AgentOps 不得 reset、clean、checkout、覆盖、提交或作为自动部署输入。

Phase 0/1 使用独立 worktree `/Users/molly/Desktop/Hermes-agentops-phase1`，同一 base SHA，分支 `codex/agentops-phase-1-foundation`。它不安装服务、不改写 `~/.hermes`、不改动运行中 Gateway 或现有 LaunchAgent。

## 2. 首批 Target 清单

| Target ID | Profile | LaunchAgent | 日志目录 | 初始 criticality | authority_mode | 当前写控制者 |
|---|---|---|---|---|---|---|
| `hermes:profile:default:gateway` | default | `ai.hermes.gateway` | `~/.hermes/logs/` | critical | observe_only | launchd + `hermes_gateway_watchdog.py` |
| `hermes:profile:feishu3:gateway` | feishu3 | `ai.hermes.gateway-feishu3` | `~/.hermes/profiles/feishu3/logs/` | noncritical (待 Owner 确认) | observe_only | launchd + `hermes_gateway_watchdog.py` |
| `hermes:profile:feishu4:gateway` | feishu4 | `ai.hermes.gateway-feishu4` | `~/.hermes/profiles/feishu4/logs/` | noncritical (待 Owner 确认) | observe_only | launchd + `hermes_gateway_watchdog.py` |
| `hermes:profile:feishu5:gateway` | feishu5 | `ai.hermes.gateway-feishu5` | `~/.hermes/profiles/feishu5/logs/` | noncritical (待 Owner 确认) | observe_only | launchd + `hermes_gateway_watchdog.py` |
| `hermes:profile:newbot:gateway` | newbot | `ai.hermes.gateway-newbot` | `~/.hermes/profiles/newbot/logs/` | noncritical (待 Owner 确认) | observe_only | launchd + `hermes_gateway_watchdog.py` |
| `hermes:cron:default` | default | n/a | `~/.hermes/cron/` | high | observe_only | Hermes Cron scheduler + individual scripts |
| `hermes:repo:desktop-hermes` | n/a | n/a | `/Users/molly/Desktop/Hermes` | critical | observe_only | Molly / Git only |
| `hermes-ai-native:gateway` | separate runtime | `com.molly.hermes-ai-native.gateway` | `~/.hermes-ai-native/` | out_of_scope_read_only | observe_only | its own launchd service |
| `hermes-ai-native:agentops-cron` | agentops | `com.molly.hermes-ai-native.agentops-cron` | `~/.hermes-ai-native/profiles/agentops/` | out_of_scope_read_only | observe_only | native Cron launchd service |
| `openclaw:gateway` | separate runtime | external OpenClaw launchd/process | OpenClaw-owned logs | out_of_scope_read_only | observe_only | `openclaw_watchdog.py` + OpenClaw |

`default` 标为 critical。其余四个 Profile 在 Phase 0 仅暂定为 noncritical，不能据此开启 R3；Owner 必须在 Phase 6 前确认实际业务关键性。

## 3. 运行与控制器职责矩阵

| 控制器 | 频率 / 生命周期 | 读取范围 | 当前写/重启动作 | 与 AgentOps 的关系 |
|---|---|---|---|---|
| 5 个 Hermes Gateway LaunchAgent | 常驻，`--replace` | 自身配置和 Gateway 生命周期 | 启动/重启所属 Gateway | 当前唯一 Gateway 生命周期控制者；Phase 1 不接管 |
| `hermes_gateway_watchdog.py` | launchd 每 15 分钟 | 5 个 Gateway 的 launchd 与日志 | 对故障 Label 执行 `launchctl kickstart -k`，带 cooldown | 已有写控制者；AgentOps 只能观察，迁移前不得并行写 |
| Hermes Cron：AIVault watchdog | 360 分钟 | AIVault MCP 状态 | 写 watchdog 自身状态；不重启 Gateway | 已知“退出 0 但业务失败”样本；Phase 2 作为观测对象 |
| Hermes Cron：OpenClaw watchdog | 5 分钟 | OpenClaw health | 可执行 OpenClaw gateway restart | out of scope，Phase 1 不接管 |
| Hermes Cron：MMD / WeChat inbox watcher | 30 / 5 分钟 | 各自业务状态 | 启动子进程 | out of scope，Phase 1 不接管 |
| `hermes-ai-native` agentops Cron | 每日 09:00 | native agentops profile | 执行 Agent 任务，非 Hermes 自愈控制面 | 只读盘点；禁止与新控制面双写 |

结论：截至本盘点时，**没有任何 Target 的写权限转移给新 AgentOps**。Phase 1 中新组件没有 Executor、Shell、launchd 或网络写接口；其唯一可写位置仅为显式传入的 AgentOps 自身状态目录和测试临时目录。

## 4. 关键 Cron 与已知“假绿”风险

当前 active 的默认 Profile Cron 包括 AIVault MCP watchdog、OpenClaw watchdog、MMD WeChat watcher、WeChat task inbox assistant、weekly data reminder。另有两个 paused 内容任务。`~/.hermes-ai-native/profiles/agentops/cron/jobs.json` 还登记一条每日 Agent 安全巡检。

默认 Cron 输出中，AIVault watchdog 的最后状态为 `ok`；历史日志已显示其 MCP 故障可被脚本吞掉后仍表现为成功。这是 Phase 2 的第一条业务断言需求：未来健康模型必须将“任务退出码”与“被测能力断言”分开，不能因脚本退出 0 自动判为健康。

## 5. 历史问题到长期资产的映射

| 已确认问题 | 自动化资产 | 当前策略 |
|---|---|---|
| Codex Responses `NoneType` 流输出 | Provider 回归测试 + model-and-tools probe | Phase 7；不自动修复 |
| Tool schema 空 `required` | 工具契约回归测试 | Phase 7；不自动修复 |
| 凭证池未传递 / reasoning 路由 | Provider/会话回归测试 | Phase 7；不自动修复 |
| Gateway 重复启动 | 进程/launchd单实例 Probe | Phase 2；R1 Runbook 仅 G4 后审阅 |
| 飞书断连、MCP 失败 | 连接 Probe + Incident fingerprint | Phase 2/3；无自动写 |
| AIVault false green | Cron business assertion 模型 + fixture | Phase 2；无自动写 |
| 压缩错误、辅助警告、后台日志噪声 | experience 质量回归样本 | Phase 7；不自动修复 |
| Book Workflow 身份、入口、R1/R2/R3、写回隔离问题 | book-workflow Review Pack + golden fixtures | Phase 7；R4 manual-only |
| dirty / 落后部署树 | Git baseline/dirty Probe | Phase 2；阻止 R3，不自动整理工作树 |

## 6. Phase 0 结论与 G0 状态

- 四份规格的独立控制面、SQLite + UDS、默认 `observe_only`、R0-R4 分层和“禁止 dirty worktree 自动改写”要求一致。
- 首批 5 个 Hermes Profile、对应 launchd Label、日志目录、仓库和关键 Cron 已登记。
- 当前写控制器已被识别：launchd 和既有 watchdog/Cron 保持其现有职责；新 AgentOps 没有写授权。
- 当前用户修改已记录为受保护资产。

**G0: implementation-authorized by active task objective; production/R1-R4 authorization remains absent.**
