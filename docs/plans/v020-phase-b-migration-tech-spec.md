# v0.20.0 Phase B 本地可靠性补丁迁移规格

## 1. 需求边界

将本机保护分支 `phase-b-lite-merged` 中已验证的 Phase B Lite 与 Weixin B-2 可靠性补丁，逐项迁移到 Hermes v0.20.0 当前运行基线 `7eefb0931` 的隔离候选分支 `candidate/v020-phaseb-migration`。

本次只在候选 worktree `E:/AI-Tools/Hermes-v020-PhaseB-Candidate` 修改。禁止切换或直接修改运行中的 `C:/Users/ZYX/AppData/Local/hermes/hermes-agent`；禁止读取凭据、auth.json、Cookie、真实任务内容或生产 state.db；禁止重启 Gateway 或进行真实微信发送。

## 2. 迁移范围与非目标

### 迁移范围

1. Cron 可靠性预算：按 Job ID 的每日启动次数、连续失败、墙钟运行时长、Agent 迭代预算；默认关闭、严格预检、事务性 reservation 与执行账本一致。
2. 元数据型文件漂移告警：只保存相对路径、哈希、大小、时间与策略版本；拒绝路径逃逸、目录与链接；只告警、不恢复、不保存正文。
3. Weixin Cron `ret=-2` 持久化延迟恢复：稳定幂等 obligation、30→60→120…秒退避（最大 900 秒）、Gateway 每 15 秒到期扫描、原子 claim、最多 3 次，成功 delivered、耗尽 abandoned。
4. 相关只读 `hermes inspect` 命令和 Windows/上下文规则兼容修复，仅在与上述补丁的现有测试明确证明仍必要时迁移。

### 非目标

- 不迁移凭据、登录状态、OpenClaw 目录或任何生产数据。
- 不替换 Hermes 主模型、平台 adapter、Profile 隔离或网关配置。
- 不引入第二套 delivery ledger、Cron 数据库或第三方队列。
- 不自动启用任何生产 Job 的预算。
- 不部署、不推送、不创建 PR；这些属于通过验收后的独立授权步骤。

## 3. 设计规则

- 优先使用 v0.20.0 现有原生 API；逐文件比较旧补丁与当前上游结构，人工解决冲突，不使用整分支合并或盲目 cherry-pick。
- 每个功能切片执行 RED → GREEN：先导入旧分支对应的最小回归测试并确认在候选基线失败，再迁移最小实现使其通过。
- 测试强制使用临时 `HERMES_HOME`，候选解释器从主 checkout 复用但导入根必须指向候选 worktree。
- 所有外部发送使用 fake adapter；禁止真实发送。
- 任一切片出现 blocker/high、测试无法隔离、或未配置 Job 行为变化时立即停止迁移并报告。

## 4. 代码地图

| 模块 | 迁移责任 |
|---|---|
| `cron/reliability.py`、`cron/executions.py`、`cron/jobs.py`、`cron/scheduler.py`、`cron/scheduler_provider.py` | Cron 预算、预检、并发 reservation、执行结算 |
| `hermes_cli/reliability_guard.py`、`hermes_cli/inspect.py`、`hermes_cli/subcommands/inspect.py`、`hermes_cli/main.py` | 元数据漂移 guard 与 inspect CLI |
| `gateway/delivery_ledger.py`、`gateway/platforms/weixin.py`、`gateway/run.py`、`tools/cronjob_tools.py` | Weixin 限流 obligation、到期 sweep、退避与状态结算 |
| `tests/cron/*`、`tests/gateway/*`、`tests/hermes_cli/*` | 各切片 RED/GREEN、隔离 E2E 与并发回归 |

## 5. 任务状态

- T-201 · P0 · ✅ 完成 — 已确认三个历史提交均未包含于候选基线；Cron/Weixin 接口无硬阻塞，`cron/scheduler.py` 与 T-204 存在顺序依赖，使用主 checkout venv + 临时 HERMES_HOME 可运行候选测试。
- T-202 · P0 · 进行中 — 迁移并 RED→GREEN 验证 Cron 预算与执行账本切片：`cron/reliability.py`、`cron/executions.py`、`cron/jobs.py`、`cron/scheduler.py`、`cron/scheduler_provider.py`。
- T-203 · P1 · 待开始 — 迁移并 RED→GREEN 验证 metadata-only 漂移 guard 与 inspect CLI，范围为 `hermes_cli/reliability_guard.py`、`hermes_cli/inspect.py`、`hermes_cli/subcommands/inspect.py`、`hermes_cli/main.py`。
- T-204 · P0 · 待开始 — 迁移并 RED→GREEN 验证 Weixin Cron 延迟恢复切片，范围为 `gateway/delivery_ledger.py`、`gateway/platforms/weixin.py`、`gateway/run.py`、`tools/cronjob_tools.py`。
- T-205 · P0 · 待开始 — 对每个切片运行定向、组合、临时 HERMES_HOME E2E、`py_compile`、`git diff --check`；运行独立只读审查。
- T-206 · P1 · 待开始 — 生成候选变更清单、测试回执、回滚说明；不部署，等待用户确认。

## 6. 验收项

- A-201 · Cron 预算 · 待验证 — 未配置的 Job 仍按 v0.20.0 原行为执行；配置无效或预算耗尽时，不创建 execution、不启动 Agent，并有可操作错误。
- A-202 · Cron 并发 · 待验证 — 同一 Job 的最后每日名额在并发 scheduler/direct-run 路径中只能被一个事务 claim。
- A-203 · 连续失败 · 待验证 — 确定性 `failed/unknown` 达阈值自动 pause；completed 与明确暂态限流/传输失败不消耗连续失败预算。
- A-204 · 漂移 guard · 待验证 — 仅写 metadata baseline；路径逃逸、目录、链接或普通文件以外的输入被拒绝；检测绝不恢复或输出文件内容。
- A-205 · Weixin 限流 · 待验证 — Cron→Weixin 可识别 `ret=-2` 创建单一稳定 obligation，首延迟 30 秒；未到期不发送，到期并发 sweep 只 claim 一次；成功 delivered，三次耗尽 abandoned。
- A-206 · 隔离与回归 · 待验证 — 所有测试在临时 HERMES_HOME 和 fake adapter 下运行，无真实平台发送；定向/组合测试、`py_compile`、`git diff --check` 通过，独立审查无 blocker/high。

## 7. 回滚与交付

候选分支可以整体删除，不影响当前运行版本。生产部署前必须保留现有完整升级前备份 `C:/Users/ZYX/AppData/Local/hermes/backups/pre-update-2026-08-04-023716.zip`，并在迁移通过后建立 v0.20.0 部署前的独立回退包。
