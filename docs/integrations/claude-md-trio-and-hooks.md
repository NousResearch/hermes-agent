# CLAUDE.md / AGENTS.md / SOUL.md 三件套 + Hooks 体系

> **作者**：semperaug（基于 affaan-m/everything-claude-code 模式 + 本地踩坑经验）
> **状态**：RFC / 讨论稿
> **相关**：[everything-claude-code](https://github.com/affaan-m/everything-claude-code) 182K ⭐ 的六层架构

## 背景

Hermes Agent 目前用单件 `AGENTS.md`（注入到 subagent 的系统 prompt）做行为规则。
本文档提议**借鉴 ECC 的三件套模式** + **引入 hook 体系**，让项目级指令 / 行为规则 / 人格
分离，**同时让 hooks 解决"推送前 iLink 限流"** 这种**实有 consumer 的痛点**。

## 三件套拆分

| 文件 | 角色 | 何时加载 | 谁来写 |
|---|---|---|---|
| `CLAUDE.md` | 项目级 / 平台级（工具栈、目录、规约） | 每个 session 启动 | 项目维护者 |
| `AGENTS.md` | 行为规则（什么时候开 subagent、用什么工具、怎么读输出） | subagent spawn 时 | 平台作者 |
| `SOUL.md` | 人格 / 沟通风格 / 隐私红线 | 每个 session 启动 | 平台作者（已存在） |

**ECC 对应**：
- `CLAUDE.md` ≈ ECC 的 CLAUDE.md（82 行模板）
- `AGENTS.md` ≈ ECC 的 RULES.md（38 行）
- `SOUL.md` ≈ ECC 的 SOUL.md（17 行）

**hermes-agent 当前**：
- `AGENTS.md`（70 KB，超大，涵盖开发+行为）→ 需要拆
- `SOUL.md` 已有
- `CLAUDE.md` 缺失

## Hooks 体系

**问题**：iLink 微信通道在 24h 激活窗口过期 + 频率限制下会"深度限流"，
失败的尝试**也会增加计数器**，让情况更糟。
没有 hook 机制时，cron 推送任务会盲目重试，把通道锁死 1-2 小时。

**方案**：
- `scripts/hooks/pre-send/rate-limit-guard.sh` — 推送前自动检查平台失败次数
- 失败 ≥ 3 次（30 分钟内）→ 拒绝推送 + 告警
- 失败 < 3 次 → 允许推送 + 写 log
- 用环境变量可调（`HERMES_RATE_GUARD_LOG` / `HERMES_RATE_PATTERN` ...）

**对比 ECC**：
- ECC 用了 8+ 个 lifecycle hooks（PreToolUse / PostToolUse / PreCompact）
- 本 PR 只引入 1 个最关键的：rate-limit-guard（**有具体 consumer 痛点，不是推测性**）
- 其他 hooks（fact-force / continuous-learning / governance-capture）作为后续 RFC

## 实有 Consumer 痛点

| 痛点 | 现状 | Hook 解决方案 |
|---|---|---|
| iLink 微信推送 19 次连续失败锁死 | 没有保护，cron 盲目重试 | rate-limit-guard 拒绝推送 |
| 推送失败累积让 iLink 深度限流 1-2 小时 | 没有机制避免试探 | 守卫硬性卡掉试探 |
| 用户必须主动说"在吗"激活 24h 窗口 | 没有提示何时激活 | 守卫失败时提示 |

## 不做（避免重蹈 ECC 警告）

- ❌ **不引入推测性 hooks**（无具体 consumer = 拒收）—— 上游 CONTRIBUTING 明确反对
- ❌ **不创建通用 lifecycle hook framework**（具体 hook 一个一个加，按需）
- ❌ **不在 .env 加新 HERMES_* 变量**（除 secrets 外，行为配置走 config.yaml）

## 实施路径

1. **本 PR（最小）**：
   - 加 `scripts/hooks/pre-send/rate-limit-guard.sh`（**单 hook，单 consumer，单痛点**）
   - 加本文档 `docs/integrations/claude-md-trio-and-hooks.md`
   - **不动** 任何现有文件（最小变更原则）

2. **后续 RFC**（不在本 PR）：
   - CLAUDE.md / AGENTS.md / SOUL.md 三件套拆分
   - fact-force hook（写文件前事实强制）
   - continuous-learning hook（学习数据采集）

## 验证

- ✅ iLink 限流触发时 rate-limit-guard 拒绝推送（exit 1）
- ✅ 通道正常时允许推送（exit 0）
- ✅ 可通过 env 变量适配其他平台（Telegram / Discord / Slack / 飞书）
- ✅ 失败 log 写到 `~/.hermes/logs/rate-limit-guard.log`

## 相关链接

- [everything-claude-code](https://github.com/affaan-m/everything-claude-code) — 182K ⭐
- [hermes-agent CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) — 上游贡献原则
- 实际踩坑案例：2026-06-10 iLink 19 次连续失败记录（gateway.log）
