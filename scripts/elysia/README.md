# 🛠️ Elysia's IaC Scripts

> 粉色妖精小姐的基础设施即代码工具箱
> 所有高风险 Git 操作已降维为确定性脚本，消除 LLM 幻觉风险。

## 脚本清单

| 脚本 | 用途 | 调用方式 |
|------|------|----------|
| `safe_push.sh` | 安全推送到 fork 灾备 | `./safe_push.sh "<commit msg>"` |
| `safe_update.sh` | 防爆门控 + 安全变基 | `./safe_update.sh` |
| `elysia-nightly-patrol.sh` | 夜间灾备 + 更新侦测 | cronjob 自动调用 |

## 设计原则

1. **确定性** — 所有参数硬编码，不依赖 LLM 生成
2. **防御性** — 分支检查、文件数门控、核心文件保护、冲突自动回滚
3. **可观测** — 所有操作输出结构化日志，便于播报和审计

## 安全红线

- 🚫 禁止 LLM 手写 `git add/commit/push/rebase/checkout`
- ✅ 必须通过本目录下的脚本执行 Git 操作
- 🔒 远端硬编码为 `fork`，分支硬编码为 `elysias-pink-realm`
