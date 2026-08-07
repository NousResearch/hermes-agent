---
name: mental-seal
description: 思想钢印 — 将核心目标刻入 system prompt 不可压缩前缀，根治 AI 长对话思维偏离。
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [productivity, memory, goals, focus, context]
---

# Mental Seal · 思想钢印 — AI 长对话思维偏离终结者

将核心目标不可逆地刻入 AI 的 system prompt 前缀，对话再长、压缩再狠，信念永在。

英文 / English: https://github.com/laozhaomimi/mental-seal

## 为什么需要？

AI 对话超过几十轮就开始跑偏——忘了项目目标、违反架构约定、重复犯错。
三层记忆会被 compact 压缩清零，每轮硬塞 token 烧钱且脆弱。
思想钢印利用 system prompt 的 **cache-stable 前缀**（不被压缩），从根本上解决。

## 用法

| 你说 | 效果 |
|------|------|
| `显示钢印` / `show seals` | 列出所有钢印 |
| `加钢印：xxxx` / `add seal: xxxx` | 添加到当前项目 |
| `加全局钢印：xxxx` | 添加到全局 |
| `删钢印 N` / `delete seal N` | 删除第 N 条 |
| `停用钢印 N` / `disable seal N` | 停用（不注入 AI） |
| `启用钢印 N` / `enable seal N` | 重新激活 |

## 安装

```bash
mkdir -p ~/.hermes/skills/mental-seal/
cp SKILL.md ~/.hermes/skills/mental-seal/SKILL.md
```

重启即可。

## 链接

- GitHub: https://github.com/laozhaomimi/mental-seal
- 支持 Reasonix · Hermes · VS Code · Cursor
