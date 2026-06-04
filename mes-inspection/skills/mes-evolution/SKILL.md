---
name: mes-evolution
description: MES 巡检技能进化 — 使用 DSPy + GEPA 自动优化巡检 Skills。
version: 1.0.0
author: MES AI Inspection
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [mes, evolution, dspy, gepa, optimization]
    category: devops
---

# MES 技能进化

## Overview

使用 hermes-agent-self-evolution 的 DSPy + GEPA 优化器，自动进化 MES 巡检技能。

## How to Run

### 列出可进化技能
```
mes_evolution(action="list")
```

### 进化单个技能
```
mes_evolution(action="evolve", skill_name="mes-nginx-check")
```

### 批量进化
```
mes_evolution(action="evolve_all")
```

### 试运行
```
mes_evolution(action="evolve", skill_name="mes-nginx-check", dry_run=true)
```

## Quick Reference

| 参数 | 说明 | 默认值 |
|------|------|--------|
| action | list/evolve/evolve_all | list |
| skill_name | 技能名称 | — |
| dry_run | 仅验证 | false |

## Prerequisites

1. hermes-agent-self-evolution 已安装
2. 模型配置正确（config.yaml 的 evolution 段）
3. API key 已设置

## Troubleshooting

| 问题 | 解决方案 |
|------|---------|
| "未安装" | `cd vendor && git clone ... && pip install -e .` |
| "Skill not found" | 检查 HERMES_AGENT_REPO 环境变量 |
| "进化超时" | 减少 iterations |
| "API key not set" | `export QWEN_API_KEY=xxx` |
