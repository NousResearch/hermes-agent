# Hermes Skills 场景预设

三套预设通过 **禁用列表** 实现：只保留白名单 + `always_keep_*` 规则，其余 skill 写入 `~/.hermes/config.yaml` → `skills.disabled`。

## 用法

```bash
# 预览
python3 scripts/apply_skills_preset.py daily --dry-run

# 应用（默认全局；daily=飞书+Kanban+日常，推荐）
python3 scripts/apply_skills_preset.py daily
python3 scripts/apply_skills_preset.py dev
python3 scripts/apply_skills_preset.py finance
python3 scripts/apply_skills_preset.py science
```

飞书 / 网关生效：`/reload-skills` 或重启 `hermes-gateway.service`。

## 预设说明

| 预设 | 用途 | 约启用数 |
|------|------|----------|
| `daily` | **默认推荐**：飞书 lark-*、Kanban、股票、轻量检索 | ~55 |
| `dev` | 开发、Docker、DB、Tavily/Brave、Playwright | ~64 |
| `finance` | 投研、财报模型、Kanban 股票、飞书 | 见 dry-run |
| `science` | K-Dense 科学包 + 文献检索 | 见 dry-run |

`daily` 启用飞书 **lark-cli**（含 `lark-slides`）+ **`feishu-finance-kanban` / `feishu-finance-nexus`**（实时飞书 + 股票池 Base），**不**启用内置 `kanban-stock-nexus`。`dev` / `finance` 见各 yaml。
