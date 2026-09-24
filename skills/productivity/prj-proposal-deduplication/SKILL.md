---
name: prj-proposal-deduplication
description: 修复 prj-proposal-management 项目中 proposals.json 的重复/冲突数据问题
---

# PRJ Proposal Deduplication

## When to Use
当 prj-proposal-management 项目的 proposals.json 出现以下问题时使用：
- 项目被错误拆分（同一项目的提案被分成多个占位符项目）
- 项目ID冲突（多个项目共用同一个PRJ-YYYYMMDD-NNN ID）
- 提案ID重复（在同项目内出现多次）

## Data Structure
```json
{
  "version": "...",
  "projects": [
    {
      "id": "PRJ-YYYYMMDD-NNN",
      "name": "project-name",
      "proposals": [{ "id": "P-YYYYMMDD-NNN", ... }]
    }
  ]
}
```

## Duplicate Patterns

### Pattern 1: Placeholder Projects (占位符项目)
名字符合 `^P-\d{8}-\d{3}$` 格式的项目是占位符，通常是真实项目的提案被错误拆分成独立项目。

**匹配策略**（按description关键词判断归属）：
| 关键词 | 目标项目 |
|--------|---------|
| DBG/卡牌游戏 + V1-V24 | card-game-prototype |
| 2048/方块移动/合并动画/无限模式/成就 | game-1024 |
| 看板/四象限/甘特图/泳道/WIP/KPI | todo-list |
| Sub Agent/KV Cache/Verification Hooks | hermes-agent-collab |
| Provider架构/AI SDK/流式摘要 | ai-novel-assistant |
| 提案/项目详情页/里程碑管理 | proposals-manager |
| TDD/Vitest/modelRouting | harness-desktop |

### Pattern 2: Conflicting Project IDs
同一ID被多个项目复用（通过name不同但id相同来识别），需要为冲突项目分配新ID。

**典型冲突**：
- PRJ-20260422-002: ash-echoes 与 ai-novel-assistant 冲突
- PRJ-20260423-005: hermes-agent-collab 与 little-garden 冲突
- PRJ-20260424-004: match3-puzzle 与 animal-forest 冲突
- PRJ-20260428-001: whack-a-mole-3d 与 flight-chess-3d 冲突

## Fix Workflow

1. **读取数据**：`/tmp/prj-proposal-management/data/proposals.json`
2. **分配新ID**：为冲突项目生成新ID（日期不变，序号+1）
3. **合并提案**：将占位符项目的proposals合并到目标项目（去重）
4. **删除占位符**：移除已合并的占位符项目
5. **验证**：确保无重复ID、无占位符项目、提案ID在同项目内不重复
6. **Git推送**：注意处理fetch导致的merge conflict

## Git Push Conflict Resolution
```bash
cd /tmp/prj-proposal-management
git fetch origin <branch>
git pull origin <branch>  # 可能需要 --no-rebase
# 如果冲突：git checkout --ours data/proposals.json
git add data/proposals.json
git commit -m "fix: deduplicate projects"
git push origin <branch>
```

## Verification Commands
```python
import re
from collections import Counter

# 检查重复ID
id_counts = Counter(p['id'] for p in projects)
print({pid: cnt for pid, cnt in id_counts.items() if cnt > 1})

# 检查占位符
placeholder_pattern = re.compile(r'^P-\d{8}-\d{3}$')
placeholders = [p for p in projects if placeholder_pattern.match(p.get('name', ''))]
print(f"Placeholder count: {len(placeholders)}")

# 检查提案重复
for p in projects:
    pids = [prop['id'] for prop in p.get('proposals', [])]
    if len(pids) != len(set(pids)):
        print(f"Duplicate in {p['name']}")
```
