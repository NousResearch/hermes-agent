---
name: prj-proposal-deduplication
description: 修复 prj-proposals-manager 项目中 proposals.json 的重复/冲突数据问题
---

# PRJ Proposal Deduplication

## When to Use
当 prj-proposals-manager 项目的 proposals.json 出现以下问题时使用：
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

### Pattern 0: Lowcase Project IDs (`p-YYYYMMDD-NNN`)
小写 `p-` 开头是占位符项目（不是标准 `PRJ-` 前缀）。同名提案应归属到对应项目后删除。

**案例**：
- `p-20260505-077` → 提案 `P-20260505-077` → 应归属 `PRJ-20260417-001` (prj-proposals-manager)
- `p-20260504-070` → 提案 `P-20260504-070` → 应归属 `PRJ-20260419-001` (temple-run)

### Pattern 1: Placeholder Projects (占位符项目)
名字符合 `^P-\\d{8}-\\d{3}$` 格式的项目是占位符，通常是真实项目的提案被错误拆分成独立项目。

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

### Pattern 3: Git Suffix in Project ID
项目ID带有 `.git` 后缀（如 `calculator-app.git`），是错误格式。需将提案移至正确项目后删除。

**案例**：`calculator-app.git`（提案 `P-20250416-003`）→ 应合并到 `PRJ-20260426-003`

### Pattern 4: Chinese Description as Project ID
用中文描述性文本作项目ID（如 `hermes-协作服务器消息转发机制重构`），是不规范的命名。需将提案移至正确项目后删除。

**判断方法**：项目ID非 `PRJ-YYYYMMDD-NNN` 格式，且为中文描述 → 很可能是占位符

### Pattern 5: Same Git Repo, Different Project IDs (Ghost Duplicates)
两个不同项目指向同一个 GitHub 仓库，但项目名和提案内容不同。这是"幽灵重复"——其中一个应该合并到另一个。

**判断方法**：
1. 两个项目的 `gitRepo` URL 相同
2. 但 `id` 和 `name` 不同，提案内容也不同

**处理原则**：
- 保留 ID 格式规范的项目（`PRJ-YYYYMMDD-NNN`），删除不规范的那个
- 提案先移后删：把要删除项目的提案移到目标项目，再删

**案例**：
- `PRJ-20260412-008` (ai-subscription-多端...) + `ai-subscription` (ai-subscription) → 合并到 `PRJ-20260412-008`，提案 `P-20260412-008` 从 `ai-subscription` 移入

### Pattern 6: Legitimate Standalone Projects (NOT duplicates)
以下情况不是重复，不要删除：
- `doc-editor` vs `tower-baby-guard` → 不同 git 仓库，是独立项目
- `preschool-puzzle`、`TradingAgents-CN`、`3d飞行棋-*` → 名称正常，属独立项目

## Fix Workflow

### Step 0: Verify against GitHub (NOT local file)
**本地文件可能落后于 GitHub**，必须先从 GitHub API 获取最新数据。
```bash
# 先确认本地是否落后
cd /home/hermes/workspace-dev/proposals/prj-proposals-manager
git fetch origin master
git log --oneline origin/master | head -5

# 如果 origin/master 有新提交，先 pull
git pull origin master
```

### Step 1: Analyze (check BEFORE making changes)
```python
import json
from collections import Counter

data = json.load(open('data/proposals.json'))
projects = data['projects']

# 1. 统计所有项目ID和名称
ids = [p['id'] for p in projects]
names = [p.get('name','') for p in projects]
print(f"Total: {len(projects)} projects")
print(f"Duplicate IDs: {[(i,c) for i,c in Counter(ids).items() if c>1]}")
print(f"Duplicate names: {[(n,c) for n,c in Counter(names).items() if c>1]}")

# 2. 检测 .git suffix 项目
git_suffix = [p['id'] for p in projects if p['id'].endswith('.git')]
print(f"Git suffix IDs: {git_suffix}")

# 3. 检测小写 p- 前缀
lowercase_p = [p['id'] for p in projects if p['id'].startswith('p-')]
print(f"Lowercase p- IDs: {lowercase_p}")

# 4. 检测非PRJ格式的占位符ID
import re
non_prj = [p['id'] for p in projects if not re.match(r'^PRJ-\d{8}-\d+$', p['id'])]
print(f"Non-PRJ-format IDs: {non_prj}")

# 5. 检测同一gitRepo但不同项目的"幽灵重复"
from collections import defaultdict
repo_to_projects = defaultdict(list)
for p in projects:
    repo = p.get('gitRepo','')
    if repo:
        repo_to_projects[repo].append(p['id'])
print(f"Shared gitRepos: {[(r, ps) for r, ps in repo_to_projects.items() if len(ps) > 1]}")
```

### Step 2: Identify targets (提案归属映射)
对每个问题项目，找出其提案应该归属到哪个目标项目。判断依据：
- 小写 `p-` 项目 → 从名称/描述找关联项目
- `.git` 后缀 → 从 gitRepo 找原项目
- 中文描述 → 从提案内容找关联项目
- 幽灵重复 → 从提案内容判断哪个是主项目

### Step 3: Move proposals FIRST, then delete
**关键顺序**：先移提案，后删项目。否则提案丢失。

```python
# 1. 收集要移动的提案
proposals_to_move = {}
for proj in projects:
    if proj['id'] in delete_ids:
        for prop in proj.get('proposals', []):
            proposals_to_move[prop['id']] = prop  # {proposal_id: proposal_obj}

# 2. 删除问题项目（但不删除提案，提案暂存到 dict）
projects = [p for p in projects if p['id'] not in delete_ids]

# 3. 把提案加到目标项目
for pid, prop in proposals_to_move.items():
    target_id = move_map[pid]  # 目标项目ID
    for proj in projects:
        if proj['id'] == target_id:
            existing_ids = [p['id'] for p in proj.get('proposals', [])]
            if pid not in existing_ids:
                proj.setdefault('proposals', []).append(prop)
            break
```

### Step 4: Verify
```python
from collections import Counter
ids = [p['id'] for p in projects]
names = [p.get('name','') for p in projects]
print(f"Final: {len(projects)} projects")
print(f"Duplicate IDs: {[(i,c) for i,c in Counter(ids).items() if c>1]}")
print(f"Duplicate names: {[(n,c) for n,c in Counter(names).items() if c>1]}")
```

### Step 5: Git Push
```bash
git add data/proposals.json
git commit -m "fix: 删除重复项目并归并提案"
git push origin master
# GitHub Actions 会自动部署 (~2min)
```

## 修复跨项目重复提案ID的典型案例

修复前（18个跨项目重复，来自12个原始ID）：
```
P-20260504-001 出现在 dont-step-white, ai-novel-assistant, flight-chess-3d
P-20260504-002 出现在 dont-step-white, ai-novel-assistant, flight-chess-3d, 3d飞行棋-v3-ai对手
P-20260502-005 出现在 flight-chess-3d, 3d飞行棋-玩家颜色选择-战绩统计
```

修复后（每个位置分配唯一ID，第一个保留原ID）：
```
dont-step-white: P-20260504-001, P-20260504-002, ...
ai-novel-assistant: P-20260505-001, P-20260505-001-2, ...（重新分配）
flight-chess-3d: P-20260504-001-2, P-20260504-002-2, ...
```

## Git Push Conflict Resolution
```bash
cd /tmp/prj-proposals-manager
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
from collections import Counter, defaultdict

# 检查重复项目ID
id_counts = Counter(p['id'] for p in projects)
print({pid: cnt for pid, cnt in id_counts.items() if cnt > 1})

# 检查占位符
placeholder_pattern = re.compile(r'^P-\d{8}-\d{3}$')
placeholders = [p for p in projects if placeholder_pattern.match(p.get('name', ''))]
print(f"Placeholder count: {len(placeholders)}")

# 检查跨项目重复提案ID
seen_proposal_ids = {}
for p in projects:
    for prop in p.get('proposals', []):
        pid = prop['id']
        if pid in seen_proposal_ids:
            print(f"CROSS-PROJECT DUPLICATE: {pid} in {seen_proposal_ids[pid]} and {p['name']}")
        else:
            seen_proposal_ids[pid] = p['name']

# 检查同项目内重复提案ID
for p in projects:
    pids = [prop['id'] for prop in p.get('proposals', [])]
    if len(pids) != len(set(pids)):
        print(f"WITHIN-PROJECT DUPLICATE in {p['name']}")
```
