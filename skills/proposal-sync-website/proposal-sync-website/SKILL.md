---
name: proposal-sync-website
description: 提案系统 CSV 数据管理 + 同步到 GitHub data/proposals.json 的完整流程。包含脚本使用、CSV 生成、Github API 坑点、cron job 配置。
---

# Proposal Sync Website

## 概述

**2026-05-05 变更：数据以 CSV 为结构化存储，markdown 仅作轻量索引。**

数据流向：
1. **GitHub → CSV**（常用）：`sync-proposals-to-website.py --csv-only` 从 GitHub 拉取 `data/proposals.json` 生成 CSV
2. **CSV → GitHub**（写入）：网站保存时通过 GitHub API 直接 PUT 到 `data/proposals.json`

目标仓库：`YeLuo45/prj-proposals-manager`（不是 proposals-manager）

---

## CSV 数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `proposals.csv` | `~/.hermes/proposals/` | 提案主数据（20字段，含 project_id 外键） |
| `projects.csv` | `~/.hermes/proposals/` | 项目主数据（id, name, proposal_count, git_repo） |
| `project_proposal_mapping.csv` | `~/.hermes/proposals/` | Project↔Proposal 映射 |

**proposals.csv 关键字段：** `id`, `title`, `status`, `project_id`, `project_name`, `git_repo`, `deployment_url`, `prd_confirmation`, `acceptance`, `last_update`

**从 CSV 统计提案数量：**
```bash
python3 -c "
import csv
with open('~/.hermes/proposals/proposals.csv') as f:
    rows = list(csv.DictReader(f))
from collections import Counter
print(Counter(r['status'] for r in rows))
print(f'Total: {len(rows)}')
"
```

---

## 同步脚本

### 脚本位置
```
~/.hermes/scripts/sync-proposals-to-website.py
```

### 关键配置
```python
REPO_OWNER = "YeLuo45"
REPO_NAME = "prj-proposals-manager"
DATA_FILE_PATH = "data/proposals.json"
BRANCH = "master"
```

### 执行方式

**生成 CSV（从 GitHub 拉取，最常用）：**
```bash
GITHUB_TOKEN=$GITHUB_TOKEN \
  python3 ~/.hermes/scripts/sync-proposals-to-website.py --csv-only
```

**同步到 GitHub（写入，不常用）：**
```bash
cd /home/hermes/.hermes/scripts
GITHUB_TOKEN=$GITHUB_TOKEN \
  python3 sync-proposals-to-website.py
```

---

## 重要 Bug：CSV git_repo 字段的读取层级

**GitHub v2 数据结构中 gitRepo 在项目层（project），不在提案层（proposal）。**

`generate_csv_files()` 函数中，`proposals.csv` 的 `git_repo` 字段必须从 `proj.get('gitRepo')` 读取，而非 `prop.get('gitRepo')`。后者永远为空（提案层无此字段），导致 `git_repo` 填充率极低。

**验证方法：** 检查 `proposals.csv` 中 `git_repo` 非空行数，应接近 175/186（仅 todolist 无仓库）。如果远低于此，说明 CSV 生成本身有问题，而非填充脚本问题。

```python
# ✅ 正确：从项目层读取
proj.get('gitRepo', '')

# ❌ 错误：从提案层读取（永远为空）
prop.get('gitRepo', '')
```

**如何验证 GitHub JSON 结构：**
```python
import json, base64, urllib.request

# GET proposals.json 并解码
r = subprocess.run(['curl', '-sL', '--tlsv1.2', '--max-time', '30',
    '-H', 'Authorization: token $GITHUB_TOKEN',
    '-o', '/tmp/gh_check.json',
    'https://api.github.com/repos/YeLuo45/prj-proposals-manager/contents/data/proposals.json'])
with open('/tmp/gh_check.json') as f:
    d = json.load(f)
p = json.loads(base64.b64decode(d['content']).decode('utf-8'))
# gitRepo 在 project 层，不在 proposal 层
print(p['projects'][0]['gitRepo'])  # 有值
print(p['projects'][0]['proposals'][0].get('gitRepo'))  # 通常为空
```

---

## GitHub API 获取 JSON 文件的正确方式

### 坑：base64 content 可能截断

直接用 curl 管道传给 `json.loads` 会失败：

```bash
# ❌ 错误：JSONDecodeError: Unterminated string
curl -sL "https://api.github.com/repos/.../contents/...?ref=master" | python3 -c "import sys,json; print(json.load(sys.stdin)['sha'])"

# ✅ 正确：先保存到文件，再解析
curl -sL --max-time 30 \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/YeLuo45/prj-proposals-manager/contents/data/proposals.json?ref=master" \
  -o /tmp/gh_proposals.json
```

### Python 解析（含截断修复）
```python
import json, base64

with open('/tmp/gh_proposals.json') as f:
    d = json.load(f)

content = d['content']
# base64 可能因特殊字符（中文/换行）导致截断，补充 padding
missing = (4 - len(content) % 4) % 4
content += '=' * missing

try:
    c = base64.b64decode(content).decode('utf-8')
    gh = json.loads(c)
except json.JSONDecodeError:
    # 降级：找最后一个完整 JSON 对象（括号深度追踪）
    depth = 0
    last_complete = 0
    for i, ch in enumerate(c):
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_complete = i + 1
    truncated = c[:last_complete]
    gh = json.loads(truncated)
```

---

## Cron Job 配置

### 创建命令（CSV 生成）
```python
cron(action='create',
     prompt='运行同步脚本，从 GitHub 拉取最新 proposals.json 并生成 CSV 文件到 ~/.hermes/proposals/。\n\n执行步骤：\n1. GITHUB_TOKEN=$GITHUB_TOKEN python3 ~/.hermes/scripts/sync-proposals-to-website.py --csv-only\n2. 检查输出确认 CSV 生成成功（32项目，186提案）\n3. 报告结果',
     schedule='0 9,18 * * *',
     name='提案CSV每日定时生成',
     repeat='forever',
     deliver='local')
```

---

## 静态网站数据读取（网站前端）

网站 `prj-proposals-manager` 前端通过 GitHub API 读取 `data/proposals.json`：

```
GET https://api.github.com/repos/YeLuo45/prj-proposals-manager/contents/data/proposals.json?ref=master
Headers: Authorization: Bearer $GITHUB_TOKEN
         Accept: application/vnd.github.v3+json
```

响应 body：
```json
{
  "content": "<base64 encoded content>",
  "sha": "<file sha>",
  "encoding": "base64"
}
```

解码后 parse JSON。注意同源策略：GitHub Pages 部署的静态网站需要 GitHub PAT 才能调用 API。

---

## 防止重复项目的机制

### 问题背景
曾因批量生成提案导致：
1. 同一项目多次写入造成重复
2. 版本碎片化：同一游戏拆成多个独立项目（game-1024 + game-1024-v2、pixelpal-v4~v17 等）

### 防护措施

#### 1. 版本碎片化合并 `consolidate_related_projects()`
自动检测并合并以下模式：
- **已知主版本映射**：`game-1024-v2` → `game-1024`、`snake-battle-v4` → `snake-battle` 等
- **pixelpal-v* 碎片**：所有 `pixelpal-v4` ~ `pixelpal-v17` → 合并到 `pixel-pal-web` (PRJ-20260420-002)
- **通用版本后缀**：检测 `name-vN` 模式，自动向基础版本合并

#### 2. 精确去重 `check_and_fix_duplicates()`
- **ID 去重**：同一 ID 出现多次时保留字典序最小的条目
- **名称去重**：名称相同但 ID 不同的条目，保留 ID 最小的

### 验证检查清单

同步后验证：
- [ ] GitHub API GET 返回 200 且 sha 变化
- [ ] JSON parse 成功，无截断错误
- [ ] proposals.json 中 `version: 2`，有 `projects` 数组
- [ ] CSV 生成成功：`projects.csv` 32行，`proposals.csv` 186行
- [ ] 无重复项目（按名称检查）
- [ ] 网站 https://yeluo45.github.io/prj-proposals-manager/ 刷新后显示正确

### 手动验证命令
```bash
# 检查项目数量和重复
GITHUB_TOKEN=$GITHUB_TOKEN \
  python3 ~/.hermes/scripts/sync-proposals-to-website.py --csv-only
```

---

## 相关文件路径

| 文件 | 路径 |
|------|------|
| 同步脚本 | `/home/hermes/.hermes/scripts/sync-proposals-to-website.py` |
| 提案 CSV | `~/.hermes/proposals/proposals.csv` |
| 项目 CSV | `~/.hermes/proposals/projects.csv` |
| 映射 CSV | `~/.hermes/proposals/project_proposal_mapping.csv` |
| 提案索引（轻量） | `~/.hermes/proposals/proposal-index.md` |
| 项目索引（轻量） | `~/.hermes/proposals/project-index.md` |
| hermes-agent repo | YeLuo45/hermes-agent (proposals 变更同步到 feature/hermesYYMMDD 分支) |

## 同步到 hermes-agent 流程

在同步到 `prj-proposals-manager` 的同时，也将 `~/.hermes/proposals` 变更推送到 hermes-agent：

```bash
# 1. 读取 GITHUB_TOKEN
GITHUB_TOKEN=$(cat ~/.hermes/tools/github-token.txt)

# 2. 在 hermes-agent 创建 feature 分支并推送
cd /home/hermes/.hermes
BRANCH_NAME="feature/hermes$(date +%y%m%d)"
git checkout -b ${BRANCH_NAME}
git add proposals/
git commit -m "sync: update proposals $(date +%Y-%m-%d)"
git push -u https://YeLuo45:${GITHUB_TOKEN}@github.com/YeLuo45/hermes-agent.git ${BRANCH_NAME}
```

**注意：** 使用 subprocess.run 绕过 terminal 阻塞，push 失败时用 GitHub REST API fallback。
| proposals.json 路径 | `data/proposals.json`（master 分支） |

> 注意：CSV 为结构化主数据，markdown 仅作轻量快速索引。同步流向：GitHub → CSV（--csv-only），写入流向：网站 → GitHub API。
