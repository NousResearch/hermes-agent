---
name: knowledge-base-restructuring
description: "Restructure a messy knowledge library into an expert system organized by business lifecycle. Covers deduplication, merging, reorganization, and maintenance governance."
tags: [knowledge-management, restructuring, deduplication, note-taking, content-strategy]
triggers:
  - user says knowledge base is messy, needs cleanup, needs restructuring
  - multiple files covering same topic with overlapping content
  - converting from standard/device/tech classification to business-flow organization
---

# Knowledge Base Restructuring

Transform a "杂货铺" (junk shop) knowledge library into an expert system organized by project lifecycle.

## Core Principle

**旧：标准 → 设备 → 技术（字典式）**  
**新：规划 → 设计 → 实施 → 运维 → 进阶（项目驱动式）**

Readers should follow a project story, not browse a dictionary.

---

## ⚠️ Pitfalls (Learned the Hard Way)

1. **合并 ≠ 删除** — When merging files, ALWAYS backup originals to a `备份_原文件/` directory first. Never delete source files without backup. User explicitly corrected this.

2. **Batch before reporting** — When executing multiple sequential tasks (e.g., merging 7 topics), complete ALL tasks first, then give ONE summary report. Do not report after each step. User explicitly asked for this.

3. **Use symlinks for reorganization** — When creating a new directory structure, use `ln -sf` to create symbolic links to original files rather than moving files. Preserves both views.

4. **execute_code for batch file ops** — Use `execute_code` with `hermes_tools` (read_file, write_file, terminal) for batch processing multiple files. More efficient than individual tool calls.

---

## 4-Step Methodology

### Step 1: 做减法 (Deduplication)

1. Scan all files: `find /path/ -name "*.md" -type f`
2. Group by topic (grep for keywords: standard numbers, product names)
3. Identify overlapping content (same standard covered in 4-5 files)
4. Create dedup checklist: `/path/去重合并清单.md`
   - List each topic group with file count before/after
   - For each file: ✅保留-合并, ⚠️重叠-合并后删除, ❌删除
   - User reviews checklist before execution

### Step 2: 理主线 (Restructure by Business Lifecycle)

Create 5-stage directory structure:

```
/path/
├── 00_全局导航.md
├── 01_规划阶段/    ← "我要建一个园区安防"
├── 02_设计阶段/    ← "我该怎么选设备"
├── 03_实施阶段/    ← "装好了，怎么验收"
├── 04_运维阶段/    ← "系统跑起来了"
└── 05_进阶前沿/    ← "我想了解未来"
```

Use symlinks: `ln -sf /path/original /path/02_设计阶段/original`

### Step 3: 加案例 (Add User Perspective)

For each key topic, add:
- User role cards (保安/校长/教育局 see different things)
- Decision logic (not just "what" but "why" and trade-offs)
- Real project cases (not "某市", use specific names)

### Step 4: 建规范 (Governance)

Each merged file gets version history table. Each topic directory gets a README with cross-references.

---

## Merge Execution Pattern

For each topic group:

```python
# 1. Create backup dir
terminal("mkdir -p /path/备份_原文件/TOPIC_原文件")

# 2. Backup originals
terminal("cp /path/ORIGINAL_DIR/*.md /path/备份_原文件/TOPIC_原文件/")

# 3. Group files by target merged file
groups = {
    "合并文件1": ["src1.md", "src2.md", "src3.md"],
    "合并文件2": ["src4.md", "src5.md"],
}

# 4. Read and merge each group
for target, sources in groups.items():
    contents = [read_file(f"/path/备份/{s}").get('content','') for s in sources]
    merged = f"# Title\n\n**版本：** v2.0\n---\n\n{''.join(contents)}"
    write_file(f"/path/TOPIC_终极指南/{target}.md", merged)

# 5. Clear original directory (backups preserved)
terminal("rm /path/ORIGINAL_DIR/*.md")
```

---

## Evaluation Framework (辣评)

When reviewing a knowledge library, check these dimensions:

| Dimension | Bad Sign | Good Sign |
|-----------|----------|-----------|
| 目录思维 vs 故事线 | Standard/device/tech classification | Business lifecycle flow |
| 内容重叠 | Same topic in 4-5 files | Merged into 1 definitive guide |
| 为写而写 | Toy code examples, template content | Real decision logic, trade-offs |
| 系统观 | Only "what", no "why" | User perspective, role cards |
| 矛盾 | Different files say different things | Consistent, version-tracked |

---

## Support Files

- `references/dedup-checklist-template.md` — Template for the deduplication checklist
- `references/evaluation-framework.md` — "辣评" framework for evaluating knowledge libraries

---

## Step 5: 专家系统升级 (Expert System Layer)

After restructuring, add three layers to transform from "素材库" to "专家系统":

### 5a. Prompt模板库
Create `00_Prompt模板库.md` with structured templates for common tasks:
- **方案设计模板** — Forces: needs analysis → architecture → device selection → cost → risk
- **故障排查模板** — Forces: classification → cause analysis → troubleshooting steps → solution
- **技术选型模板** — Forces: requirements → comparison table → decision → implementation
- **成本估算模板** — Forces: cost breakdown → price range → budget advice
- **标准解读模板** — Forces: standard info → core content → application guidance
- **快速问答模板** — Direct answer + reason + action + experience validation

Each template uses `# 角色与任务` + `# 步骤一/二/三` structure to force "consultant thinking" instead of "search engine mode".

### 5b. 经验库 (Experience Library)
Create `00_经验库_FAQ.md` with structured problem-solution pairs:
```
| 问题 | 原因 | 解决方案 | 优先级 |
```
Group by category (视频/网络/存储/平台/AI/安全). Target: 100+ entries.
Use for "answer self-check": after AI generates answer, validate against this library.

### 5c. 反馈机制 (Feedback Loop)
Create `00_反馈机制.md` with:
- Feedback channels: 👍/👎, 纠错, 需求, 建议
- Processing flow: collect → classify → analyze → update → verify
- Update cadence: experience库 weekly, Prompt templates monthly, knowledge content weekly
- Quality metrics: accuracy >95%, satisfaction >90%, feedback processing 100%

---

## ⚠️ Pitfalls (Learned the Hard Way)

1. **合并 ≠ 删除** — When merging files, ALWAYS backup originals to a `备份_原文件/` directory first. Never delete source files without backup. User explicitly corrected this.

2. **Batch before reporting** — When executing multiple sequential tasks (e.g., merging 7 topics), complete ALL tasks first, then give ONE summary report. Do not report after each step. User explicitly asked for this.

3. **Use symlinks for reorganization** — When creating a new directory structure, use `ln -sf` to create symbolic links to original files rather than moving files. Preserves both views.

4. **execute_code for batch file ops** — Use `execute_code` with `hermes_tools` (read_file, write_file, terminal) for batch processing multiple files. More efficient than individual tool calls.

5. **Cron jobs for autonomous work** — For long autonomous tasks (e.g., generating 8 reports overnight), use `cronjob(action='create')` with `schedule` set to desired time. Keep prompt self-contained with file paths and rules. Create a second cronjob at end time to report results.

6. **Don't use the model's knowledge as excuse to skip structure** — User said "你是大模型了，本身就有海量知识" — meaning use the model's built-in knowledge to ADD DEPTH (decision logic, cost analysis, real cases), not to skip the restructuring work. The value is in the structure + depth, not raw information.

7. **Detect root-vs-subdirectory duplication first** — Before restructuring, scan for folders that exist BOTH at root level AND inside numbered subdirectories (e.g., `/note/平安校园/` AND `/note/01_规划阶段/平安校园/`). These duplicates are the #1 source of confusion. Use `find /path -type d | sort` to detect, then deduplicate BEFORE merging. The subdirectory copy is usually the canonical one; the root copy is the stale leftover from a previous partial restructure.

8. **Always propose the plan before executing** — When user says "整理一下文件", do NOT immediately start moving files. First: (1) scan and analyze the full structure, (2) present a clear reorganization plan with proposed directory tree, (3) wait for user confirmation. This prevents mistakes and builds trust. User expects a planning-first workflow for structural changes.

9. **File organization topic folders (非安防) need separate top-level categories** — The 01-05 lifecycle structure works for 安防 domain files. For other domains (Agent AI, 智能农业, 海康研究), create numbered top-level categories (06, 07, 08...) rather than forcing them into the 安防 lifecycle. Each domain gets its own subtree with its own internal organization.

---

## Success Metrics

- Files reduced by ~60% (e.g. 65 → 25)
- All original content preserved in backup
- Clear navigation via `00_全局导航.md`
- Readers can follow project lifecycle linearly
- Expert system layer: Prompt templates + experience library + feedback mechanism
