---
name: study-os
description: Use Study OS tools to review Obsidian notes with spaced repetition.
platforms: [linux, macos, windows]
---

# Study OS Skill

Use this skill for learning review, mistake analysis, Anki candidate generation, and
Ebbinghaus-based spaced repetition from an Obsidian vault.

## When to Use

- User gives a problem (题目) and explicitly asks to "整理" it — **must execute**
  the Single-Problem Study Workflow. Trigger words: 整理, 分析一下这道题,
  研究一下, 帮我看看这道题, 做笔记.
- User asks to generate a learning curriculum from a textbook chapter.
- User asks to review, study, or practice from their Obsidian learning vault.
- User wants to check which examples are due for spaced-repetition review.
- User wants to log mistakes, generate weekly reports, or export Anki cards.
- A cron job triggers daily review (see Cron Setup below).

## Prerequisites

- `OBSIDIAN_VAULT_PATH` env var set, or vault at `~/Documents/Obsidian Vault`.
- `study` toolset enabled (`hermes tools` → enable "study").
- `.StudyOS/study_profile.md` exists with error taxonomy and review preferences.

## Quick Reference

| Tool | Purpose |
|------|---------|
| `study_create_curriculum` | Create standardized curriculum JSON from textbook + exercise book. |
| `study_list_curricula` | List all curricula in .StudyOS/curricula/. |
| `study_learning_queue` | New material: concepts by learning_state + examples never attempted. |
| `study_update_concept_state` | Update concept learning_state (未开始→学习中→已理解→已掌握). |
| `study_log_session` | Log a study session (duration, topics, notes created, examples). |
| `study_due_reviews` | Examples due for Ebbinghaus spaced-repetition review. |
| `study_record_review` | Record pass/fail, update intervals. |
| `study_list_notes` | Search and filter notes by tag, layer, or query. |
| `study_read_note` | Read a single note by path, title, or alias. |
| `study_extract_concepts` | Extract concepts, patterns, and tags from notes. |
| `study_log_error` | Log a mistake with cause classification. |
| `study_create_review_task` | Create a manual review task with due date. |
| `study_generate_weekly_report` | Weekly summary with cause×concept cluster analysis. |
| `study_export_anki_candidates` | Export high-value Anki card candidates. |
| `study_concept_graph` | Query concept dependency graph (cached JSON). |
| `study_review_stats` | Aggregated review + learning stats (cached JSON). |
| `study_sync_memory` | Build memory entries for cross-session recall. |
| `study_concept_graph` | Query concept dependency graph (cached in .StudyOS/concept_graph.json). |
| `study_review_stats` | Aggregated review stats (cached in .StudyOS/review_stats.json). |

## Spaced Repetition Algorithm

Intervals are calculated as: **Ebbinghaus base × review_level weight**.

| review_count | Base interval | review_level | Weight |
|-------------|--------------|-------------|--------|
| 0 | 1 day | 0 | ×0.5 |
| 1 | 2 days | 1 | ×0.7 |
| 2 | 4 days | 2 | ×1.0 |
| 3 | 7 days | 3 | ×1.3 |
| 4 | 15 days | 4 | ×1.6 |
| 5 | 30 days | 5 | ×2.5 |
| 6+ | 60 days | | |

Example: review_count=2 (base 4 days), review_level=1 (×0.7) → next review in 2 days.
Example: review_count=4 (base 15 days), review_level=4 (×1.6) → next review in 24 days.

On pass: review_count increments, interval advances.
On fail: review_count resets to 0, next review in 1 day.

## Curriculum Creation Workflow

Standardized curricula (`curricula/*.json`) are the single source of truth for what to learn.
Generate them from textbooks and exercise books — NOT from inconsistently formatted review/ 题单.

1. **Get template**: `study_create_curriculum()` → returns empty JSON template.

2. **Read source materials**: read the textbook chapter and corresponding exercise book section.
   Ask the user to provide the textbook name, chapter, and exercise book section if unclear.

3. **Identify 考点**: for each section in the textbook chapter:
   - Extract concepts (定义, 定理, 公式).
   - Identify problem types and variations.
   - Map each 考点 to representative problems from the exercise book.
   - Note which concepts are prerequisites for others.

4. **Fill the JSON**:
   ```json
   {
     "meta": {
       "topic": "一元函数微分学",
       "textbook": "张宇高等数学30讲 第9章",
       "exercise_book": "张宇1000题 A组第9章"
     },
     "sections": [
       {
         "title": "导数概念",
         "kaodian": [
           {
             "id": "kd-1",
             "name": "基本概念与微分",
             "summary": "微分定义 + 性质传递（奇偶→导函数奇偶）",
             "concepts": ["导数定义", "微分", "奇偶性"],
             "problems": [
               {"id": "1306", "source": "1993数二"},
               {"id": "1308", "source": "补充"}
             ]
           }
         ]
       }
     ]
   }
   ```

5. **Save**: `study_create_curriculum(data={...})` → validates and saves to `.StudyOS/curricula/<topic>.json`.

6. **Verify**: `study_list_curricula()` confirms the import.

After the curriculum exists, `study_learning_queue` can reference it to order concepts by dependency
and `study_concept_graph` can cross-reference 考点 with vault concept notes.

## Single-Problem Study Workflow (整理题目)

**触发即执行。** 当用户给出一道题目并要求"整理"时，必须完整执行此流程，
不可跳过任何步骤。这是对单个题目的深度消化——识别考点、梳理概念关联、
按需创建笔记——与每日批量复习不同。

### 触发词

"整理"、"分析一下这道题"、"研究一下"、"帮我看看这道题"、"做笔记"、
"总结一下"——只要用户显式给出题目并表达做笔记/分析的意图，就必须触发。

用户可能直接粘贴题目文本，或给出题号让你从 `examples/` 读取。

### Phase 1 — 强制探索（不可跳过）

在创建任何笔记之前，**必须先充分了解仓库中已有的知识和题型**。
使用 Study OS 工具进行系统探索：

**1a. 获取已有概念全量列表（快速去重）**

```text
study_concept_graph()
```

先不带参数调用，获取 `all_concepts`（全部概念名的列表）和
`isolated_concept_names`（无依赖关系的孤立概念）。

与题目中识别的候选概念交叉比对，区分「已有概念」和「潜在新概念」。
已有概念跳过创建步骤（但仍需检查完整性）；潜在新概念进入后续搜索确认。

**1b. 提取题目中的概念和题型关键词**

从题目中识别涉及的核心概念（定义、定理、公式）和题型特征（触发信号、
解题套路）。列出一个候选列表。

**1c. 搜索已有概念笔记**

```text
study_list_notes(layer="concept", query="<概念名>", search_body=true, normalize=true)
```

对候选列表中的每个概念，搜索 `/Box/` 中是否已有对应概念卡。
关注返回的 `title`、`path`、`excerpt` 判断匹配度。

- `search_body=true` 确保正文中深度讨论的概念也能被搜到（不局限于 frontmatter）。
- `normalize=true` 处理中文措辞变体（如"导数的定义"能匹配到"导数定义"），作为严格匹配失败时的 fallback。

如果一次查询无结果，尝试缩短 query（去掉修饰词）或用近义词再搜。

**1d. 搜索已有题型笔记**

```text
study_list_notes(layer="pattern", query="<题型关键词>", search_body=true, normalize=true)
```

搜索 `/Box/题型/` 中是否已有对应的题型卡。同理启用 `search_body` 和 `normalize`。

**1e. 检查概念的链入链出（概念依赖图）**

对每个候选概念，调用概念图查询：

```text
study_concept_graph(concept="<概念名>")
```

关注以下返回字段：
- `direct_prerequisites` — **链出**：该概念依赖哪些前置概念（必须先掌握）
- `direct_dependents` — **链入**：哪些概念依赖它（它薄弱会影响什么）
- `ancestor_chains` — 完整前置依赖链（如 泰勒展开 → 高阶导数 → 导数定义 → 极限定义）
- `descendant_chains` — 完整影响链（该概念被哪些下游概念依赖）
- `exercised_in` — 哪些已有例题涉及该概念
- `review_level` — 该概念下例题的平均掌握级别
- `note_count` — 有多少条笔记引用了该概念

**关键判断**：如果 `note_count > 0` 但概念本身没有独立的 `/Box/` 概念卡，
说明该概念仅在例题中被引用但缺少正式定义，需要创建概念卡。

**1f. 阅读候选匹配笔记（确认完整性）**

对搜索到的疑似匹配笔记，用 `study_read_note` 读取内容，判断是否已覆盖
题目中的所有考点。不要只看标题——读 body 确认内容的完整性。

**1g. 交叉引用周边概念**

```text
study_extract_concepts(notes=["<相关概念卡路径>", ...])
```

了解相关概念笔记中已链接了哪些概念和题型，避免孤立地创建笔记。

### Phase 2 — 决策与创建

完成 Phase 1 的全面探索后，按以下决策逻辑执行：

```
题目分析完成，已了解现有知识体系
  ↓
【知识点笔记 /Box/】
  对每个识别出的概念：
    ├─ /Box/ 中无对应概念卡 → 创建知识点笔记
    ├─ 有概念卡但不完整（缺定义/条件/混淆点/相关链接）→ 补充
    └─ 概念卡已完整 → 跳过，可考虑添加双向链接
  ↓
【题型笔记 /Box/题型/】
  仅当题目展示了可复用的解题套路时：
    ├─ 无可复用套路 → 跳过（不是每道题都需要题型卡）
    ├─ 有套路但 /Box/题型/ 中无对应卡片 → 创建题型笔记
    ├─ 有题型卡但不完整 → 补充（添加题目作为新 example 引用）
    └─ 题型卡已完整 → 跳过
  ↓
【严禁创建例题 /examples/】
  └─ 绝对不在 /examples/ 下创建 E-####.md
     └─ 除非用户显式要求："把这个加到例题库" / "创建例题笔记"
```

#### 创建知识点笔记（`/Box/`）

遵循 `Box/AGENTS.md` 的全部约定：
- `type: concept`
- 文件名：描述性中文名（如 `导数定义与存在性.md`）
- 内容必须包含：定义、定理/公式、适用条件、常见混淆点、
  与相关概念的双链（利用 Phase 1 探索到的 `direct_prerequisites`
  和 `direct_dependents` 建立链接）
- `concepts` 字段中填写该概念的前置概念（链出）
- 如果 Phase 1 探索发现已有例题引用该概念，在正文中用 `[[双链]]` 反链回去

#### 创建题型笔记（`/Box/题型/`）

遵循 `Box/AGENTS.md` 的全部约定：
- `type: pattern`
- 文件名：必须使用 `题型：...` 前缀
- 内容必须包含：触发信号（看到什么条件/结构时用这个套路）、
  步骤模板、关键变形、易错点
- `concepts` 字段链接到该题型依赖的概念卡
- 不要为了"每题都挂题型"而生造低价值题型卡

### Phase 3 — 输出总结

整理完成后，向用户报告：

1. **本题涉及的概念**（已存在 / 新创建 / 已补充）
2. **本题涉及的题型**（已存在 / 新创建 / 已补充 / 无需题型卡）
3. **概念依赖关系**：指出新概念插入的位置——
   它的前置概念（链出）和下游概念（链入）分别是什么
4. **关键易错点或变体**（口头提醒，不强制写成笔记）

### 反模式

- **跳过 Phase 1 直接创建**：未探索就创建笔记会导致重复、孤立、链接断裂
- **批量创建概念卡**：一次整理聚焦一道题的核心概念，
  不要把题目中所有出现过的名词都创建卡片
- **创建无触发信号的题型卡**：如果说不清"看到什么条件用这个套路"，
  就不是一个合格的题型
- **擅自创建例题文件**：例题入库必须由用户显式授权
- **修改既有笔记文件名**："优化编号"、"统一风格"都是禁止操作

---

## Learning New Material Workflow

This is the workflow for studying new concepts, creating notes, and attempting
examples for the first time — before they enter the Ebbinghaus review cycle.

1. **Check the queue**: `study_learning_queue()` — shows what's new.
   - Concepts ordered by dependency (prerequisites first, via `study_concept_graph`).
   - Examples ordered by difficulty.

2. **Study a concept**:
   a. Mark it as `学习中` via `study_update_concept_state(note="<concept>", learning_state="学习中")`.
   b. Read the concept card, related examples, and textbook references.
   c. When understood → `study_update_concept_state(note="<concept>", learning_state="已理解")`.
   d. When fully mastered → `study_update_concept_state(note="<concept>", learning_state="已掌握")`.

3. **Attempt new examples**: pick from `study_learning_queue` (state="未开始" examples).
   - Read the problem via `study_read_note`.
   - Attempt to solve. If stuck, use `study_concept_graph` to find prerequisite gaps.
   - On first success → `study_record_review` records the result, entering
     the example into Ebbinghaus rotation.
   - On failure → `study_log_error` with cause classification.

4. **Log the session**: `study_log_session(duration_minutes=..., topics=[...])`.

5. **Sync memory**: `study_sync_memory()` at session end.

## Daily Ebbinghaus Review Workflow

This is the primary workflow. Run it when the user says "复习" or when triggered by cron.

1. **Load profile**: `study_read_note(note=".StudyOS/study_profile.md", include_body=true)` to get error taxonomy and review preferences.

2. **Find due reviews**: `study_due_reviews()` — returns examples sorted by priority (lowest review_level first, oldest last_reviewed_at first).

3. **For each due example** (up to `max_problems` from profile, default 8):
   a. `study_read_note(note="<path>", include_body=false)` to get the problem statement.
   b. Present the problem to the user and ask for their solution or answer.
   c. Judge correctness. Be strict: concept confusion or missed conditions = fail.
   d. `study_record_review(note="<path>", passed=<bool>)` to update intervals.
   e. If failed: also call `study_log_error` with the appropriate cause from the taxonomy, and set `log_error=true` on `study_record_review`.

4. **Summary**: Report how many reviewed, how many passed/failed, and which concepts need attention next.
   If a concept appears weak across multiple examples, use `study_concept_graph(concept="<name>")`
   to check its prerequisite chain and suggest what to review first.

5. **Sync memory**: `study_sync_memory()` → pass the returned `memory_entries` to Hermes's `memory` tool with `target="memory"`. This preserves study state across sessions so the agent remembers weak areas on next startup.

## Weekly Review Workflow

1. `study_generate_weekly_report(start_date="<Monday>", end_date="<Sunday>")`.
   The report now includes:
   - **Error Patterns (cause×concept table)**: cross-tabulation of every (cause, concept) pair.
   - **⚠️ Repeated Patterns**: same cause + same concept ≥3 times — high-priority remediation.
   - **🔴 Deep Confusion**: one concept with ≥2 different causes — indicates fundamental misunderstanding.
2. For Repeated Patterns / Deep Confusion → `study_concept_graph(concept="<concept>")` to trace prerequisites and dependents.
3. For Repeated Patterns → check the concept card in `/Box` for gaps. Create targeted review tasks for affected examples.
4. For Deep Confusion → the `recommended_review_order` from concept graph tells you what to review first. Suggest re-reading from definitions up, not just patching individual errors.
5. `study_sync_memory()` → update Hermes memory with refreshed study state.
6. Propose 3–5 focused review tasks for next week, ordered by `recommended_review_order`.

## Memory Integration

`study_sync_memory` produces structured entries for Hermes's `memory` tool.
These entries use `old_text` matching via "StudyOS Math:" prefix so updates
replace stale data rather than accumulating duplicates.

After every review session (daily or weekly), call:

1. `study_sync_memory()` — get `memory_entries` array.
2. `memory(target="memory", operations=<memory_entries>)` — upsert into Hermes memory.

This ensures the agent remembers across sessions:
- Which concepts are currently weakest
- How many examples are due for review
- When the last sync happened

On session start (when the user says "开始学习" or cron fires), the agent
should check `memory` tool for existing StudyOS entries to re-establish context.

## Concept Dependency Graph

`study_concept_graph` builds a directed graph from all notes' `concepts` fields.
Edges point from a concept to its prerequisites. Two query modes:

**Summary mode** (no `concept` argument):
- Lists all concepts with dependencies
- Identifies **bottleneck concepts** — concepts many others depend on
- Shows `review_levels` per concept (min/avg across referencing examples)
- With `weak_only=true`: only returns concepts with recent errors, each with
  recommended review order

**Target mode** (`concept="<name>"`):
- Direct prerequisites — what you must understand first
- Direct dependents — what will break if this concept is weak
- Ancestor chains — full prerequisite path (e.g., 泰勒展开 → 高阶导数 → 导数定义 → 极限定义)
- Descendant chains — full impact path
- Affected examples — which examples are impacted
- Recommended review order — topological sort of the concept and all prerequisites

### Diagnostic Workflow

When the weekly report shows a Repeated Pattern or Deep Confusion on a concept:

1. `study_concept_graph(concept="<弱项概念>")` — trace its prerequisites.
2. Check `review_level` of each prerequisite. If any are ≤ 2, review them first.
3. Check `descendant_chains` to understand the blast radius — which other concepts
   depend on this one and may also need attention.
4. If the concept is a bottleneck (many dependents), prioritize it above non-bottleneck concepts.
5. Use `recommended_review_order` to plan the fix sequence.

Example:
```
study_concept_graph(concept="泰勒展开")
→ prerequisites: [高阶导数, 导数定义]
→ review_levels: 高阶导数=3.5, 导数定义=1.2  ← 先修导数定义！
→ dependents: [泰勒级数, 幂级数展开, 近似计算]  ← 影响面大
→ recommended_review_order: [导数定义, 高阶导数, 泰勒展开]
```

## Cached Data

Two JSON caches speed up repeated queries on large vaults (700+ notes):

| Cache | Path | Invalidated |
|-------|------|-------------|
| Concept graph | `.StudyOS/concept_graph.json` | Hourly TTL, or `rebuild=true` |
| Review stats | `.StudyOS/review_stats.json` | Auto after `study_record_review` |

Call `study_review_stats()` to get a quick overview:
- `progress_pct` — percentage of examples mastered (review_level=5)
- `due_today` — how many examples are due for Ebbinghaus review
- `review_streak_days` — consecutive days with at least one review
- `by_review_level` — distribution across 0-5
- Per-concept averages and due counts

These caches are read-only for the agent. `study_record_review` auto-invalidates
the stats cache; `study_concept_graph(rebuild=true)` forces a graph rebuild after
adding or editing notes.

## Concept Learning States

Concepts and patterns track progress through four states (stored in YAML `learning_state`):

| State | Meaning | Next |
|-------|---------|------|
| `未开始` | Haven't studied yet | → `学习中` when you start |
| `学习中` | Currently studying | → `已理解` when you understand |
| `已理解` | Understood, can apply | Enters review pool |
| `已掌握` | Fully mastered | Maintenance only |

Examples use `review_level` (0–5) for skill mastery. Concepts use `learning_state` for knowledge acquisition. Both feed into the dependency graph.

## Write Policy

Study OS data always goes under `<vault>/.StudyOS/`:
- `curricula/*.json` — standardized learning curricula (single source of truth)
- `errors/YYYY-MM.md` — mistake logs
- `review_tasks.md` — manual review tasks
- `reports/YYYY-Www.md` — weekly summaries with cluster analysis
- `anki_candidates/YYYY-MM-DD.md` — Anki exports
- `sessions/YYYY-MM-DD.md` — study session logs
- `concept_graph.json` — cached dependency graph (auto-refreshed hourly)
- `review_stats.json` — cached review + learning stats (auto-invalidated on review)

Do not edit source notes unless the user explicitly asks. The `study_record_review` tool safely updates
`review_count`, `last_reviewed_at`, and `next_review_at` in the note's YAML frontmatter.

## Cron Setup

To automate daily review without manual triggering:

```bash
hermes cron add "math-daily-review" \
  --schedule "every day 20:00" \
  --skills study-os \
  --workdir /home/puji/Math \
  --prompt "执行每日艾宾浩斯复习：用 study_due_reviews 找出待复习例题，逐题展示并判定，用 study_record_review 更新间隔。上限 8 题。最后输出今日总结。"
```

## Output Style

Use Simplified Chinese. Tie all recommendations to concrete notes, concepts, and review states.
