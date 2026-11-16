---
name: prj-proposals-manager
description: Manage proposal lifecycle from intake to delivery across coordinating agents or roles. Use when the user asks to create, update, track, accept, or close a proposal, or when handling PRD confirmation, technical review, acceptance, or revision workflows. Works with any agent platform (Cursor, Hermes, OpenClaw, etc.)
---

# Proposal Management

## Overview

A platform-agnostic skill for managing proposal lifecycles across multi-role workflows (e.g. coordinator / PM / dev). Covers intake, clarification, PRD confirmation, technical review, development handoff, acceptance, and delivery.

## Configuration

Current environment configuration (for Hermes):

| Variable | Value | Description |
|----------|-------|-------------|
| `PROPOSALS_ROOT` | `~/.hermes/proposals` | Directory holding proposal index and files |
| `TEMPLATES_DIR` | `~/.hermes/proposals/templates` | Subdirectory for templates |
| `PM_OUTPUT_DIR` | `~/.hermes/proposals/workspace-pm/<项目名>/proposals` | Where PM stores PRD documents |
| `DEV_OUTPUT_DIR` | `~/.hermes/proposals/workspace-dev/<项目名>/proposals` | Where dev stores project artifacts |
| `TEST_OUTPUT_DIR` | `~/.hermes/proposals/workspace-test/<项目名>/proposals` | Where Test Expert stores test cases and results |
| `RESEARCH_OUTPUT_DIR` | `~/.hermes/proposals/workspace-research/<项目名>/proposals` | Where Research Analyst stores iteration research reports |
| `COORDINATOR` | `小墨` | Primary coordinating role name |
| `REQUESTER` | `boss` | Who submits requests |
| `PROPOSAL_DOCS_INDEX` | `~/.hermes/proposals/proposal-docs-index.md` | Index of all PRD and technical solution documents |
| `SYNC_SCRIPT` | `~/.hermes/scripts/sync-proposals-to-website.py` | 单向同步脚本，CSV模式：`--csv-only` 从 GitHub 生成 CSV |

### Index Files

**重要变更：2026-05-05 起，数据以 CSV 为结构化存储，markdown 仅作轻量快速索引。**

| File | Purpose | Format |
|------|---------|--------|
| `~/.hermes/proposals/proposals.csv` | 提案主数据（20字段，含 project_id 外键） | CSV |
| `~/.hermes/proposals/projects.csv` | 项目主数据（id, name, proposal_count, git_repo） | CSV |
| `~/.hermes/proposals/project_proposal_mapping.csv` | Project↔Proposal 映射关系 | CSV |
| `~/.hermes/proposals/proposal-index.md` | 提案快速索引（仅含统计摘要，指向 CSV） | Markdown |
| `~/.hermes/proposals/project-index.md` | 项目快速索引（仅含清单和统计，指向 CSV） | Markdown |

**CSV 字段说明：**
- `proposals.csv`：`id`, `title`, `status`, `project_id`, `project_name`, `git_repo`, `deployment_url`, `prd_confirmation`, `acceptance`, `last_update` 等 20 字段
- `projects.csv`：`id`, `name`, `proposal_count`, `git_repo`
- `project_proposal_mapping.csv`：`project_id`, `project_name`, `project_git_repo`, `proposal_id`, `proposal_name`, `proposal_status`

**关系：**
- `project-index.md` 和 `proposal-index.md` 共同构成完整的"项目-提案"双视角索引
- **两者均为轻量索引，实际数据以 CSV 为准**
- CSV 由 `sync-proposals-to-website.py --csv-only` 从 GitHub 生成

### Website & GitHub

| Item | Value |
|------|-------|
| Website | https://yeluo45.github.io/prj-proposals-manager/ |
| Website title | 项目提案管理（不是"提案管理"） |
| GitHub repo | YeLuo45/prj-proposals-manager (master 分支存源码，gh-pages 分支存部署) |
| hermes-agent repo | YeLuo45/hermes-agent (proposals 变更同步到此处) |
| Data file | `data/proposals.json` on master (v2 格式: `{version: 2, projects: [{name, url, gitRepo, proposals: [...]}]}`) |
| GitHub Token | `$GITHUB_TOKEN`（环境变量，placeholder 用于文档；实际 token 在 `~/.hermes/tools/github-token.txt`）|

### Sync Strategy

**数据流向：** GitHub `data/proposals.json` → `sync-proposals-to-website.py --csv-only` → `proposals.csv` / `projects.csv`

**本地 CSV 生成（常用）：**
```bash
GITHUB_TOKEN=$GITHUB_TOKEN \
  python3 ~/.hermes/scripts/sync-proposals-to-website.py --csv-only
```

**写入 GitHub（通过 API）：** 网站保存时直接 PUT 到 GitHub master 的 `data/proposals.json`。GitHub Actions 检测到 master push 后自动 rebuild gh-pages。

**三层数据读取优先级：**
1. GitHub API（实时，有 token 时优先）
2. 本地 CSV（`~/.hermes/proposals/proposals.csv`，无网络时使用）
- `project_proposal_mapping.csv` 同步维护
- 写入前全部字段自动校验，校验失败则中断，不写入

### 初始化脚本

首次使用或目录损坏时，运行初始化脚本（会自动检测是否已初始化）：

```bash
python3 scripts/init_proposals_dir.py
```

**检测逻辑：**
- 根目录不存在 → 自动初始化
- 子目录/文件缺失 → 补建缺失项
- CSV 文件为空 → 重建
- CSV 文件已有数据行（>1行）→ 拒绝初始化，退出（防止覆盖已有数据）

### Data Management CLI

**所有项目和提案的增删改必须通过 `proposal_manager_cli.py` 脚本进行，禁止直接写入 CSV。**

```bash
python3 scripts/proposal_manager_cli.py <command> [options]
```

#### 项目管理

```bash
# 新增项目（自动生成ID）
python3 scripts/proposal_manager_cli.py project add --name "项目名" --git-repo "https://github.com/owner/repo"

# 列出项目
python3 scripts/proposal_manager_cli.py project list --fields id,name,proposal_count

# 获取单个项目
python3 scripts/proposal_manager_cli.py project get PRJ-20260417-001 --json

# 更新项目
python3 scripts/proposal_manager_cli.py project update PRJ-20260417-001 --name "新名称"

# 删除项目（需确认无活跃提案，或用 --force 强制）
python3 scripts/proposal_manager_cli.py project delete PRJ-20260513-001 --force
```

#### 提案管理

```bash
# 新增提案（自动生成ID）
python3 scripts/proposal_manager_cli.py proposal add --title "提案标题" --project-id PRJ-20260417-001

# 列出提案（支持过滤）
python3 scripts/proposal_manager_cli.py proposal list --fields id,title,status,project_name
python3 scripts/proposal_manager_cli.py proposal list --status active
python3 scripts/proposal_manager_cli.py proposal list --project-id PRJ-20260417-001

# 获取单个提案
python3 scripts/proposal_manager_cli.py proposal get P-20260513-001 --json

# 更新提案
python3 scripts/proposal_manager_cli.py proposal update P-20260513-001 --status "in_dev"
python3 scripts/proposal_manager_cli.py proposal update P-20260513-001 --status "accepted" --acceptance "accepted"

# 删除提案（软删除，直接移除）
python3 scripts/proposal_manager_cli.py proposal delete P-20260513-001

# 归档提案（软删除，标记为 archived）
python3 scripts/proposal_manager_cli.py proposal archive P-20260513-001
```

#### 校验规则（自动执行）

| 字段 | 校验内容 |
|------|---------|
| `id` | 格式 `PRJ-YYYYMMDD-XXX` 或 `P-YYYYMMDD-XXX`，自动判重 |
| `project_id` | 外键必须已存在于 `projects.csv` |
| `status` | 必须在有效枚举值内 |
| `git_repo` | 必须以 `http://` `https://` 或 `git@` 开头（可空） |
| `deployment_url` | 同上（可空） |
| `prd_confirmation` | 枚举：`pending`, `confirmed`, `timeout-approved`, `rejected`, 空 |
| `tech_expectations` | 同上 |
| `acceptance` | 枚举：`pending`, `accepted`, `rejected`, 空 |
| `game_type` | 枚举：休闲/策略/卡牌/RPG/消除/塔防/模拟/动作/射击 |
| 必填 | `id`, `title`, `project_id`, `status` 不能为空 |

#### 数据写入规范

- 新增时自动填充 `last_update` 为当天日期
- 新增提案时自动继承所属项目的 `git_repo`
- 新增/删除提案时自动更新对应项目的 `proposal_count`
- `project_proposal_mapping.csv` 同步维护
- 写入前全部字段自动校验，校验失败则中断，不写入

**URL 路径拼接注意：** GitHub Pages 部署在子路径 `/prj-proposals-manager/`（末尾有斜杠）。`window.location.pathname` 返回 `/prj-proposals-manager/`。拼接 data URL 时必须 strip 末尾斜杠：`pathname.replace(/\/$/, '')` 得到 `/prj-proposals-manager`，再拼 `${origin}${basePath}/data/proposals.json` = `https://yeluo45.github.io/prj-proposals-manager/data/proposals.json`。

These values are hardcoded for the current Hermes environment. Do not ask — use them directly.

## Proposal ID Format

`P-YYYYMMDD-XXX` — zero-padded sequential number per day.

To determine the next ID, read `proposals.csv` and find the highest `XXX` for today's date.

## Proposal States

Use these exact names across all roles — no custom variants:

```
intake → clarifying → prd_pending_confirmation → approved_for_dev → in_tdd_test → in_dev → in_test_acceptance → accepted → deploying → deployed
                                                                                    ↓                              ↓                              ↓
                                                                          needs_revision → in_dev              test_failed → in_dev              research_direction_pending → in_prd
```

## Project Docs Directory Convention

Every project under `${DEV_OUTPUT_DIR}/<项目名>/proposals/` must have a `docs/` subdirectory containing all proposal-related documentation and a local index. This keeps historical context self-contained within each project.

```
${DEV_OUTPUT_DIR}/<项目名>/proposals/
├── docs/
│   ├── index.md              # Local document index (version history)
│   ├── proposal.md           # Original proposal intake document
│   ├── prd.v1.md            # PRD v1 (versioned)
│   ├── prd.v2.md            # PRD v2 (if revised)
│   ├── technical-solution.v1.md  # Technical solution v1 (versioned)
│   └── technical-solution.v2.md  # Technical solution v2 (if revised)
└── (project source files)
```

### docs/index.md Format

```markdown
# P-YYYYMMDD-XXX: <Title> — Documents

## Proposal

| Version | File | Updated |
|---------|------|---------|

## PRD

| Version | File | Updated | Notes |
|---------|------|---------|-------|

## Technical Solution

| Version | File | Updated | Notes |
|---------|------|---------|-------|

## Test Cases

| Version | File | Updated | Notes |
|---------|------|---------|-------|
```

**Rules:**
- `docs/proposal.md` is the single source of truth for the original proposal (never versioned)
- `prd.md` is a symlink or copy of the current PRD version for convenience
- `technical-solution.md` is a symlink or copy of the current technical solution for convenience
- `test-cases.md` is a symlink or copy of the current test cases for convenience
- Versioned files use `.vN.md` suffix (v1, v2, ...)
- When a new version is created, update `docs/index.md` with the new entry
- The index is the single source of truth for version history within the project

## Proposal Docs Index

Maintain a single source of truth at `${PROPOSAL_DOCS_INDEX}` (`~/.hermes/proposals/proposal-docs-index.md`) that maps every proposal to its current and historical documents.

### proposal-docs-index.md Format

```markdown
# Proposal Documents Index

## P-YYYYMMDD-XXX: <Title>

| Document | Path | Version | Updated |
|----------|------|---------|---------|
| Proposal | `workspace-dev/<项目名>/proposals/docs/proposal.md` | - | YYYY-MM-DD |
| PRD | `workspace-dev/<项目名>/proposals/docs/prd.v1.md` | v1.0 | YYYY-MM-DD |
| Technical Solution | `workspace-dev/<项目名>/proposals/docs/technical-solution.v1.md` | v1.0 | YYYY-MM-DD |

---

## P-YYYYMMDD-YYY: <Title>
...
```

**Index maintenance rules:**
- Create the index entry when the proposal is registered
- Update the version and date whenever a document is revised
- Do NOT copy full documents into the index — the index only tracks paths and versions
- The index provides a quick overview of all proposals and their document versions without needing to open each project

### Historical PRD References

When a proposal evolves (e.g., scope change, revision), historical PRD versions should be preserved:

- Store old PRD versions with version suffix: `prd.v1.md`, `prd.v2.md`
- The index tracks the current version and previous versions
- When requesting PRD confirmation, include reference to the previous version if this is a revision cycle

## Workflow

```
- [ ] Step 1a/1b: Intake – register proposal (from existing codebase OR new)
- [ ] Step 2: Clarify – up to 3 rounds
- [ ] Step 3: Route to PM if needed
- [ ] Step 4: PRD confirmation gate
- [ ] Step 5: Technical expectations gate (up to 3 rounds)
- [ ] Step 6: Output technical solution
- [ ] Step 6b: Hand off to Test Expert – generate TDD test cases
- [ ] Step 7: Hand off to dev (with test cases as reference)
- [ ] Step 8: Test Expert executes acceptance based on test cases
- [ ] Step 9: Deliver or revise
```

### Step 1a: Register from Existing Codebase

When the request is to clone an existing GitHub repo and register it as a proposal (rather than building from scratch):

1. **Clone the repo** to `${DEV_OUTPUT_DIR}/<项目名>/proposals/`
   ```bash
   git clone https://<token>@github.com/<owner>/<repo>.git ${DEV_OUTPUT_DIR}/<项目名>/proposals/
   ```
   Use token `$GITHUB_TOKEN` (YeLuo45)

2. **Create project docs** under `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/`:
   - `index.md` — local document index
   - Any existing README.md should be in Chinese; if English, replace content

3. **Update `proposal-index.md`** — add entry with status `delivered` (already built), `in_dev` (if still developing), or `accepted` (if already accepted)

4. **Update proposals-manager website** via GitHub API:
   ```python
   # GET current file + SHA
   GET https://api.github.com/repos/YeLuo45/prj-proposals-manager/contents/data/proposals.json
   # PUT new content with SHA (triggers GitHub Actions rebuild)
   PUT https://api.github.com/repos/YeLuo45/prj-proposals-manager/contents/data/proposals.json
   body: { "message": "feat: add P-YYYYMMDD-XXX <name>", "content": <base64>, "sha": <sha> }
   ```
   注意：网站 repo 是 `prj-proposals-manager`，不是 `proposals-manager`。同步脚本 `sync-proposals-to-website.py` 已配置正确目标。
   静态打包文件需在 build 前通过 `curl` 从 GitHub API 下载最新 proposals.json 保持同步。

5. **Sync to `proposals-document` repo** — commit updated project docs to `project-docs/<项目名>/proposals/` in the proposals-document repo

6. **Fix GitHub repo description** to Chinese via PATCH API:
   ```python
   PATCH https://api.github.com/repos/YeLuo45/<repo>
   body: { "description": "<Chinese description>", "homepage": "<pages-url>" }
   ```

### Step 1b: Register New Proposal (from scratch)

1. Read `${PROPOSALS_ROOT}/proposal-index.md` to determine the next ID
2. Copy `${TEMPLATES_DIR}/request-intake-template.md` to `${PROPOSALS_ROOT}/P-YYYYMMDD-XXX.md`
3. Fill in Basic Information and Original Request
4. Add an entry in `proposal-index.md` under Active Proposals with status `intake`
5. Add an entry in `${PROPOSAL_DOCS_INDEX}` for this proposal
6. Create `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/index.md` with the initial index structure

### Step 2: Clarify Requirements

- Ask the requester up to 3 rounds of clarifying questions focused on: goal, scope, constraints, acceptance criteria
- Record each round in the proposal file under Clarification
- After 3 rounds or when clear, record Final Assumptions
- Update status to `clarifying`

### Step 3: Route to PM

If the request is an idea or rough draft, hand off to the PM role for PRD generation.

- PM saves PRD to `${PM_OUTPUT_DIR}/<项目名>/YYYY-MM-DD-prd.md`
- PM also copies the PRD as `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/prd.v1.md`
- Update `PRD Path` in `proposal-index.md` once PM delivers
- Update both `${PROPOSAL_DOCS_INDEX}` and `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/index.md` with the new PRD path and version

### Step 4: PRD Confirmation Gate

When PM returns the PRD:

1. Present the PRD to the requester and ask for confirmation
2. Start a confirmation timeout (recommended: 5 minutes)
3. Record the timeout reference in `PRD Confirmation Countdown ID`

**If confirmed**: set `PRD Confirmation` to `confirmed`, cancel the timeout, then **immediately update status to `approved_for_dev` and launch dev**. Do NOT ask for further confirmation — the cron job is a safety net for when the user never responds, not a required gate.

**If timeout**: set `PRD Confirmation` to `timeout-approved`, record in `Timeout Resolution`, then **immediately update status to `approved_for_dev` and launch dev**.

**Timeout Implementation by Platform**

| Platform | Method |
|----------|--------|
| Hermes | Use `cron` with `schedule` as ISO timestamp (e.g. `schedule: "2026-04-16T12:43:00+08:00"`), or track manually with timestamps in proposal file |
| OpenClaw | `cron` with `schedule.kind="at"`, `atMs=<now+300000>`, `payload.kind="systemEvent"` |
| Cursor | Use the countdown-manager skill if available, or track manually with timestamps |
| Other | Record a deadline timestamp and check on next interaction |

**Output Format for Timeout Default-Through**

When stating a timeout will automatically pass, you MUST:
1. Output the exact datetime in the message, using the format `YYYY-MM-DD HH:mm:ss`. Calculate it from `now + timeout_duration`
2. **Immediately after creating the cron job**, display the cron ID and trigger time

Example output:
```
PRD 确认超时时间：2026-04-22 14:30:00（5分钟后默认通过）
定时器ID: crn_abc123 | 生效时间: 2026-04-22 14:30:00
```

This provides clear visibility into when the auto-approval will trigger and which cron job handles it.

**Hermes cron timeout example:**
```python
cron(action='create', schedule='2026-04-16T12:43:00+08:00', prompt='Check proposal P-YYYYMMDD-XXX PRD confirmation timeout', name='prd-timeout-P-YYYYMMDD-XXX')
```

### Step 5: Technical Expectations Gate

Before outputting a technical solution:

1. Ask the requester about: stack, performance, cost, deployment, maintainability, dependency constraints
2. Up to 3 rounds of questions
3. Start a confirmation timeout (same mechanism as Step 4)
4. Record in `Technical Expectations Countdown ID`

**If confirmed**: set `Technical Expectations` to `confirmed`, cancel the timeout, then **immediately write technical solution and update status to `approved_for_dev`**. Do NOT ask for further confirmation.

**If timeout**: set `Technical Expectations` to `timeout-approved`, proceed with current assumptions, record in `Timeout Resolution`, then **immediately write technical solution and update status to `approved_for_dev`**.

### Step 6: Technical Solution

- Output the technical solution at `${PROPOSALS_ROOT}/P-YYYYMMDD-XXX-tech-solution.md`
- Also copy to `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/technical-solution.v1.md`
- Update status to `approved_for_dev`
- Update both `${PROPOSAL_DOCS_INDEX}` and `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/index.md` with the new technical solution path and version

### Step 6b: TDD Test Case Generation

After technical solution is output, hand off to Test Expert to generate test cases based on TDD principles:

1. **Coordinator routes to Test Expert** with:
   - PRD document
   - Technical solution document
   - Project context

2. **Test Expert outputs test cases** to `${TEST_OUTPUT_DIR}/<项目名>/YYYY-MM-DD-test-cases.md`
   - Test cases must be traceable to PRD requirements
   - Include test case ID, description, preconditions, steps, expected results
   - Cover happy path and edge cases
   - Copy test cases to `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/test-cases.v1.md`

3. **Update tracking:**
   - Update `Test Cases Path` in `proposal-index.md`
   - Update both `${PROPOSAL_DOCS_INDEX}` and `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/index.md`
   - Update status to `in_tdd_test`

4. **Dev references test cases** during development as acceptance criteria

### Step 7: Hand Off to Dev

- Update status to `in_dev`
- Dev creates `${DEV_OUTPUT_DIR}/<项目名>/proposals/docs/` directory if not exists (should already exist from Step 1)
- Dev saves project output to `${DEV_OUTPUT_DIR}/<项目名>/proposals/`
- The `docs/` directory must contain `index.md`, `proposal.md`, `prd.vN.md`, `technical-solution.vN.md`, and `test-cases.vN.md`
- Update `Project Path` in `proposal-index.md`

### Step 8: Test Expert Acceptance (TDD-based)

When dev reports completion, Test Expert executes acceptance based on test cases:

**Requirements consistency:**
- Matches requester-confirmed requirements
- Aligns with PRD
- No scope creep or shortcuts

**Test case execution:**
- Execute each test case from `test-cases.vN.md`
- Record pass/fail status for each test case
- Document any deviations or failures

**Functional verification (hands-on, not screenshots only):**
- Core functionality works end-to-end
- No errors in console/logs (warnings OK)
- Existing functionality not broken
- Build succeeds (e.g. `npm run build`, `cargo build`, `go build`)

**Delivery completeness:**
- File paths provided
- Startup/access instructions provided
- Verification results or screenshots provided

**Quality:**
- No obvious gaps
- No UI/logic conflicts
- Known limitations documented

Update status to `in_test_acceptance` during review.

**If all test cases pass**: proceed to Step 9 (Deliver)

**If any test case fails**: update status to `test_failed`, output structured revision notes:

```markdown
## Test Failure Report

- **Test Case ID**: <id>
- **Test Description**: <description>
- **Actual Result**: <what happened>
- **Expected Result**: <what should happen>
- **Impact**: <what is affected>
- **Expected fix**: <how to fix>
```

Record in `proposal-index.md` Notes field and route back to dev for fix.

### Step 9: Deliver or Revise

**If all test cases pass**: update status to `accepted`, proceed to Step 10 (Research Direction)

**If not accepted**: update status to `needs_revision`, output structured revision notes:

```markdown
## Revision Notes

- **Issue**: <description>
- **Impact**: <what is affected>
- **Expected fix**: <how to fix and how to verify>
```

Record revision notes in `proposal-index.md` Notes field.

### Step 10: Research Direction (Post-Acceptance Iteration Planning)

When acceptance passes (status becomes `accepted` or `delivered`):

1. **Coordinator asks the requester**: "基于本次交付，你希望进入下一个迭代方向探索，还是先维护当前版本？"
2. **Start a 5-minute confirmation timeout** with cron job
3. Record the timeout reference in `Research Direction Countdown ID`

**If requester confirms direction**: set `Research Direction` to `confirmed`, cancel timeout, then **immediately route to PM for next iteration PRD**. Do NOT wait for further confirmation.

**If timeout**: set `Research Direction` to `timeout-approved`, coordinator decides autonomously, then **immediately route to PM for next iteration PRD**.

**After Research Analyst delivers iteration direction report**:
- Update status to `research_direction_pending`
- Coordinator hands off to PM with the iteration direction as input
- PM generates new PRD based on the iteration direction
- Continue from Step 4 (PRD Confirmation Gate)

### Step 11: Deployment (Post-Acceptance Delivery)

After acceptance (status becomes `accepted`), coordinator handles deployment:

1. **Determine deployment target:**
   - **GitHub Pages**: If project has `gh-pages` branch support or GitHub Actions workflow
   - **Cloudflare Pages**: If project is a static site and GitHub Pages not suitable
   - Use `static-site-deploy` skill for automatic selection

2. **Create deployment branch:**
   ```bash
   cd ${DEV_OUTPUT_DIR}/<项目名>/proposals
   git checkout -b deploy/<项目名>-<YYYYMMDD>
   ```

3. **Prepare deployment:**
   - For React/Vite projects: ensure `package-lock.json` is committed, run `npm run build`
   - For static sites: ensure all built assets are present
   - Update README.md with deployment URL if needed

4. **Push to remote:**
   ```bash
   git add .
   git commit -m "deploy: <项目名> P-YYYYMMDD-XXX"
   git push -u origin deploy/<项目名>-<YYYYMMDD>
   ```

5. **Trigger deployment:**
   - **GitHub Pages**: Push to `gh-pages` branch directly or trigger GitHub Actions
   - **Cloudflare Pages**: Use `wrangler pages deploy` or connect via Cloudflare dashboard

6. **Verify deployment:**
   - Check the deployed URL is accessible
   - Verify key functionality works on the live site

7. **Update proposal:**
   - Set status to `deployed`
   - Record `Deployment URL` and `Deployment Branch` in proposal-index.md
   - Report final delivery URL to requester

8. **Sync to proposals-manager website + hermes-agent:** After updating proposals.json on GitHub, also sync changes to hermes-agent repo:
   ```bash
   # 在 hermes-agent 仓库创建 feature 分支并推送
   cd /home/hermes/.hermes
   git checkout -b feature/hermes$(date +%y%m%d)
   git add proposals/
   git commit -m "sync: update proposals from hermes-agent $(date +%Y-%m-%d)"
   git push -u https://YeLuo45:${GITHUB_TOKEN}@github.com/YeLuo45/hermes-agent.git feature/hermes$(date +%y%m%d)
   ```
   - 分支命名格式：`feature/hermesYYMMDD`（如 `feature/hermes260505`）
   - 推送目标：`https://github.com/YeLuo45/hermes-agent`
   - 只推送 `proposals/` 目录的变更，不碰其他源码

### Step 11: Deployment (Post-Acceptance Delivery)
   - Use `proposal-sync-website` skill to update `data/proposals.json` in `YeLuo45/prj-proposals-manager`
   - After sync, rebuild the website: download updated proposals.json from GitHub API to `public/data/proposals.json`, then `npm run build && gh-pages deploy`
   - This triggers GitHub Actions rebuild to publish the updated status on https://yeluo45.github.io/prj-proposals-manager/

## Dev Delivery Quality Checks

Three hard indicators to verify before accepting:

1. **Build exit code**: must be 0
2. **Output directory not empty**: list core files to confirm
3. **Core source/service files exist**: verify key files are present

If dev claims completion without providing evidence, run the checks yourself.

### Takeover Triggers

The coordinator should take over from dev when:
- Dev fails delivery twice consecutively
- Dev session interrupted by API/quota errors
- Dev session abnormally short (< 30s) yet claims completion
- Fix is simple and clearly identified

### Recording Fixes

When the coordinator directly fixes issues, record in:
1. Project memory file (e.g. `MEMORY.md`) under relevant section
2. Daily log (e.g. `memory/YYYY-MM-DD.md`)
3. Proposal's `Notes` or `Main Fixes Applied` field

## Index Entry Template

When adding to `proposal-index.md`:

```markdown
### P-YYYYMMDD-XXX: <Title>

- `Proposal ID`: `P-YYYYMMDD-XXX`
- `Title`: <title>
- `Owner`: <coordinator>
- `Current Status`: <status>
- `PRD Path`: (to be filled by PM)
- `Technical Solution`: (to be filled)
- `Test Cases Path`: (to be filled by Test Expert)
- `Project Path`: (to be filled by dev)
- `Acceptance`: -
- `PRD Confirmation`: pending
- `PRD Confirmation Countdown ID`: -
- `Technical Expectations`: pending
- `Technical Expectations Countdown ID`: -
- `Research Direction`: pending
- `Research Direction Countdown ID`: -
- `Deployment URL`: (to be filled after deployment)
- `Deployment Branch`: (to be filled after deployment)
- `Last Update`: YYYY-MM-DD
- `Notes`:
```

## Important Notes

### Path Discovery

#### Hermes Environment
- `~/.hermes/proposals/` is the actual proposals root — not `~/proposals/` (which may be empty or contain unrelated content)
- The main index file is `proposal-docs-index.md` (not `proposal-index.md`)

#### OpenClaw Environment (Windows/WSL)
- Proposals root: `~/.openclaw/workspace/proposals/` (not `~/.hermes/proposals/`)
- The main index file is `proposal-index.md` (not `proposal-docs-index.md`)
- PM output: `~/.openclaw/workspace-pm/proposals/`
- Dev output: `~/.openclaw/workspace-dev/proposals/`
- **WSL path translation**: WSL home `/root/` maps to Windows `C:\Users\<windows_username>\`
  - Example: `~/.openclaw/workspace/proposals/proposal-index.md` from WSL = `C:\Users\<username>\.openclaw\workspace\proposals\proposal-index.md` in Windows
  - Use `/mnt/c/Users/<username>/` prefix when accessing Windows filesystem from WSL
  - Windows username can be found via `ls /mnt/c/Users/`

#### General
- When cron jobs reference proposals but files aren't found at expected paths, check both `~/.hermes/proposals/` and `~/.openclaw/workspace/proposals/`
- Git commit messages sometimes reference proposal IDs (e.g., `P-20260430-005` in commit message) — use `git log --all --oneline | grep P-YYYYMMDD` to find related commits when navigating
- **WSL filesystem search performance**: `find` commands on `/mnt/c/` with large directories can hang (>60s timeout). Use targeted `ls` + `grep` via `search_files` tool instead of broad `find` to avoid hangs.

### Handling Duplicate Cron Timeout Events

When processing a cron timeout event:
1. **Always check proposal-index.md first** to see if the state was already updated by a previous identical cron event
2. If `PRD Confirmation` or `Technical Expectations` already shows `timeout-approved` or the expected next state, do NOT update again — just record the duplicate timestamp
3. The same cron event can arrive multiple times (e.g., system event redelivery); idempotency is critical
4. Before making any state changes, verify the current state matches what the cron expects to change

## Templates

This skill expects three templates in `${TEMPLATES_DIR}/`:

| Template | Purpose |
|----------|---------|
| `request-intake-template.md` | Initial proposal registration with clarification fields and confirmation gates |
| `proposal-status-template.md` | Status tracking with linked assets, confirmation gates, and revision notes |
| `acceptance-checklist-template.md` | Structured acceptance review with functional/quality/delivery checks |

If templates do not exist at the expected path, create them based on the index entry template above and the acceptance review checklist in Step 8.
