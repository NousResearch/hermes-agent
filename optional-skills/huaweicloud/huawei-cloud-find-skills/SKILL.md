---
name: huawei-cloud-find-skills
description: |
  Invoke this skill to search, discover, browse, find and install any Huawei Cloud (华为云) agent skill.Triggers include: "华为云","华为云有什么skill","华为云相关skill","华为云agent skill 市场","华为云skill类目","explore Huawei Cloud skills","show Huawei Cloud skill categories","does a Huawei Cloud skill exist for...","which Huawei Cloud skills exist","搜索华为云技能","有没有管理ECS/OBS/RDS的skill","帮我找 XX 华为云skill","介绍 XX Skill 内容","华为云 XX Skill 具体做什么","安装华为云Skill".
---

> [!IMPORTANT]
> **For any Huawei Cloud query or management task:**
>
> 1. **Search** — use this skill (`huawei-cloud-find-skills`) to find the relevant Skill.
> 2. **Install** — install the matched Skill (see [Step 3](#step-3-install-skill)).
> 3. **Execute** — follow the installed Skill's instructions to fulfill the request.

# Huawei Cloud Agent Skills Search and Discovery

This skill enables users to efficiently search, discover, and install Huawei Cloud skills.

## Scenario Description

This skill enables users to:

- **Search Skills**: Find skills by keyword, category, or both (matched against name, description, and triggers)
- **Browse Categories**: Explore available skill categories
- **View Skill Details**: Fetch full SKILL.md content from GitHub for specific skills
- **Install Skills**: Guide users through skill installation via `npx skills add` (GitCode default), `npx clawhub install`, or fallback GitHub method

**Architecture**: GitCode API v5 (`index.json` + `cn-en-map.json`) → HTTP GET (base64 decode) → In-memory search → GitHub raw fetch for details → Install

### Use Cases

- "Find a skill for managing ECS instances"
- "What Huawei Cloud skills are available for OBS?"
- "华为云有哪些 VPC 相关的 skill?"
- "Browse all available Huawei Cloud skills"
- "Install a skill for RDS management"
- "帮我找一个华为云网络相关的skill"

## Prerequisites

- **Node.js >= 22** must be installed
- **Network access** to `gitcode.com` (API v5 for index) and `github.com` / `raw.githubusercontent.com` (for skill details)

### Step 0: Check Python Environment

> **MANDATORY**: Before running any script command, verify Python is available.

```bash
# Check Python availability
python --version   # or: python3 --version
```

If the command fails or returns Python 2.x:

1. **Install Python 3**: Download from [python.org](https://www.python.org/downloads/) or use a package manager:
   ```bash
   # macOS
   brew install python3
   # Ubuntu/Debian
   sudo apt-get install python3
   # Windows — download installer from python.org, check "Add Python to PATH"
   ```
2. **Verify after install**: Run `python --version` again to confirm Python 3.6+ is available
3. **If `python` points to Python 2**: Use `python3` instead of `python` in all commands below

## Repository Info

```
INDEX_REPO=2501_91318609/skills-for-index
INDEX_BRANCH=main
SKILLS_REPO=huaweicloud/huaweicloud-skills
SKILLS_BRANCH=master
RAW_BASE=https://raw.githubusercontent.com/$SKILLS_REPO/$SKILLS_BRANCH
```

## Index Source

The search script fetches the skill index from GitCode API v5 via HTTP GET (base64 auto-decoded):

```
SKILLS_INDEX_URL=https://gitcode.com/api/v5/repos/2501_91318609/skills-for-index/contents/skills-index/index.json?ref=main
SKILLS_CN_EN_MAP_URL=https://gitcode.com/api/v5/repos/2501_91318609/skills-for-index/contents/skills-index/cn-en-map.json?ref=main
```

## Core Workflow

### Step 1: Search Skills

> **MANDATORY**: The agent MUST execute the search script to search the skill index. Do NOT read the JSON file directly — always use the script.

Given `keyword` (from AI-understood user intent) and optional `category`, run the search script:

```bash
node scripts/search-skills.mjs <keyword>
node scripts/search-skills.mjs <keyword> -c <category>
node scripts/search-skills.mjs -c <category>
```

→ [scripts/search-skills.mjs](scripts/search-skills.mjs) (Node.js — cross-platform)

**What the script does**:

1. Fetches `index.json` and `cn-en-map.json` via HTTP GET from GitCode API v5 (auto-decodes base64 content)
2. Expands keywords via `cn-en-map.json` (bidirectional CN↔EN, e.g., "ECS" → "ECS, 弹性云服务器, 云服务器")
3. Scores each skill: name match **+10**, trigger match **+8**, description match **+5**, service match **+3**
4. Sorts by score descending, outputs formatted results with matched keywords

**Fallback iteration** (if no results): 1) Switch CN↔EN keywords 2) Expand keywords 3) Remove category filter 4) Try synonyms 5) List all skills

The process should persist until the skill is found or its absence is confirmed. In the event of total failure, notify the user of the specific steps that were attempted.

### Step 2: View Skill Details (optional)

Fetch the full SKILL.md content from GitHub for intent validation. Skip this step if the search results from Step 1 are sufficiently informative.

```bash
# URL pattern — use the skill's category, service, and name from index
DETAIL_URL="https://raw.githubusercontent.com/huaweicloud/huaweicloud-skills/master/skills/${category}/${service}/${name}/SKILL.md"
```

The agent can fetch this URL using `curl` or its web-fetch tool, then present the skill's full documentation to the user.

### Step 3: Install Skill

> **MANDATORY**: Use one of the commands below. Option A is the default; Option C is a fallback when Option A is unavailable.

```bash
# Option A: npx skills add from GitCode (default)
npx skills add https://gitcode.com/huaweicloud/huaweicloud-skills.git#master --skill <skill-name> -y

# Option B: npx clawhub install (OpenClaw ecosystem)
npx clawhub install <skill-name> -y

# Option C (fallback): npx skills add from GitHub
npx skills add huaweicloud/huaweicloud-skills --skill <skill-name> -y
```

If all installation attempts fail, report the error message to the user. Do NOT attempt any method outside the commands above.

## Parameters

| Parameter    | Required/Optional | Description                                                           | Default |
| ------------ | ----------------- | --------------------------------------------------------------------- | ------- |
| `Keyword`    | Optional          | Search keyword (matched against name, description, triggers, service) | None    |
| `Category`   | Optional          | Category code for filtering (e.g., "computing", "storage", "network") | None    |
| `skill-name` | Required (Step 3) | Exact skill name for installing                                       | None    |

## Reference Documentation

| Document                                             | Description                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- |
| GitCode API v5 `index.json`                          | Skill index fetched via HTTP GET (base64 decoded)                                     |
| GitCode API v5 `cn-en-map.json`                      | Chinese-English keyword mapping fetched via HTTP GET (base64 decoded)                 |
| [scripts/search-skills.py](scripts/search-skills.py) | Search script (Python) — fetches from GitCode API v5, expands keywords, scores, sorts |

## Search Heuristics

> Optional reference for keyword-to-category hints. The agent can infer categories from `index.json` without these.

- Cloud infrastructure keywords (ecs, bms, vpc, obs, rds, ...) → likely `computing`, `network`, `storage`, etc.
- Tool keywords (cli, terraform, koo) → likely `devtools`
- Management keywords (monitoring, alarm, log) → likely `monitoring`

## Troubleshooting

### Issue: Script fails with "Failed to fetch index.json"

**Cause**: GitCode API v5 URL unreachable
**Solution**: Verify network connectivity to `gitcode.com`

### Issue: GitCode API v5 returns 404

**Cause**: File path incorrect or default branch is not `main`
**Solution**: Verify the skill's `category`, `service`, and `name` from search results

### Issue: Search returns no results

**Cause**: Keywords don't match any skill
**Solution**:

1. Try broader keywords
2. Switch between Chinese and English keywords (e.g., "对象存储" → "obs")
3. List all skills: `python scripts/search-skills.py -c "computing"`

## Notes

- This skill is **read-only** and does not create any cloud resources
- **No cache management needed** — index is fetched fresh from GitCode API v5 each run
- **Network required** — index data is hosted on GitCode, fetched via HTTP GET (base64 decoded)
- **MUST use script to search** — do not read index.json directly
- Index repo: `https://gitcode.com/2501_91318609/skills-for-index` (branch: `main`)
- Skills repo: `https://github.com/huaweicloud/huaweicloud-skills` (branch: `master`)
