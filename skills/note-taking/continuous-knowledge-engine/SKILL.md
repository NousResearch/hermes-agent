---
name: continuous-knowledge-engine
description: Build and operate 24/7 automated knowledge management systems that gather from Discord, YouTube, academic sources, version everything in Git, sync to Obsidian, and report progress via messaging platforms.
triggers:
  - User wants continuous automated learning
  - Need to aggregate knowledge from multiple sources continuously
  - User wants background learning that doesn't sleep
  - Need to version and track learning progress over time
version: "1.0.0"

---

# Continuous Knowledge Engine

Build 24/7 automated learning systems that never sleep — "50 years in five".

## 🎯 Core Philosophy

The AI works while you sleep. It gathers knowledge from Discord, YouTube, university materials, versions everything in Git, syncs to Obsidian, and reports progress via Telegram/Discord DM.

**Motto:** I don't sleep, only you need to.

---

## 📐 Architecture

```
INPUTS                    PROCESSING                    STORAGE
• Discord threads      →  Transcripts → Summaries  →  Git repo
• YouTube vídeos       →  Extract patterns          →  Obsidian vault
• Faculdade materias   →  Create memórias           →  Progress logs
• Seus prompts         →  Generate skills           →  Reports

24/7 AUTOMATION:
• Cronjobs scheduled at specific hours
• Background processes running continuously
• Auto-commit to Git with timestamps
• Progress reports → Telegram/Discord DM
```

---

## 🗂️ Directory Structure

```bash
# Source Data (where cronjobs save)
/home/ubuntu/knowledge-base/
├── discord/
│   └── threads/              # Raw Discord thread JSONs
├── youtube/
│   └── transcripts/          # YouTube video transcripts
├── youtube-learning/
│   ├── data/                 # scan-state.json, youtube-subscriptions.json
│   └── insights/             # scan-*.json with extracted patterns
├── faculdade/
│   └── materias/             # Academic materials
├── <domain>/                 # Domain-specific learning (e.g., empreendedorismo/)
│   ├── mindset/              # Subdomain categories
│   ├── startup/
│   ├── growth/
│   ├── scale/
│   ├── personal-development/
│   ├── hermes-resources/
│   ├── insights/
│   ├── README.md             # Domain guide
│   ├── STATUS.md             # Daily status
│   └── SETUP.md              # Setup instructions
├── scripts/
│   ├── knowledge-gatherer.py      # Main gathering script
│   ├── progress-reporter.py        # Telegram/Discord reporting
│   ├── learning-agent.py           # Domain-specific learning agent
│   ├── test-learning-system.py    # Test suite
│   └── git-auto-commit.sh          # Git automation
├── logs/                     # Daily logs
└── README.md                 # Documentation

# Vault (versioned data, synced from sources)
/home/ubuntu/dev/workspace/Personal-Vault-backup/
├── 02-Literature/
│   ├── YouTube/
│   │   ├── data/             # From youtube-learning/data/
│   │   ├── insights/         # From youtube-learning/insights/
│   │   └── subscriptions.json
│   └── 02-Discord/
│       └── discord-spy-db.json
└── .git/                     # Git history

# Hermes Memories (processed Discord patterns)
~/.hermes/memories/
└── discord-spy-db.json       # Discord spy patterns (292KB)
```

---

## 📅 Schedule Template

### General Knowledge Base

| Hour | Action | Platform |
|------|--------|----------|
| 06:00 | Morning wake-up, yesterday summary | Telegram |
| 08:00 | Discord scan + process | Discord DM |
| 10:00 | YouTube scan channels | Telegram |
| 12:00 | Git commit + report | Discord DM |
| 14:00 | Process new YouTube videos | Telegram |
| 16:00 | Progress report (📈 knowledge gained) | Telegram |
| 18:00 | Discord scan #2 | Discord DM |
| 20:00 | Obsidian sync | Telegram |
| 22:00 | Night report + good night | Discord DM |

### Domain-Specific Learning (e.g., Entrepreneurship)

| Hour | Job | Foco | Deliverable |
|------|-----|------|-------------|
| 02:00 | Discord Learning Scan | Discord patterns | Local |
| 03:06 | Domain YouTube Learning | YouTube videos | Local |
| 06:00 | Knowledge Morning Report | Daily summary | Local |
| 07:00 | Daily Briefing | Docker + repos | Telegram |
| 08:00-20:00 | Domain YouTube Learning | Every 2h | Local |
| 20:00 | Hermes Resource Monitor | New features | Local |
| 22:00 | Knowledge Night Report | Night summary | Local |

---

## 🎯 Domain-Specific Learning Engines

For focused learning on a domain (e.g., entrepreneurship, ML ops, marketing), create a dedicated subdirectory:

### Setup Pattern

```bash
# Create domain structure
mkdir -p /home/ubuntu/knowledge-base/<domain>/{mindset,startup,growth,scale,personal-development,hermes-resources,insights}

# Create domain README
write_file path="/home/ubuntu/knowledge-base/<domain>/README.md" content="# <Domain> Learning

## Focus Areas

1. **Subdomain 1** - Description
2. **Subdomain 2** - Description

## YouTube Channels

### Local (BR)
- Channel 1 - Focus
- Channel 2 - Focus

### International
- Channel 1 - Focus
- Channel 2 - Focus

## Learning Schedule

[Schedule table]

## Metrics

- Videos processed: 3/day
- Insights extracted: 15/week
- Skills generated: 2/month
- Action items: 5/week"
```

### Domain Learning Agent

**File:** `scripts/learning-agent.py`

**Key components:**
- Domain-specific YouTube channel list
- Insight extraction (keywords: "como", "fazer", "passo", "estratégia")
- Auto-skill generation trigger (5+ similar patterns)
- Hermes resource monitoring (for meta-learning)
- Obsidian note generation with frontmatter
- Git auto-commit

**Pattern extraction:**
```python
# Look for action-oriented phrases
action_phrases = ["como", "fazer", "passo", "primeiro", "segundo", "terceiro",
                  "importante", "foco", "estratégia", "funciona"]

# Extract key takeaways (first 50 sentences, filtered by action phrases)
for sentence in sentences[:50]:
    if any(phrase in sentence.lower() for phrase in action_phrases):
        if 50 < len(sentence) < 200:  # Length filter
            insights["key_takeaways"].append(sentence)
```

### Auto-Skill Generation Workflow

**Trigger:** 5+ similar patterns detected in domain learning

**Workflow:**
```
Pattern Detection (5+ occurrences)
         ↓
Categorize pattern (e.g., "validation", "funding", "hiring")
         ↓
Draft skill name: <domain>-<pattern>
         ↓
Compile steps from multiple sources
         ↓
Generate skill draft in /home/ubuntu/.hermes/skills/
         ↓
Add to STATUS.md: "Skill suggested: <domain>-<pattern>"
         ↓
User review → Approve → skill_manage(action='create')
```

**Example:**
```yaml
Pattern detected in 5+ videos about startup validation:
→ Create skill: startup-idea-validation
→ Include: steps from multiple sources
→ Teste: apply to idea atual
```

### Hermes Resource Monitoring (Meta-Learning)

**Purpose:** Keep learning system updated with new Hermes capabilities

**Schedule:** Daily at 20:00

**Workflow:**
```python
def check_hermes_resources():
    # List all available skills
    result = subprocess.run(['hermes', 'skills', 'list'], capture_output=True, text=True)

    # Parse output
    skills = parse_hermes_output(result.stdout)

    # Identify new/updated skills
    new_skills = compare_with_last_check(skills)

    # Document useful patterns
    for skill in new_skills:
        if is_relevant_to_domain(skill):
            add_to_hermes_resources_note(skill)
```

**Output:** `/home/ubuntu/knowledge-base/<domain>/hermes-resources/new-skills.md`

**Example note:**
```yaml
---
type: hermes-resources
updated: 2026-06-03T20:00:00
tags: [hermes, automation, learning]
---

# Hermes Resources

## Última Atualização
2026-06-03 20:00:00

## Novas Skills Encontradas

### skill-name
Description of what it does and how it applies to <domain>
```

### Domain-Specific Cronjobs

**Learning job:**
```python
cronjob(action='create',
         name='<Domain> Learning - YouTube',
         schedule='every 2h',
         deliver='local',
         prompt='You are the <Domain> Learning Agent. Your mission: scan YouTube videos about <domain>, extract key insights, save to Obsidian, and generate actionable takeaways.

# Focus Areas:
1. Subdomain 1
2. Subdomain 2
3. Subdomain 3

# Workflow:
1. Check /home/ubuntu/knowledge-base/<domain>/README.md for channel list
2. Fetch recent videos (via youtube-content skill or manual URL list)
3. Extract transcripts
4. Identify patterns, key takeaways, action items
5. Save to /home/ubuntu/knowledge-base/<domain>/<category>/
6. Generate Obsidian notes
7. If 5+ similar patterns found → suggest new skill creation

# Always report to user (Telegram: @user) with:
- Videos processed today
- Key insights found
- New patterns identified
- Skills suggested')
```

**Hermes resource monitor:**
```python
cronjob(action='create',
         name='Hermes Resource Monitor',
         schedule='0 20 * * *',
         deliver='local',
         prompt='You are the Hermes Resource Monitor. Mission: keep @user updated on new Hermes features, MCP servers, and skills.

# Workflow:
1. List all available skills: `hermes skills list`
2. Check for new MCP servers in ~/.hermes/config.yaml
3. Identify new tools or features
4. Document useful patterns in /home/ubuntu/knowledge-base/<domain>/hermes-resources/
5. Create actionable tips for current work

# Focus Areas:
- New MCP servers that could help with <domain>
- Skills that improve productivity or learning
- Patterns for automation and scaling
- Configuration best practices')
```

### Test Suite

**File:** `scripts/test-learning-system.py`

**Purpose:** Validate learning system components before deployment

**Tests:**
```python
def test_youtube_skill():
    """Test if youtube-content skill is working"""
    from hermes_tools import skill_view
    result = skill_view(name="skills/media/youtube-content")
    return result["success"]

def test_obsidian_integration():
    """Test if Obsidian is accessible"""
    obsidian_path = Path.home() / ".obsidian/Obsidian"
    return obsidian_path.exists()

def test_knowledge_base():
    """Test knowledge base structure"""
    kb_path = Path("/home/ubuntu/knowledge-base/<domain>")
    required_dirs = ["mindset", "startup", "growth", "scale", "personal-development", "hermes-resources", "insights"]
    return all((kb_path / dir_name).exists() for dir_name in required_dirs)

def test_learning_script():
    """Test learning agent script"""
    script_path = Path("/home/ubuntu/knowledge-base/scripts/learning-agent.py")
    return script_path.exists()
```

**Run:** `python3 /home/ubuntu/knowledge-base/scripts/test-learning-system.py`

**Note:** Test scripts that use `hermes_tools` module should run via `cronjob` tool with proper skill loading, not direct Python execution in terminal context.

---

## 🚀 Step-by-Step Implementation

## Cron Log Convention

- Standardize log file naming to include cron name and timestamp for easy daily rotation:
  ```
  /home/ubuntu/knowledge-base/logs/cron-<name>.log
  /home/ubuntu/knowledge-base/logs/cron-<name>-<YYYYMMDD>.log
  ```
- Example in this session:
  `/home/ubuntu/knowledge-base/logs/cron-gatherer.log`
  ```bash
  crontab entry (every 2 hours pattern):
  0 8,10,12,14,16,18,20 * * * /usr/bin/python3 /home/ubuntu/knowledge-base/scripts/knowledge-gatherer.py >> /home/ubuntu/knowledge-base/logs/cron-gatherer.log 2>&1
  ```
- This pattern reduces log fragmentation and makes `tail -f /home/ubuntu/knowledge-base/logs/cron-*.log` reliable across any number of scheduled scripts.

---

### Step 1: Create Base Structure

```bash
mkdir -p /home/ubuntu/knowledge-base/{discord/threads,youtube/transcripts,faculdade/materias,obsidian/{00-Inbox,01-YouTube,02-Discord,03-Faculdade,04-Patterns},scripts,logs}
cd /home/ubuntu/knowledge-base
git init
```

### Step 2: Create Main Knowledge Gatherer Script

**File:** `scripts/knowledge-gatherer.py`

**Key components:**
- Fetch Discord content from existing discord-learning.json
- Generate Obsidian notes with proper frontmatter
- Extract patterns and key concepts
- Create Git commits with descriptive messages
- Generate progress reports

**See:** `references/knowledge-gatherer-template.md` in this skill

### Step 3: Create Progress Reporter Script

**File:** `scripts/progress-reporter.py`

**Key components:**
- Generate daily summaries
- Morning checks (06:00)
- Night reports (22:00)
- Progress metrics (threads, videos, notes, commits)
- Send via `hermes send telegram` command

**See:** `references/progress-reporter-template.md` in this skill

### Step 4: Setup Cronjobs via Hermes

Use the `cronjob` tool directly in the session:

```
cronjob action='create' name='knowledge-morning-report' schedule='0 6 * * *' deliver='local' prompt='Execute python3 /home/ubuntu/knowledge-base/scripts/progress-reporter.py morning and send via send_message target=telegram'
```

### Step 5: Configure Data Sources

**YouTube channels** (once provided by user):
```bash
mkdir -p /home/ubuntu/knowledge-base/config
nano /home/ubuntu/knowledge-base/config/youtube-channels.txt
# Format: CHANNEL_ID per line
```

**Faculdade materias** (once provided by user):
```bash
nano /home/ubuntu/knowledge-base/config/faculdade-materias.txt
# Format: MateriaName:/path/to/materials
```

**Discord** (uses existing discord-learning.json):
- Already scanning threads via `discord-integration` skill
- Patterns stored in `~/.hermes/memories/discord-learning.json`

### Step 6: Test the System

```bash
# Test main gatherer
python3 /home/ubuntu/knowledge-base/scripts/knowledge-gatherer.py

# Test progress reporter
python3 /home/ubuntu/knowledge-base/scripts/progress-reporter.py daily

# Verify Obsidian notes created
ls /home/ubuntu/knowledge-base/obsidian/02-Discord/

# Verify Git commits
cd /home/ubuntu/knowledge-base && git log --oneline
```

---

## 🔧 Obsidian Note Format

Every generated note includes:

```yaml
---
type: discord-thread  # or youtube-video, faculdade-materia
thread_id: 1510426965700513882
created: 2026-06-03T00:42:24.053285
tags: [discord, learning]
---

# Thread/Video Title

## 📊 Stats
- Mensagens: 38
- Último scan: 2026-06-03T00:32:59

## 🎯 Patterns

### Technical Questions (N)
**User** (date):
> Question or insight excerpt

### Workflows (N)
- **Workflow name** by user
```

---

## 📊 Metrics to Track

```bash
# Total notes
find /home/ubuntu/knowledge-base/obsidian -name "*.md" | wc -l

# Discord threads
find /home/ubuntu/knowledge-base/discord -name "*.json" | wc -l

# YouTube videos
find /home/ubuntu/knowledge-base/youtube -name "*.md" | wc -l

# Git commits
git log --oneline | wc -l

# Skills generated
ls ~/.hermes/skills/ | wc -l
```

---

## ⚠️ Pitfalls

### Discord Pattern Extraction
- **Issue:** Patterns stored in `discord-learning.json` but script might not find them
- **Fix:** Verify JSON structure before processing — check for `technical_patterns` key
- **Location:** This depends on `discord-integration` skill output format

### Git Auto-Commit Conflicts
- **Issue:** Multiple cronjobs might try to commit simultaneously
- **Fix:** Add locking mechanism or schedule jobs at non-overlapping times
- **Approach:** Use `git add .` then `git commit` — Git handles atomicity

### Obsidian Sync Overwrites
- **Issue:** Re-running gatherer overwrites existing notes
- **Fix:** Check if note exists and only update if content changed
- **Implementation:** Compare file hash or message count before writing

### YouTube Transcript API Limits
- **Issue:** YouTube API has rate limits for transcript downloads
- **Fix:** Batch downloads, respect rate limits, cache transcripts
- **Tool:** Use `youtube-content` skill which handles this

### Telegram Send Failures
- **Issue:** `hermes send telegram` might fail if chat_id not configured
- **Fix:** Verify Telegram config before scheduling jobs
- **Check:** `cat ~/.hermes/config.yaml | grep -A 5 "telegram:"`

### Domain Learning Test Failures
- **Issue:** Test scripts reference `hermes_tools` module which doesn't exist in terminal context
- **Fix:** Test scripts should run via `cronjob` tool with proper skill loading, not direct Python execution
- **Alternative:** Use subprocess with `hermes skills list` instead of importing non-existent modules

### Obsidian Symlink on Cloud VM
- **Issue:** Obsidian vault not on VM; needs symlink to user's desktop Obsidian installation
- **Fix:** Create local vault on VM (`/home/ubuntu/.obsidian/Obsidian/`) for immediate use
- **Desktop sync:** User creates symlink on their desktop, not on VM
- **Command (desktop, not VM):** `ln -s /home/ubuntu/.obsidian/Obsidian ~/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/YourVault`

### Auto-Skill Generation False Positives
- **Issue:** Transient patterns (e.g., same word in different contexts) trigger skill creation
- **Fix:** Require 5+ occurrences AND semantic similarity check before suggesting skills
- **Verification:** User review required before `skill_manage(action='create')` — never auto-create

### Video Processing Claims Must Be Verified
- **Issue:** Session summary may claim videos were "moved to video-processed/" or STATUS.md was "created/updated" but the actual file operations never happened. Summaries from previous sessions are NOT reliable proof of execution.
- **Fix:** After any video processing batch, ALWAYS verify with `ls` and `cat` that files are actually in the right place and STATUS.md exists with expected content. Never trust the previous session's summary as a substitute for filesystem verification.
- **Check commands:** `ls /home/ubuntu/knowledge-base/<domain>/video-processed/` and `cat /home/ubuntu/knowledge-base/<domain>/STATUS.md`

### No-Agent Cronjob Timeout on Heavy Commands
- **Issue:** `no_agent: true` cronjobs that run heavy test commands (e.g., `vitest run --coverage`) will timeout or hang. These scripts run in a sandboxed 5-min window with no adaptive retry.
- **Fix:** Only put bounded, deterministic commands in `no_agent` scripts. If a command routinely takes >30s, either (a) move it to an agent-driven cronjob that can handle timeouts, or (b) break it into smaller incremental steps. Never schedule `vitest --coverage`, full test suites, or heavy builds in frequent (hourly/every-2h) no-agent cronjobs.
- **Example:** The `continue-coverage.py` script for the project was hourly no-agent running `vitest run --coverage` — it hung every time. Paused it; should only run during active coverage sprints.

### Hermes Resource Monitoring Overhead
- **Issue:** Checking all skills daily might miss incremental updates
- **Fix:** Track last seen skill hash, only report changes
- **Optimization:** Focus on categories relevant to domain (e.g., skills/autonomous-ai-agents for entrepreneurship)

### Cronjob Data Not Auto-Synced to Vault
- **Issue:** Background cronjobs (YouTube Learning, Discord Spy, content monitors) save data to isolated directories (`~/knowledge-base/`, `~/.hermes/memories/`). The Obsidian Vault at `~/dev/workspace/Personal-Vault-backup/` does NOT auto-sync. User expects all learning data to be versioned.
- **Fix:** Create dedicated sync cronjobs per data source. Each sync job runs AFTER the source job, copies data, and git commits if changes exist.
- **Key patterns:**
  1. **Schedule sync AFTER source**: YouTube Learning Scan at 04:00 → Sync at 05:00; Discord Spy at 02:00 → Sync at 03:00
  2. **Conditional copy**: `cp -u` only copies if source is newer
  3. **Silent on no changes**: Exit code 0 when git status is clean prevents empty commits
  4. **Atomic git operations**: `git add -A` + `git commit` handles concurrent updates safely
  5. **Git push last**: Only push after successful commit
- **See:** `references/vault-sync-pattern.md` for full templates, working examples, and directory mapping
- **Redundant jobs:** Single generic sync job (e.g., `hermes-knowledge-sync`) is redundant once specific sync jobs exist. Remove it to avoid conflicts.

---

## 🔌 Integration Points

### Discord Integration
- **Prerequisite:** `discord-integration` skill must be configured
- **Input:** `~/.hermes/memories/discord-learning.json`
- **Output:** `/home/ubuntu/knowledge-base/discord/threads/*.json`

### Obsidian Integration
- **Prerequisite:** `obsidian` skill must be loaded
- **Input:** Processed Discord threads, YouTube transcripts
- **Output:** Notes in `/home/ubuntu/knowledge-base/obsidian/`

### Telegram Integration
- **Prerequisite:** Telegram enabled in Hermes config (ensure targets like `telegram` exist in ~/.hermes/config.yaml)
- **Input:** Progress reports from scripts
- **Syntax used in scripts:** `hermes send --to telegram "<body>"`
  - **⚠️ Critical:** Always provide `--to` followed by a target; omission causes CLI exit rc=2 with unrecognized-arguments.
- **Output:** Messages to configured chat

### Pitfalls / Discoveries
- Telegram Send Args Required:
  When invoking `hermes send --to <PLATFORM>`, ensure the target exactly matches configured platforms (e.g., `telegram`). Missing `--to` causes:
  ```
  hermes send: --to PLATFORM[:channel[:thread]] is required
  ```
  Workaround used in this session: explicit `--to telegram "body"`. Embed this as required syntax for all Telegram integrations going forward.

---
## 💡 Pro Tips
- **Scheduling:** Use `deliver='local'` to save output
- **Reporting:** Use `send_message` tool in prompts

---

## 🎯 Next Extensions

### Auto-Skill Generation
Detect recurring patterns (5+ mentions) → Generate skill draft → Queue for approval → Auto-create on approval

### Knowledge Graph
Link related notes across sources (Discord → YouTube → Faculdade) using Obsidian backlinks

### RAG Indexing
Create vector index of all notes for semantic search and retrieval-augmented generation

### Cross-Platform Sync
Sync Obsidian vault to Notion, Google Drive, or other platforms for mobile access

---

## 📚 References

- `references/knowledge-gatherer-template.md` — Working Python script (from a previous session)
- `references/vault-sync-pattern.md` — Templates and patterns for syncing cronjob data to Vault with git commit automation (from a previous session)
- `references/hermes-cronjob-usage.md` — Hermes cronjob tool reference
- `references/setup-checklist.md` — User-facing setup checklist
- `references/progress-reporter-template.md` — Progress reporter script template
- `references/hermes-send-pitfalls.md` — Guide for robust `hermes send telegram` usage in scripts

---

## 🏆 Success Criteria

- [ ] Discord threads automatically captured and processed
- [ ] Obsidian notes generated with proper frontmatter
- [ ] Git commits happen automatically with descriptive messages
- [ ] Morning/night reports sent via Telegram successfully
- [ ] System runs continuously without manual intervention
- [ ] Knowledge base grows measurably over time (track metrics)
- [ ] User can see progress in Telegram/Discord DM