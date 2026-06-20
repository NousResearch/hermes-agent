---
name: skill-auditor
description: "Audit Hermes skills for quality, structure, and freshness."
version: 1.0.0
author: GUOHAO LIU (lennney)
license: MIT
metadata:
  hermes:
    tags: [Software Development, Skills, Quality, Audit, Maintenance]
    related_skills: [hermes-agent]
---

# Skill Auditor

Analyze installed Hermes skills for quality, structure compliance, and staleness. Produces actionable reports with specific fix suggestions.

## When to Use

- Before publishing or sharing skills with others
- After creating a new skill to verify it meets standards
- Periodically to find stale or low-quality skills
- When curating skills for a specific project or team

## Prerequisites

- Access to `~/.hermes/skills/` directory
- Python 3.11+ (for the audit script)

## How to Run

```bash
# Audit all installed skills
terminal(command="python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py", timeout=30)

# Audit a specific skill directory
terminal(command="python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py --path ~/.hermes/skills/my-skill", timeout=15)

# Output as JSON for programmatic use
terminal(command="python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py --json", timeout=30)
```

## Quick Reference

| Metric | What it checks | Pass criteria |
|--------|---------------|---------------|
| **description** | Length and format | ≤ 60 chars, ends with `.` |
| **sections** | Required sections present | All 7 modern sections exist |
| **tools** | References Hermes tools correctly | No raw `grep`/`cat`/`sed` |
| **size** | Line count | 50–300 lines |
| **frontmatter** | Required fields | name, description, version, author |
| **pitfalls** | Has Pitfalls section | Non-empty Pitfalls section |
| **scripts** | Helper scripts exist | Referenced scripts are present |

## Procedure

### Step 1: Run the audit script

```bash
python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py
```

The script outputs a table with per-skill scores and issues.

### Step 2: Review the report

Each skill gets a score from 0–100:

| Score | Meaning |
|-------|---------|
| 90–100 | Production ready |
| 70–89 | Good, minor fixes needed |
| 50–69 | Needs work, several issues |
| < 50 | Major problems, rewrite recommended |

### Step 3: Fix issues

Common fixes in priority order:

1. **Missing sections** — Add the 7 modern sections (When to Use, Prerequisites, How to Run, Quick Reference, Procedure, Pitfalls, Verification)
2. **Description too long** — Shorten to ≤ 60 chars, one sentence, ends with `.`
3. **Wrong tool references** — Replace `grep` → `search_files`, `cat` → `read_file`, `sed` → `patch`, `curl` → `web_extract`
4. **No Pitfalls section** — Add known failure modes and workarounds
5. **Missing scripts** — Create helper scripts for non-trivial logic

### Step 4: Re-audit

```bash
python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py --path ~/.hermes/skills/my-skill
```

## Pitfalls

- The script checks structure, not content quality — a skill can score 90+ but still have bad instructions
- `platforms:` frontmatter must match actual code imports — the script flags this but can't verify runtime behavior
- Skills in `optional-skills/` are not audited by default — pass `--path` explicitly
- The audit script reads files directly; it does not load skills into the agent

## Verification

```bash
# Should produce a report with 0 errors for a well-written skill
python3 ~/.hermes/skills/software-development/skill-auditor/scripts/audit_skills.py --path ~/.hermes/skills/software-development/skill-auditor
```

Expected: score ≥ 90, no structural errors.
