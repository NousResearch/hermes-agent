---
name: automated-pattern-detection
description: "Automated detection of recurring error patterns and automatic generation of solution playbooks/skills when thresholds are met."
platforms: [linux, macos, windows]
---

# Automated Pattern Detection System

Detects when 5+ instances of the same error/block pattern appear across sessions or channels. When threshold is exceeded, auto-generates an **entrepreneurship-playbook** skill with validated fixes and references.

## When to use

Run this when:
1. You notice a user repeatedly hitting the same blocking error (IP block, missing dependency, etc.)
2. A tool consistently fails with the SAME error pattern across different invocations
3. You identify 3+ identical errors in session search results with no resolution
4. The user says "this keeps happening" about a specific failure mode
5. You discover a workaround that should be codified as a skill

**Trigger conditions:**
- 5+ occurrences of **exact same error string** with same context across timeline
- Errors span **3+ different channels/signals** (Telegram, commits, logs, user reports)
- There exists a **validated fix path** (tested solution, documented approach)

## Workflow

### Automatic Detection Script (run periodically)

```bash
/home/ubuntu/.hermes/skills/entrepreneurship-patterns/scripts/auto-detect-5plus.sh
```

Checks:
- Session search for exact error message matches
- Cron job logs for repeated failures
- Git commit messages and error patterns
- User reports via Telegram/Discord

### Validation Steps

1. **Count occurrences**: use session_search or grep across logs
2. **Verify fix exists**: has someone successfully worked around this?
3. **Extract context**: what are the common parameters/scope?
4. **Test solution**: ensure the workaround actually works before codifying
5. **Template skill**: generate skill under `entrepreneurship-patterns/` with:
   - Frontmatter + trigger conditions
   - Step-by-step fix instructions
   - References to original error sources
   - Academic fallback alternatives if applicable

### Output

Generates a new umbrella skill at:
```
~/.hermes/skills/entrepreneurship-patterns/<domain>-playbook/SKILL.md
```

Where `<domain>` is derived from error type:
- `ip-block` → `ip-block-cloud-workaround-playbook`
- `system-chrome-missing` → `system-setup-chrome-install-playbook`
- `transcript-ui-disabled` → `youtube-transcript-forcing-playbook`

## Structure of Generated Skills

Each playbook SKILL.md includes:

```markdown
---
name: <playbook-name>
description: "Reusable playbook for <error-problem> pattern"
trigger: "5+ occurrences of <error-pattern> in 7 days"
---

# <Playbook Title>

## Problem Statement
{concise description of the recurring issue}

## Trigger Conditions
- 5+ occurrences detected via session_search/auto-detect-5plus.sh
- Validate fix actually resolves the issue
- Extract parameters from failure logs

## Quick Fix (One-liner)
```bash
{validated fix command - copy/paste ready}
```

## Step-by-Step Resolution

1. **Detect**: {how to identify this specific pattern}
2. **Escalate**: {when to use this playbook vs continue retrying}
3. **Apply**: {exact commands/JS code to run}
4. **Verify**: {how to confirm fix worked}

## Common Parameters/Scope

- Video: `https://youtube.com/watch?v=VIDEO_ID`
- Service: Chrome DevTools via `mcp_chrome_devtools_*`
- Commands: hermes CLI or shell integration

## Pitfalls & Gotchas

- **Internet-dependent**: works on Oracle Cloud but fails on restricted networks → see academic alternatives
- **Version drift**: Chrome path changes across Linux distros
- **Login state**: must use authenticated Chrome via `mcp_chrome_devtools`
- **Timeouts**: 60s timeout ceiling for MCP tools in current config

## Academic Alternatives

When fix attempts fail 3+ times:
- `entrepreneurship/academic-fallback-framework` (auto-included)
- arXiv research search
- Semantic Scholar API fallback
- YouTube course materials from universities

## Verification Script

Each playbook includes `scripts/validate-pattern.sh` that:
1. Checks if original error still occurs
2. Verifies fix works in current environment
3. Returns success/failure status for automation

### Example validator

```bash
#!/bin/bash
set -euo pipefail

# Check if youtube-transcript-api is blocked
if timeout 10 python3 -c "from youtube_transcript_api import YouTubeTranscriptApi" 2>/dev/null; then
  echo "API_ACCESSIBLE" && exit 0
fi

# Check if Chrome DevTools can access video
if timeout 15 hermes mcp_chrome_devtools_evaluate_script \
   function="() => { return 'CHROME_WORKING' }" \
   2>/dev/null | grep -q "CHROME_WORKING"; then
  echo "CHROME_WORKING" && exit 0
fi

echo "NEEDS_ESCALATION" && exit 1
```

## References Directory Structure

```
references/
├── error-matrix.md                # Decision table of patterns → playbooks
├── session-2026-06-04-issue-*.md  # Raw session transcripts containing errors
├── commands-tested.md             # Commands that successfully fixed the issue
├── vendor-docs/                   # Official docs for tools being used
│   ├── chrome-devtools.md
│   ├── youtube-transcript-api.md
│   └── mcp-protocol.md
└── bibliography.md               # Academic/research citations for fallbacks
```

## Integration with Existing Skills

1. **youtube-content**: adds `references/block-patterns-db.md` linking to pattern db
2. **entrepreneurship-patterns**: umbrella skill that ALL playbooks hang off
3. **obsidian**: receives new notes at location determined by vault path
4. **cronjob**: runs `auto-detect-5plus.sh` every 6 hours to catch patterns

## Threshold Matrix

| Pattern Type | Threshold | Scope | Fallback |
|-------------|-----------|-------|----------|
| IP Block (Cloud) | 5 occurrences | Service-wide | Academic research |
| Transcript Disabled UI | 5 occurrences | Video extraction | Manual download |
| System Dependency Missing | 5 occurrences | Fresh install | Workaround available |
| User-reported repeat error | 3 mentions | Channel-specific | Investigate |

## Example: IP Block Pattern Detection

**Error**: `RequestBlocked` / `403 Forbidden` from YouTube API
**User impact**: All video transcript extraction fails
**Root cause**: Cloud IPs blocked for upload API scope
**Detection**: Session search + grep logs
**Fix validated**: Chrome DevTools authenticated session with JS transcript extraction
**Playbook created**: `entrepreneurship-patterns/ip-block-cloud-workaround-playbook/SKILL.md`

When this playbook is referenced via `session_search` or `skill_view`, future agents automatically know:
- Alternative path when API fails
- Prefer `mcp_chrome_devtools` over `browser_navigate`
- Use JS to force transcript button click
- Validate fix with `scripts/validate-pattern.sh`

## Updating Detection Logic

Edit `scripts/auto-detect-5plus.sh` in this skill when:

1. New failure patterns emerge
2. Detection needs adjustments (e.g., watch different log files)
3. Threshold rules change (e.g., now require 7 occurrences)
4. Integration with new tools (e.g., Prometheus monitoring)

## Skill Discovery

To find relevant playbooks for a current error:

```bash
skill_view(name="entrepreneurship-patterns")
# Skim for playbook matching error type

session_search(query="RequestBlocked OR 403 Forbidden" limit=5)
# Returns sessions with the error for context mapping
```

## Maintenance

- Monitor detection false-positive rate monthly
- Remove playbooks when underlying issue is fixed
- Update references/ when new solutions emerge
- Increment skill version with each new playbook generated