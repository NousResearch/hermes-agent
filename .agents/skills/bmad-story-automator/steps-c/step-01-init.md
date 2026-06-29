---
name: 'step-01-init'
description: 'Check for existing state and route appropriately'
nextStep: './step-02-preflight.md'
continueStep: './step-01b-continue.md'
outputFolder: '{output_folder}/story-automator'
outputFile: '{outputFolder}/init-log-{timestamp}.md'
rules: '../data/orchestrator-rules.md'
scripts: '../scripts/story-automator'
ensureStopHook: '../scripts/story-automator'
stateHelper: '../scripts/story-automator'
settingsFile: '{project-root}/.claude/settings.json'
---

# Step 1: Initialize

**Goal:** Verify safeguards, check for existing state → resume or start fresh.

---

## Do

### 1. Verify Stop Hook Installation

**CRITICAL:** The Stop hook prevents premature stopping during orchestration.

Use script to ensure the Stop hook exists:
```bash
result=$("{ensureStopHook}" ensure-stop-hook --settings "{settingsFile}" \
  --command "{scripts} stop-hook" --timeout 10)
ok=$(echo "$result" | jq -r '.ok')
changed=$(echo "$result" | jq -r '.changed')
verification_state=$(echo "$result" | jq -r '.verificationState // "verified"')
message=$(echo "$result" | jq -r '.message // ""') # Helper returns provider-specific restart/setup guidance for Claude or Codex.
```
The settings path is used for Claude; Codex resolves `.codex/hooks.json` and `.codex/config.toml` from the project root.

**IF ok == false:** Report error and STOP.

**IF changed == true:**
Display:
```
**Stop Hook Installed**

<message from helper>

This prevents the orchestrator from randomly stopping mid-workflow.

⚠️ **Please restart this active agent session** for the hook to take effect.

After restarting, run the story-automator workflow again.
```
**HALT** - Do not proceed until user restarts

**IF verification_state == "pending_trust":**
Display:
```
**Stop Hook Pending Codex Trust**

<message from helper>

Trust this project in Codex, then restart Codex and run the story-automator workflow again.
```
**HALT** - Do not proceed until Codex can run the hook

**IF changed == false:**
Display: "✓ Stop hook verified"
Continue to step 2

### 2. Load Rules
Load `{rules}` once. These apply to all subsequent steps.

### 3. Check for Existing State
Search `{outputFolder}` for `orchestration-*.md` files.

Use deterministic state listing:
```bash
state_list=$("{stateHelper}" orchestrator-helper state-list "{outputFolder}")
latest_incomplete=$(echo "$state_list" | jq -r '.files | map(select(.status == "COMPLETE" | not)) | sort_by(.lastUpdated) | last | .path // empty')
```

**IF latest_incomplete is non-empty:**
- Display: "**Found existing orchestration in progress.**"
- Show: epic name, current story, current step, last updated
- → Load `{continueStep}`
- **STOP** (don't continue below)

**IF none found:**
- Continue to step 4

### 4. Welcome
Display:
```
**Welcome to Story Automator.**

I'll automate story implementation by spawning isolated sessions,
handling code review loops, and committing completed stories.

Everything is logged for full resumability.
```

### 5. Check Sprint Status (MANDATORY)
```bash
has_status=$("{stateHelper}" orchestrator-helper sprint-status exists)
sprint_ok=$(echo "$has_status" | jq -r '.exists')
```

**IF sprint_ok == false:** ABORT immediately.

Display:
```
**❌ Sprint status file not found.**

Expected: `_bmad-output/implementation-artifacts/sprint-status.yaml`

This file is required before running the story automator.
Please run the **sprint-planning** workflow first to generate it.
```
**HALT** - Do not proceed.

**IF sprint_ok == true:**
- Store for later reference during preflight
- Will be used to check if earlier stories need completion

### 6. Setup
Ensure `{outputFolder}` exists.

Append an initialization entry to `{outputFile}`:
```bash
printf \"[%s] init: stop-hook=%s existing_state=%s\\n\" \
  \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"${changed}\" \"${latest_incomplete}\" >> \"{outputFile}\"
```

**Note:** Marker file path is resolved by `orchestrator-helper marker path` in step-02b-preflight-finalize after epic/story context is established.

---

## Then
→ Load `{nextStep}`
