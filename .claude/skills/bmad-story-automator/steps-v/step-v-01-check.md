---
name: 'step-v-01-check'
description: 'Validate orchestration state document integrity and session health'
nextStep: './step-v-02-report.md'
outputFolder: '{output_folder}/story-automator'
rules: '../data/orchestrator-rules.md'
stateFilePattern: '{outputFolder}/orchestration-*.md'
outputFile: '{outputFolder}/orchestration-{epic_id}-{timestamp}.md'
validateState: '../scripts/story-automator'
listSessions: '../scripts/story-automator'
deriveProjectSlug: '../scripts/story-automator'
tmuxCommands: '../data/tmux-commands.md'
---

# Validation Step 1: Check State Integrity

**Goal:** Validate an orchestration state document for structural integrity and session health.

## MANDATORY EXECUTION RULES

- 🛑 **DO NOT BE LAZY** - CHECK EVERY FIELD AND SESSION
- 📖 Validate ALL required fields, not just a sample
- 🚫 DO NOT skip any validation checks
- ✅ Report ALL issues found, not just the first one

---

## Do

### 1. Load Rules
Load `{rules}` once for context on expected state structure.

### 2. Request State Document
```
**Which orchestration would you like to validate?**

Found state documents in `{outputFolder}`:
[List all orchestration-*.md files with: name, status, last updated]

Pattern: `{stateFilePattern}`

Enter filename or number to select:
```

**Wait.**

### 3. Load and Parse State
Load the selected state document (resolved as `{state_path}` for this run). Extract frontmatter:
- `epic`, `epicName`, `storyRange`
- `status`, `currentStory`, `currentStep`
- `stepsCompleted`, `lastUpdated`
- `projectContext`, `aiCommand`, `agentConfig`, `overrides`
- `activeSessions`, `completedSessions`

### 3a. Helper CLI Contract Check (Required)

Before running validation commands, verify helper interfaces in parallel:
```bash
tmp_help_validate=$(mktemp)
tmp_help_sessions=$(mktemp)
tmp_help_slug=$(mktemp)

("{validateState}" validate-state --help >"$tmp_help_validate" 2>&1) &
pid_validate=$!
("{listSessions}" list-sessions --help >"$tmp_help_sessions" 2>&1) &
pid_sessions=$!
("{deriveProjectSlug}" derive-project-slug --help >"$tmp_help_slug" 2>&1) &
pid_slug=$!

wait "$pid_validate"; status_validate=$?
wait "$pid_sessions"; status_sessions=$?
wait "$pid_slug"; status_slug=$?

if [ "$status_validate" -ne 0 ] || [ "$status_sessions" -ne 0 ] || [ "$status_slug" -ne 0 ]; then
  rm -f "$tmp_help_validate" "$tmp_help_sessions" "$tmp_help_slug"
  echo "validation helper CLI contract changed"
  exit 1
fi

rm -f "$tmp_help_validate" "$tmp_help_sessions" "$tmp_help_slug"
```

If any check fails: **STOP and report "validation helper CLI contract changed"**.

### 4. Run Structure + Session Baseline in Parallel

Run structure validation and session inventory concurrently, then aggregate results.

```bash
tmp_validation=$(mktemp)
tmp_sessions=$(mktemp)

("{validateState}" validate-state --state "{state_path}" > "$tmp_validation") &
validation_pid=$!

project_slug_json=$("{deriveProjectSlug}" derive-project-slug --project-root "{project-root}") || {
  rm -f "$tmp_validation" "$tmp_sessions"
  echo "derive-project-slug failed"
  exit 1
}
project_slug=$(printf '%s' "$project_slug_json" | jq -r '.slug')
("{listSessions}" list-sessions --slug "$project_slug" > "$tmp_sessions") &
sessions_pid=$!

wait "$validation_pid"; validation_status=$?
wait "$sessions_pid"; sessions_status=$?

if [ "$validation_status" -ne 0 ] || [ "$sessions_status" -ne 0 ]; then
  rm -f "$tmp_validation" "$tmp_sessions"
  echo "state validation or session inventory failed"
  exit 1
fi

validation=$(cat "$tmp_validation")
sessions=$(cat "$tmp_sessions")
rm -f "$tmp_validation" "$tmp_sessions"
```

### 5. Validate Structure + Session Consistency (Single Diff Pass)

**Required Fields Check:**

| Field | Present | Valid |
|-------|---------|-------|
| epic | ✅/❌ | non-empty string |
| epicName | ✅/❌ | non-empty string |
| storyRange | ✅/❌ | array |
| status | ✅/❌ | valid enum |
| lastUpdated | ✅/❌ | ISO date |
| aiCommand or agentConfig | ✅/❌ | at least one runtime command source is present |

**Valid status values:** INITIALIZING, READY, IN_PROGRESS, PAUSED, COMPLETE, ABORTED

**Record issues:**
- Missing required fields
- Invalid field values
- Malformed YAML

Single-pass structure issue extraction (compact output):
```bash
field_issues=$(echo "$validation" | jq -r '.issues[]? | select(.type=="missing_field" or .type=="invalid_value" or .type=="yaml_error") | "\(.type): \(.field // .message)"')
```

Using `{tmuxCommands}` semantics and `sessions` output, compare state vs live sessions in one pass:
```bash
state_sessions=$(echo "$validation" | jq -r '.activeSessions[]?.sessionId // empty' | sort -u)
live_sessions=$(echo "$sessions" | jq -r '.sessions[]?.name // empty' | sort -u)

orphaned_refs=$(comm -23 <(echo "$state_sessions") <(echo "$live_sessions"))
untracked_live=$(comm -13 <(echo "$state_sessions") <(echo "$live_sessions"))
```

**Session consistency checks:**

| Check | Result |
|-------|--------|
| Active sessions in state but not in T-Mux | Orphaned references |
| T-Mux sessions not in state | Untracked sessions |
| Status=IN_PROGRESS but no active sessions | Stale state |

### 6. Carry Forward Validation Context

Carry forward to `{nextStep}`:
- `state_path`
- `validation`
- `sessions`
- `orphaned_refs`
- `untracked_live`
- Any structure/session issues identified

### 7. Auto-Proceed

Display: "**Structure and session baseline complete. Proceeding to progress validation and final report...**"

---

## Then
→ Load and execute `{nextStep}`
