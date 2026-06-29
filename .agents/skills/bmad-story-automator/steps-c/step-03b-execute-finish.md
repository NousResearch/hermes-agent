---
name: 'step-03b-execute-finish'
description: 'Finalize each story (commit/status), trigger retrospective when epic complete, and finish execution loop'
nextStep: './step-03c-execute-complete.md'
scriptsDir: '../scripts/story-automator'
outputFile: '{output_folder}/story-automator/orchestration-{epic_id}-{timestamp}.md'
---

# Step 3b: Finalize Story + Wrap Execution

**Goal:** After code review completes for a story, commit changes, verify sprint status, update progress, and finish the loop.
**Interaction mode:** Deterministic autonomous execution.

---

## Story Loop (Continue from Step 3)

### E. Git Commit

**Required:** Commit after every story (do not skip).

```bash
commit=$("{scriptsDir}" commit-story --repo "{project-root}" --story {story_id} --title "{title}")
ok=$(echo "$commit" | jq -r '.ok')
```

- If `ok == true`:
  ```bash
  # Update Story Progress: mark git-commit done
  tmp_state=$(mktemp)
  sed "s/^| ${story_id} |.*$/| ${story_id} | done | done | done | done | done | in-progress |/" "{outputFile}" > "$tmp_state" && mv "$tmp_state" "{outputFile}"
  ```
  → proceed to F
- If `ok == false` → log warning and escalate

### F. Verify Sprint Status

```bash
# Check sprint-status with story file fallback (v1.4.0)
normalized=$("{scriptsDir}" orchestrator-helper normalize-key {story_id})
story_key=$(echo "$normalized" | jq -r '.key')
status=$("{scriptsDir}" orchestrator-helper sprint-status get "$story_key")
is_done=$(echo "$status" | jq -r '.done')

# Fallback: trust story file if sprint-status disagrees
if [ "$is_done" != "true" ]; then
    file_done=$("{scriptsDir}" orchestrator-helper story-file-status {story_id} | jq -r '.status')
    [ "$file_done" = "done" ] && is_done="true"
fi
```

- If `is_done == false` → return to Code Review Loop (Step 3, section D)
- If `is_done == true` → proceed to G

### G. Story Complete
Display: "**✅ Story {N} complete.**"
```bash
"{scriptsDir}" orchestrator-helper state-update "{outputFile}" \
  --set lastUpdated="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "- **[$(date -u +%Y-%m-%dT%H:%M:%SZ)]** Story {story_id}: ✅ complete (commit + sprint-status verified)" >> "{outputFile}"

# Update Story Progress: mark story done
tmp_state=$(mktemp)
sed "s/^| ${story_id} |.*$/| ${story_id} | done | done | done | done | done | done |/" "{outputFile}" > "$tmp_state" && mv "$tmp_state" "{outputFile}"
```
Display: `[story {N}/{total}] finalize -> done`

### H. Check Epic Completion & Trigger Retrospective (Multi-Epic Support)

After each story completes, check if ALL stories in this epic are now done. Retrospective only triggers when every story in the epic has passed code review and sprint status confirms all are "done".

#### H.1 Check All Stories Done

```bash
# Run epic-level check in parallel with per-story checks
tmp_epic_status=$(mktemp)
("{scriptsDir}" orchestrator-helper sprint-status check-epic {epic_number} > "$tmp_epic_status") &
epic_status_pid=$!

# Get all stories for this epic and verify each is done
epic_stories=$("{scriptsDir}" orchestrator-helper get-epic-stories {epic_number} --state-file "{outputFile}")
stories_ok=$(echo "$epic_stories" | jq -r '.ok')
story_count=$(echo "$epic_stories" | jq -r '.count')
all_done=true

if [ "$stories_ok" != "true" ] || [ "$story_count" -eq 0 ]; then
    all_done=false
else
    tmp_story_checks=$(mktemp)
    echo "$epic_stories" | jq -r '.stories[]' \
      | xargs -I{} -P 4 sh -c '
          status=$("'"{scriptsDir}"'" orchestrator-helper sprint-status get "{}")
          done=$(echo "$status" | jq -r ".done")
          [ "$done" = "true" ] && echo "{}|done" || echo "{}|not_done"
        ' > "$tmp_story_checks"

    if rg -q '\|not_done$' "$tmp_story_checks"; then
      all_done=false
    fi
    rm -f "$tmp_story_checks"
fi
```

#### H.2 Secondary Verification via Sprint Status

```bash
# Double-check: use result from parallel epic-level check
wait "$epic_status_pid"
epic_status=$(cat "$tmp_epic_status")
rm -f "$tmp_epic_status"

epic_complete=$(echo "$epic_status" | jq -r '.allStoriesDone')
epic_ok=$(echo "$epic_status" | jq -r '.ok')

# Both checks must pass
if [ "$all_done" = "true" ] && [ "$epic_ok" = "true" ] && [ "$epic_complete" = "true" ]; then
    trigger_retro=true
else
    trigger_retro=false
fi
```

#### H.3 Trigger Retrospective (Only When Epic Fully Complete)

**IF trigger_retro == true:**

1. Display: "**✅ Epic {epic_number} complete! All stories passed code review. Triggering retrospective (YOLO mode)...**"
2. Log: `- **[{timestamp}]** Epic {epic_number}: ALL STORIES DONE - triggering retrospective`

```bash
# CRITICAL: Use build-cmd to get full YOLO prompt with doc verification.
# Pass `"$retro_model"` quoted so bracketed IDs like `claude-opus-4-7[1m]`
# survive shell expansion intact.
retro_selection=$("{scriptsDir}" orchestrator-helper retro-agent --state-file "{outputFile}")
retro_agent=$(echo "$retro_selection" | jq -r '.primary')
retro_model=$(echo "$retro_selection" | jq -r '.model // ""')
if [ -n "$retro_model" ]; then
    cmd=$("{scriptsDir}" tmux-wrapper build-cmd retro {epic_number} --agent "$retro_agent" --model "$retro_model")
else
    cmd=$("{scriptsDir}" tmux-wrapper build-cmd retro {epic_number} --agent "$retro_agent")
fi
session=$("{scriptsDir}" tmux-wrapper spawn retro "" {epic_number} --agent "$retro_agent" --command "$cmd")

# Monitor with safe failure (never escalate on retro failure)
retro_timeout=60
[ "$story_count" -gt 10 ] && retro_timeout=90
result=$("{scriptsDir}" monitor-session "$session" --json --agent "$retro_agent" --timeout "$retro_timeout")
"{scriptsDir}" tmux-wrapper kill "$session"

retro_status=$(echo "$result" | jq -r '.final_state')

if [ "$retro_status" = "completed" ] || [ "$retro_status" = "success" ]; then
    echo "- **[{timestamp}]** Epic {epic_number} retrospective: completed successfully" >> "{outputFile}"
else
    echo "- **[{timestamp}]** Epic {epic_number} retrospective: skipped (reason: $retro_status)" >> "{outputFile}"
fi
```

3. Update state document with retrospective status:
```yaml
retrospectives:
  epic-{epic_number}:
    status: "completed" | "skipped"
    reason: "{reason_if_skipped}"
    timestamp: "{timestamp}"
```

4. **Continue to next story regardless of retrospective result** (retrospectives never block)

**IF trigger_retro == false:**
- Continue to next story (epic not yet complete)

**IMPORTANT RULES:**
- **ALL stories must be done**: Retrospective only triggers when every story in the epic shows "done" in sprint status
- **Use configured retro agent**: Resolve retrospective agent from `agentConfig` before spawn
- **Never escalate; non-blocking**: If retrospective fails for any reason, log warning and continue

**END FOR EACH**

## Then
→ After all stories complete, load and execute `{nextStep}`
