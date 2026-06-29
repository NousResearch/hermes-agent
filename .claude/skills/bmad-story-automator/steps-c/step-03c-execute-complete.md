---
name: 'step-03c-execute-complete'
description: 'Post-loop completion summary, parallelism notes, and transition to wrapup'
nextStep: './step-04-wrapup.md'
scriptsDir: '../scripts/story-automator'
outputFile: '{output_folder}/story-automator/orchestration-{epic_id}-{timestamp}.md'
executionPatterns: '../data/execution-patterns.md'
retryStrategy: '../data/retry-fallback-strategy.md'
triggers: '../data/escalation-triggers.md'
---

# Step 3c: Execution Complete

**Goal:** Summarize results after all stories finish, persist final status, and transition to wrapup.
**Interaction mode:** Deterministic auto-proceed.

---

## All Complete

Display:
```
**All {count} stories completed!**

If `{count} <= 10`:
| Story | Status |
|-------|--------|
{summary_table}

If `{count} > 10`:
- Completed: {completed_count}
- Warnings: {warning_count}
- Escalations: {escalation_count}
- See state log for full per-story table.

Proceeding to wrap-up...
```

```bash
"{scriptsDir}" orchestrator-helper state-update "{outputFile}" \
  --set status=EXECUTION_COMPLETE --set lastUpdated="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "- **[$(date -u +%Y-%m-%dT%H:%M:%SZ)]** All stories complete — execution finished" >> "{outputFile}"
```

## Parallelism & Escalation

**Parallelism:** When `overrides.maxParallel > 1`, batch independent stories into concurrent groups:
1. Check story dependency graph — only stories with no shared file dependencies can run in parallel
2. Spawn up to `maxParallel` tmux sessions simultaneously (each runs steps A→F independently)
3. Wait for all sessions in the batch to complete before starting the next batch
4. Epic completion check (H) runs only after all batches finish

See `{executionPatterns}` for forbidden patterns and session isolation rules.

**Escalation:** See `{triggers}` for trigger definitions and `{retryStrategy}` for retry/fallback patterns. Escalation only after exhausting all retry attempts.

## Auto-Proceed to Wrap-up

Display: "**Execution loop complete. Proceeding to wrap-up...**"

```bash
"{scriptsDir}" orchestrator-helper state-update "{outputFile}" \
  --set currentStep=step-04-wrapup \
  --set lastUpdated="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

## Then
→ Immediately load and execute `{nextStep}`
