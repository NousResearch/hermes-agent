---
name: 'step-v-02-report'
description: 'Validate story progress consistency and present final validation report'
outputFile: '{output_folder}/story-automator/orchestration-{epic_id}-{timestamp}.md'
---

# Validation Step 2: Progress Consistency + Final Report

**Goal:** Validate story-progress consistency using the selected state document and present a consolidated validation report.

## MANDATORY EXECUTION RULES

- 🛑 **DO NOT BE LAZY** - CHECK EVERY STORY IN RANGE
- 📖 Validate ALL progress checks, not just samples
- 🚫 DO NOT skip stalled/skipped-step checks
- ✅ Include structure/session findings from step-v-01 in final report

---

## Do

### 1. Load Validation Context from Step 1

Use carried-forward context:
- `state_path`
- `validation`
- `sessions`
- `orphaned_refs`
- `untracked_live`
- Prior structure/session issues

Load the selected state document again (resolved as `{state_path}` for this run) to verify progress details.

### 2. Validate Story Progress Thoroughly

Run a single prefilter pass first and keep parent context compact:
```bash
# Focused extraction before deep checks
progress_focus=$(rg -n "done|in_progress|blocked|review|create|dev|automate|commit|ERROR|WARN|FAIL" "$state_path" | head -n 200)
if [ -z "$progress_focus" ]; then
  progress_focus=$(tail -n 200 "$state_path")
fi
```

Return only compact progress fields to the final report synthesis:
- `story_count`
- `progress_rows`
- `inconsistency_count`
- `stalled_count`
- `critical_issues[]`

For each story in `storyRange`:
- Check progress table has an entry for the story
- Verify task sequence is coherent (`create -> dev -> automate -> review -> commit`)
- Flag impossible regressions (for example, `review=done` while `dev` missing)
- Detect potentially stuck stories (same `currentStep` for too long without action-log movement)

Deterministic checks (example pattern):
```bash
# Example only: derive summary values from state/action log without loading full logs
story_count=$(echo "$validation" | jq -r '.storyRangeCount // 0')
progress_rows=$(rg -n "^[[:space:]]*\\|[[:space:]]*[0-9]+\\.[0-9]+" "$state_path" | wc -l | tr -d ' ')
```

If `story_count >= 4`, run per-story consistency checks in parallel and return compact rows only:
```bash
story_ids=$(echo "$validation" | jq -r '.storyRange[]?')
tmp_progress=$(mktemp)
printf "%s\n" "$story_ids" | xargs -I{} -P 4 sh -c \
  'id="$1"; file="$2"; rg -n -F "| ${id} |" "$file" | head -n 1 | sed "s/^/${id}|/"' _ "{}" "$state_path" \
  > "$tmp_progress"
progress_rows=$(wc -l < "$tmp_progress" | tr -d ' ')
rm -f "$tmp_progress"
```

### 3. Consolidate Findings

Create final status buckets:
- **Structure:** from `validation`
- **Sessions:** from `sessions`, `orphaned_refs`, `untracked_live`
- **Progress:** from step-2 checks above

Mark severity:
- **CRITICAL:** malformed state / irrecoverable sequence corruption
- **WARNING:** stale or inconsistent but recoverable
- **INFO:** healthy or minor notes

### 4. Present Final Validation Report

```
**Validation Report: {epicName}**

**Structure:** ✅ Valid / ⚠️ Issues found
**Sessions:** ✅ Healthy / ⚠️ Anomalies detected
**Progress:** ✅ Consistent / ⚠️ Inconsistencies found

[If issues:]
**Issues Found:**
1. {issue description}
2. {issue description}

**Recommendations:**
- {recommendation}
```

### 5. Complete

Display: "**Validation complete.** Review the issues above and use the edit workflow to apply fixes if needed."

**End validation.**

---

## Then
→ End workflow (validation completed in 2 steps)
