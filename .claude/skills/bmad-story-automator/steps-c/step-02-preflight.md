---
name: 'step-02-preflight'
description: 'Gather epic, story selection, and complexity analysis'
nextStep: './step-02a-preflight-config.md'
outputFolder: '{output_folder}/story-automator'
outputFile: '{outputFolder}/preflight-{epic_id}-{timestamp}.md'
parseEpic: '../scripts/story-automator'
parseStoryRange: '../scripts/story-automator'
parseStory: '../scripts/story-automator'
stateHelper: '../scripts/story-automator'
defaultEpicPath: '{output_folder}/planning-artifacts/epics.md'
defaultSprintStatusFile: '{output_folder}/implementation-artifacts/sprint-status.yaml'
complexityRules: '../data/complexity-rules.json'
complexityScoring: '../data/complexity-scoring.md'
preflightRequirements: '../data/preflight-requirements.md'
---
# Step 2: Pre-flight (Epic + Complexity)

**Goal:** Gather epic, story range, complexity analysis, and custom instructions.
**Interaction mode:** Collaborative discovery and clarification.

---

## 🚨 BEFORE STARTING: Load Requirements

**CRITICAL:** Load and read `{preflightRequirements}` FIRST. It contains MANDATORY sequence rules, FORBIDDEN patterns, and verification gates that MUST be followed.

---

## Do

### 1. Confirm Epic File
```
**Epic source**

Default epic file: `{defaultEpicPath}`
Use this file? [Y/n]
```

If user confirms (Y/Enter), set `epic_path="{defaultEpicPath}"`.
If user says no, ask for epic file path and set `epic_path` from response.
If confirmed default does not exist, tell user and request explicit path.

**Wait.**

### 2. Review Epic
Parse epic file deterministically:
```bash
epic_json=$("{parseEpic}" parse-epic --file "{epic_path}")
epic_name=$(echo "$epic_json" | jq -r '.epicTitle')
story_count=$(echo "$epic_json" | jq -r '.count')
story_titles=$(echo "$epic_json" | jq -r '.stories[] | "\(.storyId) \(.title)"')
story_ids_csv=$(echo "$epic_json" | jq -r '.stories[] | .storyId' | paste -sd, -)
sprint_exists=$("{stateHelper}" orchestrator-helper sprint-status exists)
story_status_rows="(sprint-status unavailable at {defaultSprintStatusFile})"
if [ "$sprint_exists" = "true" ]; then
  story_status_rows=$(echo "$epic_json" | jq -r '.stories[] | .storyId' | while read -r sid; do
    status_json=$("{stateHelper}" orchestrator-helper sprint-status get "$sid")
    st=$(echo "$status_json" | jq -r '.status // "unknown"')
    printf -- "- %s | %s\n" "$sid" "$st"
  done)
fi
```

Display:
```
**Epic:** {epic_name}

Stories found:
1. {storyId} {title}
2. {storyId} {title}
...

Total: {story_count}

Current sprint-status ({defaultSprintStatusFile}):
{story_status_rows}

Which stories? (e.g., `1-3`, `all`, `1,3,5`)
```
If user hesitates, suggest `all` as default and confirm.

**Wait.**

### 3. Read Stories and Compute Complexity (MANDATORY - DO NOT SKIP)

> **🚨 CRITICAL:** This step MUST use the Python helper for complexity scoring. NEVER manually assess complexity by reading story content.

For each story in range, extract complexity **programmatically**:

**3a. Parse story range:**
```bash
range_json=$("{parseStoryRange}" parse-story-range --input "{user_selection}" --total "$story_count" --ids "$story_ids_csv")
selected_ids=$(echo "$range_json" | jq -r '.storyIds[]')
selected_count=$(echo "$range_json" | jq -r '.count')
first_story_id=$(echo "$range_json" | jq -r '.storyIds[0]')
epic_id=$(echo "$first_story_id" | cut -d. -f1)
```

**3b. Get complexity for EACH story using Python helper:**
```bash
# Initialize accumulator - REQUIRED
stories_json='[]'

# For EACH story_id in selected_ids, run:
story_json=$("{parseStory}" parse-story --epic "{epic_path}" --story "$story_id" --rules "{complexityRules}")

# Extract and accumulate - REQUIRED
story_title=$(echo "$story_json" | jq -r '.title')
story_level=$(echo "$story_json" | jq -r '.complexity.level')
story_score=$(echo "$story_json" | jq -r '.complexity.score')
story_reasons=$(echo "$story_json" | jq -r '.complexity.reasons // []')
stories_json=$(echo "$stories_json" | jq -c --arg id "$story_id" --arg title "$story_title" --arg level "$story_level" --argjson score "$story_score" --argjson reasons "$story_reasons" \
  '. + [{storyId:$id,title:$title,complexity:{level:$level,score:$score,reasons:$reasons}}]')
```

Refer to `{complexityScoring}` for scoring criteria and thresholds.

**Parallelism Policy (MANDATORY):**

- If `selected_count >= 4`: run per-story complexity parsing in parallel subprocesses (max 4 workers).
- If `selected_count < 4`: run sequentially.
- In both modes, return only summary fields to parent context: `storyId`, `title`, `complexity.level`, `complexity.score`, `complexity.reasons`.

```bash
# Deterministic threshold
if [ "$selected_count" -ge 4 ]; then
  # Parallel mode (max 4 workers)
  tmp_story_complexity=$(mktemp)
  printf "%s\n" $selected_ids | xargs -I{} -P 4 sh -c '
    "{parseStory}" parse-story --epic "{epic_path}" --story "{}" --rules "{complexityRules}" \
      | jq -c "{storyId:.storyId,title:.title,complexity:.complexity}"
  ' > "$tmp_story_complexity"
  stories_json=$(jq -s '.' "$tmp_story_complexity")
  rm -f "$tmp_story_complexity"
else
  # Sequential mode
  stories_json='[]'
  for story_id in $selected_ids; do
    story_json=$("{parseStory}" parse-story --epic "{epic_path}" --story "$story_id" --rules "{complexityRules}")
    stories_json=$(echo "$stories_json" | jq -c --argjson s "$(echo "$story_json" | jq -c '{storyId:.storyId,title:.title,complexity:.complexity}')" '. + [$s]')
  done
fi
```

**3c. Display Complexity Matrix (REQUIRED):**

Display the Complexity Matrix using the template from `{preflightRequirements}`.

**3d. VERIFICATION GATE:**

Follow the verification gate from `{preflightRequirements}` before proceeding.

---

### 4. Custom Instructions
```
**Any custom instructions?**

Examples:
- "Always run tests after changes"
- "Prioritize stories 3 and 5"
- "Be extra careful with database migrations"
- "Use strict typing throughout"

Enter instructions or 'none':
```
If user is unsure, recommend `none` and continue.

**Wait.**

Store response as `custom_instructions` (use "" for none).

### 5. Proceed to Configuration

Persist preflight snapshot before continuing:
```bash
mkdir -p "{outputFolder}"
cat > "{outputFile}" <<EOF
# Preflight Snapshot

- Timestamp: {timestamp}
- Epic path: {epic_path}
- Epic name: {epic_name}
- Story count: {story_count}
- Selected count: {selected_count}
- Selected IDs: {selected_ids}
- Custom instructions: {custom_instructions}

## Complexity Summary
$(echo "$stories_json" | jq -r '.[] | "- \(.storyId) | \(.complexity.level) | score=\(.complexity.score)"')
EOF
```

Carry forward: `epic_path`, `epic_name`, `story_count`, `story_ids_csv`, `range_json`, `selected_ids`, `selected_count`, `stories_json`, `epic_id`, `first_story_id`, `custom_instructions`.

---

## Then
→ Load and execute `{nextStep}`
