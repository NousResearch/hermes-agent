---
name: 'step-02b-preflight-finalize'
description: 'Finalize preflight and start execution'
nextStep: './step-03-execute.md'
outputFolder: '{output_folder}/story-automator'
outputFile: '{outputFolder}/orchestration-{epic_id}-{timestamp}.md'
stateHelper: '../scripts/story-automator'
ensureMarkerGitignore: '../scripts/story-automator'
deriveProjectSlug: '../scripts/story-automator'
markerFormat: '../data/marker-file-format.md'
---

# Step 2b: Pre-flight Finalize

**Goal:** Finalize preflight artifacts, create marker, and start execution.
**Interaction mode:** Deterministic auto-proceed.

---

## Do

### 1. Create Complexity + Agents Files

Derive deterministic filenames:
```bash
state_base=$(basename "{outputFile}" .md)
complexity_path="{outputFolder}/complexity-${state_base}.json"
agents_dir="{outputFolder}/agents"
agents_path="$agents_dir/agents-${state_base}.md"
```

Write complexity file:
```bash
mkdir -p "$(dirname "$complexity_path")"
echo "$stories_json" | jq -c '{stories:.}' > "$complexity_path"
```

Build deterministic agents file:
```bash
mkdir -p "$agents_dir"
"{stateHelper}" orchestrator-helper agents-build \
  --state-file "{outputFile}" \
  --complexity-file "$complexity_path" \
  --output "$agents_path" \
  --config-json "$agent_config_json"
```

Update state frontmatter with file paths:
```bash
agents_path_json=$(printf '%s' "$agents_path" | jq -R '.')
complexity_path_json=$(printf '%s' "$complexity_path" | jq -R '.')
"{stateHelper}" orchestrator-helper state-update "{outputFile}" \
  --set "agentsFile=$agents_path_json" \
  --set "complexityFile=$complexity_path_json"
```

### 2. Create Marker and Begin Execution

**Create marker file** (see `{markerFormat}` for JSON structure):
```bash
# Resolve the active marker path for the selected runtime layout and gitignore it.
marker_info=$("{stateHelper}" orchestrator-helper marker path)
marker_entry=$(echo "$marker_info" | jq -r '.entry')
"{ensureMarkerGitignore}" ensure-marker-gitignore --gitignore ".gitignore" --entry "$marker_entry"

# Create marker
project_slug=$(echo "$("{deriveProjectSlug}" derive-project-slug --project-root "{project-root}")" | jq -r '.slug')
"{stateHelper}" orchestrator-helper marker create --epic "$epic_id" --story "$first_story_id" \
  --remaining "$selected_count" --state-file "{outputFile}" \
  --project-slug "$project_slug" --pid "$$" --heartbeat "{timestamp}"
```

Set status="IN_PROGRESS", log "Execution started".
Update frontmatter (append `step-02b-preflight-finalize`, set `lastUpdated`).

---

## Then
→ Load, read entire file, and execute `{nextStep}`
