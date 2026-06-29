---
name: 'step-02a-preflight-config'
description: 'Configure agents and execution settings, then create state document'
nextStep: './step-02b-preflight-finalize.md'
stateTemplate: '../templates/state-document.md'
outputFolder: '{output_folder}/story-automator'
outputFile: '{outputFolder}/orchestration-{epic_id}-{timestamp}.md'
buildStateDoc: '../scripts/story-automator'
agentConfigPrompts: '../data/agent-config-prompts.md'
agentConfigPresets: '../data/agent-config-presets.json'
---
# Step 2a: Pre-flight Configuration

**Goal:** Configure agents and execution settings, then create the orchestration state document.
**Interaction mode:** Guided configuration (collaborative inputs, deterministic state creation).

---

## Prerequisites

- Step 2 completed.
- Variables available: `epic_id`, `epic_name`, `range_json`, `stories_json`, `selected_count`, `custom_instructions`.

---

## Do

### 1. Configure Execution Preferences

> **PREREQUISITE:** Step 2 (preflight) MUST be complete. The Complexity Matrix MUST have been displayed. If not, STOP and complete step 2 first.

```
**Execution Settings:**

1. **Skip the 'automate' step (test automation)?** [N]o (default) / [Y]es
2. **Max parallel sessions?** (tmux sessions running concurrently, default: 1)

Enter choices (e.g., `N 1` or `Y 3`):
```

**Wait.**

Store responses as `skip_automate` (true/false) and `max_parallel` (integer).

### 2. Configure Agent (Complexity-Aware)

Using the complexity data from `stories_json`, present agent configuration options that reference the actual complexity breakdown.

**2a. Check for Saved Presets**

```bash
presets_result=$("{buildStateDoc}" agent-config list --file "{agentConfigPresets}")
preset_count=$(echo "$presets_result" | jq -r '.count')
```

Store `preset_count` — this determines whether [L]oad option appears in the menu.

**2b. Present Complexity-Based Agent Options**

Display prompts from `{agentConfigPrompts}`, selecting the appropriate table variant:
- If `skip_automate` is false: show table WITH `auto` column
- If `skip_automate` is true: show table WITHOUT `auto` column
- If `preset_count > 0`: include [L]oad saved option
- If `preset_count == 0`: omit [L] option

**Wait.**

**2c. Handle Selection**

- **IF S:** Build `agent_config_json` from defaults (no save prompt).
- **IF U or C:** Follow Uniform/Custom prompts from `{agentConfigPrompts}`, build `agent_config_json`, then proceed to **2d (Save Prompt)**.
- **IF L:** Follow Load Saved Preset prompt from `{agentConfigPrompts}`. Load preset config as `agent_config_json` (no save prompt).

```bash
# Example shape with complexity-based config (auto column included when not skipped)
agent_config_json='{
  "complexityBased": true,
  "low": {"create":{"primary":"...","fallback":"..."},"dev":{...},"auto":{...},"review":{...}},
  "medium": {"create":{...},"dev":{...},"auto":{...},"review":{...}},
  "high": {"create":{...},"dev":{...},"auto":{...},"review":{...}},
  "auto": {"skip": $skip_automate}
}'
```

Store:
- `agent_config_json` = full config object
- `primary_agent` = default primary (for backwards compatibility)

**2d. Save Prompt (U/C only)**

Only when user chose **[U]niform** or **[C]ustom**, follow the Save Configuration prompt from `{agentConfigPrompts}`:

```bash
# If user provides a name:
"{buildStateDoc}" agent-config save --file "{agentConfigPresets}" --name "$save_name" --config-json "$agent_config_json"
```

### 3. Review

Display configuration summary:
- Epic and story range
- Custom instructions (if any)
- Agent configuration
- Execution settings

Pause for confirmation before starting execution.

### 3b. Confirm Autonomous Start (Optional Checkpoint)

Before creating state and launching autonomous phases, confirm:
```
Proceed with autonomous execution after preflight? [Y/n]
```

**Wait.**

- If `Y`/Enter: continue.
- If `n`: return to Step 1 (settings) for adjustments.

### 4. Create State Document

From `{stateTemplate}`:
- Generate: `orchestration-{epic_id}-{timestamp}.md`
- Fill frontmatter with all config
- Initialize story progress table
- Set status: "READY"
- Save to `{outputFolder}`

Deterministic creation:
```bash
agent_cmd="claude --dangerously-skip-permissions"
if [ "$primary_agent" = "codex" ]; then agent_cmd="codex exec --full-auto"; fi

config_json=$(jq -n \
  --arg epic "$epic_id" \
  --arg epicName "$epic_name" \
  --argjson storyRange "$(echo "$range_json" | jq '.storyIds')" \
  --arg status "READY" \
  --arg currentStory "null" \
  --arg currentStep "preflight" \
  --arg aiCommand "$agent_cmd" \
  --arg customInstructions "$custom_instructions" \
  --argjson overrides "{\"skipAutomate\":$skip_automate,\"maxParallel\":$max_parallel}" \
  --argjson agentConfig "$agent_config_json" \
  '{epic:$epic,epicName:$epicName,storyRange:$storyRange,status:$status,currentStory:null,currentStep:$currentStep,aiCommand:$aiCommand,customInstructions:$customInstructions,overrides:$overrides,agentConfig:$agentConfig}'
)

state_result=$("{buildStateDoc}" build-state-doc --template "{stateTemplate}" --output-folder "{outputFolder}" --config-json "$config_json")
state_path=$(echo "$state_result" | jq -r '.path')
```

Display: "**State document created.**"
Record: `state_path` is the resolved `{outputFile}` for this run.

### 5. Auto-Proceed to Finalize

Persist any preflight notes to `{outputFile}`, update frontmatter (append `step-02-preflight` and `step-02a-preflight-config`, set `lastUpdated`).

---

## Then
→ Load, read entire file, and execute `{nextStep}`
