---
# Orchestration State Document
epic: ""
epicName: ""
storyRange: []
status: "INITIALIZING"
currentStory: null
currentStep: null
stepsCompleted: []
lastUpdated: ""
createdAt: ""

# Configuration
aiCommand: ""  # Deprecated: use agentConfig
overrides:
  skipAutomate: false
  maxParallel: 1
customInstructions: ""  # User-provided instructions for orchestration
agentsFile: ""  # Deterministic per-story agent selections
complexityFile: ""  # Persisted story complexity data
policyVersion: 0
policySnapshotFile: ""
policySnapshotHash: ""
legacyPolicy: false

# Agent Configuration (v3.0.0)
agentConfig:
  defaultPrimary: "auto"      # auto resolves to the active runtime provider: claude | codex
  defaultFallback: false      # Default fallback: claude | codex | false (disabled)
  # Per-task overrides (optional)
  # perTask:
  #   create:
  #     primary: "codex"
  #     fallback: "claude"
  #   dev:
  #     primary: "claude"
  #     fallback: false
  #   auto:
  #     primary: "codex"
  #     fallback: false
  #   review:
  #     primary: "claude"
  #     fallback: false
  # Complexity-based overrides (optional, WIN per task)
  # complexityOverrides:
  #   low:
  #     create:
  #       primary: "claude"
  #       fallback: false
  #   medium:
  #     dev:
  #       primary: "claude"
  #       fallback: false
  #   high:
  #     review:
  #       primary: "claude"
  #       fallback: false
  # Codex-specific (applied automatically when agent is codex):
  # - 1.5x timeout multiplier (60min → 90min)
  # - 1.5x wait time cap (2min → 3min between polls)
  # - Natural language prompts instead of command syntax

# Session Tracking
activeSessions: []
completedSessions: []
---

# Orchestration Log: {{epicName}}

## Configuration

**Epic:** {{epic}}
**Story Range:** {{storyRange}}
**Created:** {{createdAt}}

**Overrides:**
- Skip Automate: {{overrides.skipAutomate}}
- Max Parallel: {{overrides.maxParallel}}

**Custom Instructions:**
{{customInstructions}}

---

## Story Progress

| Story | create-story | dev-story | automate | code-review | git-commit | Status |
|-------|--------------|-----------|----------|-------------|------------|--------|
<!-- Progress rows will be appended here -->

---

## Action Log

<!-- Timestamped action entries will be appended here -->

---

## Session References

| Session ID | Story | Step | Status | Started | Completed |
|------------|-------|------|--------|---------|-----------|
<!-- Session entries will be appended here -->

---

## Pending Decisions

<!-- Escalations awaiting user input will be listed here -->

---

## Learnings & Recommendations

<!-- Populated during wrapup phase -->
