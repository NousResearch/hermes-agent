---
title: 'Module Plan'
status: 'ideation'
module_name: ''
module_code: ''
module_description: ''
architecture: ''
standalone: true
expands_module: ''
skills_planned: []
config_variables: []
created: ''
updated: ''
---

# Module Plan

## Vision

<!-- What this module does, who it's for, and why it matters -->

## Architecture

<!-- Architecture decision and rationale -->
<!-- Options: single agent with capabilities, multiple agents, hybrid, orchestrator pattern -->
<!-- Document WHY this architecture was chosen — future builders need the reasoning -->

### Memory Architecture

<!-- Which pattern: personal memory only, personal + shared, or single shared memory? -->
<!-- If single shared memory: include the full folder structure -->
<!-- If shared memory: define the memory contract below -->

### Memory Contract

<!-- For each curated file in the memory folder, document: -->
<!-- - Filename and purpose -->
<!-- - What agents read it -->
<!-- - What agents write to it -->
<!-- - Key content/structure -->

### Cross-Agent Patterns

<!-- How do agents hand off work to each other? -->
<!-- Is the user the router? Is there an orchestrator? Service-layer relationships? -->
<!-- How does shared memory enable cross-domain awareness? -->

## Skills

<!-- For each planned skill, create a self-contained brief below. -->
<!-- Each brief should be usable by the Agent Builder or Workflow Builder WITHOUT conversation context. -->

### {skill-name}

**Type:** {agent | workflow}

**Persona:** <!-- For agents: who is this? Communication style, expertise, personality -->

**Core Outcome:** <!-- What does success look like? -->

**The Non-Negotiable:** <!-- The one thing this skill must get right -->

**Capabilities:**

| Capability | Outcome | Inputs | Outputs |
| ---------- | ------- | ------ | ------- |
|            |         |        |         |

<!-- For outputs: note where HTML reports, dashboards, or structured artifacts would add value -->

**Memory:** <!-- What does this agent read on activation? Write to? Daily log tag? -->

**Init Responsibility:** <!-- What happens on first run? Shared memory creation? Domain onboarding? -->

**Activation Modes:** <!-- Interactive, headless, or both? -->

**Tool Dependencies:** <!-- External tools with technical specifics -->

**Design Notes:** <!-- Non-obvious considerations, the "why" behind decisions -->

---

## Configuration

<!-- Module-level config variables for the setup skill. -->
<!-- If none needed, explicitly state: "This module requires no custom configuration beyond core BMad settings." -->

| Variable | Prompt | Default | Result Template | User Setting |
| -------- | ------ | ------- | --------------- | ------------ |
|          |        |         |                 |              |

## External Dependencies

<!-- CLI tools, MCP servers, or other external software that skills depend on -->
<!-- For each: what it is, which skills need it, and how the setup skill should handle it -->

## UI and Visualization

<!-- Does the module include dashboards, progress views, interactive interfaces, or a web app? -->
<!-- If yes: what it shows, which skills feed into it, how it's served/installed -->

## Setup Extensions

<!-- Beyond config collection: web app installation, directory scaffolding, external service configuration, starter files, etc. -->
<!-- These will need to be manually added to the setup skill after scaffolding -->

## Integration

<!-- Standalone: how it provides independent value -->
<!-- Expansion: parent module, cross-module capability relationships, skills that may reference parent module ordering -->

## Creative Use Cases

<!-- Beyond the primary workflow — unexpected combinations, power-user scenarios, creative applications discovered during brainstorming -->

## Ideas Captured

<!-- Raw ideas from brainstorming — preserved for context even if not all made it into the plan -->
<!-- Write here freely during phases 1-2. Don't write structured sections until phase 3+. -->

## Build Roadmap

<!-- Recommended build order with rationale for why each skill should be built in that order -->

**Next steps:**

1. Build each skill using **Build an Agent (BA)** or **Build a Workflow (BW)** — share this plan document as context
2. When all skills are built, return to **Create Module (CM)** to scaffold the module infrastructure
