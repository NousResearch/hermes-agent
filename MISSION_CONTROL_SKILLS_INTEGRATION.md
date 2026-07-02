# Mission Control Operating Surface — Hermes Skills Integration

**Date integrated:** July 2, 2026  
**Source:** `/home/matt/workspace/github/mission-control/.claude/skills/`  
**Destination:** `/home/matt/.hermes/hermes-agent/skills/`  
**Status:** ✓ Active and discoverable

---

## Overview

Fifteen mission-critical skills have been imported from the Mission Control project into the Hermes agent system. These skills enable orchestrated multi-agent workflows, design signal scouting, and professional content generation — all deployed agenically across the Hermes operating surface.

The skills are immediately available as slash commands (`/skill-name`) in the CLI, gateway, TUI, and desktop app. They automatically inherit Hermes' multi-agent delegation, session memory, and caching infrastructure.

---

## Imported Skills

### 1. **Orchestration & Workflow** (1 skill)

#### `hands-free`
Convert a desired end result into a precise `/goal [...]` invocation that a **Lead orchestration agent** executes autonomously.

- **Trigger:** User states an outcome they want driven to completion hands-free
- **Input:** Plain-language desired result (e.g., "ship the landing page", "I want to end up with searchable docs")
- **Output:** A self-contained `/goal` block for a Lead agent to execute step-by-step
- **Key capabilities:**
  - Bounded exploration (2–4 high-leverage questions)
  - Linear milestone spine + parallel sub-agent ribs
  - Explicit approval/review gates for outward-facing work
  - Built-in assumptions and verification

**References:**
- `references/goal-spec.md` — canonical `/goal [...]` structure
- `references/agent-roster.md` — catalog of skilled sub-agents and parallelization rules
- `references/exploration.md` — context-refinement question checklist
- `examples/example-run.md` — worked example from request to `/goal`

---

### 2. **Content & Writing** (4 skills)

#### `scribe-brand-voice-guardian`
Enforce brand voice consistency across all content. Use before finalizing any piece destined for public consumption.

- **Purpose:** Verify tone, terminology, and personality match brand guidelines
- **Input:** Draft content (copy, social posts, email, landing page, etc.)
- **Output:** Critique with corrected voice + revised version

#### `scribe-editor-critic`
Professional editorial review. Catches redundancy, clarity issues, flow, tone consistency, and missed opportunities.

- **Purpose:** Second-pass editing for completeness and craft
- **Input:** Draft or semi-final content
- **Output:** Annotated feedback + sharp, publication-ready revision

#### `scribe-platform-adapter`
Adapt content to platform-specific norms, formats, and audience expectations.

- **Purpose:** Reformat content for multi-platform distribution (Twitter, LinkedIn, newsletter, blog, etc.)
- **Input:** Canonical source content or broad idea
- **Output:** 2–5 platform-specific versions with audience tone and format baked in

#### `scribe-social-post-writer`
Write high-engagement social media posts from raw material, research, or existing content.

- **Purpose:** Generate shareable social content that drives traffic and conversation
- **Input:** Topic, angle, reference link, or repurposing source
- **Output:** 3–5 distinct post options with hook, body, CTA, and hashtag strategy

---

### 3. **Design Signal Scouting** (10 skills)

The design signal scouting cluster monitors emerging design, technology, and creative trends across footwear, product design, 3D, CAD, creative AI, and adjacent domains. These skills work together in a coordinated system:

**Signal flow:** `signal-scout` → (`visual-trend-analyzer` | `technology-radar` | `pain-point-miner`) → `trend-clusterer` → `content-opportunity-generator` | `inspiration-curator` → `weekly-creative-brief`

#### `signal-scout`
Entry point. Run a scouting cycle to discover high-value design and technology signals.

- **Trigger:** "What's new in [domain]?", "Find me signals on X", "Scan for anything interesting in Blender/footwear/creative AI"
- **Process:**
  1. Generate broad and narrow searches
  2. Search primary and specialist sources
  3. Collect and deduplicate candidates
  4. Verify material claims
  5. Classify and score (0–40 range)
  6. Route to deeper analysis skills as needed
- **Output:** Scored signals with categories, evidence, momentum, and potential use

#### `visual-trend-analyzer`
Analyze emerging visual language, motifs, aesthetics, and design direction changes.

- **Input:** Image references, design collections, mood boards, or visual signals
- **Output:** Pattern analysis with design implications and commercial potential

#### `technology-radar`
Track tools, software, workflows, and emerging capabilities in design software and creative coding.

- **Input:** New releases, open-source projects, feature announcements, community discussion
- **Output:** Tool assessments with adoption roadmap and skill implications

#### `pain-point-miner`
Extract actionable pain points from designer communities (Reddit, Discord, forums, GitHub issues).

- **Input:** Community signals, complaints, wishlist threads, "how do I?" posts
- **Output:** Clustered pain points with frequency, severity, and potential solutions

#### `trend-clusterer`
Group related signals into coherent trends; identify emerging themes across multiple independent sources.

- **Input:** Multiple scattered signals from signal-scout, visual-trend-analyzer, technology-radar, pain-point-miner
- **Output:** Consolidated trend buckets with pattern analysis and commercial implications

#### `content-opportunity-generator`
Identify actionable content opportunities from signals, trends, and gaps in existing coverage.

- **Input:** Signals, trends, pain points, and audience interests
- **Output:** Ranked content ideas with format, angle, SEO potential, and audience reach

#### `inspiration-curator`
Surface design inspiration and creative references that match a specific brief or exploration direction.

- **Input:** Aesthetic direction, technical specification, or problem statement
- **Output:** Curated inspiration from diverse sources with annotations on why each is relevant

#### `source-manager`
Maintain and expand the intelligence network — track sources, assess reliability, identify gaps in coverage.

- **Input:** Current source list and coverage assessment
- **Output:** Updated source strategy with new recommendations, deprecations, and coverage gaps

#### `signal-feedback-loop`
Convert signals into feedback on existing products, work, and strategy.

- **Input:** Signals + current work/product
- **Output:** Structured feedback with validation against market reality

#### `weekly-creative-brief`
Distill a week's signals, trends, and opportunities into an actionable creative brief.

- **Input:** Weekly signal collection + strategic priorities
- **Output:** Briefing document with top signals, trends, content opportunities, and recommended actions

---

## How to Use

### Quick Start

1. **Orchestration:** 
   ```
   /hands-free I want to ship a research report on emerging footwear manufacturing techniques with visualizations and a creative brief
   ```

2. **Content:**
   ```
   /scribe-brand-voice-guardian [paste draft]
   /scribe-social-post-writer emerging AI color-rendering techniques in Blender
   ```

3. **Signal Scouting:**
   ```
   /signal-scout what's new in procedural geometry for product design?
   /weekly-creative-brief
   ```

### Integration with Agents

These skills work seamlessly with Hermes' multi-agent system:

- Use `/hands-free` to architect a complex outcome → it emits a `/goal` command
- The Lead orchestration agent parses the `/goal` and deploys sub-agents in parallel
- Sub-agents use the scribe and scouting skills to execute their assigned milestones
- Results feed back into the conversation, memory, and Wiki

### In the Dashboard & Desktop App

All skills are available:
- **CLI:** slash commands + autocomplete
- **TUI:** `/skill-name` in the composer
- **Dashboard:** embedded in the chat interface
- **Desktop app:** slash palette (curated in `apps/desktop/src/lib/desktop-slash-commands.ts`)

---

## File Structure

Each skill follows the Hermes skill structure:

```
skill-name/
├── SKILL.md                    # Main skill definition (frontmatter + body)
├── references/                 # Supporting reference materials
│   ├── operating-model.md      # (design-signal-scout cluster)
│   ├── goal-spec.md            # (hands-free)
│   ├── agent-roster.md         # (hands-free)
│   └── exploration.md          # (hands-free)
└── examples/                   # Worked examples
    └── example-run.md          # (hands-free)
```

All reference materials are copied verbatim from the source. They remain editable — if you discover gaps or want to expand the roster, update the files directly in `~/.hermes/hermes-agent/skills/skill-name/references/`.

---

## Deployment & Agentic Activation

### Self-Contained `/goal` Format

The `hands-free` skill emits `/goal` commands that follow this structure (see `references/goal-spec.md` for the full spec):

```
/goal
title: [Goal Title]
description: [What we're building and why]
milestones:
  - title: [Milestone 1]
    description: [What gets done]
    subtasks:
      - task: [Task description]
        owner: [Agent/Skill]
        definition_of_done: [Measurable completion criteria]
      - task: [...]
        owner: [...]
        definition_of_done: [...]
    gate: [optional approval/review step]

  - title: [Milestone 2]
    ...

discovery_window: [Ask the user 2–4 context-refining questions]
success_criteria:
  - [Criterion 1]
  - [Criterion 2]
assumptions:
  - [Assumption 1; user corrects before launch]
```

### Agent Roster (for orchestration)

The `agent-roster.md` in `hands-free/references/` lists all available sub-agents and their specialties:

- **Scribe Agent** — writing, editing, brand voice
- **Design Signal Scout Agent** — research, trend analysis, opportunity discovery
- **Technical Agent** — implementation, debugging, optimization
- **Content Agent** — blog posts, tweets, newsletters
- (Plus routing rules for parallelization)

When the Lead agent encounters a subtask with `owner: [agent-name]`, it delegates to that agent in parallel with others at the same milestone level.

---

## Maintenance & Evolution

### Adding New Sub-Agents to the Roster

Edit `skills/hands-free/references/agent-roster.md`:
1. Add the agent name and specialty
2. List the tasks it's best suited for
3. Document any parallelization constraints (dependencies with other agents)
4. Save and commit

No code changes needed — the orchestrator reads the roster at runtime.

### Expanding Signal-Scout Coverage

Edit `skills/design-signal-scout-signal-scout/SKILL.md`:
1. Add new domains to the **Search Themes** section
2. Add new patterns to **Discovery Queries**
3. Adjust the **Signal Threshold** if criteria have changed

The skill is fully self-contained; updates take effect immediately.

### Updating Brand Voice or Tone

Edit `skills/scribe-brand-voice-guardian/SKILL.md` and references:
1. Update tone definitions, terminology, or personality traits
2. Add examples of correct vs. incorrect voice
3. Document any off-brand pitfalls to watch for

---

## Prompt Caching & Performance

All 15 skills are now discoverable and cached by Hermes' prompt caching layer:

- **First call** to a skill loads its full SKILL.md + references into the prompt context
- **Subsequent calls** reuse the cached prefix (same session)
- **Multi-session memory** stores which skills you use frequently (available via `memory` tool)

No warm-up needed — skills are live from the moment this integration completes.

---

## Gateway & Platform Availability

All skills are immediately available across all Hermes platforms:

| Platform | Access | Notes |
|----------|--------|-------|
| **CLI** | `/skill-name` | Autocomplete works; skill commands injected as user messages |
| **TUI** | `/skill-name` in composer | Full slash palette with curation |
| **Gateway (Telegram, Discord, Slack, etc.)** | `/skill-name` | Slash commands routed through `_SlashWorker` |
| **Desktop app** | Slash palette | Curated in `desktop-slash-commands.ts` |
| **Dashboard** | Embedded TUI | Same as TUI; PTY-backed |

Skill commands always preserve prompt caching — they're injected as user messages, not system-prompt mutations.

---

## Troubleshooting

### Skills not appearing in autocomplete
Restart Hermes CLI or reload the gateway:
```bash
hermes /reload
# or restart the process
```

### Skill reference files not found
Verify the skill directory structure:
```bash
ls -la ~/.hermes/hermes-agent/skills/hands-free/references/
```
All `.md` files should be present. If missing, re-copy from the source:
```bash
cp -r /home/matt/workspace/github/mission-control/.claude/skills/[skill-name] ~/.hermes/hermes-agent/skills/
```

### `/goal` command not parsed correctly
Ensure the YAML in the `/goal` block matches the spec exactly. Common issues:
- Indentation (use 2 spaces, not tabs)
- Missing colons after field names
- Quoted strings without escaping internal quotes

See `examples/example-run.md` for a working template.

---

## Quick Reference

| Skill | Trigger | Output |
|-------|---------|--------|
| `hands-free` | "I want X done" | `/goal [...]` command |
| `signal-scout` | "What's new in Y?" | Scored signals + categories |
| `scribe-brand-voice-guardian` | "Check this draft" | Voice critique + revision |
| `scribe-editor-critic` | "Improve this copy" | Editorial feedback + polish |
| `scribe-platform-adapter` | "Make this for Twitter/LinkedIn" | Platform-specific variants |
| `scribe-social-post-writer` | "Write about X" | 3–5 social post options |
| `visual-trend-analyzer` | [images + design brief] | Visual pattern analysis |
| `technology-radar` | "Track tools in Z" | Tool assessments + roadmap |
| `pain-point-miner` | "What are designers complaining about?" | Clustered pain points |
| `trend-clusterer` | [multiple signals] | Consolidated trend buckets |
| `content-opportunity-generator` | [signals + brief] | Ranked content ideas |
| `inspiration-curator` | [aesthetic direction] | Curated visual references |
| `source-manager` | "Update intelligence network" | Source strategy + gaps |
| `signal-feedback-loop` | [signals + product] | Structured market feedback |
| `weekly-creative-brief` | "What should we focus on?" | Weekly action brief |

---

## Next Steps

1. **Verify** skills are discoverable in your environment:
   ```bash
   cd ~/.hermes/hermes-agent
   ls skills/ | grep -E "(hands-free|scribe|design-signal)"
   ```

2. **Test** a quick skill invocation:
   ```bash
   hermes "use the hands-free skill to turn this into a /goal command: I want to build a research report on emerging footwear tech"
   ```

3. **Set up signal-scouting automation** (optional):
   - Create a cron job that runs `/weekly-creative-brief` on Monday mornings
   - Route output to Telegram or Discord for async reading

4. **Customize the agent roster** for your specific sub-agents and team structure

5. **Extend coverage** by editing signal-scout themes or adding new scribe variants

---

**Integration complete.** All skills are now live, discoverable, and ready for agentic deployment.
