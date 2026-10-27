---
name: skill-graph
description: "Use when a request needs domain workflow discovery."
version: 2.0.0
author: Hauzer S. Lee
license: MIT
platforms: [linux, macos, windows]
category: hermes
metadata:
  hermes:
    tags:
      - hermes
      - skills
      - discovery
      - 意图路由器
      - routing
      - classification
    relations:
      - type: complemented_by
        target: quality-gate
        properties:
          reason: Routing + validation complete the pipeline
          strength: strong
---

# Skill Graph — Intent Routing + Discovery

Load this skill first when a request needs domain workflow discovery. Skip it
for the prompt protocol's built-in-tool short circuit, and load a matching
pre-injected candidate directly when the user message already supplies one.

## Protocol

```
1. skill_load("skill-graph") → read the routing tables below
2. Classify the input using Phase 1
3. Find the matching entry in Phase 4 routing table
4. Call skill_graph_search() with the query from that entry
5. skill_load("result-name") → execute
6. skill_load("quality-gate") → final validation

Use `skill_graph_search()` before broad filesystem discovery. Known-file reads
and direct execution do not require a second graph search.
Never plan from scratch — the graph has the skills you need.
```

## Loader Protocol

After loading a skill via `skill_load()`, verify it fits the current task:

- **Scope mismatch?** If the loaded skill is a pipeline / multi-step
  wrapper but the task is a single simple operation — go back and
  search again with narrower terms.
- **Description mismatch?** If the skill's description clearly doesn't
  match the task — call `skill_graph_search()` again with different
  keywords.
- **Good match?** Continue with execution.

Pipelines should declare their scope explicitly in the description
so the agent can judge fit before executing.

## Phase 1: Input Classification

| Type | Description | Initial route |
|------|-------------|---------------|
| 1A Task mgmt | Task name/path/hash mentioned | `skill_graph_search("task workflow")` |
| 1B Execute | "run"/"do it"/"commit" | → Phase 2, then Phase 4 |
| 1C Design discussion | Suggestion / proposal / question | Discuss only, don't execute |
| 1D Info query | "What is"/"show me"/"check project" | **Search for a skill first** |
| 1E Meta | Change config/skill/memory | Handle directly |

**Key: 1D info queries must not just "answer directly"**. Many info
queries ("show the project", "explain the architecture") have a matching
domain skill. Search the graph first, then answer following its guidance.

## Phase 2: Intent Resolution (1B only)

| Signal | Action |
|--------|--------|
| "commit" | Git workflow (commit, no push) |
| "push" | Allow push |
| "stop" / "wait" | Stop immediately, wait silently |
| All done | `skill_load("quality-gate")` |

## Phase 3: Pre-flight Check

| Target | Tool | Check |
|--------|------|-------|
| Task directory | task-framework tools | Read TASK_MEMORY.md first |
| Git repo | Git commands | Pre-change sync |
| Skill file | skill_manage | — |
| Rule file | read_file / write_file | — |

## Phase 4: Routing

After classification, search with intent keywords — NOT the user's
exact words. The graph scores results by FTS5, term matches, and
relationship traversal, so **provide multiple relevant keywords**
to increase the chance of matching the right skill:

| User said | Don't search | Search instead |
|-----------|-------------|----------------|
| "show the eir project" | "show the eir project" | `skill_graph_search("project overview structure architecture discovery")` |
| "what's wrong with this bug" | "what's wrong" | `skill_graph_search("debug root cause analysis investigation")` |
| "help design the database" | "help design the database" | `skill_graph_search("database schema data model naming conventions")` |

### Routing table

The table below maps common intent types to keyword-rich search queries.
Add more keywords relevant to the specific context — more terms increase
the graph's ability to find the right match via its scoring pipeline.

| User intent | Search query (add more context keywords!) |
|------------|------------------------------------------|
| View / understand project | `skill_graph_search("project overview structure analysis document")` |
| Create / manage tasks | `skill_graph_search("task workflow management lifecycle")` |
| Git operations | `skill_graph_search("git commit push branch workflow")` |
| Code review | `skill_graph_search("code review static analysis audit quality")` |
| Write PRD / requirements | `skill_graph_search("product requirements document specification")` |
| Debug a program | `skill_graph_search("debug root cause analysis investigation")` |
| Design / prototype | `skill_graph_search("design prototype mockup wireframe")` |
| Architecture analysis | `skill_graph_search("architecture design analysis discovery")` |
| Domain research | `skill_graph_search("domain analysis market research feasibility")` |
| Deploy a service | `skill_graph_search("deploy service server container")` |
| Video production | `skill_graph_search("video production recording editing")` |
| Database design | `skill_graph_search("database schema data model naming")` |

## Domain-Specific Routing

For routing extensions specific to the installed skill ecosystem
(scene activation, troupe entity lookup, pipeline context),
load the file configured at `skills.config.skill-graph.extensions_file`
in config.yaml. Relative paths resolve under the active profile's
`HERMES_HOME`; no extensions file is loaded unless one is configured.

This file extends the Phase 4 Routing Table and Phase 3 Pre-flight Check
with domain-specific entries.

## Slash Commands

The plugin registers ``/skill-graph`` with subcommands:

- ``/skill-graph search <query>``  Search skills by intent
- ``/skill-graph list``            List all skills in graph (name + description)
- ``/skill-graph status``          Show graph stats (counts, DB size)
- ``/skill-graph config``          Show configuration (paths, scanned dirs)
- ``/skill-graph rebuild``         Force full graph rebuild

## Tools

- ``skill_graph_search(query)`` — Find skills by intent (PREFERRED over skills_list())
- ``skill_load(name)`` — Load a skill's full content (alternative to skill_view())

## Configuration

Add extra skill directories in ``config.yaml``:

```yaml
skills:
  config:
    skill-graph:
      source_dirs:
        - ~/path/to/extra/skills
```

The active profile's ``$HERMES_HOME/skills/`` directory and bundled skills
are always scanned. Additional shared locations must be configured explicitly.

## Understanding Results

Each search result has a ``relevance`` field:

| Value | Meaning |
|-------|---------|
| ``direct`` | FTS5 match in name/description/tags |
| ``tag_match`` | Tag-based match |
| ``expansion`` | Graph traversal from another matched skill |
| ``term_match`` | Auto-extracted keyword match (English + Chinese) |

## Adding Relations to Skills

To make the graph smarter, add a ``relations`` field to any SKILL.md:

```yaml
metadata:
  hermes:
    relations:
      - type: depends_on
        target: systematic-debugging
        properties:
          reason: "must find root cause before fixing"
          strength: strong
```

### Relation Types

| Type | Meaning | Auto-reverse? |
|------|---------|---------------|
| ``depends_on`` | This skill needs another to work | → ``supported_by`` |
| ``supported_by`` | Another skill supports this one | → ``depends_on`` |
| ``complemented_by`` | Works well with another skill | Symmetric |
| ``alternative_to`` | Alternative approach for same task | Symmetric |
| ``similar_to`` | Semantically similar | Symmetric |
| ``supersedes`` | Replaces an older skill | → ``superseded_by`` |
| ``used_in_workflow`` | Part of a larger workflow | Symmetric |
| ``belongs_to_domain`` | Domain category relationship | Symmetric |

## Pitfalls

- The graph is rebuilt on every session start — changes to SKILL.md take
  effect after restarting Hermes (``/reset`` or exit+relaunch).
- Legacy ``related_skills`` fields are auto-converted to ``similar_to``
  relations. Ensure they're accurate.
- Chinese text is indexed by FTS5 but doesn't benefit from porter stemming.
  Use explicit tags and relations for Chinese-dominated skills.

## Creating Skills

When creating a new skill with a `skill_manage` `create` operation,
Hermes writes it to the active profile's `$HERMES_HOME/skills/<name>/`.
The plugin refreshes affected graph nodes after successful `skill_manage`
batches. Run `/skill-graph rebuild` after out-of-band filesystem changes.
