---
title: Hermes Agent Knowledge Intake Router
tags:
  - hermes-agent
  - knowledge-center
  - intake-router
status: active
updated: 2026-05-24
---

# Knowledge Intake Router

The Knowledge Intake Router is the missing layer above the restored HermesAgent
Knowledge Center. It classifies raw knowledge before it becomes a permanent
artifact and must route through the old HermesNous graph that is now renamed as
`~/ObsidianVault/HermesAgent`.

The router must not rebuild the short-lived standalone vault layout. Its default
Obsidian targets are the existing layer folders: `sources`, `knowledge`,
`lessons`, `patterns`, `playbooks`, `review-queue`, `training-packs`, and
`skills`.

## Destinations

| Destination | Meaning | Next action |
|---|---|---|
| `skill_candidate` | Repeatable procedural capability | Review, then create or patch a Hermes Skill |
| `agent_candidate` | Role, persona, routing, delegation, or ownership behavior | Review, then create or update an Agent profile |
| `domain_knowledge` | Cross-project reusable technical knowledge | Add to review queue, then promote to domain KB |
| `workspace_knowledge` | Portfolio, strategy, roadmap, glossary, or project-relationship knowledge | Review, then merge into workspace notes |
| `playbook_candidate` | Global operating procedure or runbook | Review, then write a playbook |
| `project_note` | Project-specific context | Keep with project context/project card |

## Runtime Tool

Tool: `capture_knowledge`

Actions:

| Action | Behavior |
|---|---|
| `classify` | Return destination, confidence, domains, rationale, related files |
| `capture` | Classify, merge into an existing high-confidence note, or write a pending review-queue note |
| `list` | List captured intake notes |
| `sync_maps` | Sync Skill cards, Agent profile cards, relation index, and routing DB |

`domain_knowledge` captures can also enqueue the existing `review_knowledge`
flow so user approval still gates promotion.

Captures can include `source_url` and `source_title`. The router stores source
lineage in frontmatter, records a short content hash, and uses the URL for
duplicate detection. If the content cannot be fetched from a source such as a
login-gated Facebook post, store the URL first and add pasted content later.

## Merge-First Policy

Before creating another permanent note, the router searches likely merge
targets:

| Destination | Merge candidates checked first |
|---|---|
| `workspace_knowledge` | `knowledge/`, `docs/`, `patterns/`, `sources/`, `review-queue/` |
| `domain_knowledge` | `knowledge/`, `patterns/`, `lessons/`, `sources/`, `review-queue/` |
| `playbook_candidate` | `playbooks/`, `patterns/`, `lessons/`, `review-queue/` |
| `project_note` | `sources/`, `knowledge/`, `lessons/`, `reports/`, `review-queue/` |
| `skill_candidate` | `skills/`, `docs/`, `review-queue/` |
| `agent_candidate` | `docs/`, `skills/conductor/`, `review-queue/` |

Signals:

- same source URL;
- title/content token overlap;
- destination match;
- workspace/domain/project relation context.

Scores >= `0.8` are safe merge recommendations. Lower scores remain review
candidates so the user can decide.

## Obsidian Outputs

| Path | Purpose |
|---|---|
| `~/ObsidianVault/HermesAgent/review-queue/intake-index.md` | Intake queue index |
| `~/ObsidianVault/HermesAgent/review-queue/intake-*.md` | Pending items only when no safe merge exists |
| `~/ObsidianVault/HermesAgent/review-queue/knowledge-routes.json` | Route snapshot |
| `~/ObsidianVault/HermesAgent/docs/OBSIDIAN_LINK_INDEX.md` | Knowledge Center link index |
| `~/ObsidianVault/HermesAgent/docs/AI_SKILL_ROUTER.md` | Skill/agent routing |
| `~/ObsidianVault/HermesAgent/docs/SKILL_GRAPH.md` | Skill/knowledge relation graph |
| `~/.hermes/knowledge_routes.json` | Runtime routing DB snapshot |

## Classification Inputs

The classifier uses deterministic local heuristics:

- destination keywords, such as Skill sections (`When to Use`,
  `Prerequisites`, `Procedure`, `Pitfalls`, `Verification`);
- Agent words, such as role, profile, responsibility, routing, handoff,
  delegate, orchestrator, worker;
- workspace words, such as portfolio, roadmap, strategy, project cluster,
  ecosystem, dependency, decision principle, glossary, taxonomy, and
  workspace-wide Thai terms such as `ภาพรวม` and `โปรเจกต์ทั้งหมด`;
- playbook/runbook/incident/rollback/phase terms;
- domain matching through `DomainRelevanceMatcher`;
- source project domain tags from Obsidian project notes;
- related Skill cards from the runtime skills root.

No network calls are used.

## Verification

Current verification command:

```bash
scripts/run_tests.sh tests/knowledge_center/ -q
```

Latest local result: 95 tests passed on 2026-05-24.
