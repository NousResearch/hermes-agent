---
name: knowledge
description: Use for looking up, saving, indexing, reorganizing, or cleaning durable YallaPlay knowledge in the yallaplay-wiki submodule.
version: 1.1.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [knowledge, wiki, obsidian, documentation, indexing, yallaplay]
    related_skills: [analytics, codebase-readonly, coder]
---

# Knowledge Skill

## Overview

Use this skill whenever the task is to find, preserve, reorganize, or clean durable YallaPlay knowledge. The wiki is the durable knowledge base for Claudio; memory is only for user preferences and stable personal/environment facts.

This skill implements the current `yallaplay-wiki/` contract: domain-first layout, self-contained pages, relative links, and one canonical page per thing.

## When to Use

- Looking up project knowledge before answering.
- Adding or editing wiki pages, indexes, dictionary entries, glossary entries, findings, methodologies, schema annotations, or runbooks.
- Reorganizing wiki layout or fixing links.
- Deciding whether a discovery belongs in memory, a skill, the wiki, or nowhere.

Load the domain skill too when the knowledge belongs to Analytics, LiveOps, Engineering, Collaboration, or Infrastructure.

## Required Context

Read before editing the wiki:

- `yallaplay-wiki/README.md` — page contract and layout rules.
- `yallaplay-wiki/index.md` — top-level routing.
- `yallaplay-wiki/dictionary.md` and `glossary.md` for terms.
- The target domain index, e.g. `products/index.md`, `operations/index.md`, `tools/index.md`, `engineering/index.md`, or `reference/warehouse/index.md`.
- `yallaplay-wiki/CLAUDE.md` if working directly inside the wiki submodule.

## Wiki Layout Rules

- Top-level folders are domains, not document types.
- Game systems live under `products/core/<system>/`, `products/spades/<feature>/`, or `products/rummy/<feature>/` with facet files: `index.md`, `design.md`, `engineering.md`, `analytics.md`, `findings.md`, `facts.md`.
- Warehouse schema/reference lives under `reference/warehouse/`: curated human pages (`index.md`, `events.md`, `aggregates.md`) plus generated DDL in `ddl/`.
- Tool pages are canonical under `tools/product/` or `tools/team/`, not duplicated in engineering/operations.
- Journal pages are records, not canonical reference. Distill reusable facts from them into domain pages.

## Lookup Workflow

1. **Search first.** Use `python3 tools/wiki_search.py <terms>` and, when relevant, `--domain <domain>`. Completion: likely existing canonical pages are identified.
2. **Open indexes.** Use top-level/domain indexes for routing before creating new pages. Completion: destination domain is justified.
3. **Read the destination.** Inspect the most-specific page before editing. Completion: no duplicate canonical page is created.
4. **Answer with links.** Cite repo-relative wiki paths when useful. Completion: future reader can navigate to source.

## Capture Workflow

Save only reusable knowledge:

- **Definitions:** metric/business concept meaning, usually in `glossary.md`, `dictionary.md`, or a domain reference page.
- **Methodology:** repeatable investigation approach, caveat, or bias trap.
- **Findings:** concrete result with future value and caveats.
- **Queries:** reusable SQL/query pattern near the domain that uses it.
- **Schema/reference:** durable table/column/tool behavior under `reference/warehouse/` or the relevant tool page.
- **Operations:** incidents, A/B tests, runbooks under `operations/`.

Do not save secrets, unsupported speculation, one-off scratch output, stale task progress, PR numbers, or facts that will expire in days.

## Agent Quick Context → Compiled Skills

For high-frequency workflows, store the source snippet in the wiki and compile it into a git-tracked skill:

1. Add an `agent_skill:` block to the wiki page frontmatter with `name`, `category`, `description`, `tags`, and `related_skills`.
2. Add a `## Agent quick context` section containing the compact runtime fast path: trigger terms, tables/events, denominator defaults, known traps, and query skeletons.
3. Run `python3 tools/compile_wiki_skills.py --name <skill-name> --profile-symlink-root ~/.hermes/profiles/claudio-lab/skills/claudio-authored`.
4. Commit the generated repo skill under `skills/yallaplay/<skill-name>/SKILL.md`; the profile path should be a symlink back to that repo directory.

Maintenance rule: update the wiki section first, regenerate the skill, and treat the generated skill as a compiled cache. If the wiki and generated skill disagree, the wiki wins.

## Editing Workflow

1. **Classify durable vs transient.** Completion: the target belongs in wiki, skill, memory, or nowhere.
2. **Find canonical owner.** Completion: most-specific domain/page is selected.
3. **Read current content.** Completion: edit does not duplicate or contradict existing page.
4. **Patch narrowly.** Completion: internal links are relative and page remains self-contained.
5. **Update entry points.** Add/update dictionary entries, relevant domain index, and log when adding/changing substantial pages. Completion: a term-first and structure-first reader can find it.
6. **Verify links.** For touched pages, check relative links resolve. Completion: no broken links introduced.

## Tools

- `python3 tools/wiki_search.py <query>` — compact search.
- `python3 tools/wiki_search.py <query> --domain analytics --files-only` — domain-routed file discovery.
- `python3 tools/wiki_new.py <kind> <title>` — draft a page from a safe template.
- `python3 tools/fetch_schemas.py` — refresh generated warehouse DDL under `reference/warehouse/ddl/`.

## Safety

- Do not overwrite wiki files without reading them first.
- Do not link to another repo's markdown/knowledge files; absorb needed content inline. Code path references are allowed.
- Do not store secrets, personal data, or unsupported speculation.
- Keep generated DDL separate from curated prose.
- When moving pages, update all internal links and the relevant index/dictionary entries.

## Common Pitfalls

1. **Document-type top-level folders.** Prefer domain owners (`reference/warehouse/`) over root folders like `/schema`.
2. **Duplicate canonical pages.** Link to the owner instead of restating.
3. **Dictionary bloat.** Dictionary points; glossary/domain pages explain.
4. **Memory misuse.** Wiki stores organizational knowledge; memory stores user preferences/environment facts.
5. **Broken relative links after moves.** Resolve links from the moved file's new directory.

## Verification Checklist

- [ ] `README.md` contract was followed.
- [ ] Most-specific canonical owner was used.
- [ ] Dictionary/index/log updates were made when substantial pages changed.
- [ ] Internal links are relative and resolve.
- [ ] No secrets, unsupported speculation, or one-off scratch output were saved.
