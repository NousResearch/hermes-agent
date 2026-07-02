---
name: yallaplay-llm-wiki
description: "Use when applying the Karpathy/llm-wiki durable-knowledge workflow to the yallaplay-wiki repo: orient, query, ingest, cross-reference, lint, and log while preserving YallaPlay's domain-first wiki contract."
version: 1.0.0
author: Claudio
license: Proprietary
metadata:
  hermes:
    tags: [yallaplay, wiki, knowledge-base, markdown, ingest, lint]
    related_skills: [llm-wiki, knowledge, yallaplay-wiki-knowledge]
---

# YallaPlay LLM Wiki Workflow

## Overview

Use this skill to apply the `llm-wiki` discipline to `yallaplay-wiki/` without importing the generic `SCHEMA.md`, `raw/`, `entities/`, `concepts/`, or `queries/` layout.

The operating model is: orient first, answer from compiled knowledge, ingest primary sources into the record layer, distill reusable facts into canonical domain pages, keep links/indexes current, and log durable changes. The schema of record is the local `yallaplay-wiki/README.md` contract.

## When to Use

- The user asks to use the LLM Wiki / Karpathy wiki pattern with YallaPlay knowledge.
- Querying, ingesting, linting, or reorganizing `yallaplay-wiki/`.
- Turning meeting notes, research, analytics findings, or implementation discoveries into durable company knowledge.
- Auditing whether the wiki behaves like a compounding knowledge base rather than a pile of markdown files.

Also load the relevant domain skill when the subject is Analytics, LiveOps, Engineering, Collaboration, Infrastructure, Observability, or product-specific work.

## Wiki Path

Prefer the repo wiki when present:

- `WIKI_PATH=/home/ubuntu/git/yallaplay-hermes-agent/yallaplay-wiki`
- fallback: `./yallaplay-wiki` from the Hermes pilot repo

Do not assume `~/wiki`; that is the generic `llm-wiki` default and is wrong for Claudio unless the user explicitly points there.

## Contract Mapping from Generic llm-wiki

| Generic `llm-wiki` concept | YallaPlay wiki equivalent |
|---|---|
| `SCHEMA.md` | `README.md` + `CLAUDE.md` when inside the submodule |
| `index.md` | root `index.md` plus domain `index.md` files |
| `log.md` | root `log.md` append-only build/ingest log |
| `raw/` | `journal/` for internal primary records; `literature/sources/` for external canon; generated snapshots beside their owning reference page |
| `entities/`, `concepts/` | most-specific domain owner: `products/`, `people/`, `engineering/`, `tools/`, `operations/`, `market/`, `literature/`, `reference/` |
| `queries/` | reusable SQL/analysis patterns near the owning domain, often `reference/warehouse/`, product `analytics.md`, or methodology pages |
| `[[wikilinks]]` | relative markdown links, e.g. `[Piggy Bank](products/core/piggy_bank/index.md)` |
| generic frontmatter | YallaPlay frontmatter from `README.md`: `title`, `domain`, `status`, `source_refs`, `see_also`, `built` |

Never create new root folders named `raw`, `schema`, `entities`, `concepts`, `queries`, `findings`, or `scratch` unless the user explicitly changes the wiki contract.

## Mandatory Orientation

Before answering from or editing the wiki:

1. **Read the contract.** Read `yallaplay-wiki/README.md`; read `yallaplay-wiki/CLAUDE.md` if operating directly inside the wiki submodule. Completion: the page contract and hard rules are known.
2. **Read entry points.** Read root `index.md`, `dictionary.md`, and `glossary.md` as needed. Completion: term-first and structure-first routes are known.
3. **Search before opening pages.** Run `python3 tools/wiki_search.py <terms>` when available; otherwise use `search_files` over `yallaplay-wiki/*.md`. Completion: likely canonical pages and duplicates are identified.
4. **Read the owner.** Open the most-specific domain index and target page before creating or editing. Completion: no duplicate canonical page is created.
5. **Check recent log for substantial work.** Read the tail of `log.md` when ingesting, reorganizing, or auditing. Completion: recent related changes are known.

## Query Workflow

When the user asks a knowledge question:

1. **Route by domain.** Use `dictionary.md`, root `index.md`, domain indexes, and `tools/wiki_search.py`. Completion: relevant pages are listed.
2. **Read canonical pages, not just search snippets.** Completion: answer is grounded in full page context.
3. **Synthesize with citations.** Cite repo-relative paths such as `yallaplay-wiki/products/core/piggy_bank/analytics.md`. Completion: a future reader can navigate to every material claim.
4. **File only reusable synthesis.** If the answer is a non-trivial methodology, definition, finding, or comparison that would be painful to rederive, save it to the canonical owner and update entry points. Otherwise, leave it in chat. Completion: wiki does not accumulate one-off scratch.
5. **Log filed work.** Append `## [YYYY-MM-DD] query | <subject>` or an appropriate operation line only when files change. Completion: durable changes are discoverable.

## Ingest Workflow

When adding source material:

1. **Classify the source.** Internal meeting/support/ops source goes under `journal/`; external design/research canon goes under `literature/sources/`; generated technical snapshots live beside their owning reference page (for example `reference/warehouse/ddl/`). Completion: the source has the right epistemic layer.
2. **Preserve records, distill reference.** Records are dated and immutable; reference pages are maintained and deduplicated. Completion: primary material is not overwritten by synthesis.
3. **Find existing owners.** Search the dictionary, glossary, domain indexes, and content before creating pages. Completion: canonical owners are known.
4. **Patch narrow canonical pages.** Add durable facts, caveats, definitions, query patterns, or findings to the most-specific page. Completion: content is self-contained and not duplicated elsewhere.
5. **Use YallaPlay frontmatter and links.** New/changed pages satisfy the local page contract and use relative markdown links. Completion: Obsidian/GitHub navigation works.
6. **Update entry points.** For substantial changes, update `dictionary.md`, relevant domain `index.md`, and `log.md`. Completion: readers can find the knowledge by term and by structure.
7. **Ask before mass updates.** If an ingest will touch 10+ pages or restructure a domain, confirm scope first. Completion: visible broad edits are user-approved.

## Capture Rules

Save to the wiki when the knowledge is reusable for future Claudio runs or teammates:

- Definitions and shared vocabulary: `glossary.md` or the canonical domain page, with `dictionary.md` pointing to it.
- Product-system knowledge: `products/core/<system>/` or game-specific product folders, split into persona facets.
- Analytics methods and caveats: `reference/analytics_rules.md`, product `analytics.md`, or `reference/warehouse/`.
- Warehouse annotations: `reference/warehouse/` with generated DDL under `reference/warehouse/ddl/`.
- Tool behavior: `tools/product/` or `tools/team/` canonical pages.
- Incidents, A/B tests, and runbooks: `operations/`.
- External game-design canon: `literature/`, with source material in `literature/sources/`.

Do not save secrets, unsupported speculation, stale task progress, PR numbers, temporary TODOs, or facts likely to expire within days.

## Lint / Health-Check Workflow

Adapt generic `llm-wiki` lint to YallaPlay rules:

1. **Frontmatter validation.** Every page has required local fields from `README.md`; game facets have their facet-specific fields. Completion: missing fields are reported by path.
2. **Relative-link validation.** Every internal markdown link resolves from the file's directory. Completion: broken links are listed with source and target.
3. **No external knowledge-file links.** Pages do not link to markdown/knowledge files in other repos; allowed outward references are repo-root-relative code paths only. Completion: violations are listed.
4. **Index and dictionary coverage.** New canonical pages and important terms appear in the relevant domain index and `dictionary.md`. Completion: missing entry points are listed.
5. **Canonical-owner audit.** Flag likely duplicate pages or repeated explanations across domains. Completion: suggested owner is named.
6. **Facet completeness.** Product-system folders have expected facet files and each facet can stand alone cold. Completion: missing facets/orientation blocks are listed.
7. **Log hygiene.** Substantial durable changes have a `log.md` entry. Completion: missing or stale log entries are reported.

Group findings by severity: broken links and contract violations first; duplicate canonical pages and missing indexes next; style and drift issues last.

## Editing Rules

- Read before editing; never overwrite a wiki page cold.
- Prefer narrow patches over rewrites.
- Keep pages self-contained: do not point to another knowledge repo for explanation.
- Inline concrete config values, formulas, and metric definitions with the `built` date when recording point-in-time snapshots.
- Keep generated artifacts separate from curated prose.
- When moving pages, update all relative links and entry points in the same change.
- Use `yallaplay-wiki/log.md` for durable wiki changes, not Hermes memory.

## Verification Checklist

- [ ] `README.md`/`CLAUDE.md` contract was read when needed.
- [ ] Existing canonical owner was searched and read before creating content.
- [ ] Destination follows domain-first layout, not generic `llm-wiki` artifact folders.
- [ ] New/changed pages satisfy YallaPlay frontmatter and self-containment rules.
- [ ] Relative markdown links introduced or touched resolve.
- [ ] Dictionary, glossary, domain index, and log were updated when substantial content changed.
- [ ] No secrets, unsupported speculation, stale task progress, or one-off scratch was saved.

## Common Pitfalls

1. **Importing the generic layout.** `llm-wiki` is the workflow; `yallaplay-wiki/README.md` is the schema.
2. **Using wikilinks.** YallaPlay uses relative markdown links, not `[[page]]` links.
3. **Creating type folders at root.** Root folders are domains. Put generated/supporting artifacts under the domain that owns them.
4. **Confusing records with reference.** `journal/` preserves dated source material; domain pages distill maintained knowledge.
5. **Answering from snippets.** Search identifies candidates; read canonical pages before synthesis.
6. **Skipping entry points.** A page not reachable from dictionary/index/log is effectively hidden from future agents.
7. **Saving transient work.** Completed tasks, PRs, and temporary findings belong in session history or outputs, not the wiki.
