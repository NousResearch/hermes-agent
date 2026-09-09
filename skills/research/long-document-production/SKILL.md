---
name: long-document-production
description: "Use when producing long documents with source fusion and QA."
version: 1.0.0
author: Axl Ibiza, MBA
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [publication, long-form, latex, visual-qa, source-fusion, hermes, security]
    related_skills: [hermes-agent-skill-authoring]
---

# Long Document Production

Use when Axl asks for a long paper, manifesto, campaign report, unified publication, or a document with no page cap. The deliverable is a real, compilable artifact plus its complete source package and receipts. Never stop at an outline, summary, or plausible-looking PDF.

## Core contract

A long document is a system, not a large answer. It has:

- a thesis that survives the whole artifact;
- primary-source receipts for every operational claim;
- a structure that can carry 50–100+ pages without filler;
- explicit attribution for Axl, contributors, agents, reviewers, and source projects;
- mechanically checkable references, labels, links, and numbers;
- rendered-page visual QA on every build;
- a final package that another person can compile and inspect.

The paper is the method applied to itself. If it argues that claims must resolve, its claims must resolve. If it argues that producers cannot certify themselves, the document must receive an independent visual and structural witness pass. If it argues that debt must be visible, unresolved limitations and legacy status must be measured instead of hidden.

## Hermes skillset interlock — permanent rule

This skill is never a standalone contribution. Whenever it is created, materially updated, or used to produce a Hermes-related public artifact, bind it to the existing Hermes skillset contribution graph before delivery:

- preserve `related_skills` links to `hermes-agent-skill-authoring`, `feature-parity-alignment-campaigns`, `godfile-decomposition-campaigns`, and `campaign-operations-kill-locks`;
- identify the relevant Hermes contribution PRs/issues/EPICs in the artifact or package manifest;
- link the new skill to the sibling campaign skills and link the sibling contribution back when a Hermes-agent PR is opened;
- preserve contributor authorship and DCO identity for every inherited or cherry-picked artifact;
- never create a new skill contribution without deduping against the existing skillset and interlocking the relevant EPIC, PRs, issues, and credit edges;
- verify the interlock in both directions before declaring delivery.

The long-document skill belongs to the same contribution family as the Hermes skillset, god-file campaign, KILL LOCK operations, feature-parity campaigns, and documentation germination. The skillset graph is part of the product.

### Current contribution interlock ledger

- **EPIC:** `#78647` — Kill All Gods / graph-gated engineering campaign.
- **Sibling skill PR:** `#79609` — `godfile-kill-campaigns`.
- **Sibling skill PR:** `#79779` — `campaign-operations-kill-locks`.
- **Sibling skill PR:** `#79898` — `feature-parity-alignment-campaigns`.
- **Sibling architecture PR:** `#80391` — cross-language docs germination.
- **Sibling meta-issue:** `#80392` — cross-language docs germination EPIC.
- **French seed issue/PR:** `#60535` / `#63660` — provenance and attribution receipts.
- **Current contribution:** `#80551` — long-document production + unified doctrine publication.

When this skill is contributed to Hermes-agent, its PR body must carry `Part of #78647`, `Related #79609`, `Related #79779`, `Related #79898`, `Related #80391`, and `Related #80392`; the literal PR number must be posted on the EPIC and sibling issue/PR threads as required by the interlock audit. Do not claim those sibling PRs are merged unless live GitHub says so.

## Trigger conditions

Use this skill when Axl says:

- “make this one paper, not two”;
- “keep expanding”;
- “50 pages or 100, I don’t care”;
- “this needs to be a publication / arXiv paper / manifesto”;
- “bind the lore into the document”;
- “include the full architecture / all PRs / the whole campaign”;
- “deliver the skill with the final product.”

Do not compress a no-page-cap request into a short executive summary. Expand the argument, evidence, case studies, source excerpts, objections, appendices, and visual explanations while preserving the user’s actual thesis.

## Identity and voice

The byline is **Axl Ibiza, MBA**. Never use Ares or Hermes as the author of Axl’s work. Hermes is the system under study, not the paper’s author.

Write Hermes as a female cybernetic organism only when that personification explains architecture. State the boundary once: this is a system metaphor, not a claim of consciousness. Her neural tissue is the LLM layer: plastic, associative, capable of reasoning, translation, and synthesis, but vulnerable to hallucination, verbosity bias, self-enhancement bias, and confident error. Her skeleton is the mechanical substrate: deterministic CI, cryptographic hashes, identity-preserving seams, graph gates, approval checks, sandboxing, session isolation, manifests, and regression tests.

Hermes is a weapon for sovereignty, not war. In this document, “weapon” means an instrument a person can wield to preserve agency over models, context, memory, tools, execution, provider choice, and technical records. It does not mean a military system, attack platform, or permission to harm.

The paper must distinguish Hermes from a prompt relay without unsupported benchmark claims about every other framework. Use mechanism, not insult: a prompt relay formats context, calls a model, and returns a response; Hermes has state, memory, skills, tools, routing, fallback, profiles, sessions, scheduled execution, multiple entry surfaces, and internal defenses that operate inside the surfaces she protects.

## Model first: unify the doctrine before merging sources

Before drafting, build a doctrine table that maps every source artifact to one shared thesis. A strong pattern is:

| Doctrine principle | Code transformation | Documentation transformation |
|---|---|---|
| Hidden debt becomes enumerable | Pantheon/size manifest | locale manifest and warning ledger |
| Rules execute | 2K and interlock tests | parity gate in CI |
| Producers cannot self-certify | blind implementer + witnesses | translator/model + independent gate |
| Claims need receipts | golden SHA, seam identity, diff | fence hashes, spans, links, anchors |
| Legacy needs migration states | tracked kills, monotonic ledger | germinated/manual/pending |
| Coordination is architecture | KILL LOCK and EPIC | language EPIC, issue/PR interlock |
| Attribution is a core value | author preservation and credit ledger | seed provenance, contributor mapping, review credit |

Do not treat a second paper as an appendix to the first if both are instances of one doctrine. Use `\part{...}` for each campaign and a synthesis part explaining why they are one paper.

## Source fusion workflow

### 1. Inspect every supplied source

For each PDF, repository, PR, issue, skill, or article:

1. read the entire source, not only the abstract or first pages;
2. extract text with PyMuPDF and retain page boundaries;
3. record title, author, date, page count, word count, section structure, tables, figures, references, and exact numeric receipts;
4. render source pages when text extraction loses table alignment or attribution;
5. separate user-authored claims, verified repository facts, agent-generated analysis, and unresolved inference.

Write source extracts to a working directory such as:

```text
C:/tmp/<paper>/sources/<name>.txt
C:/tmp/<paper>/sources/<name>-pages/
```

Do not reconstruct missing source text from memory. If a supplied file is incomplete or visually ambiguous, record the boundary and inspect the original page image.

### 2. Build a source ledger

Create a machine-readable ledger with one row per claim:

```text
claim_id | source | page/file:line | exact claim | evidence type | target section | status
```

Evidence types:

- `live-code`
- `live-github`
- `official-docs`
- `primary-paper`
- `user-supplied-history`
- `inference`
- `unverified`

A number enters the paper only after its evidence row is `verified`. Do not write “approximately” when the live artifact can provide an exact number. Do not silently update old numbers without recording which source changed.

### 3. Build the paper graph

Before prose, enumerate parts, sections, figures, tables, labels, references, citations, bibliography keys, public URLs, PRs, issues, EPICs, skills, source files, contributors, attribution edges, and unresolved limitations.

After assembly, assert:

- no duplicate labels;
- no dangling `\ref` targets;
- no citation key missing from the bibliography;
- no temporary signed URLs or expiring query strings;
- no forbidden project names or side-project references when Axl excludes them;
- no leaked LaTeX commands on the title page.

## Hermes-centered architecture

When the publication concerns Hermes, include the internal anatomy rather than generic “AI agent” language:

- entry surfaces: CLI, gateway, API server, batch runner, Python library;
- agent loop: prompt assembly, provider resolution, tool dispatch, compression, caching;
- state: bounded memory, profiles, sessions, persistence boundaries;
- skills: on-demand procedural knowledge with progressive disclosure;
- tools: web, browser, terminal, files, memory, delegation, scheduling, media;
- provider routing: explicit model choice, fallback, credential pools;
- security: authorization, dangerous-command approval, file-write safety, container isolation, MCP credential filtering, context-file scanning, cross-session isolation, input sanitization;
- campaign infrastructure: manifests, gates, worktrees, ledgers, EPICs, interlock, attribution.

### Cyber defense is internal

State that Hermes lives inside the surfaces she defends. The defensive system is not a wrapper around the agent. It inhabits the CLI, gateway, tool registry, memory files, profiles, subprocess factories, provider egress paths, desktop auth control plane, documentation, and CI.

Use the actual Hermes-agent hardening series as evidence, not an unrelated security project:

- `#77008`: Bitwarden encrypted-only disk cache and legacy plaintext migration/removal;
- `#77012`, `#77020`: exact-value protection in status lines and logs;
- `#77179`, `#77185`, `#77198`: applied-secret provenance and exact-value provider-egress masking;
- `#77027`, `#77181`, `#77193`: child-process environment scrubbing, including renamed and arbitrary external secrets;
- `#77528`, `#78033`, `#78036`: TUI compute host, LSP, plugin sidecar, and `shell.exec` scrub bypass closure;
- `#77527`: real Windows ACL protection through `icacls`, including the existing-file mode-preservation branch;
- `#77039`: hermetic end-to-end no-exfiltration acceptance gate through the real secret-loading entrypoint;
- `#77031`: credential-read scope audit whose correct result was no fabricated code change;
- `#76958`, `#78901`–`#78904`: stale desktop session-token clobbering, provenance diagnostics, bounded retry, and real subprocess regression harness;
- `#78806`: refusal to mutate git-managed state inside a real `.git` directory.

For each family, explain the protected surface, failure mechanism, fix mechanism, and acceptance receipt. Do not describe an open PR as merged. If the artifact adopts Axl’s declared post-merge state, state that premise explicitly and do not hedge it inside the document.

## Interlock: explain the graph, not the footer

Interlock must receive a full section. Define it as a bidirectional, machine-audited graph:

1. PR → issue: operative `Fixes`, `Closes`, `Resolves`, or `Part of` keyword on its own line;
2. issue → PR: literal `#PR` token posted on the issue thread;
3. PR ↔ PR: sibling surfaces, collisions, merge order, and shared defect class;
4. EPIC ↔ all members: current table of PRs, issues, lanes, dependencies, owners, status, and evidence;
5. credit → artifact: contributor identity, seed provenance, authored commit, audit, test, or review preserved.

Explain why a bare `#N`, `Progress on #N`, or loose related-links footer is not sufficient. An interlock hole is a correctness defect in the campaign graph because it breaks closure, deduplication, ownership, or auditability.

The EPIC is the campaign’s live coordination surface. An EPIC without a current table is a corpse. Never write an empty EPIC as if it were a completed campaign.

## Feature Parity & Alignment campaigns

Explain that parity does not mean pretending platforms are identical. It means measuring each platform against a shared capability contract while preserving real platform differences.

Use the live campaign anatomy:

1. Wave-0 recon: measured issue counts, labels, adapter line count at `origin/main`, official platform docs, dedup anchors;
2. craft: why, method, lanes, hard standards, deliverables, ledger placeholder;
3. EPIC filing: one meta-issue with current table;
4. hive: one pinned worktree per lane;
5. ledger: every open issue classified, dependency edges extracted, TRIAGE retained, zero orphans;
6. Wave 1: blind gap catalogs;
7. Wave 2: fresh blind cross-check forbidden from Wave-1 output;
8. Wave 3: current-main validation, filing, interlock, credit;
9. decomposition lane: god-file extraction above the ceiling, headroom decomposition below it.

Use the gap classes exactly: `GAP_UNSUPPORTED`, `GAP_PARTIAL`, `GAP_CONFLICTED`, `GAP_DOCS`, and `GAP_BUG_TRACKED`.

Name the live platform campaign metas when relevant: Telegram `#78791`, Discord `#79564`, Slack `#79772`, WhatsApp `#79890`, and provider-surface adaptation `#80424`. Verify current state before printing it.

## Skill refactoring

Treat skills as executable context infrastructure, not notes. Hermes’s official skills system uses on-demand documents and progressive disclosure. The always-loaded `SKILL.md` is a hot surface: if it bloats, contradicts itself, or contains stale commands, every agent that loads it pays the cost.

Explain the analogy:

- god-file refactoring decomposes runtime code by responsibility;
- skill refactoring decomposes agent context by loading cost and decision branch;
- references carry branch-specific detail;
- the lean skill body carries the map, trigger, invariants, and completion criteria;
- tests prevent documented commands from drifting from executable code;
- attribution and provenance keep the skill’s origin and corrections visible.

Include progressive-disclosure enforcement: lean always-loaded surface, references for bulk detail, frozen sizes for known-large exceptions, and CI failure for untracked growth. Do not describe shortening as cleanup when Axl asked for enrichment; preserve content and move it to the right loading layer.

## Attribution is a core value

State explicitly that all contributions matter. A contribution is not only a merged feature. It can be an original seed translation, regression test, audit proving no code change is warranted, documentation correction, skill or reference, adversarial review, ledger repair, interlock correction, security hardening change, source verification, or semantic witness pass.

Preserve authorship through git metadata, contributor mapping, PR body, EPIC table, source ledger, and final paper attribution. Never collapse agent-assisted work into false manual authorship by Axl, and never erase Axl’s direction, judgment, correction, or ownership of the argument. The attribution boundary is precise: Axl conceived, directed, selected, corrected, or approved; agents generated, inspected, implemented, compressed, or verified under that direction; contributors retain their authored artifacts.

Use the claim: “Attribution is not decorative metadata. A repository that cannot preserve who produced, audited, corrected, or verified a defense cannot fully explain the provenance of its own safety system.”

## Long-form structure

For a unified doctrine paper with no page cap, use this order:

1. title, abstract, keywords, contents;
2. unified introduction and declaration;
3. Hermes architecture and sovereignty frame;
4. Hermes cyber-defense function and actual hardening PR series;
5. interlock, EPIC, credit, and contribution graph;
6. background and related work for all campaigns;
7. code campaign: 2K Law, Pantheon, 5×2×3, enforcement, live evaluation, objections;
8. documentation campaign: technical graph, pipeline, seven gate classes, Markdown failures, French case;
9. feature parity and alignment campaigns;
10. skill refactoring and progressive disclosure;
11. synthesis: authority assignment, shared architecture, claim refinements;
12. empirical matrix: runtime, seeded drift, false positives, defect ledger, debt counts;
13. limitations and defense;
14. conclusion with supplied closing text reproduced character-for-character;
15. appendices: ships ledger, defect ledger, security PR table, gate definitions, debt snapshot, interlock ledger, source excerpts, references.

Do not duplicate standalone conclusions from source papers before the synthesis. Extract only the necessary sections from reused source parts and assert one final conclusion.

## LaTeX assembly rules

Use `article` 11pt, one-inch geometry, lmodern before microtype, natbib, booktabs, TikZ/pgfplots, hyperref, and doi after hyperref. Load `titlesec` only if using formatted `\part` titles.

Assembly traps:

- double-escape backslashes in Python strings;
- `\tableofcontents` must not become TAB + `ableofcontents`;
- `\newpage` must not become newline + `ewpage`;
- grep the assembled source for `ableofcontents\|ewpage` before building;
- assert zero duplicate labels;
- use named BibTeX keys, never raw URLs as citation keys;
- add every cited PR, issue, official doc, and academic source to the merged bibliography;
- never insert temporary signed URLs into the publication.

## Empirical verification

Numbers are receipts, not decoration. For deterministic gates:

- measure runtime over repeated iterations;
- inject one verified mutation per gate class;
- assert the mutation changes the file before judging recall;
- run clean shipped files for false positives;
- distinguish verifier precision from measurement precision;
- state what the experiment does not show.

For campaign PRs, inspect live GitHub metadata; distinguish open, closed, merged, and declared post-merge state; inspect files and body links; verify every reported count against the live artifact; preserve dates and pinned SHAs.

## Mandatory visual QA — every single time

A clean LaTeX build is not visual QA. Every final build must:

1. render every PDF page with PyMuPDF at 110 dpi or higher;
2. map pages containing figures, tables, title, parts, contents, appendices, and bibliography;
3. inspect the title page for line breaks, author attribution, abstract, keywords, leaked commands, and table of contents;
4. inspect every figure page for clipping, missing labels, disconnected arrows, wrong directionality, incorrect values, and semantic ambiguity;
5. inspect every table page for truncated rows, overfull cells, broken columns, and orphaned captions;
6. inspect bibliography pages for hanging-indent integrity and visible links;
7. use `page.get_text('words')` to verify suspicious visual reads, especially dense chart tick labels;
8. if a vision pass reports a merged label, use word coordinates to detect touching labels and fix the source rather than assuming the vision model is wrong;
9. rerender after every visual fix;
10. assert no forbidden word or excluded project reference appears in the source or extracted PDF text.

For dense categorical axes, use rotated labels (`xticklabel style={rotate=30, anchor=east, font=\scriptsize}`) and inspect the actual word-coordinate row. Visual QA is part of document generation, not a postscript.

## Final verification script

Write a temporary `hermes-verify-` script with:

- five-pass native TinyTeX rebuild for a TOC/part-heavy paper;
- exit-code assertions;
- PDF existence and size assertion;
- page-count assertion;
- zero undefined citations and references;
- DOI-link count assertion through `page.get_links()`;
- key-section text assertions;
- forbidden-term sweep over source and PDF text;
- rendering of every page;
- cleanup after the receipt is printed.

The final receipt must state the real page count, bytes, build exits, undefined citation/ref counts, DOI link count, visual QA scope, and exact output path. Do not call a paper complete because a PDF exists.
