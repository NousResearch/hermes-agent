# All Gods Must Die - Hermes Skillset and Doctrine Interlock

This publication, its long-document production skill, and its successor doctrine sources are part of the Hermes contribution graph, not standalone artifacts.

## Contribution graph

| Edge | Artifact | Relationship |
|---|---|---|
| EPIC | #78647 | Graph-gated engineering / Kill All Gods campaign |
| Sibling skill PR | #79609 | God-file kill campaign / 5x2x3 decomposition |
| Sibling skill PR | #79779 | KILL LOCK operations / bidirectional audit |
| Sibling skill PR | #79898 | Feature Parity & Alignment campaigns |
| Sibling architecture PR | #80391 | Cross-language docs germination pipeline |
| Sibling EPIC | #80392 | Cross-language docs germination campaign |
| Seed issue / PR | #60535 / #63660 | French documentation provenance and contributor authorship |
| Current contribution | #80551 | Long-document production + unified doctrine publication |
| Doctrine refinement source | `Task Completion Verification.txt` | Negative receipt proving that prepared/local verification is not target-state completion |
| Task 7 case-study object | #85002 | Named-profile configuration: current canonical PR and exact-head receipt at the observed reconciliation |
| Task 10 historical object | #85523 | Closed, unmerged provenance; explicitly superseded |
| Task 10 canonical successor | #90236 | Current-main closure candidate and exact-head receipt at the observed reconciliation |

## Required contribution edges

- PR -> EPIC: `Part of #78647`.
- PR -> sibling contributions: `Related #79609`, `Related #79779`, `Related #79898`, `Related #80391`, `Related #80392`.
- EPIC -> PR: literal PR number posted on the EPIC thread.
- Sibling surfaces -> current PR: literal PR number posted on sibling PR/issue threads where the contribution is relevant.
- Skill -> skillset: `related_skills` includes the authoring, campaign, parity, KILL LOCK, publication, source-verification, and quality skills.
- Credit -> artifact: Axl Ibiza, MBA is the author of the doctrine and publication; contributors and source projects remain attributed in the paper and source ledger.
- Release -> predecessor: v1.1 carries an explicit `derives_from` / `refines` edge to immutable v1.0; it does not silently replace v1.0 bytes.
- Case study -> live objects: the dated local-only/blocked receipt and the later GitHub objects remain separate events in the mutation journal.
- Supersession: #85523 -> #90236 is typed and explicit; historical provenance does not become active ownership.

## Core publication surfaces

- `website/docs/guides/all-gods-must-die.md` - original long LLM-readable guide surface.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.pdf` - immutable 68-page v1.0 publication.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.tex` - reproducible v1.0 source.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.bib` - merged bibliography.
- `skills/research/long-document-production/SKILL.md` - reusable production skill.

## Task Completion Verification source-controlled surfaces

- `website/docs/guides/task-completion-verification.md` - architecture guide for exact-object completion semantics.
- `docs/research/task-completion-verification-amendment.md` - canonical source-readable formal amendment.
- `docs/research/all-gods-must-die-adversarially-verified-transformation-v1.1.tex` - composition source preserving v1.0 and appending the generated Amendment I PDF.
- `docs/research/task-completion-record.schema.json` - machine-auditable completion-record contract.
- `docs/research/task-completion-verification-ledger.json` - dated local-blocker and current GitHub reconciliation records.
- `docs/research/all-gods-must-die-v1.1-release-manifest.json` - source digests, generated-output digests, typed lineage, and packaging-time state.
- `docs/research/all-gods-must-die-interlock.md` - this graph and exact lifecycle boundary.

## Generated release outputs

The following were built, preflighted, rendered, and inspected locally. Their digests are in the release manifest. Their repository publication is a separate predicate and must not be inferred from source publication:

- `task-completion-verification-amendment.pdf` - 14 pages.
- `all-gods-must-die-adversarially-verified-transformation-v1.1.pdf` - 83 pages: release cover + v1.0 + Amendment I.

The skill, original publication, successor sources, schemas, guide, and interlock must not be merged as isolated files while the graph edges are absent. The graph is part of the contribution.

Signed-off-by: Andrex Ibiza, MBA <84248988+andrexibiza@users.noreply.github.com>

Part of #78647
Related #79609
Related #79779
Related #79898
Related #80391
Related #80392
Related #60535
Related #63660

## Doctrine amendment: the sixth law

> **No task is complete until the requested state exists at the authoritative target and is verified on the exact object.**

Derivative rule:

> **No receipt may be inherited from a local draft, previous head, adjacent commit, parent object, superseded artifact, or intended write.**

The completion model is a typed state vector across materialization, integrity, governance, integration, operation, and lineage. Generic words such as `done`, `fixed`, `shipped`, and `green` are inadmissible unless they resolve to a namespaced predicate and exact receipts.

The original god-file `SHIPPED` standard is preserved as `KAG_SHIPPED`: an open, individually interlocked, campaign-complete PR. It does not imply `MERGED`, `PUBLISHED`, `RELEASED`, or `OPERATIONALLY_VERIFIED`.

## Verification state

- Source tree contains the original skill, guide, PDF, LaTeX, BibTeX, and this interlock manifest.
- Original guide is registered in `website/sidebars.ts`.
- Forbidden excluded side-project term: zero matches in the original publication package.
- Original publication build: 68 pages, five native TinyTeX passes, zero undefined citations/references, 25 DOI hyperlinks, all pages rendered for visual QA.
- Amendment build: 14 pages; PDF preflight passes; all pages rendered and visually inspected.
- Composed v1.1 build: 83 pages; release cover plus v1.0 component plus Amendment I; boundary pages and final declaration inspected.
- JSON Schema: all five case-study records validate under Draft 2020-12.
- Source and generated-output hashes are recorded in `all-gods-must-die-v1.1-release-manifest.json`.
- Merge state of sibling PRs is not asserted here; live GitHub state must be checked at review time.
- Repository source submission, exact commit identity, binary publication, and CI status must be read from PR #80551 and its exact Git tree after mutation. This file cannot certify its own publication.

## Contribution principle

All contributions matter: feature code, security hardening, tests, audits, documentation, translations, skills, reviews, ledger repairs, interlock corrections, provenance work, negative receipts, and truthful blocker reports. Attribution is a core value and a correctness edge.

**Interlock is bidirectional or it is incomplete. Completion is exact-object bound or it is unproven.**
