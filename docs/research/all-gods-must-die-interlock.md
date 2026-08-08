# All Gods Must Die — Hermes Skillset Interlock

This publication and its long-document production skill are part of the Hermes contribution graph, not standalone artifacts.

## Contribution graph

| Edge | Artifact | Relationship |
|---|---|---|
| EPIC | #78647 | Graph-gated engineering / Kill All Gods campaign |
| Sibling skill PR | #79609 | God-file kill campaign / 5×2×3 decomposition |
| Sibling skill PR | #79779 | KILL LOCK operations / bidirectional audit |
| Sibling skill PR | #79898 | Feature Parity & Alignment campaigns |
| Sibling architecture PR | #80391 | Cross-language docs germination pipeline |
| Sibling EPIC | #80392 | Cross-language docs germination campaign |
| Seed issue / PR | #60535 / #63660 | French documentation provenance and contributor authorship |
| Current contribution | #80551 | Long-document production + unified doctrine publication |

## Required edges

- PR → EPIC: `Part of #78647`.
- PR → sibling contributions: `Related #79609`, `Related #79779`, `Related #79898`, `Related #80391`, `Related #80392`.
- EPIC → PR: literal PR number posted on the EPIC thread.
- Sibling surfaces → current PR: literal PR number posted on sibling PR/issue threads where the contribution is relevant.
- Skill → skillset: `related_skills` includes the authoring, campaign, parity, KILL LOCK, publication, source-verification, and quality skills.
- Credit → artifact: Axl Ibiza, MBA is the author of the doctrine and publication; contributors and source projects remain attributed in the paper and source ledger.

## Publication surfaces

- `website/docs/guides/all-gods-must-die.md` — long LLM-readable guide surface.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.pdf` — compiled publication.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.tex` — reproducible source.
- `docs/research/all-gods-must-die-adversarially-verified-transformation.bib` — merged bibliography.
- `skills/research/long-document-production/SKILL.md` — reusable production skill.

The skill and publication must not be merged as isolated files while the interlock edges are absent. The graph is part of the contribution.

Signed-off-by: Andrex Ibiza, MBA <84248988+andrexibiza@users.noreply.github.com>

Part of #78647
Related #79609
Related #79779
Related #79898
Related #80391
Related #80392
Related #60535
Related #63660

## Verification state

- Source tree contains the skill, guide, PDF, LaTeX, BibTeX, and this interlock manifest.
- Guide is registered in `website/sidebars.ts`.
- Forbidden excluded side-project term: zero matches.
- Final publication build: 68 pages, five native TinyTeX passes, zero undefined citations/references, 25 DOI hyperlinks, all pages rendered for visual QA.
- Merge state of sibling PRs is not asserted here; live GitHub state must be checked at review time.

## Contribution principle

All contributions matter: feature code, security hardening, tests, audits, documentation, translations, skills, reviews, ledger repairs, interlock corrections, and provenance work. Attribution is a core value and a correctness edge.

**Interlock is bidirectional or it is incomplete.**
