# Upstream awesome-design-md comparison

Snapshot date: 2026-07-20

Upstream repository: `VoltAgent/awesome-design-md`
Upstream commit: `664b3e78fd1a298ba11973822da988483256d4b4` (`2026-06-16T17:21:08+03:00`)

## Summary

`popular-web-designs` is a Hermes-ready adaptation of `VoltAgent/awesome-design-md`. At this snapshot, the bundled catalog covers every upstream design slug. It is not a byte-for-byte mirror: Hermes templates remove upstream YAML frontmatter and add implementation notes, CDN font substitutions, and practical HTML/CSS generation guidance.

| Source | Count |
|---|---:|
| Hermes `skills/creative/popular-web-designs/templates/*.md` | 74 |
| Upstream `design-md/*/DESIGN.md` | 74 |
| Shared design slugs | 74 |
| Upstream-only design slugs | 0 |
| Hermes-only design slugs | 0 |

Completeness is tied to the commit above. Upstream evolves independently, so future claims must compare against a new pinned snapshot.

## Designs added in this refresh

The original 2026-06-08 audit found 19 upstream-only designs. The pinned upstream snapshot now contains one additional design, `nintendo-2001`, so this refresh adds 20 templates:

- `binance`
- `bmw-m`
- `bugatti`
- `dell-1996`
- `ferrari`
- `hp`
- `lamborghini`
- `mastercard`
- `meta`
- `nike`
- `nintendo-2001`
- `playstation`
- `renault`
- `shopify`
- `slack`
- `starbucks`
- `tesla`
- `theverge`
- `vodafone`
- `wired`

## Format differences

Hermes templates:

- use `templates/<slug>.md` files loaded via `skill_view(name="popular-web-designs", file_path="templates/<slug>.md")`;
- identify `VoltAgent/awesome-design-md` as the source in every newly imported template and point to the preserved MIT notice;
- normalize each title to `# Design System: <name>`;
- include a Hermes implementation notes block with role-specific open substitutes, paste-ready Google Fonts links when appropriate, system stacks for historically faithful templates, and HTML/CSS generation guidance;
- preserve the upstream design analysis below that Hermes-specific header, apart from documented editorial normalizations;
- are optimized as practical visual vocabularies for generated web artifacts.

The refresh makes two spelling-only editorial normalizations in the imported body text:

- `shopify.md`: upstream `Shopifi` → `Shopify`;
- `slack.md`: upstream `Slacc` → `Slack`.

The other 18 imported analysis bodies are preserved byte-for-byte after removing upstream frontmatter and any source H1.

Upstream files:

- use `design-md/<slug>/DESIGN.md` and usually a companion `README.md`;
- are formal design analyses and may include large YAML frontmatter blocks;
- describe proprietary fonts directly and do not include Hermes artifact-generation instructions.

## Safe refresh strategy

When refreshing this bundled skill from upstream:

1. Pin the upstream commit and compare slug sets before claiming completeness.
2. Preserve the Hermes title and implementation-notes format rather than copying raw `DESIGN.md` files unchanged.
3. Keep each frontmatter description to one sentence of at most 60 characters, per `AGENTS.md`.
4. Use open font substitutes and verify every Google Fonts URL in the added templates.
5. Update the catalog count, provenance snapshot, corresponding generated website skill page, and aggregate catalog row when its description changes.
6. Prefer small reviewable batches for future large gaps when practical; this refresh intentionally closes the complete known 20-template gap.
7. Re-run skill and documentation smoke tests after editing bundled skills.
