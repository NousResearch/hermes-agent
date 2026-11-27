# LATENS Slice 1 Site Foundation

Issue: chipoto69/latens#2
Task: t_24194b8d
Status: draft implementation artifact, local-only until founder approval

## Purpose

Slice 1 establishes the minimum public site spine for LATENS, FERROGLYPH, and OZVENA before any generated artifacts are published. It defines stable routes, lexicon canonical URLs, and provenance metadata so later editions can be cited, reviewed, redacted, and approved without changing their public URLs.

## Site foundation

The site is intentionally static-first for Slice 1. A later app can render these manifests as pages, but the first durable contract is the route and provenance data in this directory.

Top-level durable routes:

| Route | Owner concept | Public purpose | Private data allowed? |
|---|---|---|---|
| `/` | LATENS | Entry page explaining the project scope and review gates | No |
| `/latens` | LATENS | Public index of LATENS editions/artifacts | No |
| `/ferroglyph` | FERROGLYPH | Public index of FERROGLYPH editions/artifacts | No |
| `/ozvena` | OZVENA | Public index of OZVENA editions/artifacts | No |
| `/lexicon` | Shared | Human-readable glossary index | No |
| `/lexicon/{term}` | Shared | Canonical definition page for a kickoff term | No |
| `/artifacts/{artifact_id}` | Shared | Public artifact landing page with sanitized provenance | No |
| `/editions/{edition_id}` | Shared | Public edition landing page linking approved artifacts | No |
| `/provenance/{run_id}` | Shared | Public provenance view for a run after approval/redaction | No |
| `/review/{run_id}` | Internal | Private review packet before approval | Yes; never public |

## Rendering contract

1. Public pages only read `packet_scope: "public"` packets and the top-level `public` object from `provenance.schema.json`.
2. Public provenance packets must not include the top-level `private_review` object. Review pages may read `private_review` fields only from `packet_scope: "internal_review"` packets, must require local/operator access, and must not be indexed or deployed without a separate approval gate.
3. Each public artifact page must link to:
   - its edition route,
   - its canonical concept route (`/latens`, `/ferroglyph`, or `/ozvena`),
   - every lexicon term referenced by the artifact,
   - its sanitized public provenance route.
4. Every canonical URL must be lowercase, hyphenated, and stable. Renames require redirects, not replacement.
5. Claims must remain descriptive. No medical, financial, legal, or scientific-performance claims are allowed unless a source reference and approval explicitly support them.

## Files in this slice

- `site-routes.json` — durable public/private route manifest.
- `lexicon-routes.json` — kickoff term to canonical URL map.
- `provenance.schema.json` — JSON Schema for public/private provenance packets.
- `provenance-example.public.json` — example sanitized public packet.
- `qa-security-checklist.md` — gate evidence checklist for QA/security review.
- `../../../tests/hermes_mission/test_latens_slice1_artifacts.py` — validation tests for the manifests and schema.

## Explicit non-goals

- No deploy or publish.
- No push or PR publish without Rudy's explicit YES.
- No account creation, credential changes, or production dispatch.
- No use of secrets or private source material in public files.
