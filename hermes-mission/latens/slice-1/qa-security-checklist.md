# LATENS Slice 1 QA and Security Checklist

Issue: chipoto69/latens#2
Task: t_24194b8d

This checklist is the review gate for the Slice 1 site/provenance foundation. Each item is phrased as evidence QA/security can verify from local files before any push, PR publish, merge, deploy, or production dispatch.

## Public-data leakage

- [ ] Public route files contain only relative paths and public-safe descriptions.
- [ ] Public provenance packets use `packet_scope: "public"` and expose only `schema_version`, `packet_scope`, and the `public` object.
- [ ] Public provenance packets do not include the top-level `private_review` object.
- [ ] Raw prompts, operator notes, private repo details, and credential context stay in internal-review-only packets or local review storage only.
- [ ] No tokens, cookies, API keys, personal data, or private source bodies appear in committed artifacts.

## Unsafe claims

- [ ] Emotion signatures are described as editorial tags, not diagnoses or user inferences.
- [ ] Source refs include `claim_scope` so public claims cannot exceed their cited support.
- [ ] No medical, financial, legal, scientific-performance, or safety claims are present without approval notes and source refs.

## Credential handling

- [ ] Public examples omit `private_review`; internal-review packets set `private_review.credential_handling.secrets_present` to false before any approval.
- [ ] No `.env`, key file, token, or generated credential is added by Slice 1.
- [ ] Review routes are marked internal and are not public render targets.

## Publish/deploy boundaries

- [ ] Route policy states `deploy_requires_founder_yes: true`.
- [ ] Founder publication approval is a required approval object and remains pending in examples.
- [ ] No remote push, PR publish, merge, deploy, account creation, spend, credential change, or production dispatch occurred during implementation.

## Accessibility

- [ ] Route descriptions are human-readable and can become page headings/landmarks.
- [ ] Lexicon definitions avoid image-only or color-only meaning.
- [ ] Later renderers must expose each route with a semantic heading and link text matching the canonical term.

## Mobile behavior

- [ ] Routes are static/data-first and do not require hover, drag, wide tables, or desktop-only interactions.
- [ ] Later renderers should treat tables as progressively enhanced content and keep lexicon pages readable at narrow widths.

## Link integrity

- [ ] Every lexicon `canonical_path` begins with `/lexicon/`.
- [ ] Every lexicon `concept_route` exists as either a route manifest path or a documented route template.
- [ ] Route manifest paths are unique.
- [ ] Dynamic route templates use `{term_slug}`, `{artifact_id}`, `{edition_id}`, or `{run_id}` consistently.

## Console cleanliness

- [ ] Slice 1 adds only static manifests/docs/tests; there is no browser runtime code that can emit console errors.
- [ ] When a renderer is added, QA must run a browser console pass before publication.

## Local validation command

Run from `/Users/rudlord/.hermes/hermes-agent`:

```bash
python3 -m pytest tests/hermes_mission/test_latens_slice1_artifacts.py -q
```
