# Software Development Skills Front Door

This folder contains the reusable workflow skills that ship with Hermes.

## Repo-review / external-library path

Use these in order:

1. `github-repo-review-decision-filter` — fast yes/no triage
2. `github-repo-review-deep-review` — deeper evidence-based review
3. `external-skill-library-curation` — adapt a useful library into Hermes-safe structure

## Rule of thumb

- Start with the decision filter.
- Only do the deep review if the repo passes the filter.
- Only do the curation step if the source is worth turning into a Hermes bundle or skill set.

## Safety

Default to defensive-only material in the main Hermes surface.
Keep offensive or dual-use material out of the default path unless a restricted lane is explicitly requested.
