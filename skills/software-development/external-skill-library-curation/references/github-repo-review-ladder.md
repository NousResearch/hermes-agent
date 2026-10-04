# GitHub Repo Review Ladder

A short practical ladder for using the repo-review skills in sequence.

## Step 1: Fast filter

Use `github-repo-review-decision-filter` to answer:
- Is the repo relevant to Hermes?
- Is the structure reusable?
- Is the repo active enough?
- Is the default surface safe?

Decision outputs:
- Keep going
- Reference only
- Adaptable
- Skip

## Step 2: Deep review

Use `github-repo-review-deep-review` if the repo passed the filter.
Check:
- discovery surface
- structure
- content quality
- safety and risk
- Hermes fit
- evidence-backed recommendation

## Step 3: Curation

Use `external-skill-library-curation` if the repo contains reusable workflows or a skill catalog.
Separate:
- reusable now
- reusable after adaptation
- reference only
- do not import

Then build:
- front door
- index/catalog
- adaptation note
- role-specific notes
- changelog entry

## Default policy

Prefer conservative cleanup and a defensive-only Hermes surface.
