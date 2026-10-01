# Codex implementation review

- Status: active
- Scope: `.codex/scripts/codex-review.sh`, development guidance
- Introduced: employee implementation

## Downstream intent

Implementation changes receive a read-only Codex review after focused checks.
The copied helper uses the existing Codex CLI login, prints progress, and fails
stalled reviews without altering the developer's service-tier settings.

## Reconciliation

Keep the helper and [review workflow](../../workflows/codex-review.md) when
merging upstream guidance. Reviewers must not recursively launch reviewers or
mutate the checkout. Keep development Codex authentication separate from the
employee server's Hermes Codex login.

## Validation

Run `tests/scripts/test_codex_review.py` and the helper itself on the staged
change. Fix valid findings and repeat the review before landing.
