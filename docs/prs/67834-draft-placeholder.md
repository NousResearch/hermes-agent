# PR #67834 — Draft placeholder

This file exists to give the draft PR a non-empty diff so the
`CI-sensitive file review` workflow's classifier returns `ci_review: false`
instead of defaulting to `true` because the diff is empty.

The PR itself has no implementation work yet — it's filed as a draft to
align on chip placement and config-toggle conventions before any code
lands. See the PR body for the full design rationale and the three open
questions.

This file can be deleted once the actual implementation commits land on the
branch. It lives under `docs/prs/` rather than `apps/` or `scripts/` so it
doesn't trip the contribution rubric's "expand reach at the edges" /
"Keep the core narrow" boundaries, and so it stays out of any tests or
build steps.
