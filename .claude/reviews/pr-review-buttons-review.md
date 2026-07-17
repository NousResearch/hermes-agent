# Code Review: pr_review_buttons plugin (local, uncommitted)

**Reviewed**: 2026-07-17
**Author**: Jonathan Tomek
**Branch**: feat/deadline-radar (working tree)
**Decision**: APPROVE with comments

## Summary
New Hermes plugin that turns the cron PR-review digest into interactive Slack
Approve / Request-changes / Comment buttons; clicking posts the verbatim staged
review to GitHub via `gh`. Own PRs are excluded at scan time and defensively
refused at click time. Clean architecture (pure functions + dependency
injection), 40 passing tests, ruff clean. No CRITICAL/HIGH issues.

## Findings

### CRITICAL
None.

### HIGH
None.

### MEDIUM
- **plugins/pr_review_buttons/store.py** — concurrent `stage()` calls are
  read-modify-write with no lock. Not a bug in practice: the cron agent stages
  sequentially in one loop and the write itself is atomic (temp + os.replace),
  so a file is never torn — but two truly-parallel stagers could lose one
  entry. Documented limitation; acceptable for a single-operator flow.

### LOW
- **blocks.py** — if a batch exceeds 50 blocks even after the title-only
  fallback, Slack would reject it. The cron caps at 6 PRs/run, so unreachable
  in practice; noted as an assumption rather than fixed.
- **Emoji in Slack-facing strings** (actions.py context lines, blocks labels) —
  these are intentional: they render in Slack messages, not code output, and
  match the existing review-output template's convention.

## Resolved during review (from the two review-agent passes + live testing)
- Auth fails CLOSED when `SLACK_ALLOWED_USERS` unset (was open). This handler is
  the only gate — the adapter's plugin wrapper does not run the gateway's own
  interactive-auth. Verified gateway env has the allowlist.
- Stale-guard fails CLOSED on a `None` head SHA (can't verify → don't post).
- Anonymous clicks (missing user id) rejected before the allowlist check.
- `slack_sdk` dependency removed — `slackio` now uses stdlib `urllib`, because
  the cron spawns the installed `hermes` whose interpreter lacks `slack_sdk`
  (root cause of the `No module named 'slack_sdk'` publish failure). Verified
  publish works through the installed binary.
- `submit_review` success is judged by rc AND stderr: `gh pr review` exits 0
  even when it fails to create the review, so trusting rc reported "posted"
  while nothing landed. Verified on real PR #3221.
- Own-PR handling changed from "post as comment fallback" to "refuse, post
  nothing" per operator preference. Two layers: scan-time author filter
  (primary) + click-time `own_pr` refusal (defensive). Verified: live scan
  with empty seen-state surfaces others' PRs but not the operator's #3221.

## Validation Results

| Check | Result |
|---|---|
| Tests (pytest, 40) | Pass |
| Lint (ruff) | Pass |
| Import smoke (all modules) | Pass |
| cron jobs.json valid JSON | Pass |
| Scan script `bash -n` | Pass |
| Own-PR exclusion (live) | Pass |
| Own-PR refusal (unit) | Pass |

## Files Reviewed
- `plugins/pr_review_buttons/__init__.py` (Added) — register CLI + 3 action handlers
- `plugins/pr_review_buttons/actions.py` (Added) — async click handler
- `plugins/pr_review_buttons/blocks.py` (Added) — Block Kit builders
- `plugins/pr_review_buttons/cli.py` (Added) — `hermes prreview stage|publish|list`
- `plugins/pr_review_buttons/github.py` (Added) — `gh` wrapper, own-PR refusal
- `plugins/pr_review_buttons/slackio.py` (Added) — stdlib Slack Web API client
- `plugins/pr_review_buttons/store.py` (Added) — pending-review JSON store
- `plugins/pr_review_buttons/plugin.yaml` (Added) — manifest
- `tests/plugins/test_pr_review_buttons.py` (Added) — 40 tests
- `~/.hermes/scripts/pr_review_scan.sh` (Modified, outside repo) — own-PR author filter
- `~/.hermes/cron/jobs.json` (Modified, outside repo) — job 3e6360441c63 prompt → stage/publish
