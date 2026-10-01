# Codex Review Workflow

Use this workflow for implementation review before committing or pushing.

The spawned review agent follows `docs/workflows/reviewing.md` and must not run
the checks listed in this document.

## Routine Flow

1. Run focused checks for the changed behavior and affected area, plus every
   required check for each surface touched.
2. Stage the complete intended change set.
3. Run `./.codex/scripts/codex-review.sh`.
4. Triage every finding. Fix valid findings at the root cause rather than
   blindly applying the review text.
5. After every review fix, rerun the applicable checks, restage, and return to
   step 3.
6. When review is clean, or remaining findings are explicit false positives,
   commit and push.

Review agents can take a while. The helper prints a heartbeat every 60 seconds
by default and watches its captured log for progress. Three consecutive
heartbeats without log growth are treated as a stuck failed review: the helper
terminates it, preserves the full log, and exits non-zero. Rerun the review
after inspecting the preserved log.

## Focused Checks

Focused checks exercise the changed behavior and its realistic regression
surface. Select test files or directories based on behavior and dependencies:
include direct tests, related component tests, and tests for affected callers,
consumers, or integration boundaries. Do not default to an entire repository or
package test suite.

- Documentation or repository guidance only: `git diff --check`.
- Python: `scripts/run_tests.sh <test-file-or-directory> -q`.
- TUI: `npm run --prefix ui-tui test -- <test-file>` and
  `npm run --prefix ui-tui typecheck` when types changed.
- Web: `npm run --prefix web test -- <test-file>` and
  `npm run --prefix web typecheck` when types changed.
- Desktop renderer: `npm run --prefix apps/desktop test:ui -- <test-file>` and
  `npm run --prefix apps/desktop typecheck` when types changed.
- Desktop main process: run the relevant `node --test <test-file>` command from
  `apps/desktop`.

Add a runtime smoke check when tests do not exercise the real boundary changed.

## Required Checks

Always run:

- `git diff --check`

For Python or shared agent/runtime changes:

- `ruff check .`
- `python scripts/check-windows-footguns.py --all`

For each TypeScript surface touched:

- TUI: `npm run --prefix ui-tui typecheck`,
  `npm run --prefix ui-tui lint`, and
  `npm run --prefix ui-tui build`.
- Web: `npm run --prefix web typecheck`, `npm run --prefix web lint`, and
  `npm run --prefix web build`.
- Desktop: `npm run --prefix apps/desktop typecheck`,
  `npm run --prefix apps/desktop lint`,
  `npm run --prefix apps/desktop build`.
- Bootstrap installer: `npm run --prefix apps/bootstrap-installer typecheck`
  and `npm run --prefix apps/bootstrap-installer build`.
- Shared package: `npm run --prefix apps/shared typecheck`.
- Website: `npm run --prefix website typecheck`,
  `npm run --prefix website lint:diagrams`, and
  `npm run --prefix website build`.

Run checks only for surfaces touched by the change. Documentation-only changes
do not require the Python or TypeScript suites.

## Modes

- Default: run `./.codex/scripts/codex-review.sh` after staging the full intended
  change set. The helper reviews staged, unstaged, and untracked local changes.
- Main-based review: run `./.codex/scripts/codex-review.sh --base-main` only when
  the user explicitly asks for a review against `origin/main`.

Do not choose the main-based mode proactively during routine implementation work.
