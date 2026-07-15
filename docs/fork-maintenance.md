# Fork maintenance

This checkout is maintained as a **fork**, not as an upstream contribution
target. This document is fork-only (it does not exist upstream).

## Remotes

- `origin` = `NousResearch/hermes-agent` — upstream open source.
- `fork`   = `graysurf/hermes-agent` — our fork; all our work lands here.

## Model

- **Never patch upstream directly.** Every local fix is committed to a feature
  branch and pushed to `fork`.
- Do **not** rely on upstream taking our changes. Contributing back to
  NousResearch is optional, never assumed.
- Stay current by periodically fetching `origin` and merging/rebasing
  `origin/main` into the working branch. Upstream moves fast (hundreds of
  commits between syncs), so treat this as a deliberate occasional task, not a
  per-change step.

## Reconciling local work with an incoming commit on the same files

This is the pattern for when one machine (e.g. the Mac) has already pushed a
commit to `fork` that touches files you have uncommitted work on locally (e.g.
on sympoies):

1. `git fetch fork` (and `git fetch origin` when syncing upstream).
2. `git stash push -- <overlapping files>` — stash only the overlapping files,
   leaving unrelated in-progress work in place.
3. `git merge --ff-only <remote-branch>` (or `git rebase`) to bring the incoming
   commit in.
4. `git stash pop`, then resolve any 3-way conflicts by keeping both sides.
5. Run the affected tests, then commit.

## Conventions

- **Commits** go through the `semantic-commit` CLI (direct `git commit` is
  hook-blocked in this environment).
- **Language:** code, comments, and docstrings are English-only — no CJK text
  in the tree.
- **Tests:** `uv run --locked --extra dev python -m pytest <path>`. The repo
  uses `uv`; the binary is at `~/.hermes/bin/uv` and is not on `PATH` by
  default. Lint with `uv run --locked --extra dev ruff check <path>`.
- **Platform adapters** live in `plugins/platforms/<name>/adapter.py`.
- **Gateway** runs as user systemd services `hermes-gateway-<profile>.service`.
  Restart after adapter changes with `hermes gateway restart --profile <name>`;
  the service boots from this source tree, so a restart picks up local edits.
  The webhook is live once its process owns the listen socket (e.g.
  `ss -ltnp | grep :8646` for LINE), even before the "webhook listening" log
  line settles.
