# Project Rules

Technical workflow rules for changes in this branch and theme workspace.

## Branching & Git Hygiene
- **Never commit directly to `main`.** Always work on a dedicated local feature/fix branch (e.g., `theme/one-dark-sync`) and integrate via Pull Request only when explicitly approved.
- **No push or PR without explicit approval.** Keep all work local until instructed otherwise.
- **One focused commit per change.** Keep commits atomic, conventional (`feat(desktop):`, `fix(desktop):`, `docs:`), and strictly scoped to the task.
- **Never rewrite published history.** Prefer clean forward fix commits over force-pushes.

## Backups & Rollback Safety
- **Back up before editing.** Before modifying any source or test file, copy the pre-edit originals (preserving relative paths) to durable storage outside the checkout (`~/.hermes/local-patches/one-dark/<timestamp>-<slug>/`) and generate a `SHA256SUMS` manifest.
- **Verify integrity.** Confirm copied files match source checksums (`sha256sum -c`) before proceeding.

## Permissions & Hooks
- **Always use `request-review` execution mode.** Never use `--dangerously-skip-permissions`.
- **Pre-commit hooks (`lefthook`):**
  - Docs (`.md`) and stylesheet (`.css`) commits must pass without `--no-verify`.
  - `--no-verify` is permitted **only** on `.ts`/`.tsx` commits in a sandbox where `lefthook`'s `uv` bootstrap fails due to read-only filesystem restrictions, **only** after explicit maintainer approval, and **must** be documented in the commit message body.

## Security & Privacy
- **Never read, search for, log, or commit tokens, passwords, API keys, or credential paths.**
- **Stop on authentication walls.** If an inspection target redirects to a login screen, stop and report instead of searching for credentials or reusing personal browser profiles.

## Build, Test & Runtime Boundaries
- **Build with `--no-install`:** Use `apps/desktop/scripts/build-webapp.mjs --no-install` with the pinned runtime matching `pm/lock.json`.
- **Run targeted Vitest suites** covering every modified theme and terminal module before reporting completion.
- **Never restart the gateway automatically.** The operator restarts the gateway and validates the live UI.
