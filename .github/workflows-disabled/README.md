# Disabled workflows (fork cleanup)

These workflows were intentionally moved out of `.github/workflows/` to disable them in this fork and reduce CI noise/maintenance.

## Disabled in this fork
- `deploy-site.yml`
- `docker-publish.yml`
- `nix-lockfile-fix.yml`
- `skills-index.yml`
- `osv-scanner.yml`

## Why
- Upstream-gated or release/deploy specific paths that are not part of the fork's core merge-signal.
- Secrets-heavy and/or write-side-effect workflows not needed for day-to-day PR validation.
- Disabled/inert workflows with no operational value in this fork.

## Re-enable
Move a workflow file back into `.github/workflows/` and push a branch/PR.
