# Rollback / quarantine

## Operator-state rollback

- Root SOUL backup: `/home/sikmindz/.ares/kit-backups/ares-soul-kit-20260822T041947Z/root-SOUL.md`.
- Remove only the newly created `/home/sikmindz/.ares/profiles/explorer` and `/home/sikmindz/.ares/profiles/public` if the operator explicitly requests profile rollback.
- Remove only the kit-installed `/home/sikmindz/.ares/skills/strategy/ares-operating-modes/` directory if requested.
- Restore profile config backups:
  - `explorer/config.yaml.before-codex-20260822T045102Z`
  - `public/config.yaml.before-codex-20260822T045102Z`

## Source rollback

Revert only the scoped source files named in `CHANGE_RECEIPT.json` after reviewing the current diff. Do not reset the checkout globally; `README.md`, `.tmp_codemod_marker`, and `_write_test` are unrelated dirty state.

## Forbidden rollback shortcuts

Do not delete or rewrite existing specialist profiles, user SOUL files, credentials, memory, sessions, cron state, gateway state, or the active runtime release as part of this rollback. No commit, push, deployment, release switch, or gateway restart occurred in this pass.
