# Scope & Technical Boundaries (Teto)

Hard boundaries defining what is out of scope for this workstream.

## Out of Scope — Do Not Do
- **No changes on `main`.** Never checkout, commit to, reset, or merge `main`.
- **No remote operations without explicit approval.** Do not `git push`, open PRs, or alter git remotes.
- **No dependency installation or lockfile edits.** Always build with `--no-install` reusing the existing `.build/webapp-workspace`; never edit `package.json`, `pnpm-lock.yaml`, or `pm/lock.json`.
- **No unsolicited changes outside the theme & integrated terminal scope.** Do not modify chat tool cards, Bot Mode workspace routing, backend gateway code, or upstream root `README.md`.
- **No upstream xterm.js patching.** Known xterm.js canvas mouse-selection offset under CSS `zoom !== 1` is documented in `.ai/pitfalls.md` and left untouched.
- **No chasing accepted visual edge cases.** Known small residual terminal spacing at `125%` UI scale is accepted and documented in `.ai/pitfalls.md`; do not add layout workarounds.
- **No service restarts or system mutations.** Never restart the Hermes gateway, install system packages/browsers, or modify global git config.
- **No credential access.** Never read tokens, secrets, `.env` files, or personal browser profiles.

## Mandatory Stop Conditions
Stop immediately and report to the maintainer if any of the following occurs:
1. A fix requires modifying files outside the authorized scope.
2. A destructive action or history rewrite would be needed.
3. A commit hook requires `--no-verify` without prior approval for that file type.
4. An unresolved build/test failure or authentication wall blocks verification.
