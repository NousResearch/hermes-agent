# Bot Mode native delivery matrix

Production Electron/preload/backend/tool execution; loopback scripted inference only.
Scratch HOME, HERMES_HOME, Python facade and CLI shims; no user profiles or credentials.
Run from the repository root after installing root npm dependencies and building Desktop:

```sh
cp evals/botmode-dm-matrix/probe-dm-matrix.spec.ts apps/desktop/e2e/
git apply evals/botmode-dm-matrix/mock-trigger.patch
(cd apps/desktop && npm run build)
# Set DISPLAY, current XAUTHORITY, XDG_RUNTIME_DIR and VIRTUAL_ENV first.
(cd apps/desktop && HERMES_DESKTOP_CDP_PORT=off BOT_DM_SERVICE_PATH=1 \
  BOT_DM_EVIDENCE=/tmp/botmode-dm-matrix-proof \
  npx playwright test e2e/probe-dm-matrix.spec.ts --reporter=list)
git apply -R evals/botmode-dm-matrix/mock-trigger.patch
rm apps/desktop/e2e/probe-dm-matrix.spec.ts
```

`BOT_DM_SERVICE_PATH=1` creates a disposable Python venv facade with an adjacent
worktree-pinned `hermes`; PATH contains a different inert `hermes` that exits 2
and records every invocation. This simulates the reported stale-launcher service
shape without modifying the shared interpreter or the user's installed launcher.
Omit that flag only on the baseline: a PATH shim then pins all children to this tree.

## Receipts

Base: `40f2702b22a343cbc95647efd0bdffe8e0b3d9e9`.
Normal PATH: two native cases passed. Default canonical Desktop owner remained
unchanged while default -> quiet Beta CLI -> unowned Gamma CLI executed, Beta
received its own completion and default received the Beta completion. Named
persisted Bot Chat IDs remained `matrix-beta` and `matrix-gamma`; both used the
scratch OS HOME as cwd, not the profile state directory. Gamma opened in Desktop
and rendered its attributed input. This refutes #105323's `--in ~` ownership premise.

Incoming live-owner case also passed without reload: wait for the attributed
sender card to mount, then expand it. A rejected earlier probe expanded before
asynchronous reconciliation had mounted the card; its final DOM already contained
`Message from beta` and `show message`. Reload was never required.

Stale PATH on base: real default message_agent acknowledged `sent`, wrong launcher
recorded `-p beta chat --in ~ -c Bot Chat --create-if-missing -Q --query-file ...`,
and Beta had zero input rows. With #100673 cherry-picked: same matrix, two passed;
both nested children used the adjacent worktree entrypoint, no wrong-path marker.

Artifacts retained separately under `/tmp/botmode-dm-matrix/{verified,service-base,service-fixed}`:
SQLite snapshots, active lease snapshots, scoped process checkpoints, prompts,
DOM text and screenshots. The `.live.json` ticket path is only for live-owner
admission; unowned CLI fallback does not promise that receipt format.

This establishes Linux native Electron behavior, not native Windows parity,
real-model reasoning quality, or the separate 420-second tool-timeout report.
No native Windows host was discovered in the available host inventory.
