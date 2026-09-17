# Embedded Browser Hide / Restore smoke test

From `apps/desktop`, with root workspace dependencies installed:

```sh
node scripts/browser-hide-restore/smoke.mjs
```

This launches a separate Electron process with a temporary `userData` directory
and a deterministic loopback page. It imports the real pane renderer, preview
contribution mirror, guest ownership and automation modules. It does not start a
Hermes backend, read an installed browser profile, or use authentication.

The checks cover:

- The browser's Hide menu and restore rail retain the same guest/webContents.
- Hidden guests retain their viewport, unsaved DOM state and running JavaScript.
- In-app click/type/read automation works while hidden, without revealing the pane.
- Hidden panes stay out of document-wide visible-pane lookup and host focus.
- Ordinary tab-cache eviction does not destroy live browser guests.
- Closing a hidden browser unregisters its automation and destroys its webContents.

Logs, screenshots and a machine-readable result are written to the repository's
`.hide-restore-artifacts/` directory. These are local verification artifacts, not
source files to commit. The fixture closes its Electron process and servers and
removes temporary user data after each run.

This is an Electron component integration test, not a packaged-app/backend E2E.
It deliberately retains the existing window/session routing and browser partition
policy. It does not prove multi-profile backend authorization, OS-specific browser
credential imports, or persistence of live page state across app restart.
