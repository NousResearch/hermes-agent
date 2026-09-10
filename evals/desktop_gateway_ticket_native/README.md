# Native Electron local-gateway ticket probe

Run from repository root on the actual target OS:

```sh
uv sync --locked --python 3.11 --extra dev
npm ci
uv run --no-sync python evals/desktop_gateway_ticket_native/run.py --output native-ticket-results
```

Do not use `--ignore-scripts` or `ELECTRON_SKIP_BINARY_DOWNLOAD`: Electron's native binary is required. The driver resolves Electron workspace-local first through Node's normal resolution. An existing native binary can be supplied with `--electron /absolute/path/to/electron` (on macOS, the app bundle's `Contents/MacOS/Electron`; on Windows, `electron.exe`). `--python` may select an existing repo-compatible venv launcher. Do not resolve the launcher's symlink to its base interpreter.

`build.mjs` uses the locked esbuild to bundle `electron-main.ts` plus the actual production `local-gateway.ts` and `local-gateway-python.ts`. It runs in **Electron main** (`process.type === 'browser'` and a real `process.versions.electron`), not Node with a forged platform. The Windows callback uses production `mintGatewayTicketWithPython`; POSIX retains the production Unix socket implementation. HTTP requests call production `nativeGatewayHttpHeaders`, and WS requests call production `mintLocalGatewayTicket`.

The driver launches two ordinary `python -m gateway.run` processes, using temporary HOME/USERPROFILE/HERMES_HOME and an allowlisted child environment without provider credentials. No service installation/start commands, live user profiles, model requests, or production patches are used. The sibling profile has its own live control listener. Daemons and owned descendants are terminated before their temporary state is deleted. Linux uses actual Electron's headless Ozone backend (no visible desktop required).

Assertions:
- Config GET returns 200; replaying the same native HTTP grant returns 401/403.
- WS `runtime.describe` reports the exact discovered instance/profile/epoch; a second handshake with the same ticket returns 401/403.
- Wrong instance and live sibling profile/original-instance combinations fail to mint for both purposes.
- A freshly minted HTTP grant cannot select the sibling config (403).
- Fresh HTTP succeeds after the negative checks, proving listener survival across repeated clients.

`receipt.json`, `electron.log`, and two daemon logs are written to the requested output directory. Tickets are never printed or retained in these receipts. This is the native main bootstrap surface, **not** full Desktop renderer/preload/startup UI coverage.

## Temporary native CI

`workflow.yml` is deliberately outside `.github`. The parent can copy it to `.github/workflows/native-ticket-temp.yml` on an `e2e/native-ticket-*` branch. Its push trigger runs actual `windows-latest` and `macos-latest` jobs; dispatch on a new nondefault workflow alone is insufficient. It uses `uv sync --locked` and root `npm ci`, uploads per-OS receipts, and needs no secrets. Never merge the temporary workflow. Inspect both downloaded receipts, not just job conclusions.

## Explicit limits

Linux execution does not validate the Windows named-pipe branch or macOS Unix-socket permissions; run the native jobs. The wrong-profile combination tests descriptor binding against a real other owner, not a fabricated owner subject. A Windows **wrong OS-owner SID** negative is not implemented: it requires an actual second-account process/token serving a permissive decoy pipe under an owned profile name (and an OS-verified differing SID), rather than the normal daemon's same-user DACL. The Linux host has no such Windows identity. Do not interpret same-user wrong-instance/profile checks, a mocked SID, or a DACL-only refusal as proof of the Python client's server-SID check. The receipt preserves `wrongOsOwnerSid.status = not_exercised` even when all implemented checks pass.
