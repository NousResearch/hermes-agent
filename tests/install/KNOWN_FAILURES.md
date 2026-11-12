# Confirmed historical upgrade limitations

These failures cannot be repaired by changing the update target: the failing code is already loaded from the starting release. The workflow still executes each original update path. Only a match on the exact starting commit, method pair, failed assertion, and fresh log signatures produces a non-red known-failure receipt. Other errors still fail. The results table shows matched cases as `known [n]`, with the explanation and evidence in a footnote at the bottom. These cases are counted separately from passed upgrades.

The machine-readable rules in `e2e-assets/known-failures.json` own the matcher and report footnote text. This document explains their historical evidence. Logs are rotated before each attempt so an earlier failure cannot classify a later one.

## Windows launcher self-lock

Classification: **unfixable in the update target for the exact released `hermes.exe update` path**.

| Starting release | Released commit | Install → update | Verified failing job |
|---|---|---|---|
| `v2026.3.12` | `a370ab8391ca5f8de7ebbc449f05cb0df36ade7c` | `installer-script` → `hermes-update` | [101514756800](https://github.com/ethernet8023/hermes-agent/actions/runs/34043635705/job/101514756800) |
| `v2026.4.8` | `86960cdbb0148145890e2ee90b4e157fa899f6e1` | `installer-script` → `hermes-update` | [101514755527](https://github.com/ethernet8023/hermes-agent/actions/runs/34043635705/job/101514755527) |

The running console launcher holds `venv/Scripts/hermes.exe` open. The old updater pulls the new checkout, then asks uv to replace that same executable during an editable install. Windows rejects the replacement with `Access is denied. (os error 5)`. The old updater's ZIP fallback repeats the dependency install and encounters the same lock.

Evidence required: the CLI update phase failed, the traceback identifies the running `hermes.exe/__main__.py`, and uv reports failure to remove that install's `Scripts/hermes.exe` with OS error 5. A generic access-denied error on another file does not match.

The March call is in the released `hermes_cli/main.py:1678-1683`, with the ZIP fallback at `1571-1576`. April calls `_install_python_dependencies_with_optional_fallback`, whose released body at `3295-3321` runs the installs without launcher quarantine. Those function objects were loaded before the checkout changed. May's sampled CLI update passed; do not classify it from this record.

Re-running the installer is a separate tested upgrade route. Invoking the old CLI through its venv Python is a possible recovery route, but is not silently substituted for the console-launcher leg.

## July Windows app offers only a manual update for script installs

Classification: **unfixable in the update target for the exact released app-button path**.

Starting release: `v2026.7.1`, commit `7c1a029553d87c43ecff8a3821336bc95872213b`.

| Install → update | Verified failing job |
|---|---|
| `installer-script` → `hermes-desktop-app-update` | [101514755236](https://github.com/ethernet8023/hermes-agent/actions/runs/34043635705/job/101514755236) |
| `installer-script+desktop` → `hermes-desktop-app-update` | [101514760893](https://github.com/ethernet8023/hermes-agent/actions/runs/34043635705/job/101514760893) |
| `installer-script+desktop` → `open-app-update` | [101514756508](https://github.com/ethernet8023/hermes-agent/actions/runs/34043635705/job/101514756508) |

These script installs have no staged updater. The released Electron code (`apps/desktop/electron/main.cjs:2212-2214`) logs `no staged updater; surfacing manual` and returns `{ ok: true, manual: true, command }`. It does not start an update. Each job's `logs/desktop.log` records that branch followed by `[updates] manual: hermes update`; no target checkout/result signal appears.

Evidence required: an app-update leg from this released commit and those explicit manual-update log entries. A hand-off timeout without the manual message is not this limitation. Desktop-installer installs have a different staged-updater path and are not covered by this classification.

## macOS June Hermes-Setup.dmg updater refused by the v2026.7.30 update lock

Classification: **unfixable in the update target for the exact released app-update path from the published `Hermes-Setup.dmg`**.

Starting release: `v2026.7.30` (v0.19.1), commit `cc4cab2f592e60a197e796506de9168f74baf3ea`.

| Install → update | Verified failing job |
|---|---|
| `desktop-installer@latest` → `hermes-desktop-app-update` | [125232826636](https://github.com/NousResearch/hermes-agent/actions/runs/41431962361/job/125232826636) |
| `desktop-installer@latest` → `open-app-update` | [125232827364](https://github.com/NousResearch/hermes-agent/actions/runs/41431962361/job/125232827364) |

`desktop-installer@latest` is the published `Hermes-Setup.dmg` (June 2026 build). It copies itself to `HERMES_HOME/hermes-setup`; the v0.19.1 app's Update now spawns that copy with `--update` and exits. The June binary writes the update marker with its own PID, then runs the old checkout's `hermes update`. v2026.7.30's `update_lock.py` recognises its orchestrator only through `HERMES_UPDATE_HANDOFF_PID`, which the June binary predates (the ancestor-PID fallback landed a day after the tag). The pre-pull CLI therefore exits 2 within a second with `✗ Another Hermes update is already running (PID <hermes-setup>)`, and hermes-setup shows "UPDATE DID NOT FINISH". Nothing fetches and no completion receipt is written. The refusal runs in v0.19.1 code, so the update target cannot repair it. The same DMG legs from v0.22.5 and HEAD → NEXT pass, as do the installer-script v2026.7.30 legs.

Evidence required (`macos-desktop-e2e.sh` rotates `HERMES_HOME/logs/bootstrap-installer.log` before the attempt and collects it on failure): the assertion `app-driven update exited 1; transcript above`, plus both of these in the fresh `bootstrap-installer.log`: the `bootstrap.log: ✗ Another Hermes update is already running (PID …` refusal and hermes-setup's `update stage stage=update state=Failed … Hermes is still running. Close all Hermes windows and try the update again.` A completion-receipt timeout without the refusal (for example a fetch failure) is not this limitation. Diagnosis: [debug run 41435547785](https://github.com/NousResearch/hermes-agent/actions/runs/41435547785/job/125244635706).

Recovery: run `hermes update` from a terminal, or re-run the installer. Both are separate tested routes (`desktop-installer@latest` → `hermes-update` / `installer-script`).

## Not classified as unfixable

The July desktop-installer → app-update failure was a driver lifetime bug, not a released-updater exception. The driver treated an expected page closure as failure and could exit before Playwright released its launch process. On Windows, inherited pipes delayed the `close` event even after the launch process exited with code 0. Playwright then ran its tree-kill cleanup. The driver now waits independently of the closing page, releases its pipe handles after process exit, and waits for `close` before it exits. [The real July rerun](https://github.com/ethernet8023/hermes-agent/actions/runs/34075042380/job/101599434616) reached the target commit, cleared the update marker, passed the CLI check, and relaunched the app.

Onboarding click failures, zoom drift, native permission dialogs, AutoHotkey window waits, stale update markers, autostash conflicts, network failures, and generic timeouts remain actionable or unclassified until diagnosed. They must not inherit a historical label because they occurred on an old release.
