# Windows Desktop Update Verification Checklist

Phase 4 task 4.6: manual verification of the desktop update flow on Windows.

**Phase 4 is not complete until this doc has a filled-in PASS column.**

## Prerequisites

- Fresh Windows 10/11 VM (no prior Hermes install)
- Git Bash or WSL available
- A GitHub release with bundles + hermes-updater published (from phase 0's CI workflow)
- The `HERMES_SIGNING_KEY` secret set in the repo (for signed bundles)

## Steps

| # | Step | Expected | PASS |
|---|------|----------|------|
| 1 | Download `hermes-updater-win-x64.exe` from the GitHub release | File downloads successfully | ☐ |
| 2 | Run `hermes-updater.exe install --channel stable` | Creates `$HERMES_HOME/versions/<v>/`, `current.txt`, `bin/hermes.exe` | ☐ |
| 3 | Verify `hermes --version` works from a new terminal | Prints version, exit 0 | ☐ |
| 4 | Launch the desktop app from the slot's `desktop/` directory | App window appears, connects to backend | ☐ |
| 5 | Check for updates (Settings → Updates or auto-check) | App detects no update (already current) OR detects a newer version if available | ☐ |
| 6 | If an update is available: click Apply | Tauri progress window (or updater console) shows download → verify → stage → flip | ☐ |
| 7 | App exits during the flip | Process terminates cleanly (no crash dialog) | ☐ |
| 8 | App relaunches automatically into the new version | New window appears, `getVersion()` reports the new version | ☐ |
| 9 | Verify `current.txt` now names the new version | `cat $HERMES_HOME/current.txt` shows new version | ☐ |
| 10 | Verify `previous.txt` names the old version | `cat $HERMES_HOME/previous.txt` shows old version | ☐ |
| 11 | Run `hermes-updater rollback` | `current.txt` reverts to old version, `hermes --version` reports old version | ☐ |
| 12 | Check for stale `hermes-updater.old.exe` files | `hermes-updater.old.exe` swept on next updater run (or absent) | ☐ |
| 13 | Verify the old slot is still present in `versions/` | `ls $HERMES_HOME/versions/` shows both versions | ☐ |
| 14 | Run `hermes-updater gc --keep 1` | Old slot (not current/previous) is removed | ☐ |
| 15 | Relaunch desktop app after rollback | App launches on the rolled-back version | ☐ |

## Notes

- The Tauri progress window is the Windows GUI shell for the updater (phase 4 task 4.2). It shows `--report json` events as progress bars/status text.
- The `.hermes-update-in-progress` marker file should be present only during the flip+restart critical section (seconds, not minutes).
- If the updater crashes mid-apply, `current.txt` should be untouched — the staging dir gets cleaned on the next run.
- The `hermes-updater.old.exe` sweep is the Windows two-step: rename running exe → `.old.exe`, move new into place, delete `.old.exe` best-effort.

## Results

Fill in the PASS column above after running the checklist on a real Windows VM.
Record any failures with the exact error message and the step number.
