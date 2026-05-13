# Scripts

This directory contains repository maintenance scripts, install helpers, diagnostics, and integration utilities.

## Safety Rules

Before running a script:

- read the script or its documentation;
- confirm the target environment;
- prefer `--help` and `--dry-run` when available;
- avoid running state-changing scripts against production without approval;
- do not paste secrets into command lines that may be logged;
- capture verification output after the script completes.

## Categories

### Test and Quality

- `run_tests.sh` — canonical pytest wrapper with CI-like hermetic settings.
- `lint_diff.py` — lint changed files.

### Install and Setup

- `install.sh` — Linux/macOS/WSL/Termux installer.
- `install.ps1` — native Windows PowerShell installer.
- `install.cmd` — Windows command wrapper.
- `setup_open_webui.sh` — Open WebUI setup helper.
- `install_psutil_android.py` — Android/Termux psutil helper.

### Release and Build

- `release.py` — release automation.
- `build_model_catalog.py` — model catalog generation.
- `build_skills_index.py` — skills index generation.

### Diagnostics and Profiling

- `check-windows-footguns.py` — Windows compatibility checks.
- `discord-voice-doctor.py` — Discord voice diagnostics.
- `keystroke_diagnostic.py` — terminal keystroke diagnostics.
- `profile-tui.py` — TUI profiling helper.
- `benchmark_browser_eval.py` — browser evaluation benchmark helper.

### Utilities

- `hermes-gateway` — gateway launcher/helper.
- `kill_modal.sh` — Modal cleanup helper.
- `sample_and_compress.py` — trajectory/sample compression helper.
- `contributor_audit.py` — contributor audit helper.

### Integrations

- `whatsapp-bridge/` — WhatsApp bridge Node project.
- `lib/node-bootstrap.sh` — shared Node bootstrap helper.

### Lab Area

- `lab/` — reserved for IT automation lab scripts.
- `templates/` — safe starter templates for new scripts.

## New Script Checklist

- [ ] Has a clear purpose and owner.
- [ ] Has `--help` or usage text.
- [ ] Supports `--dry-run` if it changes state.
- [ ] Validates required commands and inputs.
- [ ] Avoids printing secrets.
- [ ] Uses deterministic exit codes.
- [ ] Has a related runbook in `docs/runbooks/` when operational.
- [ ] Documents verification and rollback steps.
