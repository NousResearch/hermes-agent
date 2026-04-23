## What does this PR do?

Fixes a false-negative in `hermes dashboard` where the CLI can report **"Web UI build failed"** on Windows even when the frontend build artifacts are actually produced successfully.

If `npm run build` exits non-zero but `web/dist/index.html` exists after the build step, we treat the build as successful and continue (this avoids blocking `hermes dashboard` on a misleading exit code).

## Related Issue

Fixes #13244

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix
- [ ] 📝 Documentation update
- [x] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [ ] 🎯 New skill (bundled or hub)

## Changes Made

- `hermes_cli/main.py`
  - Remove stale `web/dist/` before building to avoid masking failures with old artifacts
  - If `npm run build` returns non-zero but `web/dist/index.html` exists, treat it as a successful build

- `tests/hermes_cli/test_web_ui_build.py`
  - Add regression tests covering the “non-zero exit code but dist exists” success path

## How to Test

### Local tests (Linux)

Ran in repo venv: `./.venv312` (Python 3.12.7)

```bash
./.venv312/bin/python -m pytest -q tests/hermes_cli/test_web_ui_build.py
```

Result: `2 passed` (warnings only; no failures)

### Manual repro (Windows)

1. On Windows PowerShell, run `hermes dashboard` in a fresh install where `HERMES_WEB_DIST` is not set.
2. Confirm the CLI reports `✓ Web UI built` and the dashboard starts, even if the underlying `npm run build` returns a non-zero exit code while still producing `web/dist/index.html`.
3. (Optional) Validate failure behavior by ensuring no `web/dist/index.html` is produced and confirming `hermes dashboard` exits with an error.

## Checklist

### Code

- [ ] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [ ] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes (required for bug fixes, strongly encouraged for features)
- [ ] I've tested on my platform: Windows 11 (manual repro steps described above)

### Documentation & Housekeeping

- [ ] I've updated relevant documentation (README, `docs/`, docstrings) — or N/A
- [ ] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A
- [ ] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — or N/A
- [ ] I've updated tool descriptions/schemas if I changed tool behavior — or N/A

