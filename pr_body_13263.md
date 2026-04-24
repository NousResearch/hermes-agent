## What does this PR do?

Fixes skill-routing behavior when the model emits a skill slug as a tool call (for example `text2sql` or `github-auth`).
The agent now rewrites this invalid call into `skill_view(name=...)`, so skill loading proceeds instead of failing with an unknown-tool retry loop.

## Related Issue

Fixes #13263

## Type of Change

- [x] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix
- [ ] 📝 Documentation update
- [x] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [ ] 🎯 New skill (bundled or hub)

## Changes Made

- Updated `run_agent.py` to add `_repair_skill_name_tool_call()` that maps skill-like invalid tool names to `skill_view` with a `name` argument.
- Integrated this repair into the invalid-tool pre-validation path so the model can recover in the same turn.
- Added `tests/run_agent/test_skill_like_tool_call_repair.py` with coverage for rewrite, argument preservation, and skip behavior when `skill_view` is unavailable.

## How to Test

1. Run `scripts/run_tests.sh tests/run_agent/test_skill_like_tool_call_repair.py`.
2. In CLI chat with skills enabled, ask a query that previously caused a skill slug to be emitted as a tool call.
3. Confirm the run no longer fails on `Unknown tool '<skill-name>'` and proceeds via `skill_view`.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [ ] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes (required for bug fixes, strongly encouraged for features)
- [x] I've tested on my platform: Linux (CentOS 8 environment)

### Documentation & Housekeeping

- [x] I've updated relevant documentation (README, `docs/`, docstrings) — or N/A
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — or N/A
- [x] I've updated tool descriptions/schemas if I changed tool behavior — or N/A

## Screenshots / Logs

- `scripts/run_tests.sh tests/run_agent/test_skill_like_tool_call_repair.py` currently fails in this environment due to missing `.venv`/`venv`.
