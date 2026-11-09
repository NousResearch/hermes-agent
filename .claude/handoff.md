# Handoff: 2026-11-09 — repair stale source version identity

## Next session focus

Diagnose `source ./activate` completing PM provisioning but failing final shell composition under
zsh with `hermes_bootstrap_python:8: no matches found:
/Users/matt/Workspace/active/hermes-agent/../tools/python-*/bin/python3`. Reproduce from the current
checkout, trace why the bootstrap resolver searches beside the repository instead of the
install-scoped tool store, and verify intent with `git log -p -S '<symbol>'` before changing it.

## Completed this session

- Fixed mutable source stamps that keep an old release label after a missing stable tag arrives at
  the same `HEAD`; full identity falls back to live Git while SHA-only probes remain one
  `rev-parse`.
- Added behavior coverage for same-HEAD tag arrival and the source-stamped startup fast path.
- Commit: `be03ddc335` (`fix(version): refresh source identity when release tag arrives`).
- Opened NousResearch/hermes-agent PR #151590.

## Key decisions

- Validate only stable tags pointing directly at `HEAD`: this repairs the proven stale-stamp case
  without weakening packaged-stamp authority.
- Keep release validation out of `installed_commit_sha()` because update/fleet startup callers need
  only commit freshness and explicitly promise a one-Git-call path.
- Design and verification details are recorded in the plan doc.

## Artifacts

- Plan: `docs/plans/2026-11-08-stale-source-version-stamp.md`
- PR: https://github.com/NousResearch/hermes-agent/pull/151590

## Open blockers

- The configured `origin` SSH URL rejects the available key. Authenticated HTTPS push works with
  GitHub CLI's credential helper.

## Test command

```bash
HERMES_PYTHON=/Users/matt/.hermes/installs/84ffe32793cc3ecd/test-environment/gen-340b13b43f18438da18102cc4441f745/venv/bin/python scripts/run_tests.sh tests/hermes_cli/test_version_info.py tests/hermes_cli/test_source_stamp.py tests/hermes_cli/test_startup_fast_guards.py tests/hermes_cli/test_version_info_startup_git.py
/Users/matt/.hermes/installs/84ffe32793cc3ecd/test-environment/gen-340b13b43f18438da18102cc4441f745/venv/bin/python scripts/check
```
