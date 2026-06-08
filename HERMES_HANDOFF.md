# Hermes Handoff — E2E agentic workflow hardening

Date: 2026-06-08
Branch: `fix/e2e-agentic-workflow-hardening`
Repository: `/opt/hermes-agent`

## Summary

Implemented and verified regression coverage for profile-scoped skill loading, Coder dashboard base-path routing, dashboard readiness checks, stale dashboard detection, and steer import-graph safety. Documented the autonomous-agent worktree/branch/PR workflow, cleaned generated Node core dumps out of the repository without deleting them, and preserved the desktop image-attachment fallback work that was present in the checkout.

## Changed files

- `.gitignore`
  - Ignores `ui-tui/core.*` so local Node/TUI core dumps do not appear as commit candidates.
- `tools/skills_tool.py`
  - Resolves the active profile skills directory at call time when the module-level `SKILLS_DIR` still equals its import-time snapshot.
  - Preserves test monkeypatch behavior when `SKILLS_DIR` is explicitly patched.
- `tests/tools/test_skills_tool.py`
  - Adds regressions for context-local `HERMES_HOME` skill discovery and `skill_view()`.
- `hermes_cli/dashboard_auth/public_paths.py`
  - Allows unauthenticated access to the read-only `/api/readiness` probe.
- `hermes_cli/web_server.py`
  - Adds `/api/readiness` and runtime import-graph readiness data in `/api/status`.
  - Adds Coder base-path middleware for HTTP and WebSocket dashboard routes.
  - Adds Coder-managed update guidance and avoids spawning `hermes update` when `HERMES_CODER_MANAGED_UPDATE` is enabled.
  - Makes dashboard attached gateway opt-in via `HERMES_DASHBOARD_ATTACH_GATEWAY=1` and maps wildcard binds to concrete loopback hosts for child clients.
- `hermes_cli/main.py`
  - Detects stale dashboard processes launched as `hermes -p <profile> dashboard` or equivalent module/script forms.
- `tests/hermes_cli/test_update_stale_dashboard.py`
  - Adds stale-dashboard matcher regressions for global profile flags and false positives.
- `tests/hermes_cli/test_web_server.py`
  - Adds readiness, base-path HTTP/WebSocket, Coder-managed update, and dashboard gateway opt-in regressions.
- `tests/run_agent/test_steer.py`
  - Adds import-graph regression for `STEER_CHANNEL_NOTE` exposure.
- `website/docs/user-guide/git-worktrees.md`
  - Documents the default autonomous-agent workflow: one worktree, one branch, draft PR, CI/review gate, merge only through PR.
- `package-lock.json`
  - Existing npm lockfile metadata changes were preserved and included because the operator asked to commit everything accordingly.
- `apps/desktop/src/app/session/hooks/use-prompt-actions.ts`
  - Falls back from `image.attach` to `image.attach_bytes` when a localhost/tunnel gateway cannot resolve a local screenshot path.
- `apps/desktop/src/app/session/hooks/use-prompt-actions.test.tsx`
  - Adds regression coverage for the local-tunnel image byte-upload fallback.

## Cleanup performed

Moved generated Node core dumps out of the repo instead of deleting them:

- `ui-tui/core.151721` → `/tmp/hermes-core-quarantine/20260608-e2e-agentic-workflow/core.151721`
- `ui-tui/core.158582` → `/tmp/hermes-core-quarantine/20260608-e2e-agentic-workflow/core.158582`

These were ELF core files from `/usr/local/bin/node --expose-gc /opt/hermes-agent/ui-tui/dist/entry.js` and were not committed.

## Verification

Final verification command:

```bash
git diff --check && ./scripts/run_tests.sh \
  tests/tools/test_skills_tool.py \
  tests/hermes_cli/test_web_server.py \
  tests/hermes_cli/test_update_stale_dashboard.py \
  tests/run_agent/test_steer.py
```

Observed final result after the independent review fix:

```text
403 tests passed, 0 failed
```

Independent pre-commit review initially found a blocking base-path auth-ordering issue. Fixed by registering the Coder base-path middleware as the outermost ASGI middleware and adding protected prefixed API coverage:

- `tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_protected_api_under_coder_base_path_still_requires_token`
- `tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_runtime_readiness_redacts_import_exception_details`

Static added-line scans for hardcoded secrets, shell injection, eval/exec, pickle, and simple SQL string formatting returned no matches.

Desktop verification for the preserved image-attachment fallback:

```bash
npm run type-check --workspace apps/desktop
npm run test:ui --workspace apps/desktop -- use-prompt-actions
```

Observed result:

```text
tsc -b passed
src/app/session/hooks/use-prompt-actions.test.tsx: 12 tests passed
```

Additional direct runtime/profile check:

```text
{'success': True, 'name': 'impeccable', 'skill_dir': '/home/coder/.hermes/profiles/vvb-agent/skills/impeccable'}
```

## Agentic workflow rule going forward

Coding agents should not share a dirty checkout. Each autonomous coding task should use:

1. clean parent clone;
2. dedicated git worktree;
3. dedicated branch;
4. scoped commits;
5. draft PR;
6. CI/review gates;
7. merge only through PR after approval.

Kanban coding cards should request `workspace_kind=worktree` with a stable `workspace_path` under the team's Hermes worktree root.

## Remaining notes

- Branch `fix/e2e-agentic-workflow-hardening` was rebased onto `origin/main` after both local commits, then re-verified.
- No production/staging secrets, deployments, releases, registry pushes, auto-merges, or destructive cleanup were performed.
