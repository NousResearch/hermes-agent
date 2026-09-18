# Agent Inbox: local implementation and verification

## Status

The isolated Agent Inbox implementation is locally verified on Windows. It is not published, merged, or installed in the consolidated TEST BUILD. The existing package and launcher have not been replaced during this closeout.

Source: `C:/w/hermes-agent-inbox`, branch `feature/agent-inbox-recovery`.
Base: `cdceca42e107f0e51ab9ff50e1cc881ad76b577e`.

## User-facing behavior

- A permanent status-bar Inbox button opens a dismissible panel.
- Needs you separates pending approvals/questions from routine Automation state.
- The panel names its active profile and connection scope. It is not an all-profile or all-process inbox.
- Open session navigates to the durable session and closes the overlay. It does not approve, answer, pause, resume, or start work.
- Escape closes the overlay and returns keyboard focus to the Inbox button.
- Partial reads retain usable rows and expose Refresh. Refresh sends a new request, shows loading state and preserves the active profile.
- Connection/profile transitions clear old rows, reject late responses, and allow the new scope to fetch even while an old request is pending.
- Truncated coverage is not reported as all quiet. Initial connection state is not described as all clear.
- The wider list rail avoids unnecessary truncation of ordinary session names; the panel stacks at narrow widths using the existing shared overlay primitives.

## Final verification

| Gate | Result |
| --- | --- |
| Backend Inbox, contracts and protocol | 125 passed |
| Frontend Inbox, lifecycle and gateway switch | 66 passed |
| Renderer, Electron and E2E TypeScript projects | Passed |
| Production Desktop build | Passed, including main/preload bundling and dist assertion |
| Electron Inbox E2E | 22 passed |
| ESLint for Inbox source and unit tests | Passed |
| Git whitespace check | Passed |
| Seeder refuses the main profile as destination | Passed |

Commands were run from the isolated worktree, with the isolated Python interpreter. Backend tests used an explicit test HERMES_HOME. Electron fixtures used disposable homes created by the existing test harness.

The build emits bundler performance advisories; Vitest emits the existing Vite config-loader migration warning. Linting the surrounding statusbar integration file also reports an existing hook dependency warning outside the added Inbox block. These are not represented as a warning-free full-repository audit.

## Evidence quality

- `real-automation-detail.png`: actual renderer reading a session and goal persisted into a disposable SQLite database through the real backend.
- `inbox-populated-detail.png` and narrow/populated captures: real renderer, response-stub approvals/questions/automation. They are UI evidence, not live request integration proof.
- `fixture-partial-after-retry.png`: response-stub partial result; Refresh is required, clicked, and a new inbox.list request plus retained rows are asserted.
- Navigation checks the durable route, overlay dismissal, and unchanged durable session count via read-only SQLite.
- Focus checks use the actual focused button, not generic active-element presence.

Current logs and screenshots remain under `.inbox-work/` as local evidence, not source payload. The committed seeding helper is under `apps/desktop/e2e/fixtures/` and no longer depends on an untracked script or a hard-coded Windows virtualenv.

## Corrections during final review

1. Closed the overlay after Open session.
2. Replaced conditional/no-op interaction assertions with strict checks.
3. Removed WebSocket fixture races that sent both a stub and the real request with the same ID.
4. Added loading, scope-reset and truncation-badge regressions.
5. Kept Refresh scoped to the active profile.
6. Avoided a generated contract name collision with the existing PendingApproval type.
7. Corrected a path-case test to respect platform-specific path semantics rather than assuming every OS is case-insensitive.
8. Widened the Inbox list without altering shared panel layout globally.

## Isolation incident

An earlier worker seeder saved two fixture-only orphan goal keys to the main profile because it inherited HERMES_HOME. This was disclosed immediately. Exact values were backed up and only the verified orphan fixture records were removed with value-guarded predicates; read-back confirmed removal. No user chat rows were removed by that cleanup. Earlier blanket claims that daily state had never been touched were incorrect.

The replacement helper validates a disposable OS-temp sandbox, sets HERMES_HOME before importing Hermes, receives explicit sandbox environment, and checks both session and goal directly in that sandbox's SQLite file. Protected-home refusal and real sandbox seeding were exercised.

## Boundaries and remaining release work

This closes the local Inbox implementation/interaction pass, not the release of every unmerged contribution. Linux/macOS execution and full-repository testing were not performed. Persisted scans are bounded (200 rows by default, maximum 1000) and report truncation. Live pending requests are visible only to the gateway process that owns them. The panel currently uses English copy and the host app's existing modal visual treatment.

Next release stage: reconcile all intended unmerged features against current upstream, integrate them in consolidated test-build source, validate that combination, then produce and exercise the versioned package. Publishing a PR and replacing the existing test launcher remain separate approval boundaries.

Worker implementation used OpenCode MiMo V2.5 Free after the Union Alpha route failed. Parent handled orchestration, UI review, runtime acceptance and the final targeted corrections. No paid fallback worker was used.
