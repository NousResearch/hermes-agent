# Hermes Agent — Five-Contribution Implementation Plan

## Document Purpose

Plan five separately reviewable upstream contributions. This document authorizes no implementation by itself. The planning task changes only this file; all code, tests, documentation updates, runtime rehearsals, and upstream communication below belong to subsequent implementation work.

The scope is the five proposed new contributions in the supplied audit. Its four suggestions to help existing terminal, backup, Matrix, and middleware PRs are excluded.

## Source and Planning Baseline

### Repository State

| Item | Verified during planning on October 2, 2026 |
|---|---|
| Checkout | `/Users/novrusshehaj/Github/OpenSource/hermes-agent` |
| Branch | `chore/document-correction` |
| HEAD | `0a374d167424cdc730ce9761368b62255b551e58` |
| Initial working tree | Clean; no tracked or untracked changes reported by Git |
| Origin | `git@github.com:NovrusShehaj/hermes-agent.git` |
| Upstream | `git@github.com:NousResearch/hermes-agent.git` |
| Live upstream snapshot | `4e3fcd5cd7e40c37cb6f9a21a76fc57a7361957a`, committed October 2 at 18:41:19 UTC |
| Upstream comparison | GitHub reported 54 commits ahead of this checkout |

No fetch, checkout, reset, merge, rebase, commit, or push was performed. The remote SHA was read through GitHub, not installed into this worktree. Later implementation must start from a deliberately refreshed base, preserving existing work.

### Audit Baseline

The audit refers to October 2, 2026, commit `0a374d1674`. The local revision still matches it, but its claim of parity with live upstream is now stale. The inspected local branch is `chore/document-correction`; the supplied audit does not establish its local branch name.

Repository searches, including hidden/ignored-file discovery excluding dependency and Git internals, did not locate `Hermes Agent Contribution Opportunities.md` or an existing file at this plan's destination. The complete embedded fallback in the request is therefore the source. There was no repository copy to reconcile.

### Verification Performed

- Read local root contribution rules, `CONTRIBUTING.md`, desktop engineering and source rules, TUI gateway rules, plugin rules, and applicable developer documentation. Reviewed `SECURITY.md` after upstream added an explicit security-reporting pointer.
- Traced test-interpreter selection through `scripts/run_tests.sh`, `scripts/_activation.sh`, `activate`, and `pm/testenv.py`.
- Traced Bot Mode editor and creation callers, MCP setup, SDK route selection, backend MCP contracts/handlers, and existing routing/profile tests.
- Traced client PCM capture, remote wake detection, event dispatch, composer start requests, voice recording/transcription, pause/resume, and plugin event/disposal/visibility infrastructure.
- Inspected pinned resolution, persistence, drag order, sidebar preferences, palette data overlay, title fallback, locale context, and relevant tests.
- Read live issues #131518, #131522, #131485, their comments/assignees, targeted PR searches, and PR #131607. Rechecked related OAuth PR #120549.
- Compared local HEAD with the fixed upstream snapshot. Relevant wake, Bot MCP setup, pin resolver, and test-runner paths were absent from that comparison's changed-file list. `tui_gateway/methods_tools.py` and generated contracts changed for `browser.manage`, not the MCP setup defect; root rules acquired a security-reporting section. Shared desktop/i18n files have other upstream changes and must not be overwritten.

These are source and GitHub observations. No application was launched, microphone recorded, test suite executed, routing bug reproduced, dependency installed, or upstream message posted during planning.

### Upstream / Duplicate-Work Status

| Phase | Verified status | Consequence |
|---|---|---|
| 1 | Targeted title searches found no exact test-environment documentation PR. | Repeat search before editing; negative search is not proof of exclusivity. |
| 2 | Exact symbol and Bot/MCP/API-key searches found no matching fix. [#120549](https://github.com/NousResearch/hermes-agent/pull/120549) remains open and changes OAuth payload scoping in `api/mcp.ts`, not Bot Mode's API-key component. | Preserve that work; recheck both add/key/test and OAuth paths on the implementation base. |
| 3 | [#131518](https://github.com/NousResearch/hermes-agent/issues/131518) open, no assignees or comments; no matching issue-reference/pre-roll PR found. | Coordinate scope and obtain a real remote-backend microphone rehearsal. |
| 4 | [#131522](https://github.com/NousResearch/hermes-agent/issues/131522) open, no assignees or comments. No API agreement recorded there. | Public API design approval remains a prerequisite to implementation. |
| 5 | [#131485](https://github.com/NousResearch/hermes-agent/issues/131485) open, reporter intends to implement. New [#131607](https://github.com/NousResearch/hermes-agent/pull/131607), author `MohamadKanso`, is open, with no comments/reviews at inspection. | Join or coordinate a follow-up; do not publish a competing implementation. |

PR #131607 proposes unconditional alphabetical resolution and disables manual dragging. Its body and changed-file list were inspected; its implementation was not independently tested. That proposal does **not** satisfy this task's requirement to retain both manual and alphabetical modes. Resolve this difference with the reporter, author, and maintainers before Phase 5 coding.

Repeat read-only checks immediately before each contribution, using issue numbers plus symbols/symptoms. Example commands, run from the repository:

```bash
gh api repos/NousResearch/hermes-agent/commits/main --jq .sha
gh issue view 131518 -R NousResearch/hermes-agent --json state,body,comments,assignees
gh issue view 131522 -R NousResearch/hermes-agent --json state,body,comments,assignees
gh issue view 131485 -R NousResearch/hermes-agent --json state,body,comments,assignees
gh pr view 131607 -R NousResearch/hermes-agent --json state,body,comments,reviews,files
gh search prs -R NousResearch/hermes-agent '"131518"' --limit 20
gh search prs -R NousResearch/hermes-agent '"131522"' --limit 20
gh search prs -R NousResearch/hermes-agent '"131485"' --limit 20
gh search prs -R NousResearch/hermes-agent '"McpSetupButton"' --limit 20
gh search prs -R NousResearch/hermes-agent '"test" "environment" "docs" in:title' --limit 20
```

Search all PR states; inspect nearby merged/closed work rather than relying solely on open-issue status. Communication requires separate authorization; this plan records coordination tasks, not messages already sent.

### Planning Assumptions and Limitations

- Paths and symbols below describe the inspected checkout. Reconcile them with the implementation base first. Do not apply old whole-file replacements to newer code.
- **Verified existing file (E):** every repository path proposed for modification is existing unless explicitly marked N or D below. Grouped test commands refer to verified existing directories/files.
- **Verified new file needed (N):** a narrow new responsibility or absent regression harness is justified by inspection; the proposed path does not yet exist and is a plan, not an existing repository fact.
- **Implementation-time discovery (D):** placement or contract depends on a design checkpoint; resolve it before coding that portion. The phrase **To verify during implementation** marks an unexecuted or unsettled claim.
- Timeouts, buffer bounds, API names, and new preference names explicitly described as proposed are design decisions to validate, not existing behavior.
- Native audio, two-backend integration, and public API coordination are real gates. Mocks cannot satisfy them. If unavailable, report the phase incomplete at that gate.

## Executive Summary

| Phase | Result | Size / main difficulty | Recommended entry condition |
|---|---|---|---|
| 1 | Accurate managed-vs-explicit test-interpreter instructions across four documents | Small; environment precedence | Refresh base and reproduce the documented selection behavior |
| 2 | Bot MCP add/key/test/probe operations stay on the selected bot's backend/profile | Medium; alias identity and async lifetime | Capture a failing base routing regression |
| 3 | Client-captured speech survives wake detection and voice startup as one utterance | Large; continuous capture, bounded PCM, ownership | Approve bounded audio design and arrange remote microphone testing |
| 4 | A visible plugin voice page can own a wake turn and reliably release/re-arm it | Medium/large; public API and cleanup | Agree API semantics, build on Phase 3 ownership contract |
| 5 | Optional locale-aware alphabetical pins coexist with recoverable manual order | Small/medium; persistence and coordination | Resolve overlap with #131607 first |

## Cross-Phase Engineering Principles

1. Keep model prompts, history, toolsets, and cache prefixes unchanged. None of these contributions needs a new core model tool.
2. Respect root and area rules. Use existing nanostores, routing SDK, plugin context, profile scope, and test infrastructure. Do not introduce another state library or generic event-interception framework.
3. Start bug fixes with a base-revision reproduction. Prefer one or two focused behavioral regression tests per defect, with table-driven edge cases. Do not test source text, freeze catalogs/counts, or fake the host OS.
4. UI `connectionId`, logical/display profile, backend `targetProfile`, transport instance, and session identity are distinct. Capture ownership before asynchronous work and preserve it through completion and teardown.
5. Run targeted tests first, affected suites once changes stabilize, then the stated broader gate. Record skips, failures, native test host, and exact revision. A test command below is a future procedure, not a passed result.
6. Python validation always uses `scripts/run_tests.sh`. Desktop scripts verified in `apps/desktop/package.json`: `test:ui`, `test:desktop:platforms`, `typecheck`, `lint`, `build`, and `dev`. Install JS workspaces from the root using existing project instructions; never create a second desktop lockfile. PM owns Python environments.
7. Large existing files receive wiring only. New audio and SDK policy belongs in narrow siblings. If an extraction is necessary under the root size rule, make it mechanical and isolated within the relevant contribution; update affected documentation in that same contribution.
8. No production home, credentials, or user messages in fixtures. Use disposable homes/runtime directories, dummy keys, local fixtures, and a separate Electron user-data directory. Preserve any work encountered when implementation begins.
9. For any newly demonstrated security-boundary violation, consult current `SECURITY.md` before public reproduction details. The static routing mismatch alone is not a proven exploit or credential disclosure.

## Dependency and Contribution Sequence

Execution order: **Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5**. Phases 1, 2, and 5 are technically independent; Phase 5 coordination can happen early without coding a competing PR.

Phase 3 establishes one **internal** wake-capture transaction: bounded retained audio, captured transport/target, single consumer, cancellation, and idempotent cleanup. Phase 4 adds only plugin selection and a public adapter over that transaction. Phase 3 must work and be mergeable with no plugin claims; Phase 4 must not replace its recorder or add a second microphone/re-arm manager.

Agree the internal ownership states before Phase 3 implementation, but do not expose speculative SDK methods in PR 3. Draft Phase 4 against the reviewed PR 3 interface or wait for it to merge. Reconcile overlapping `wiring.tsx`, `wake-word.ts`, and composer cleanup edits explicitly. Do not silently run both plugin and chat handlers for one wake event.

## Phase 1 — Correct the Developer Test-Environment Documentation

### Objective

Make the developer instructions accurately describe explicit interpreter selection and the managed test environment. Change documentation only; preserve runner behavior, hermetic environment scrubbing, PM ownership, and Bash-on-Windows execution.

### Verified Repository Evidence

| Existing file / symbol | Established behavior or gap |
|---|---|
| `scripts/run_tests.sh`: `_has_pytest`, interpreter-selection block | With no `__HERMES_ACTIVATED`, an executable `HERMES_PYTHON` that imports pytest wins. Otherwise checks activation freshness and `__HERMES_TEST_PYTHON`, sourcing `activate --` when needed. |
| `scripts/_activation.sh`: `hermes_activation_current` | Validates checkout ownership and input stamps; a branch/lock change can require reactivation. |
| `activate`; `pm/testenv.py`: `ensure_testenv`, `testenv_root` | Activation builds a side test environment with dev/test groups and default `all` extras, separate from application generations. Activation's bootstrap search for `.venv` is not runner selection of that venv as the test interpreter. |
| `CONTRIBUTING.md`: Manual development and test environment; Running tests | Incorrect automatic `.venv` discovery statement and obsolete Windows fallback precedence. |
| `AGENTS.md`: Development Environment; Testing | Obsolete `.venv` → `venv` → installed-venv probe description; independent-build instructions need explicit selection/context. |
| `website/docs/developer-guide/contributing.md` | Mirrors the stale discovery/Windows text; also permits bare pytest where root rules require the wrapper. |
| `website/docs/reference/package-management.md`: Test and editor environments | Managed environment description is current; independent-build example says to run the wrapper without clearly selecting the freshly built interpreter. Clarify this bridge rather than rewriting the section. |
| `README.md` | Points to CONTRIBUTING; no conflicting test-interpreter selection found. No change currently justified. |
| `tests/scripts/test_run_tests_shell.py`, `tests/scripts/test_hermes_python_prologue.py`, `tests/pm/test_activate_scripts.py` | Existing shell failure-propagation, activation freshness, and script behavior coverage. |

The audit understated the affected documentation: the website contributor mirror and Windows text also need synchronization. Provider documentation links to the contributor section; retain that anchor. Direct `.venv` commands in runtime/worktree examples are not automatically wrong and are not targets of a blanket replacement.

### Problem / Gap

A contributor can build `.venv`, run the canonical runner, and incorrectly assume that interpreter was selected. An already activated shell also does not obey the unactivated `HERMES_PYTHON` override branch. The documentation must state these conditions precisely.

### Scope

#### In Scope

The four documentation files identified above, their directly relevant snippets and anchors, and verification of the documented workflows.

#### Out of Scope

Runner/PM behavior changes, Python dependency edits, deletion of existing environments, unrelated installation/worktree documentation, and new documentation snapshot tests.

### Dependencies and Preconditions

- Repeat duplicate searches and inspect current runner after refreshing the base.
- Use a disposable development home and runtime directory; have the supported bootstrapped Python and Bash available. Native Windows evidence is needed for Windows-specific claims, not an emulated platform flag.
- No requirement to rewrite a working README link or unrelated translated document merely to increase the diff.

### Detailed Implementation Steps

- [ ] Re-read the runner's complete selection branch and activation scripts on the implementation base; record the explicit-override conditions and stale-activation behavior.
- [ ] Update `CONTRIBUTING.md` to lead with activation-managed tests and separately describe independent environments. State that an independent interpreter must be selected via `HERMES_PYTHON` in an unactivated shell, whether inside or outside the checkout.
- [ ] Replace the obsolete root `AGENTS.md` probe chain and align its test-build example with the same distinction. Preserve all existing test isolation and PM dependency rules.
- [ ] Synchronize `website/docs/developer-guide/contributing.md`, including Windows/Bash guidance and the contradictory bare-pytest exception. Preserve `manual-development-and-test-environment` links used by provider docs.
- [ ] Clarify the independent-environment example in `website/docs/reference/package-management.md`: show explicit selection, and explain that an already activated shell continues using the managed test environment. Use a fresh unactivated shell or supported `deactivate`, not hand-edited activation internals.
- [ ] Search README files, root/area instructions, website docs, and scripts for the exact obsolete claims. Classify each hit as test selection, bootstrap selection, or an unrelated explicit runtime path; edit only genuine contradictions.
- [ ] Rehearse the revised commands before final wording. Record selected interpreter evidence and runner exit status; keep environment creation and any generated build output outside the documentation diff.

### Test Changes

None expected. Do not add tests asserting prose or code-string shape. Reuse the existing shell/activation suites and runtime rehearsal. If verification exposes a code defect, report it separately rather than expanding a documentation PR.

### Validation Procedure

#### Automated Validation

From the root with a prepared test environment:

```bash
scripts/run_tests.sh tests/scripts/test_run_tests_shell.py tests/scripts/test_hermes_python_prologue.py tests/pm/test_activate_scripts.py -q
rg -n 'discovers repository|probes `.venv`|no local `.venv`|HERMES_PYTHON' CONTRIBUTING.md AGENTS.md website/docs/developer-guide/contributing.md website/docs/reference/package-management.md
git diff --check
```

Review search results semantically; a zero-match search returns 1 and is not itself a test failure. Before the PR, build the English documentation with `npm run build:fast` from `website` after preparing that package's dependencies. Inspect any generated files and exclude unrelated output.

#### Manual / Environment Validation

In a disposable instance, prepare `HERMES_HOME` and `HERMES_RUNTIME_DIR` as instructed in the PM developer workflow. Test three paths: unactivated + explicit interpreter containing pytest; cold managed activation without explicit selection; warm activation with a different valid `HERMES_PYTHON` (the managed interpreter must still win). A fourth negative control uses an explicit interpreter without pytest and verifies documented fallback/error behavior. Run `tests/scripts/test_run_tests_shell.py` via the wrapper in each applicable case. Rehearse the PowerShell environment assignment plus Bash invocation natively before claiming Windows validation. Do not run these setup procedures against the user's production home.

### Edge Cases and Failure Modes

An executable without pytest; stale activation stamps; inherited activation from another worktree; a `.venv` used only to bootstrap PM; Windows path conversion; user interpretation that test extras modify the packaged app; a documentation build generating unrelated files.

### Acceptance Criteria

- [ ] All four identified documentation surfaces describe the actual two selection paths and activated-shell precedence.
- [ ] Existing section links still resolve; README delegation remains correct.
- [ ] Both explicit and managed workflows have isolated-home execution evidence.
- [ ] The PR contains documentation only and makes no unverified native-platform claims.

### Pull-Request Boundary

PR 1 is independently mergeable documentation correction. Include before/after instructions and interpreter-selection receipts. No runtime changes or new tests are needed solely to validate wording.

### Definition of Done

The docs match the current runner, searches find no remaining relevant contradictory examples, targeted shell/activation checks and the English docs build pass, and the isolated command rehearsals demonstrate the two workflows. Record any native Windows gap explicitly; do not call that platform validated without its receipt.

## Phase 2 — Fix Bot Mode's API-Key MCP Setup for Remote/Profile-Scoped Bots

### Objective

Every setup probe, catalog add, key write, and connection test acts on the selected bot's captured connection and backend profile. Foreground navigation must not retarget the operation. Preserve OAuth, local compatibility, lazy bot creation, error envelopes, and existing plugin-owned-server restrictions.

### Verified Repository Evidence

| Existing file / symbol | Role |
|---|---|
| `apps/desktop/src/plugins/hermes-bots/mcp-setup.tsx`: `McpSetupButton`, `mcpRpc`, `mcpSetupSupported`, `beginKeys`, `submitKeys`, `resolveProfile` | `host.request()` uses ambient gateway. An object-valued `profile` is forwarded verbatim by add/key/test. Support is cached in one module boolean and probed with `{}`. |
| `profile-config.tsx`: `AdvancedProfileConfig` | Resolves `botRoute`, backend name, and `backendScope`; hands the reduced scope to MCP setup. |
| `create-dialog.tsx`: `requestForTarget`, `ensureAgentCreated`, MCP capability row | Can create remotely through a full descriptor, but passes only the new slug to `McpSetupButton`; creation path must be fixed too. |
| `routing.ts`: `botConnectionRoute`, `resolveBotConnectionRoute`, `scopedBotParams`, `requestForBot`, `botBackendProfileScope` | Existing explicit routing and logical-to-backend parameter mapping. The two-field REST-style scope is not a full SDK RPC route. |
| `types.ts`: `ProfileRoute`; `apps/desktop/src/sdk/index.ts`: `requestPluginProfile`, `host.requestProfile`, `host.request` | Full RPC descriptor requires `connectionId`, logical `profile`, and `targetProfile`; `host.request` reads `$gateway`. Profile-only overload is limited to legacy/sole-local topology. |
| `apps/desktop/src/store/gateway.ts`: `requestGatewayForAgent`, `requestGatewayForProfile` | Real routing boundary below the SDK; exercise it rather than mocking away every routing decision. |
| `tui_gateway/methods_tools.py`: `_mcp_rpc`, `_profile_scoped_rpc`, `_str_arg`, add/set_api_key/test handlers | Profile-scoped config/secret operations. `contracts/tools_mcp_plugins.py` declares name-string profile parameters. An object can fail dispatch validation; direct handler tests may instead stringify it. Neither is a valid route. |
| `mcp-setup.tsx`: `beginOAuth`; `apps/desktop/src/api/mcp.ts` | OAuth uses captured scope through `host.completeMcpOAuth`; do not revert this newer path based on the stale component TODO mentioning OAuth. |

Identity contract: connection ID selects a registered backend; logical profile identifies the roster/alias route; `targetProfile` is the profile name sent on the wire. A route such as `(A, friendly-bot, default)` must dial A's friendly alias and send `profile: "default"`, not `"friendly-bot"` or the descriptor object.

### Problem / Gap

Expected: editing a bot on A while chatting on B writes/tests A's bot profile. Current: reduced object scope enters a string parameter, requests use the foreground socket, and one backend's unsupported probe can hide setup on another. New Bot creation also loses its chosen connection before credential setup.

This is a verified static mismatch, not a runtime reproduction from this planning pass.

### Scope

#### In Scope

Bot Mode API-key setup, both callers, owner-bound capability checks, async cancellation/stale UI handling, and focused routing regressions. OAuth changes only where a common caller contract needs adaptation without altering its auth protocol.

#### Out of Scope

Generic gateway routing redesign, accepting descriptor objects in backend profile fields, credential schema changes, global MCP reload behavior, unrelated OAuth fixes, or a new connection manager.

### Dependencies and Preconditions

- Refresh upstream and inspect exact-symbol searches plus #120549 and nearby MCP PRs.
- Reproduce on the base: two connections with identically named `work` profiles, foreground B, editor for A; capture dispatch error or wrong target using dummy credentials.
- Prepare two isolated `serve` instances and a controlled MCP fixture endpoint that records which dummy credential it receives. Also prepare one multiplexed instance with launch and secondary homes.

### Detailed Implementation Steps

- [ ] Record failing add/key/test/probe behavior before changing code; distinguish contract-validation failure from handler behavior.
- [ ] Replace the MCP component's ambiguous reduced scope with a captured operation target containing the full existing `ProfileRoute` and explicit backend profile. Reuse `requestForBot`/`scopedBotParams` where the caller has a roster row; otherwise retain the complete descriptor rather than reconstructing one from a backend name.
- [ ] In `AdvancedProfileConfig`, pass the already resolved bot route and target. An orphaned source-scoped bot must report an unavailable target; never fall back to the ambient gateway for a credential write.
- [ ] In `create-dialog.tsx`, carry the selected connection through `ensureAgentCreated` into a complete setup target. Preserve `requestForTarget` semantics and the newly created slug. Capture selection before creation awaits; if selection changes, retire the old UI completion without guessing a new owner.
- [ ] Capture one immutable target at setup start and reuse it across catalog add, all key writes, test, and completion. Ensure `params.profile` is the backend name. Pin OAuth to its existing captured scope and adapt shared types only as necessary.
- [ ] Use the existing foreground spawn priority for explicit setup actions so backend startup does not wait in the passive queue. Preserve server-side error payload handling; a fulfilled RPC with `ok: false` must not be treated as successful configuration.
- [ ] Remove the process-wide `_mcpRpcSupported` boolean. Prefer a component/target-generation probe with cancellation over a new global cache. Reprobe on target/backend replacement; only a confirmed unknown method means unsupported. Transient/auth/network errors remain retryable errors, not permanent unsupported results.
- [ ] Give API-key flows the same lifetime discipline as OAuth: invalidate on target replacement/unmount, clear typed key values when switching targets, suppress stale success/onDone, and never continue later steps against a new target. An already accepted key write is not rolled back; report partial completion without logging key contents.
- [ ] Keep the component thin. If route normalization needs a helper, use the existing `routing.ts` when generally applicable; otherwise a small topical sibling is D, not a new generic SDK API.

### Test Changes

Existing coverage to reuse/extend:

- `apps/desktop/src/plugins/hermes-bots/routing.test.ts`: real Bot routing, mocked SDK boundary; verify alias-to-target mapping.
- `apps/desktop/src/sdk/profile-routing.test.ts` and `apps/desktop/src/store/gateway-profile-request.test.ts`: full-descriptor and actual routing contracts. Keep at least one regression using real SDK/store routing with only native transport/dial boundaries replaced.
- `tests/tui_gateway/test_mcp_profile_rpcs.py`: real registered MCP handlers and temp config/.env writes. Existing tests bypass full dispatch; complement with real transport validation rather than treating them as frontend-to-backend E2E.
- `apps/desktop/src/lib/mcp-dashboard-oauth.test.ts`: existing OAuth non-regression.
- **N:** `apps/desktop/src/plugins/hermes-bots/mcp-setup.test.tsx`, because no Bot MCP component test was found. Render the actual component/callers; two main behavioral cases can cover (1) A→B→A routing/probe identity including logical aliases, and (2) async target replacement/cancellation. Parameterize add failure, key failure, test failure, unsupported backend, and new-bot creation.

Never validate only that `host.requestProfile` was called: assert the actual selected connection and final string-valued wire profile, then verify filesystem effects in real integration.

### Validation Procedure

#### Automated Validation

From `apps/desktop`, after adding the new regression file:

```bash
npm run test:ui -- src/plugins/hermes-bots/mcp-setup.test.tsx src/plugins/hermes-bots/routing.test.ts src/sdk/profile-routing.test.ts src/store/gateway-profile-request.test.ts src/lib/mcp-dashboard-oauth.test.ts
npm run typecheck
npm run lint
```

From the root:

```bash
scripts/run_tests.sh tests/tui_gateway/test_mcp_profile_rpcs.py -q
git diff --check
```

Before review, run the full desktop `npm run test:ui`; run the broader `tests/tui_gateway/` directory if backend code/contracts changed. Keep a base-failing/patched-passing receipt for the specific routing regression.

#### Manual / Environment Validation

1. Connect A and B, each with profile `work`, and distinct dummy credentials on a test MCP endpoint. Include an alias whose logical profile differs from backend `targetProfile`.
2. Through the actual desktop Bot editor run add→set key(s)→test on A, then B, then A. Inspect each disposable `config.yaml`/`.env` and fixture receipt: only the intended profile changes, launch/sibling homes remain unchanged.
3. Switch foreground chat/connection while add and test are pending. The captured operation stays with its bot or is explicitly cancelled; no subsequent write migrates to the foreground.
4. Repeat New Bot creation on remote B while foreground is A. Confirm credential setup follows the newly created remote profile.
5. Probe a backend without lifecycle RPCs followed by one with support, then return to the first. Verify unsupported state does not leak. Test reconnect/replacement with the same connection label.
6. Repeat launch-profile and secondary-profile cases on one multiplexed backend; exercise OAuth once to ensure its scope remains intact. Store only redacted evidence.

### Edge Cases and Failure Modes

Same profile names across connections; friendly aliases targeting `default`; orphaned or removed connections; stale promise completion after reopening an editor; unsupported vs offline backend; partial multi-key writes; duplicate catalog adds on retry; empty key fields; older SDK without explicit routing; changed bot target during lazy creation; failure envelopes nested in `result`.

### Acceptance Criteria

- [ ] All four operations—probe, add, key write, and test—use one captured connection/profile target.
- [ ] No descriptor object reaches `params.profile`; aliases resolve to backend names.
- [ ] A→B→A with same-named profiles and foreground switching passes actual backend checks.
- [ ] Edit Bot and New Bot work; transient probe errors remain recoverable.
- [ ] Stale UI callbacks cannot enable/configure the newly selected bot.
- [ ] Existing local and OAuth behavior passes regression checks.

### Pull-Request Boundary

PR 2 is independently mergeable, normally frontend-only plus regression coverage. Reference the actual reproduction and adjacent OAuth work without claiming that PR fixes this path. Do not widen backend contracts to accommodate malformed UI payloads.

### Definition of Done

Base regression demonstrated, corrected routing and lifetime tests pass, full desktop UI/type/lint checks complete, and two-backend plus multiplexed-home receipts prove destination/key/test isolation. No production credentials appear in logs, fixtures, or PR evidence. Related upstream work has been rechecked.

## Phase 3 — Preserve Speech Immediately After the Wake Phrase

### Objective

Preserve the first client-captured voice utterance across remote wake detection and desktop voice startup, including speech spoken before the event reaches the renderer. Continue existing voice behavior for later turns and explicit mic starts. Keep capture bounded, owner-scoped, cancellable, and released exactly once.

### Verified Repository Evidence

| Existing file / symbol | Established role |
|---|---|
| `apps/desktop/src/lib/wake-client-capture.ts`: `startClientWakeCapture`, `downsampleTo16k`, `floatToInt16LE`, queue/drain | Browser Float32 channel 0 is resampled to 16 kHz; sends base64 signed int16 little-endian mono. Default frame 1280 samples/80 ms; queue capped at 24 frames and batches at four. Queue drops old unsent frames; it is not a retained recording. |
| `apps/desktop/src/store/wake-word.ts`: `clientCapture`, `stopClientCapture`, `maybeStartClientCapture`, `resumeWakeAfterVoice` | Owns capture handle. Several operations consult ambient `$gateway`; re-arm can restart client capture and retries status/start. |
| `tui_gateway/methods_voice.py`: `_wake_detect_handler`, wake feed/pause/resume handlers | Pauses detector and verifies transport ownership before emitting phrase/profile/start-new-session. Feed accepts 16 kHz PCM, max 64,000 bytes per request. |
| `tools/wake_word.py`: `WakeWordDetector.feed`, `_run`, `pause`, `feed_audio` | Queues/splits engine frames, processes them asynchronously, and owns a transport-bound detector. Detection emits no client sample cursor or exact phrase-end alignment. |
| `apps/desktop/src/app/contrib/wiring.tsx`: `handleGatewayEventWithPlugins` | Stops client capture immediately, plays wake sound, selects/new-creates profile session, requests voice start. Cross-profile `ensureGatewayProfile` can be fire-and-forget. |
| `apps/desktop/src/store/composer.ts`: start-request atom and `takeVoiceConversationStart` | One-shot integer request, no audio or owner payload. |
| `apps/desktop/src/app/chat/composer/hooks/use-composer-voice.ts` | Consumes request in main composer; pauses wake via ambient gateway and calls re-arm on completion. |
| `use-voice-conversation.ts`: listening start, `handleTurn`, conversation owner/generation | Awaits mic barrier, starts recorder, gates empty/silent clips, transcribes/submits using a captured owner and rejects stale completions. |
| `use-mic-recorder.ts`: `useMicRecorder`, `start`, `stop`, `cancel` | Opens a new getUserMedia stream; uses MediaRecorder MIME negotiation (WebM/Opus, MP4, Ogg, WAV), returns `MicRecording` Blob/duration/speech flags. |
| `apps/desktop/src/lib/voice-barge-in.ts` | Existing precedent: continuous pre-trigger recording and one complete utterance to avoid losing first syllables. Its playback detector is not the wake detector. |
| `apps/desktop/src/api/system.ts`: `transcribeAudio`; `lib/voice-client-direct.ts`: `transcribeAudioClientDirect` | Existing transcription routes; preserve captured-owner configuration and fallback behavior. |

The audit's gap is confirmed statically. Runtime audio loss has not been reproduced here. Event arrival time is not wake-phrase end time. There is no evidence supporting exact sample-level trimming at that point.

### Problem / Gap

Stopping one stream and awaiting another loses the interval between them. Starting a buffer only after `wake.detected` also loses speech during detector/network latency. Joining raw PCM bytes to an encoded MediaRecorder Blob produces an invalid audio file; joining two recordings without a sample boundary can duplicate or omit speech.

### Scope

#### In Scope

Desktop **client-capture mode against a remote backend**, bounded pre-event retention, continuous first-utterance handoff, owner/generation propagation, endpoint/cleanup integration, and real audio validation. Local backend deployments using client capture share the same path.

#### Out of Scope

Rewriting all voice recording, changing wake engines, adding a new STT provider, changing later-turn barge-in, or claiming server-local PortAudio capture is fixed. Server-local capture keeps its existing path; extending it requires a separate backend audio transport decision. Document this limit on #131518.

### Dependencies and Preconditions

- Recheck #131518 and related audio changes, coordinate the bounded client-capture scope, and capture a real base recording/transcript.
- Access to native Electron microphone permissions, a remote backend, a functioning wake model, and a test STT route is required.
- Review the following internal design before implementation. It deliberately solves retention and continuous ownership together; performance bounds are proposed and must be measured.

### Detailed Implementation Steps

- [ ] Instrument a disposable base run with monotonic timing only: input samples, queued feed, received wake, recorder ready, first transcription. Do not add permanent audio telemetry or log raw audio/credentials.
- [ ] Retain a bounded ring of the **locally captured samples before network queue drops**, using a monotonic sample cursor. Proposed default: five seconds of 16 kHz mono int16 (160,000 PCM bytes), plus separately bounded first-utterance storage. Verify resampler continuity at 44.1/48 kHz across callback boundaries; carry fractional phase if the current per-block averaging would distort a recording. Do not modify wake feed limits or queue recovery unnecessarily.
- [ ] Keep the source stream alive on a valid wake; stop detector feeding without stopping the track. Transfer a single-use capture lease to the first voice utterance. Represent states explicitly: listening → offered → recording-first-utterance → stopped/cancelled. Stop/start/re-arm, old onError callbacks, and duplicate wake events must be generation-checked.
- [ ] Use one continuous PCM segment for the entire first utterance and encode it once as a valid mono PCM WAV for existing transcription APIs. Extend the recorder's optional start input to consume this lease and produce the existing `MicRecording` result. Ordinary starts and subsequent utterances retain MediaRecorder. Do not concatenate PCM with Opus/MP4 or open a second getUserMedia stream while the lease owns the mic.
- [ ] Reuse existing recorder VAD/silence/max-recording policy through an adapter, including `heardSpeech`, meter-failure semantics, duration, and cancel behavior. The buffered speech must participate in speech detection: a user who finishes speaking before UI startup must still yield a transcribable clip. Verify the level metric rather than applying the existing normalized threshold directly to raw PCM RMS.
- [ ] Define explicit limits for unconsumed offers (proposed 10 seconds) and first-utterance capture (use the existing configured recording cap, with a finite internal hard ceiling established during the design review). Account for ring bytes, retained utterance, encoding copies, and recorder shutdown; bound memory even if UI startup hangs or silence never arrives. Limit exhaustion must visibly cancel/retry, not silently present a truncated transcript as complete.
- [ ] Preserve the entire retained prefix rather than cutting at renderer event arrival. The current protocol has no phrase-end cursor. A short wake-phrase lead-in is acceptable; exact phrase removal is not promised. Do not blindly strip matching words from arbitrary user text. Measure supported latency against the retention window and document behavior outside it. If exact phrase alignment becomes an agreed requirement, it is a D protocol extension requiring backend sample metadata and compatibility negotiation, not an assumed existing field.
- [ ] Replace the wake-origin integer-only start request with an owner-addressed, single-consumption handoff while preserving the existing explicit-start API. Capture the detector transport and target profile separately: a wake may target another profile, but pause/resume must use the original owning transport. Await target profile/session readiness before allowing its composer to consume audio.
- [ ] Bind transcription and prompt submission to the intended owner using the existing conversation owner/generation mechanism. Foreground changes either retain the explicit target or abort; never let a newly mounted unrelated composer consume the clip. Expire orphaned offers and release their tracks.
- [ ] Adjust `wiring.tsx` to call a narrow handoff function instead of embedding a recorder/controller. Pin `wake-word.ts` feed/re-arm requesters to the lease's transport rather than looking up `$gateway` during each asynchronous step. Reconcile status only into the matching current UI scope.
- [ ] Prevent the wake confirmation sound from becoming request audio: test echo cancellation with the retained stream; prefer existing visual indication or delayed/gated sound during the handoff if reliable separation is unavailable. Do not insert an intentional speech-capture gap to play a chime.
- [ ] Finish/cancel through one idempotent cleanup: release PCM buffers, callbacks, timers, track, AudioContext and pending promises, then run the existing config-aware re-arm against the original owner if still valid. `/wake off`, owner disconnect, or another surface's lease must win over a stale retry.
- [ ] Add focused documentation to `website/docs/user-guide/desktop.md` describing client-capture behavior, limits, and the lack of server-local parity. Behavioral configuration, if approved, goes through existing config/schema/UI plumbing; no new environment variable. Default limits remain internal unless maintainers request an exposed preference.

**New-file placement:** **N** `apps/desktop/src/lib/wake-audio-handoff.ts` and `wake-audio-handoff.test.ts`: ring/sample accounting, single-owner PCM lease, and WAV encoding are a coherent testable responsibility absent today. Keep UI/profile orchestration out of that file. A small store sibling for owner-addressed offers is **D** if `wake-word.ts` cannot own it cleanly; choose one owner, not duplicated state in composer and wake stores.

### Test Changes

Extend verified tests at their existing boundaries:

- `lib/wake-client-capture.test.ts`: fake browser audio input, real resampling/queue/ring code; sequence-coded samples retained despite delayed feed and queue overflow, clean teardown, no failure callback from retired capture.
- New handoff test: known sample sequences before/during/after event, boundary uniqueness, bounded retention, valid decoded WAV sample count/rate/channels, cancellation and timeout.
- `store/wake-word.test.ts`, `store/wake-word-client-capture.test.ts`: original-transport re-arm, status generation, enable-off race, no double capture.
- `app/chat/composer/hooks/use-mic-recorder.test.tsx`: leased first recording vs ordinary stream acquisition, stop/cancel results and speech wholly in prefix.
- `use-voice-conversation-boundary.test.tsx`, `use-voice-conversation-rearm.test.tsx`, and `use-voice-conversation-stale-reply.test.tsx`: first utterance delivered once to captured owner, pending startup/cancellation, next normal turn, and stale transcription rejection.
- `lib/voice-barge-in.test.ts`: existing full-duplex behavior remains intact.
- `tests/tools/test_wake_word.py` and `tests/tui_gateway/test_tui_gateway_server.py`: existing detector ownership/pause and gateway wake checks. Extend only if the implemented backend contract changes; use real imports and temporary homes.

Mock microphone/network boundaries for deterministic tests, not the handoff policy being tested. Source-shape assertions cannot prove sample continuity.

### Validation Procedure

#### Automated Validation

From `apps/desktop`:

```bash
npm run test:ui -- src/lib/wake-audio-handoff.test.ts src/lib/wake-client-capture.test.ts src/store/wake-word.test.ts src/store/wake-word-client-capture.test.ts src/app/chat/composer/hooks/use-mic-recorder.test.tsx
npm run test:ui -- src/app/chat/composer/hooks/use-voice-conversation src/lib/voice-barge-in.test.ts
npm run typecheck
npm run lint
```

Before review run full `npm run test:ui`, then `npm run build` for the real Electron rehearsal. If backend changes were justified, run from root:

```bash
scripts/run_tests.sh tests/tools/test_wake_word.py tests/tools/test_wake_word_stream_recovery.py tests/tui_gateway/test_tui_gateway_server.py -q
```

A wire change also requires `scripts/gen_gateway_contracts.py` regeneration under the prepared interpreter and `scripts/run_tests.sh tests/tui_gateway/contracts/ -q`; never hand-edit generated TS/OpenRPC. The planned frontend-only conservative retention does not require that change.

#### Manual / Environment Validation

Use an isolated Electron user-data directory and disposable home/runtime, on native macOS plus any additional supported host being claimed. With a real remote backend, say **“Hey Hermes, what time is it?”** without pausing. Repeat with a longer distinctive phrase (“Hey Hermes, alpha seven beta nine, what time is it?”) to make missing/duplicated words obvious.

Capture at least ten trials per normal and delayed-start condition, recording only consented test audio, transcript, owner, timing, and result. Include user finishing before the composer mounts, cold profile startup, A→B target-profile routing, foreground switch, cancellation during mic handoff, backend disconnect, permission/device failure, repeated wake, and wake-off during the voice turn. Compare decoded sample boundaries in deterministic tests and audible/STT word continuity in real trials; STT text alone cannot prove no duplicate PCM. The first utterance must be delivered once, the next normal turn must work, and re-arming must allow another wake without a manual toggle.

Do not describe unbounded network delay as lossless. Establish the tested latency envelope and demonstrate honest failure outside it. Verify remote STT accepts WAV through the real endpoint and the configured provider route; mocked upload acceptance is insufficient.

### Edge Cases and Failure Modes

Detector processing lag precedes event delivery; feed queue drops; callback resampling drift; wake chime bleed; prefix-only speech; VAD ending while UI still loads; duplicate event; stale capture error stopping a replacement stream; delayed getUserMedia resolving after cancellation; profile switch redirecting re-arm; malformed sample rate; buffer/encoding allocation growth; multi-window detector ownership; old backend with only current event fields.

### Acceptance Criteria

- [ ] Deterministic decoded audio contains expected post-wake samples once, without gaps at the ownership boundary.
- [ ] A real microphone and remote backend preserve the test phrase without a deliberate pause; trial receipts record failures as well as successes.
- [ ] Only one capture owner exists throughout the first utterance; later/manual voice starts still work.
- [ ] Retention, utterance, timeout, and memory bounds are explicit and tested.
- [ ] Prefix-only speech is transcribed; stale/cancelled clips are not submitted.
- [ ] Correct target receives transcription, original detector owner receives cleanup/re-arm, and wake-off is respected.
- [ ] No buffer/track/context survives completion, cancellation, timeout, unmount, or owner loss.

### Pull-Request Boundary

PR 3 references #131518 and implements client-capture handoff only. Its internal lease has a real consumer (normal chat) and no public plugin claim API. Include audio architecture, tested latency/memory limits, native audio receipts, and explicitly unverified capture modes. Keep unrelated barge-in/provider work out.

### Definition of Done

Base audio loss is documented, sample/lifecycle regressions and full desktop checks pass, bounded first-utterance capture works through actual remote STT, native microphone trials show preserved nonduplicated post-wake speech, and next-turn/re-arm/cleanup behavior is verified. Unit-only completion is insufficient.

## Phase 4 — Let a Visible Plugin Page Handle and Re-Arm Wake-Word Events

### Objective

Expose a narrow, lifecycle-owned SDK capability for the voice-console consumer in #131522 to take a wake turn while its contributed page is visible. Preserve passive event subscriptions and normal chat fallback. Use Phase 3's capture transaction and existing backend wake lease rather than a second microphone manager.

### Verified Repository Evidence

- `apps/desktop/src/contrib/events.ts`: `emitGatewayEvent` is a passive isolated fan-out; listener return values do not consume events. `trackGatewayEventDisposers` tracks registrations made during plugin registration.
- `contrib/plugin.ts`: `PluginContext`, `createPluginContext`, `onDispose`, and `onEvent` provide plugin-attributed cleanup even for later registrations. `contrib/runtime-loader.ts` and `contrib/plugins.ts` manage runtime/bundled plugin registration and teardown.
- `sdk/index.ts`: `host.onEvent`, `host.paneVisibility`, and explicit routing are public. `paneVisibility` describes contributed panes, not an arbitrary plugin page's route ownership.
- `app/routes.ts`: `ROUTES_AREA`, `RouteContribution`, `contributedRoutes` retain contribution provenance and render functions. `app/contrib/surfaces.tsx`: `WorkspaceSurface` mounts plugin routes through `ContribBoundary` in the workspace page shell.
- `contrib/react/boundary.tsx`: render failure isolation. Mounted content can be kept alive; mount alone must not imply visible wake ownership.
- `app/contrib/wiring.tsx` currently observes then unconditionally performs chat wake handling. `store/wake-word.ts::resumeWakeAfterVoice` already reconciles enabled/status/start and respects another owner's lease.
- `website/docs/developer-guide/desktop-plugin-sdk.md` is the existing public authoring reference. No agreed wake-claim API was found in #131522.

### Problem / Gap

A plugin can observe detection but cannot prevent navigation into chat, receive a bounded ownership lease, or reliably re-arm after its own turn. Making generic event listeners cancellable would unnecessarily change every event's contract.

### Scope

#### In Scope

A visible contributed **page** wake claim, plugin-owned disposal, first-wake delivery, release/re-arm/fallback, types/docs, and an external consumer rehearsal. A page can be hidden despite being mounted; use host visibility/route identity.

#### Out of Scope

Global plugin event cancellation, tool/model schema changes, arbitrary background wake listeners, new wake detectors, registering the reporter's third-party voice product in core, or a general priority/arbitration framework.

### Dependencies and Preconditions

- Review current #131522 and ask the consumer to confirm lifecycle and audio requirements through an authorized communication step.
- **Maintainer/API-design checkpoint before code:** agree public shape, visible-page ownership, cancellation, success boundary, timeout, compatibility, and fallback. No agreement was recorded during planning.
- Reuse reviewed Phase 3 ownership primitives. If PR 3 is still open, declare the dependency and avoid parallel conflicting implementations.

### Detailed Implementation Steps

- [ ] Submit a small API sketch and state table for review. Prefer a **scoped `PluginContext` capability** over a bare global `host.voice` method because `ctx` already knows provenance and automatically disposes late registrations. Proposed spelling: `ctx.voice.claimWake({ routeId }, handler): () => void`; names and exact types remain subject to that checkpoint.
- [ ] Define registration vs event ownership separately. Registration makes a page eligible; it does not own the mic. At detection, the host confirms that the registration belongs to the active, visible contributed route and matching connection/profile scope, then offers that one event. Hidden/disabled/unloaded pages cannot claim.
- [ ] Proposed handler contract for review: return `false` synchronously to decline, or a promise for the owned turn. The lease contains captured owner/event information and an `AbortSignal`; no credentials or mutable global gateway. A fulfilled owned turn completes/re-arms; a pre-handling rejection releases to chat fallback after cleanup. A `markHandled()` boundary records that the consumer has begun delivering the utterance, preventing replay into chat after a later error. The method is not a generic event interceptor.
- [ ] Resolve **audio ownership at the API checkpoint**, before publishing types: Phase 3's retained capture cannot remain open while the plugin independently opens another mic. Offer one narrowly scoped first-utterance operation/handle from the same transaction; the plugin may consume it or explicitly discard/release it before using its own recorder. Do not expose an unbounded raw audio stream merely to support one console. Exact adapter placement and whether the initial SDK supports transferred audio are D; passing a wake without a safe capture disposition is not acceptable.
- [ ] Document event semantics: claim = host grants one active page a transaction; handle = consumer uses it; release before handling = relinquish to normal chat once; completion/re-arm = release consumer capture then reconcile original detector; cancellation after handling = no chat replay, visible failure and safe re-arm. A plugin does not permanently disable wake configuration.
- [ ] Implement policy in a narrow SDK sibling and delegate to Phase 3's owner-bound wake transaction. Preserve `emitGatewayEvent` as observation; do not allow its listeners to suppress core behavior. `wiring.tsx` calls the claim dispatcher and invokes existing chat handling only on the explicit fallback outcome.
- [ ] Resolve visible route authority in `WorkspaceSurface`/the contributed-route registry, including the workspace pane's actual visibility. Validate route provenance against the registering plugin; do not trust an arbitrary route ID or a plugin-supplied boolean alone. Integrate only the minimal page-activity signal required. Existing `host.paneVisibility` remains useful for panes but does not alone identify the current plugin route.
- [ ] Allow at most one registration per plugin route (reject duplicate active registration with a clear error or atomically replace and dispose it; choose at API review). The foreground contributed route is the winner, not last-mounted hidden content. Multiple unrelated consumers on the same route are not a supported priority system. Across windows, preserve the backend transport lease as authority.
- [ ] Wire unload/disable/hot-reload through `PluginContext` disposers; route hide/unmount/render failure/profile change aborts the offered/active lease. Establish bounded acquisition and turn deadlines based on the existing capture limits. A hung promise must not leave wake paused indefinitely.
- [ ] Ensure abort is followed by actual owned-resource disposal, not just an emitted signal. Host-managed audio is stopped by the host; plugin-owned recorders must provide completion/cleanup acknowledgment under the agreed contract. Do not open fallback capture concurrently with unconfirmed release; show a recoverable failure when cleanup cannot be established.
- [ ] Make release/re-arm idempotent and target the original detector transport only. If disconnected, disabled, or no longer owner, retire without waking another backend. Preserve `/wake off`, capture failure notice, and normal UI indicator behavior.
- [ ] Update public types and `desktop-plugin-sdk.md` with feature detection, a complete mounted-visible-page example, try/finally cleanup, error outcomes, old-host fallback, and the rule that passive `onEvent` is not a claim. Exercise the actual external consumer or a disposable runtime-plugin fixture through the normal loader.

**File placement:** **N** `apps/desktop/src/sdk/wake.ts` and `wake.test.ts`, justified as a new narrow public capability with policy kept out of the large SDK facade. Extend E `contrib/plugin.ts` for scoped exposure and E `sdk/index.ts` only for necessary type/export wiring. **D** exact host page-activity helper/test placement under existing route/surface ownership; select after confirming the current visibility source, not a new generic plugin framework.

### Test Changes

- New SDK wake tests exercise real registration/arbitration/lease states with controlled capture and transport boundaries: accept, decline, throw, rejection, handled failure, timeout, duplicate detection, idempotent completion, two registrations, and wrong owner.
- Extend `contrib/plugin.test.ts` and `contrib/runtime-loader.test.ts`: unload/reload/disable removes claims and aborts in-flight leases, including claims registered after `register()` returns.
- Extend `app/contrib/surfaces.app-routes.test.tsx` or a D colocated rendered-route test if that existing harness cannot exercise actual visibility: hidden/mounted page, route switch, boundary failure, and profile switch cannot keep ownership.
- Extend `store/wake-word.test.ts` / `wake-word-client-capture.test.ts` for original-owner cleanup/re-arm. Re-run Phase 3 first-utterance and normal-chat tests with zero plugin registrations.
- Test normal chat fallback once, not both chat and plugin handling. A pre-handling plugin failure must preserve an unconsumed handoff; a handled failure must not replay it.

### Validation Procedure

#### Automated Validation

From `apps/desktop`, after new SDK test creation:

```bash
npm run test:ui -- src/sdk/wake.test.ts src/contrib/plugin.test.ts src/contrib/runtime-loader.test.ts src/app/contrib/surfaces.app-routes.test.tsx src/store/wake-word.test.ts src/store/wake-word-client-capture.test.ts
npm run test:ui -- src/lib/wake-audio-handoff.test.ts src/app/chat/composer/hooks/use-voice-conversation
npm run typecheck
npm run lint
```

Before review run full `npm run test:ui` and `npm run build`, plus the English website build for SDK documentation. Backend changes are not expected; if introduced, use the Phase 3 Python/contract gates. Do not claim the source-inspection portions of old tests constitute lifecycle E2E.

#### Manual / Environment Validation

Load the consumer as an ordinary runtime plugin on a disposable desktop. Open its page, speak a wake phrase, verify no new chat/navigation and one plugin turn; finish and wake again without toggling. Repeat with no claimant, hidden page, explicit decline, thrown handler, hung handler, plugin disable/reload, route change, profile change, backend disconnect, and wake-off. Confirm fallback is available when the page did not handle the event and that post-handling failures never duplicate the user's request. Exercise an older desktop through feature detection: plugin remains loadable and normal chat behavior remains available. Repeat real microphone/remote-backend handoff coverage where the SDK consumes retained audio.

### Edge Cases and Failure Modes

Mounted but hidden page; focus in another pane; route ID impersonation/mistake; duplicate registrations; late claim after timeout; promise rejection after submission; render boundary replacing a page; stale re-arm against a new gateway; two windows; consumer forgetting cleanup; releasing during a pending getUserMedia; first-utterance handoff consumed twice.

### Acceptance Criteria

- [ ] Maintainers and consumer agree API/lifecycle semantics before implementation.
- [ ] Only an eligible visible page can own the event; unowned events follow normal chat once.
- [ ] Unmount, unload, failure, timeout, profile/route change release ownership and audio resources.
- [ ] Re-arm respects original transport, current config, and other-surface ownership.
- [ ] Retained audio has an explicit single consumer; no duplicate mic or transcription path exists.
- [ ] Existing passive events and plugins remain compatible; older hosts are feature-detectable.
- [ ] Public docs and actual runtime consumer demonstrate repeated wake/finish/wake behavior.

### Pull-Request Boundary

PR 4 references #131522, depends on the reviewed internal lifecycle from PR 3, and adds only the public claim adapter plus integration. Ship independently after that base is ready. Do not bundle the external voice-console product. Preserve contributor credit when incorporating existing work.

### Definition of Done

API review is recorded, public types/docs are complete, deterministic arbitration/disposal tests and full desktop checks pass, the real plugin loader and microphone/backend flow demonstrate claim→handle→release→re-arm, and no-claim/error fallback is verified without duplicate delivery. An unreviewed SDK sketch is not complete implementation.

## Phase 5 — Add an Alphabetical Sorting Option for Pinned Conversations

### Objective

Provide an explicit **Manual / Alphabetical** pinned-conversation preference. Manual remains the default for existing users and preserves their stored ordering. Alphabetical mode derives display order without rewriting that ordering, reacts to renames/locale changes, and is consistent in sidebar and command palette.

### Verified Repository Evidence

| Existing file / symbol | Role / invariant |
|---|---|
| `apps/desktop/src/app/chat/sidebar/session-index.ts`: `buildSessionByAnyId`, `resolvePinnedSessions` | Local ordered IDs first, backend-only pins afterward; deduplication, lineage aliases, and pending-unpin fence are load-bearing. |
| `apps/desktop/src/store/layout.ts`: `$pinnedSessionIds`, `setPinnedSessionOrder`, `resetSidebarView`, `$sidebarViewCustomized` | Manual pin IDs are connection-scoped with `includeProfile: false`; recents ordering and general display preferences have separate state. Drag updates preserve omitted/filtered IDs. |
| `sidebar/index.tsx`: pinned resolution, `reorderPinned`, pinned `SessionsSection` | Owns resolved rows and enables drag when more than one pin is shown. |
| `sidebar/filter-menu.tsx`: `SidebarFilterMenu` | Existing preference UI with radio controls, nanostore subscriptions, reset, and localized text. |
| `app/command-palette/index.tsx`: `liveRows`, `pinnedSessions` | Overlays live title/pin updates onto fetched rows, then uses the same pin resolver; data window is not necessarily identical to the sidebar. |
| `apps/desktop/src/lib/chat-runtime.ts`: `sessionTitle` | Display title = trimmed explicit title, then preview, then `Untitled session`. Sort the displayed title, not raw nullable title. |
| `apps/desktop/src/i18n/context.tsx`: `useI18n`, `locale` | Reactive app locale; don't silently use host locale where the UI selected another language. |
| PR #131607 | New unconditional-sorting proposal overlaps this phase and removes manual drag. Its proposal conflicts with the requested two-mode contract. |

### Problem / Gap

Current manual ordering is intentional and tested. The requested new opt-in mode improves scanning of many pins. A global `.sort()` in the existing resolver or replacing stored IDs with alphabetical IDs would destroy the manual-order contract.

### Scope

#### In Scope

One display preference, shared pure ordering policy, sidebar/palette adoption, drag eligibility, live title/locale updates, persistence/reset, localization, and regression tests.

#### Out of Scope

Reordering Recents/Projects/cron/messaging lists, backend schema changes, changing pin identity or server pin sync, automatic alphabetical-only migration, and a separate sort implementation per UI.

### Dependencies and Preconditions

- **Coordination checkpoint:** discuss #131485 and #131607 with reporter/author/maintainers through an authorized step. Prefer contributing the optional-mode change to the existing PR or an agreed follow-up after it merges.
- Agree manual default, preference scope, and the existing-menu location. If upstream chooses alphabetical-only, document that conflict and seek an agreed follow-up; do not silently abandon this task's manual mode or open a duplicate.
- Refresh the implementation base after coordination and inspect its actual pin/drag code. Preserve author attribution if building on accepted work.

### Detailed Implementation Steps

- [ ] Capture current manual-order contracts using `session-index.test.ts`, `layout-pinned-order.test.ts`, and `layout-connection-scope.test.ts` before changing the display.
- [ ] Add a validated enum preference in `store/layout.ts`, proposed values `manual` and `alphabetical`, default `manual`. Prefer `persistentAtom` like the neighboring global sidebar display preferences: the mode is presentation, while the pin IDs remain connection-scoped. Confirm this product choice at coordination; do not copy IDs between scopes. Invalid/missing stored values fall back to manual.
- [ ] Include the preference in `resetSidebarView` and `$sidebarViewCustomized` without resetting or sorting `$pinnedSessionIds`. Resetting the view changes the mode, not the saved manual sequence.
- [ ] Add a small pure presentation-order helper beside `resolvePinnedSessions`, taking resolved rows, mode, and explicit UI locale. Manual returns the resolved order. Alphabetical returns a copy; never mutate input arrays or stored IDs.
- [ ] Compare `sessionTitle(row)` with `Intl.Collator(locale, { numeric: true, sensitivity: 'base' })`. Use deterministic nonlocalized ID comparison as the final tie-breaker; prefer durable lineage identity, then actual row ID, so compressed tips do not unnecessarily reshuffle equal titles. Do not fall back to timestamp/recency ties. Verify custom locale values are valid BCP-47 or use the existing normalized locale fallback.
- [ ] Apply the same helper after existing membership resolution in both sidebar and command palette. Add reactive dependencies for preference, locale, live title/preview, and pin membership. Preserve the palette's existing fetched-data window; require consistent relative ordering for common rows, not a new all-history query.
- [ ] Add a clearly labeled Pinned order radio group in `SidebarFilterMenu`, distinct from Recents ordering. Use current translation structures under `src/i18n/`; update all bundled locales and verify the locale registry. Do not hardcode Portuguese globally; test Portuguese via the selected UI locale.
- [ ] Gate pinned drag/reorder handlers on manual mode while retaining them. A stale drag that completes after switching to alphabetical must not rewrite the manual sequence. Keep unpin, rename, keyboard navigation, and other sections functional.
- [ ] Test Manual→Alphabetical→Manual round trips before/after rename, pin/unpin, reload, connection switch, and filtered/project views. Newly pinned items may append through existing manual persistence; alphabetical display must never save its computed permutation.
- [ ] Update `website/docs/user-guide/desktop.md` with both modes, persistence, and drag behavior. Integrate with #131607's final shape instead of duplicating its comparator/changes.

All proposed production paths above are E. **N** `apps/desktop/src/app/command-palette/pinned-order.test.tsx` is justified if the current palette has no rendered pin-order test (only unrelated highlight-watcher coverage was found). Prefer a narrow rendered regression with query/network boundaries mocked; do not extract a controller merely to test a pure comparator twice. Exact palette harness setup is D.

### Test Changes

- Extend `sidebar/session-index.test.ts`: both modes; natural numbered titles; Portuguese accents/case; equal titles; preview/untitled fallback; stable identity ties; unchanged arrays; preserve all existing alias/unpin/dedup tests.
- Extend `store/layout-pinned-order.test.ts`, `layout-sidebar-view.test.ts`, and `layout-connection-scope.test.ts`: enum fallback/persistence, reset, mode round trip, hidden pin preservation, independent A/B pin sequences.
- Extend `sidebar/filter-menu.test.tsx`: selected mode, reactive locale labels, keyboard interaction, reset.
- Extend `sidebar/chat-sidebar.integration.test.tsx` or the existing row/drag harness as appropriate: alphabetical disables reorder; manual re-enables it; rename changes display immediately and failed rename rollback restores it. Keep a stale-drag completion assertion at the actual persistence boundary.
- Add the N palette regression with live rename overlay and preference/locale changes. Compare common-row order with sidebar policy; avoid assertions that require identical pagination.

### Validation Procedure

#### Automated Validation

From `apps/desktop`:

```bash
npm run test:ui -- src/app/chat/sidebar/session-index.test.ts src/store/layout-pinned-order.test.ts src/store/layout-sidebar-view.test.ts src/store/layout-connection-scope.test.ts src/app/chat/sidebar/filter-menu.test.tsx
npm run test:ui -- src/app/chat/sidebar/chat-sidebar.integration.test.tsx src/app/command-palette/pinned-order.test.tsx
npm run typecheck
npm run lint
```

After targeted checks run full `npm run test:ui` and the documentation English build. Review translation key changes using the existing `i18n:keys` script only if regeneration is required, and include solely intended generated changes. No Python suite is needed for a frontend-only preference unless implementation actually changes Python.

#### Manual / Environment Validation

Create pins in an intentionally nonalphabetical order. Use `Árvore 2`, `arvore 10`, mixed case, equal titles, and untitled/preview rows in a Portuguese locale, then another supported locale. Switch modes, rename a session, reopen palette, refresh/reconnect, restart the app, and switch A→B→A connections. Manual order must return unchanged except for explicit pin/unpin/manual operations. Verify drag is unavailable in alphabetical mode, restored in manual mode, and changing Recents order does not alter pinned mode. Test keyboard access and a second window for persistent preference synchronization according to the chosen scope.

### Edge Cases and Failure Modes

Equal collation results; numeric substrings; empty titles; server-only pins; missing local IDs; compression aliases; pending unpin; rename rollback; invalid stored enum; app locale differing from OS locale; hidden pins in filtered views; palette pagination; preference change during drag; connection switch with same session IDs.

### Acceptance Criteria

- [ ] Both modes remain available; default/migration follows the agreed manual-preserving policy.
- [ ] Alphabetical comparison is locale-aware, natural, deterministic, and based on displayed titles.
- [ ] Sidebar and palette use one policy and react to rename/locale changes.
- [ ] Switching modes and restarting never overwrites the saved manual order.
- [ ] Drag is enabled only in manual mode, including late-drop protection.
- [ ] Existing pin identity, sync/unpin fences, filtering, and unrelated ordering remain valid.
- [ ] The contribution is coordinated with #131607 rather than a competing duplicate.

### Pull-Request Boundary

PR 5 is either an agreed addition to #131607 or a separately reviewed follow-up implementing the optional preference on its merged base. It is independent of the voice work. Identify this as a QoL feature, not a repair of intentional manual ordering. Include a short mode-switch/rename recording and preservation tests.

### Definition of Done

Coordination and default/scope decisions are recorded, both modes pass behavior/persistence/drag tests, desktop checks and documentation validation pass, real UI mode/rename/restart/connection rehearsals preserve manual order, and the resulting contribution is ready for review without duplicating upstream work.

## Cross-Phase Validation Matrix

| Contribution | Primary subsystem | Automated evidence | Required manual/integration evidence | External dependency | Key regression risk | Merge/coordination prerequisite |
|---|---|---|---|---|---|---|
| 1 | PM/test documentation | Existing shell/activation tests; English docs build | Explicit vs managed interpreter in isolated home; native Windows for Windows claim | Supported bootstrapped Python/PM, Bash, docs dependencies | Misstated activated-shell precedence | Updated upstream docs/runner check |
| 2 | Bot UI → SDK → gateway MCP | Base-failing component/routing cases, actual route boundary, real handler/temp-home suite | A→B→A same profile names; aliases; create flow; foreground switch; multiplex | Two disposable backends, controlled MCP endpoint | Wrong key destination or cached capability | Duplicate/OAuth refresh; reproduction |
| 3 | Client audio → wake → first voice utterance | Sample continuity/WAV/limits; lifecycle/owner tests; full UI | Real mic + remote STT; delayed startup; cancellation; repeated wake | Native mic, wake model, remote backend/STT | Lost/duplicate audio or unreleased mic | #131518 scope; internal lease design |
| 4 | Plugin SDK/route visibility/wake ownership | Registration/disposal/arbitration plus Phase 3 regression | Runtime plugin claim/fallback/re-arm; errors/unload/profile switch | Consumer and maintainer API agreement | Hidden plugin owns event; double handling | Reviewed API and Phase 3 base |
| 5 | Pin state/presentation/preferences | Resolver, stores, menu, sidebar/palette integration; full UI | Mode/rename/restart/locale/A→B→A | Reporter/PR author coordination | Manual order destroyed or views disagree | Resolve #131607 overlap and two-mode policy |

## Expected PR Sequence

| PR | Scope | Dependency / reviewer evidence |
|---|---|---|
| 1 | Developer documentation correction | Independent; selection matrix and isolated command receipts |
| 2 | Bot Mode MCP API-key backend-routing fix | Independent; base failure, wire target, dummy-key file/endpoint receipts |
| 3 | Wake-phrase audio handoff | Independent from public SDK; sample accounting, bounded memory, real audio proof |
| 4 | Plugin wake-event claim/release/re-arm SDK capability | Reviewed PR 3 lifecycle; API discussion link, real consumer, fallback/cleanup proof |
| 5 | Pinned-session alphabetical sorting preference | Coordinate with #131607; may be a follow-up, not a fifth competing PR |

PRs 3 and 4 remain separate. Rebase/merge operations are future authorized implementation work, not part of this planning task. Refresh each proposed diff against its actual base and preserve contributor authorship if adopting external commits. Do not require an oversized combined voice PR merely because both touch a dispatcher.

## Issue / PR Coordination Matrix

| Topic | Known reference | Required next action | Work allowed before coordination |
|---|---|---|---|
| Test docs | No exact PR found | Refresh search; optionally identify a maintainer preferred wording | Read-only reproduction and documentation draft |
| Bot API-key routing | Adjacent #120549 | Verify API-key path still fails; avoid OAuth duplication | Isolated base reproduction and failing regression |
| Post-wake audio | #131518 | Confirm client-capture scope, tested latency/retention limits, consumer test access | Audio lifecycle experiments in disposable environments |
| Plugin wake claims | #131522 | Agree scoped SDK, visibility, capture disposition, failure/fallback/timeout | Design sketch and consumer requirements; no public API implementation yet |
| Optional alphabetical pins | #131485, #131607 | Coordinate ownership and preserve manual mode via existing PR or accepted follow-up | Inspect/review existing proposal; no competing implementation |

No coordination listed here has been performed. A lack of issue comments is not maintainer approval.

## Repository Areas Expected to Change

Every entry marked E was verified to exist. N paths are proposed new files justified above. D placement requires the stated design decision; do not interpret it as a discovered file.

| Phase | Classification | Expected surfaces |
|---|---|---|
| 1 | E | `CONTRIBUTING.md`, `AGENTS.md`, `website/docs/developer-guide/contributing.md`, `website/docs/reference/package-management.md` |
| 2 | E | `apps/desktop/src/plugins/hermes-bots/mcp-setup.tsx`, `profile-config.tsx`, `create-dialog.tsx`, possibly `routing.ts` and `types.ts` in the same directory; existing routing and profile tests |
| 2 | N / D | N Bot `mcp-setup.test.tsx`; D small target helper only if existing routing cannot own it cleanly |
| 3 | E | `apps/desktop/src/lib/wake-client-capture.ts`, `store/wake-word.ts`, `store/composer.ts`, `app/contrib/wiring.tsx`, composer voice/recorder hooks and listed tests; `website/docs/user-guide/desktop.md` |
| 3 | N / D | N `apps/desktop/src/lib/wake-audio-handoff.ts` and `.test.ts`; D owner-offer store sibling if needed. Backend contract changes are conditional, not assumed. |
| 4 | E | `apps/desktop/src/contrib/plugin.ts`, `sdk/index.ts`, `app/contrib/surfaces.tsx`, `app/contrib/wiring.tsx`, listed plugin/runtime tests; `website/docs/developer-guide/desktop-plugin-sdk.md` |
| 4 | N / D | N `apps/desktop/src/sdk/wake.ts` and `.test.ts`; D minimal visible-page activity helper/test and audio adapter placement after API review |
| 5 | E | `apps/desktop/src/store/layout.ts`, `app/chat/sidebar/session-index.ts`, `index.tsx`, `filter-menu.tsx`, `app/command-palette/index.tsx`, existing listed tests, bundled `src/i18n/` locale files, `website/docs/user-guide/desktop.md` |
| 5 | N / D | N palette `pinned-order.test.tsx`; D exact harness if existing integration can cover it without a new file |

Directories in this table are grouping shorthand, not authorization to edit every file. No package/dependency change is currently justified. If a design requires one, use existing bounded/pinned dependency and PM policies and explain why the installed primitives are insufficient.

## Final Completion Checklist

- [ ] Refresh upstream safely and record exact base before each contribution begins; preserve unrelated work.
- [ ] Repeat all-state duplicate searches and read issue/PR ownership/design comments.
- [ ] Reproduce documentation-selection/routing/audio defects on the applicable base before fixing; retain failing evidence where practical.
- [ ] Keep each contribution narrow; exclude the four existing-work suggestions outside this plan.
- [ ] Validate Python changes through `scripts/run_tests.sh`, with temporary homes and native OS markers where relevant.
- [ ] Validate desktop changes through existing Vitest scripts, typecheck/lint, and the indicated broader UI suite.
- [ ] Verify actual two-backend and multiplexed profile routing, including A→B→A with identical profile names.
- [ ] Complete real microphone plus remote-backend/STT validation for post-wake speech, including loss/duplication and cleanup evidence.
- [ ] Review public wake SDK before implementation; verify runtime plugin unload/fallback/re-arm and feature detection.
- [ ] Coordinate pinned sorting ownership with #131485/#131607 and preserve manual ordering.
- [ ] Document each contribution's behavior, scope, compatibility, test results, and any unverified gates.
- [ ] Give each contribution appropriate regression evidence and independent upstream-review readiness; do not equate code written with completion.
- [ ] Review final diffs for unrelated generated files, secrets, stale docs, and accidental changes to prompt/cache invariants.

## Open Questions / External Blockers

1. **Upstream movement:** live main was already 54 commits ahead. Revalidate before implementation; these coordinates are a fixed snapshot, not a promise about current main.
2. **Phase 2 runtime receipt:** static routing mismatch is established, but no real desktop reproduction occurred here. Confirm dispatch behavior, alias mapping, and New Bot flow before final fix shape.
3. **Phase 3 bounds and audio quality:** proposed five-second retention and ten-second unconsumed deadline need native measurement. Verify PCM WAV acceptance on actual configured STT routes and first-utterance VAD without assuming raw PCM and current meter levels are interchangeable.
4. **Detection alignment:** current wake events carry no sample cursor or exact phrase-end timing. Conservative retained lead-in is the planned compatibility-preserving choice. Exact wake-phrase trimming would require separate evidence and potentially a negotiated wire change.
5. **Audio infrastructure:** microphone permission, a remote backend, wake model, and test STT access are required to complete Phase 3. Their availability was not exercised during planning.
6. **Phase 4 API agreement:** no maintainer decision exists in the inspected issue. Context-scoped registration, visibility authority, first-utterance delivery vs explicit discard, handling-commit semantics, and cleanup acknowledgment need agreement before public code.
7. **Phase 5 conflict:** #131607 proposes alphabetical-only behavior. This plan requires both modes. Resolve through collaboration or an accepted follow-up; do not claim that PR already satisfies the requested feature.
8. **Platform evidence:** native Windows documentation rehearsal and additional audio hosts remain To verify during implementation. Static tests and fake browser audio do not establish native device behavior.

Planning completion means this document is saved and structurally checked. None of the implementation acceptance boxes above was marked passed during this planning task.
