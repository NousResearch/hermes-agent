# Desktop Files #104199 delivery recipe

This directory is the public delivery representation for the Desktop Files owner. It keeps the #104199 source branch owner-only while making the reviewed dependency composition reproducible.

## Ownership and pins

The recipe composes these immutable inputs in this order:

1. runtime / #106742: `485d5f6848c2598ca00fa7704e50e8ae66d0983a`;
2. Desktop continuity plus the normal-forward #97846 route chain:
   - continuity: `68f92bbf8e4b3cc6b9eff5f5dff2c28704958b48`;
   - route parent: `0c19759cd268170214535da5079c602c82fb1159`;
   - published route tip: `e2bfdfa8b8133d39f0380cb016397dedbe5b0e79`;
   - monotonic classic-log owner candidate: `8f0cec384e30a3cef5da31c2e64191c9e7ecf8bd`;
   - cancellation owner: `937b08778122ca5e250adc8072e4cb20209d8303`;
3. the exact #97846-owned `desktop-cancellation-dependent-overlay.patch`, applied to lower substrate `e76b349554ed86a02f2470c31d3912fcd9b5b332` **before Files**;
4. the direct 32-path #104199 owner source at `101128c012a8b16c4bfd003d5ead31aaa9830262`;
5. `desktop-files-dependent-overlay.patch`, the four Files-owned consumer insertions into paths supplied by the lower owners.

Both the monotonic `appendGroupChatEntry` correction and the cancellation correction live in #97846, not in the Files owner source. The lower merge map in `compose.sh` selects exact line slices from the fetched runtime and monotonic #97846 commit. Its 46 literal lines remain only reviewed cross-owner signature/UI adaptations. The cancellation package adds only a README plus a contextual six-path patch with SHA-256 `a25a14f89aa0961b38318ccd5f08d2237d329805eb75f42cafed7bf6900d775f`; the recipe verifies that two-path package scope and consumes it lower-first. No runtime-owned whole source body is copied into #97846 or embedded in this script. Cancellation has no semantic dependency on Files.

The Files owner source is generated directly with:

```bash
git diff --binary \
  485d5f6848c2598ca00fa7704e50e8ae66d0983a \
  101128c012a8b16c4bfd003d5ead31aaa9830262
```

The recipe requires that diff to have 32 paths and SHA-256 `ebac33db6267df28a689cc8b6e37bd2a05a262dd7296978b35656f433adb7a20`, applies it exactly once, and only then applies the four-path dependent overlay. The contextual overlay has SHA-256 `1e0e24264574c65a8500086c6f92d55a16893d7de4cedd289e1884e375341dfd`; its ordered added/deleted lines preserve the accepted Files overlay. Both dependent patches use context, not `--unidiff-zero`, so lower-first application does not misplace test hunks. Every stage requires its exact staged paths and tree.

## Run from public refs

After the existing #97846 and #104199 refs contain the named pins:

```bash
apps/desktop/scripts/delivery/104199/compose.sh /tmp/hermes-desktop-files-104199
```

The script creates a new disposable repository and never pushes. Its public defaults are:

- runtime: `NousResearch/hermes-agent`, `feat/unified-gateway-runtime`;
- lower owner: `dokterdok/hermes-agent`, `feat/bot-mode-desktop-continuity-20260829`;
- Files owner: `dokterdok/hermes-agent`, `feat/desktop-group-files-20260905`.

Each moving ref is fetched once. Every immutable pin must exist and be reachable from that fetched tip. Before publication, reviewers may exercise the exact local candidates without changing recipe bytes:

```bash
DESKTOP_FILES_VERIFY=compose \
DESKTOP_FILES_RUNTIME_REMOTE=/path/to/runtime-repository \
DESKTOP_FILES_RUNTIME_REF=refs/heads/runtime-branch \
DESKTOP_FILES_LOWER_REMOTE=/path/to/lower-repository \
DESKTOP_FILES_LOWER_REF=refs/heads/local/pr-97846-lower-ordering-owner \
DESKTOP_FILES_OWNER_REMOTE=/path/to/owner-repository \
DESKTOP_FILES_OWNER_REF=refs/heads/local/pr-104199-final-delivery \
apps/desktop/scripts/delivery/104199/compose.sh /tmp/hermes-desktop-files-104199
```

Verification modes:

- `compose`: retrieval, pinning, lower conflict resolution, cancellation lower-first, Files overlays, and exact path/hash/tree checks; no suite replay.
- `lower`: stop before Files, install dependencies, then run all three TypeScript projects, six-path ESLint and the focused causal regressions on the lower-only state.
- `focused` (default): compose all stages, install dependencies, run the same changed-path checks on the assembled product, then build renderer/Electron assets.
- `full`: additionally reproduce the precise inherited availability-test failure below; this is not an all-tests-green mode.

Neither the unchanged 228-test lower selection nor the 36-test Files selection is repeated.

## Exact trees and verification

The composition asserts these trees:

- lower substrate with the #97846 regression: `e76b349554ed86a02f2470c31d3912fcd9b5b332`;
- corrected lower-only product: `74868e3bd713445e192eb19efa298785c63fc616`;
- after the direct 32-path owner delta: `c8b4bc45f1bd439594d399d7cfcdc3b249ad1c76`;
- final Files product: `a9f73b2ced6d620803adf701dfa6938b8c7db39a`.

Earlier candidates were rejected because existing-message recovery bypassed active queue ownership, renamed commands retained stale names, and recovery could release its fence while queued behind a cancelled predecessor. Recovery now uses the same queue as fresh sends; settlement observes its exact active/pending ownership rather than only the room's `running` flag. Pending work is removed before fence release, expired work is discarded before activation, and teardown respects the current authority and epoch. Stop retires its active/pending command fences rather than allowing a delayed retry. Commands bind stable identity while resolving the current room name for validity, cancellation and settlement. The new tests first reproduced the failures, then passed with exact interruption, queue-survival and subsequent-usability assertions.

The corrected lower-only state passed 19 selected causal tests (99 excluded by name filter), all three TypeScript projects and changed-source ESLint with zero warnings. The actual executable recipe then reproduced the final tree above and passed its own 19 selected tests, all three TypeScript projects, six-path zero-warning ESLint, renderer/Electron build and `assert-dist-built`. These are separate lower-only and assembled-product results over the same selected cases, not additive unique coverage. The recipe was exercised using exact local candidate refs before publication; public defaults must retrieve those same immutable pins. No Files/SDK source behavior was changed.

Focused and full modes run the exact setup and build path:

```bash
npm ci --ignore-scripts --no-audit --no-fund
cd apps/desktop
npm run typecheck
# focused eslint and vitest selections are listed verbatim in compose.sh
npm run build
```

Only full mode also runs `group-availability.test.tsx` separately and requires the inherited exact-pin status to stay visible: 11 pass, 1 fail, status 1. The failing case is `retains classic availability in the chat header`; the fixture expects `2 of 4 available` but the accepted #97846 parent requires durable authority and renders `Group driver unavailable. Update or reconnect the owning gateway.` Status 1 alone is not accepted: the recipe validates the exact 11/1 count, test name, expected text, and actual text before continuing. This known failure was not rerun during the bounded cancellation correction; focused mode reports `DEPENDENCY_RENDER_TEST_STATUS=not-run`, not a repaired or total-suite-green claim.

## Four dependent paths

The overlay contains only:

- `apps/desktop/src/plugins/hermes-bots/group-chat-view.render.test.tsx`;
- `apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx`;
- `apps/desktop/src/plugins/hermes-bots/hosted-room-runtime.ts`;
- `apps/desktop/src/plugins/hermes-bots/types.ts`.

It is intentionally a dependency-applied artifact rather than source copied into the bare Files branch. The recipe and this documentation are delivery metadata; they are not included in the composed product tree.

## #97846 cancellation paths

The correction overlay contains only:

- `apps/desktop/src/plugins/hermes-bots/desktop-room-command-runtime.ts`;
- `apps/desktop/src/plugins/hermes-bots/desktop-room-mailbox-integration.test.ts`;
- `apps/desktop/src/plugins/hermes-bots/group-chat-view.render.test.tsx`;
- `apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx`;
- `apps/desktop/src/plugins/hermes-bots/group-rounds.test.ts`;
- `apps/desktop/src/plugins/hermes-bots/group-rounds.ts`.

It is stored in the #97846 owner tip because that branch owns the corrected behavior. The recipe consumes it on the lower-only substrate before either Files delta, so the Files script contains no literal cancellation implementation.

## Limits

- #104199 is an owner-only dependent branch, not a standalone green branch.
- The default public invocation cannot succeed until the parent publishes and reads back the new normal-forward #97846 pin and the replacement #104199 candidate on their existing refs.
- No new PR or branch is required for the lower stack, and this recipe performs no public write, merge to main, deployment, restart, or native action.
- Backend #104198 is an independent already-published input and is neither copied nor retested here.
