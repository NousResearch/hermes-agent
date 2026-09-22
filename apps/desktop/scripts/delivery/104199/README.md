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
3. the direct 32-path #104199 owner source at `101128c012a8b16c4bfd003d5ead31aaa9830262`;
4. `desktop-files-dependent-overlay.patch`, the four dependency-owned consumer insertions needed after the lower owners are present.

The monotonic `appendGroupChatEntry` correction and its repeated-clock regression live in #97846, not in this patch or the Files owner source. The lower merge map in `compose.sh` selects exact line slices from the fetched runtime and #97846 commits. Its 46 literal lines are only reviewed cross-owner signature/UI adaptations. It replaces the former 86 KB 13-path semantic patch and carries no copy of the monotonic implementation.

The Files owner source is generated directly with:

```bash
git diff --binary \
  485d5f6848c2598ca00fa7704e50e8ae66d0983a \
  101128c012a8b16c4bfd003d5ead31aaa9830262
```

The recipe requires that diff to have 32 paths and SHA-256 `ebac33db6267df28a689cc8b6e37bd2a05a262dd7296978b35656f433adb7a20`, applies it exactly once, and only then applies the four-path dependent overlay. The zero-context overlay has SHA-256 `6584161cab98f38fbb21921c1ce8ea64ec201d52c618d27e0f56e522b3033879`; its ordered added/deleted lines equal the reviewed overlay `05ef49ab1f58d3b70ab52dc338dfd2004be8237678030974cc43304c12bf5126` exactly, with hunk positions regenerated for the final lower substrate. `compose.sh` uses Git's explicit `--unidiff-zero` mode and still requires the exact four staged paths and final tree.

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

`DESKTOP_FILES_VERIFY=compose` exercises retrieval, pinning, lower conflict resolution, the direct owner delta, the four-path overlay, and every exact-tree assertion without repeating unchanged suites. The default, `DESKTOP_FILES_VERIFY=full`, additionally executes setup, typecheck, focused lint, impacted lower tests, Files tests, and the renderer/Electron build.

## Exact trees and verification

The composition asserts these trees:

- lower substrate with the #97846 regression: `e76b349554ed86a02f2470c31d3912fcd9b5b332`;
- after the direct 32-path owner delta: `994ef94a3afae0adc78259dce6a7a11c0c615157`;
- final product: `19b08e1c6be010cbc05716bc24bb4de163313d71`.

The prior full-build-tested owner tree was `d07898f5b857a845b9dd5cb8c53c7b6bcc9a38c1`. The final product differs from it in exactly one file: `group-chat.test.ts` gains the 20-line #97846 repeated-clock regression. Every production blob is byte-identical to the tested tree, so the previous typecheck/lint, 228 lower tests, 36 Files tests, and complete renderer/Electron build remain applicable. The new focused regression was separately proven red before the lower fix and green after it.

Full mode runs the exact setup and build path:

```bash
npm ci --ignore-scripts --no-audit --no-fund
cd apps/desktop
../../node_modules/.bin/tsc -p . --noEmit
# focused eslint and vitest selections are listed verbatim in compose.sh
npm run build
```

It also runs `group-availability.test.tsx` separately and requires the inherited exact-pin status to stay visible: 11 pass, 1 fail, status 1. The failing case is `retains classic availability in the chat header`; the fixture expects `2 of 4 available` but the accepted #97846 parent requires durable authority and renders `Group driver unavailable. Update or reconnect the owning gateway.` The recipe does not exclude, rewrite, or relabel that result.

## Four dependent paths

The overlay contains only:

- `apps/desktop/src/plugins/hermes-bots/group-chat-view.render.test.tsx`;
- `apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx`;
- `apps/desktop/src/plugins/hermes-bots/hosted-room-runtime.ts`;
- `apps/desktop/src/plugins/hermes-bots/types.ts`.

It is intentionally a dependency-applied artifact rather than source copied into the bare Files branch. The recipe and this documentation are delivery metadata; they are not included in the composed product tree.

## Limits

- #104199 is an owner-only dependent branch, not a standalone green branch.
- The default public invocation cannot succeed until the parent publishes and reads back the new normal-forward #97846 pin and the replacement #104199 candidate on their existing refs.
- No new PR or branch is required for the lower stack, and this recipe performs no public write, merge to main, deployment, restart, or native action.
- Backend #104198 is an independent already-published input and is neither copied nor retested here.
