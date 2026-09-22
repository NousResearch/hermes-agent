#!/usr/bin/env bash
set -euo pipefail

# Public, disposable composition for Desktop Files #104199. The script fetches
# only existing owner refs, pins immutable commits, writes no remote, applies the
# #97846-owned six-path cancellation correction before Files, then applies the
# direct 32-path Files owner delta and its four dependent paths exactly once.
TARGET=${1:?usage: compose.sh TARGET_DIR}
VERIFY=${DESKTOP_FILES_VERIFY:-focused}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEPENDENT_OVERLAY="$SCRIPT_DIR/desktop-files-dependent-overlay.patch"
CANCELLATION_OVERLAY_PATH=apps/desktop/scripts/delivery/97846/desktop-cancellation-dependent-overlay.patch

RUNTIME_REMOTE=${DESKTOP_FILES_RUNTIME_REMOTE:-https://github.com/NousResearch/hermes-agent.git}
LOWER_REMOTE=${DESKTOP_FILES_LOWER_REMOTE:-https://github.com/dokterdok/hermes-agent.git}
OWNER_REMOTE=${DESKTOP_FILES_OWNER_REMOTE:-https://github.com/dokterdok/hermes-agent.git}
RUNTIME_REF=${DESKTOP_FILES_RUNTIME_REF:-refs/heads/feat/unified-gateway-runtime}
LOWER_REF=${DESKTOP_FILES_LOWER_REF:-refs/heads/feat/bot-mode-desktop-continuity-20260829}
OWNER_REF=${DESKTOP_FILES_OWNER_REF:-refs/heads/feat/desktop-group-files-20260905}

RUNTIME_SHA=485d5f6848c2598ca00fa7704e50e8ae66d0983a
CONTINUITY_SHA=68f92bbf8e4b3cc6b9eff5f5dff2c28704958b48
ROUTE_PARENT_SHA=0c19759cd268170214535da5079c602c82fb1159
ROUTE_SHA=e2bfdfa8b8133d39f0380cb016397dedbe5b0e79
LOWER_SHA=8f0cec384e30a3cef5da31c2e64191c9e7ecf8bd
CANCELLATION_OWNER_SHA=b0157bd84bd1a082966394907a35ac83c066614b
OWNER_SHA=101128c012a8b16c4bfd003d5ead31aaa9830262
DIRECT_OWNER_PATCH_SHA=ebac33db6267df28a689cc8b6e37bd2a05a262dd7296978b35656f433adb7a20
DEPENDENT_OVERLAY_SHA=1e0e24264574c65a8500086c6f92d55a16893d7de4cedd289e1884e375341dfd
CANCELLATION_OVERLAY_SHA=8ad9da6a90a4de55c07b8d7e15054aa67db3c98db3a107dd9374d771550cbb52
EXPECTED_LOWER_TREE=e76b349554ed86a02f2470c31d3912fcd9b5b332
EXPECTED_CORRECTED_LOWER_TREE=02ef3e6596c87239d6b1076d11b850022e7a1ab4
EXPECTED_DIRECT_OWNER_TREE=0c73eccdb65112117ecd5f6826ca33a298eea492
EXPECTED_FILES_TREE=40eaa689442d786e6a8679c98d75910669c193d4
EXPECTED_FINAL_TREE=40eaa689442d786e6a8679c98d75910669c193d4
TESTED_OWNER_TREE=40eaa689442d786e6a8679c98d75910669c193d4

case "$VERIFY" in
  compose|lower|focused|full) ;;
  *) printf 'DESKTOP_FILES_VERIFY must be compose, lower, focused or full, got %s\n' "$VERIFY" >&2; exit 2 ;;
esac

require_ref() {
  local expected=$1 ref=$2 actual
  actual=$(git rev-parse "$ref")
  test "$actual" = "$expected" || {
    printf 'pin mismatch: %s expected %s got %s\n' "$ref" "$expected" "$actual" >&2
    exit 2
  }
}

pin_reachable_commit() {
  local name=$1 expected=$2 fetched_tip=$3
  git cat-file -e "$expected^{commit}" 2>/dev/null || {
    printf 'fetched tip %s did not retrieve pin %s\n' "$fetched_tip" "$expected" >&2
    exit 2
  }
  git merge-base --is-ancestor "$expected" "$fetched_tip" || {
    printf 'pin %s is not reachable from fetched tip %s\n' "$expected" "$fetched_tip" >&2
    exit 2
  }
  git update-ref "refs/inputs/$name" "$expected"
  require_ref "$expected" "refs/inputs/$name"
}

require_paths() {
  local expected=$1 actual=$2 label=$3
  test "$actual" = "$expected" || {
    printf 'unexpected %s paths\n--- expected ---\n%s\n--- actual ---\n%s\n' "$label" "$expected" "$actual" >&2
    exit 3
  }
}

require_tree() {
  local expected=$1 label=$2 actual
  actual=$(git write-tree)
  test "$actual" = "$expected" || {
    printf '%s tree mismatch: expected %s got %s\n' "$label" "$expected" "$actual" >&2
    exit 4
  }
}

verify_cancellation() {
  npm run typecheck
  ../../node_modules/.bin/eslint \
    src/plugins/hermes-bots/desktop-room-command-runtime.ts \
    src/plugins/hermes-bots/desktop-room-mailbox-integration.test.ts \
    src/plugins/hermes-bots/group-chat-view.render.test.tsx \
    src/plugins/hermes-bots/group-chat-view.tsx \
    src/plugins/hermes-bots/group-rounds.test.ts \
    src/plugins/hermes-bots/group-rounds.ts --max-warnings=0
  ../../node_modules/.bin/vitest run --project ui \
    src/plugins/hermes-bots/desktop-room-mailbox-integration.test.ts \
    src/plugins/hermes-bots/group-rounds.test.ts \
    src/plugins/hermes-bots/group-chat-view.render.test.tsx \
    --maxWorkers=2 --retry=0 \
    -t 'interrupts recovered mailbox work|follows an in-flight mailbox rename|pending mailbox command|active mailbox thread|lease abort|replacement during Stop|delayed mailbox drive|post-turn commit|follows a rename|uses the active queue item|resolves session-scoped|follows queue identity|does not pass the latest display'
}

test ! -e "$TARGET"
test "$(sha256sum "$DEPENDENT_OVERLAY" | cut -d' ' -f1)" = "$DEPENDENT_OVERLAY_SHA"
mkdir -p "$TARGET"
git -C "$TARGET" init -q
git -C "$TARGET" config user.name 'Desktop Files integration recipe'
git -C "$TARGET" config user.email '5354424+dokterdok@users.noreply.github.com'
git -C "$TARGET" remote add runtime "$RUNTIME_REMOTE"
git -C "$TARGET" remote add lower "$LOWER_REMOTE"
git -C "$TARGET" remote add owner "$OWNER_REMOTE"
git -C "$TARGET" fetch --no-tags runtime "$RUNTIME_REF:refs/inputs/runtime-tip"
git -C "$TARGET" fetch --no-tags lower "$LOWER_REF:refs/inputs/lower-tip"
git -C "$TARGET" fetch --no-tags owner "$OWNER_REF:refs/inputs/owner-tip"
cd "$TARGET"

pin_reachable_commit runtime "$RUNTIME_SHA" refs/inputs/runtime-tip
pin_reachable_commit continuity "$CONTINUITY_SHA" refs/inputs/lower-tip
pin_reachable_commit route-parent "$ROUTE_PARENT_SHA" refs/inputs/lower-tip
pin_reachable_commit route "$ROUTE_SHA" refs/inputs/lower-tip
pin_reachable_commit lower "$LOWER_SHA" refs/inputs/lower-tip
pin_reachable_commit cancellation-owner "$CANCELLATION_OWNER_SHA" refs/inputs/lower-tip
pin_reachable_commit owner "$OWNER_SHA" refs/inputs/owner-tip

test "$(git rev-parse "$ROUTE_PARENT_SHA^")" = "$CONTINUITY_SHA"
test "$(git rev-parse "$ROUTE_SHA^")" = "$ROUTE_PARENT_SHA"
test "$(git rev-parse "$LOWER_SHA^")" = "$ROUTE_SHA"
test "$(git rev-parse "$CANCELLATION_OWNER_SHA^^")" = "$LOWER_SHA"
expected_cancellation_owner_paths=$(printf '%s\n' \
  apps/desktop/scripts/delivery/97846/README.md \
  "$CANCELLATION_OVERLAY_PATH" | LC_ALL=C sort)
cancellation_owner_paths=$(git diff --name-only "$LOWER_SHA..$CANCELLATION_OWNER_SHA" | LC_ALL=C sort)
require_paths "$expected_cancellation_owner_paths" "$cancellation_owner_paths" 'cancellation owner'
git merge-base --is-ancestor "$RUNTIME_SHA" "$OWNER_SHA"
test "$(git rev-list --first-parent --count "$RUNTIME_SHA..$OWNER_SHA")" -eq 5

git checkout -q -b integration/files-104199 refs/inputs/runtime
set +e
git merge --no-commit --no-ff refs/inputs/lower
merge_status=$?
set -e
test "$merge_status" -eq 1
expected_conflicts=$(printf '%s\n' \
  apps/desktop/src/plugins/hermes-bots/bot-row.tsx \
  apps/desktop/src/plugins/hermes-bots/create-dialog.tsx \
  apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx \
  apps/desktop/src/plugins/hermes-bots/group-chat.ts \
  apps/desktop/src/plugins/hermes-bots/group-membership.ts \
  apps/desktop/src/plugins/hermes-bots/group-round-members.ts \
  apps/desktop/src/plugins/hermes-bots/group-rounds.ts \
  apps/desktop/src/plugins/hermes-bots/group-turns.ts \
  apps/desktop/src/plugins/hermes-bots/plugin-panes.test.tsx \
  apps/desktop/src/store/gateway.ts | LC_ALL=C sort)
actual_conflicts=$(git diff --name-only --diff-filter=U | LC_ALL=C sort)
require_paths "$expected_conflicts" "$actual_conflicts" 'lower merge conflict'

# Materialize the reviewed lower merge postimages from exact line slices of the
# two fetched public owners. Only 46 literal lines are integration adaptations;
# the monotonic timestamp behavior is not among them and comes from LOWER_SHA.
python3 - <<'PY'
from pathlib import Path
import subprocess

MANIFEST = {'apps/desktop/src/plugins/hermes-bots/bot-row.tsx': [['runtime', 0, 56], ['lower', 56, 59], ['runtime', 58, 495], ['lower', 475, 487], ['literal', '      ? b.group.you\n'], ['lower', 488, 490], ['literal', "  const preview = last ? `${speaker}: ${stripPreviewMarkdown(last.text) || '…'}` : b.group.memberCount(members.length)\n"], ['runtime', 504, 507], ['lower', 494, 505], ['runtime', 508, 515], ['literal', '      aria-label={[group, b.group.memberCount(members.length), hosted ? hostWarning : availabilityLabel]\n'], ['lower', 508, 513], ['runtime', 519, 540], ['lower', 526, 554], ['runtime', 568, 652]], 'apps/desktop/src/plugins/hermes-bots/create-dialog.tsx': [['lower', 0, 60], ['runtime', 53, 61], ['lower', 67, 75], ['runtime', 62, 1036], ['lower', 1045, 1062], ['runtime', 1041, 1068], ['lower', 1089, 1136], ['runtime', 1107, 1109], ['literal', '                for (const group of removableCurrent) {\n'], ['runtime', 1110, 1117], ['lower', 1144, 1601]], 'apps/desktop/src/plugins/hermes-bots/group-availability.test.tsx': [['lower', 0, 45], ['literal', '  return (\n    <GroupRow\n      active={false}\n      group={GROUP}\n      members={selected}\n      needsYou={false}\n      onDisband={noop}\n      onNewSection={noop}\n      onOpen={noop}\n    />\n  )\n'], ['lower', 46, 172]], 'apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx': [['lower', 0, 83], ['runtime', 79, 80], ['lower', 83, 121], ['runtime', 104, 132], ['lower', 142, 460], ['runtime', 371, 383], ['lower', 471, 563], ['runtime', 446, 464], ['lower', 567, 756], ['runtime', 613, 700], ['lower', 842, 953], ['runtime', 787, 850], ['lower', 978, 1032], ['literal', '          ) : summaryActivity ? (\n'], ['runtime', 875, 878], ['lower', 1036, 1329], ['literal', '        ? b.group.you\n'], ['lower', 1330, 1344], ['runtime', 1069, 1070], ['lower', 1345, 1401], ['runtime', 1139, 1157], ['lower', 1413, 1549], ['runtime', 1281, 1303], ['lower', 1564, 1608], ['runtime', 1344, 1376], ['lower', 1638, 1669], ['runtime', 1397, 1426], ['lower', 1697, 1711], ['runtime', 1439, 1467]], 'apps/desktop/src/plugins/hermes-bots/group-chat.ts': [['lower', 0, 68], ['runtime', 42, 51], ['lower', 77, 1027], ['runtime', 757, 776], ['lower', 1044, 1678], ['runtime', 1360, 1380], ['lower', 1696, 2005]], 'apps/desktop/src/plugins/hermes-bots/group-membership.ts': [['runtime', 0, 7], ['lower', 7, 63], ['runtime', 59, 276], ['lower', 244, 436], ['runtime', 390, 391], ['lower', 437, 489]], 'apps/desktop/src/plugins/hermes-bots/group-round-members.ts': [['lower', 0, 26], ['runtime', 25, 26], ['lower', 26, 134], ['runtime', 132, 160], ['lower', 154, 170], ['runtime', 176, 191], ['lower', 184, 193], ['runtime', 200, 303]], 'apps/desktop/src/plugins/hermes-bots/group-rounds.ts': [['runtime', 0, 8], ['lower', 6, 16], ['runtime', 16, 19], ['lower', 18, 30], ['runtime', 22, 32], ['lower', 34, 79], ['runtime', 60, 363], ['lower', 383, 465], ['runtime', 365, 432], ['lower', 523, 559], ['runtime', 433, 437], ['lower', 563, 567], ['literal', '  fence?: GroupCommandFence,\n  failedMembers = new Set<string>()\n'], ['lower', 568, 611], ['runtime', 454, 456], ['lower', 612, 782], ['runtime', 616, 672], ['lower', 837, 917], ['runtime', 674, 682], ['lower', 925, 956], ['runtime', 693, 710], ['lower', 960, 1050], ['runtime', 718, 719], ['lower', 1054, 1077], ['literal', '  queueGroupChatDrive(group, members, target, fence)\n'], ['runtime', 761, 767], ['literal', '  pending: Map<string, { fence?: GroupCommandFence; members: GroupMember[] }>\n'], ['runtime', 768, 775], ['literal', 'function queueGroupChatDrive(group: string, members: GroupMember[], thread: string, fence?: GroupCommandFence) {\n'], ['runtime', 776, 782], ['literal', '    bindGroupCommandFence(fence, thread, $groupChats.get()[group]?.epoch || 0)\n    active.pending.set(thread, { fence, members })\n'], ['runtime', 783, 794], ['literal', '  const drive: GroupChatDrive = {\n    pending: new Map([[thread, { fence, members }]]),\n    failedMembers: new Set(),\n    binding\n'], ['runtime', 87, 89], ['runtime', 795, 799], ['literal', '  bindGroupCommandFence(fence, thread, $groupChats.get()[group]?.epoch || 0)\n'], ['runtime', 799, 805], ['literal', '        const [nextThread, next] = drive.pending.entries().next().value!\n'], ['runtime', 806, 809], ['literal', '        await runGroupChatRounds(group, next.members, nextThread, next.fence, drive.failedMembers)\n'], ['runtime', 810, 825]], 'apps/desktop/src/plugins/hermes-bots/group-speaker-display.test.tsx': [['lower', 0, 67], ['literal', '      <GroupRow\n        active\n        group="Board"\n        members={members}\n        needsYou={false}\n        onDisband={vi.fn()}\n        onNewSection={vi.fn()}\n        onOpen={vi.fn()}\n      />\n'], ['lower', 68, 293]], 'apps/desktop/src/plugins/hermes-bots/group-turns.test.ts': [['lower', 0, 65], ['literal', "    await room.turns.ensureGroupChatSession('Alpha', member, 'legacy')\n"], ['lower', 66, 99], ['literal', "      expect(await room.turns.ensureGroupChatSession('Alpha', member, 'legacy', fence)).toEqual({ runtime: null })\n"], ['lower', 100, 112], ['runtime', 68, 1151]], 'apps/desktop/src/plugins/hermes-bots/group-turns.ts': [['lower', 0, 16], ['runtime', 13, 20], ['lower', 17, 107], ['runtime', 109, 143], ['runtime', 483, 484], ['lower', 131, 151], ['runtime', 154, 184], ['lower', 168, 195], ['runtime', 207, 244], ['lower', 228, 458], ['runtime', 469, 667], ['lower', 636, 652], ['runtime', 677, 722], ['lower', 697, 862], ['runtime', 877, 910], ['lower', 895, 908], ['runtime', 923, 924], ['lower', 909, 976], ['literal', '    const { runtime, stored } = await ensureGroupChatSession(group, member, thread, fence)\n'], ['lower', 977, 1066], ['runtime', 1058, 1136]], 'apps/desktop/src/plugins/hermes-bots/plugin-panes.test.tsx': [['lower', 0, 39], ['runtime', 30, 31], ['literal', '  undismissPane: vi.fn(),\n'], ['lower', 39, 41], ['literal', '  stopDesktopRoomCommandRuntime: vi.fn()\n'], ['lower', 43, 54], ['runtime', 43, 53], ['lower', 63, 491], ['runtime', 251, 322]], 'apps/desktop/src/store/gateway.ts': [['runtime', 0, 36], ['lower', 28, 47], ['runtime', 46, 104], ['lower', 100, 278], ['runtime', 264, 280], ['lower', 281, 826], ['runtime', 784, 791], ['lower', 832, 849], ['runtime', 807, 1016], ['lower', 1045, 1403], ['runtime', 1190, 1210], ['lower', 1416, 1711], ['runtime', 474, 475], ['lower', 1654, 1658], ['runtime', 1498, 1500], ['literal', '    return Boolean(isOpen(attached.gateway) && !signal?.aborted && applyActive(g.primaryProfile, activationEpoch))\n'], ['runtime', 1501, 1943], ['lower', 2159, 2201]]}
EXPECTED_BLOBS = {'apps/desktop/src/plugins/hermes-bots/bot-row.tsx': 'ecbc2b6445413ac49076140bf3b83f6823140579', 'apps/desktop/src/plugins/hermes-bots/create-dialog.tsx': '0ec7febf97e7fb64018c9e2aaf2f074d3ceb4a23', 'apps/desktop/src/plugins/hermes-bots/group-availability.test.tsx': '2f3ccde4daff49b1deb4bfe0eda931de798fc377', 'apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx': '8183b4dbae1354e6e830833ea706a08c6c916cb2', 'apps/desktop/src/plugins/hermes-bots/group-chat.ts': '791b07ce78ee77deb78e5cfb897da5b3ca402494', 'apps/desktop/src/plugins/hermes-bots/group-membership.ts': '85a014e2fda12399a0d310cb908ade4ad3a18ea6', 'apps/desktop/src/plugins/hermes-bots/group-round-members.ts': '38286bb841678c4fcd288dd94f894297f1bfd59c', 'apps/desktop/src/plugins/hermes-bots/group-rounds.ts': '3ebf668934d58c9fcfb624e9229bfbfa7772e7ba', 'apps/desktop/src/plugins/hermes-bots/group-speaker-display.test.tsx': '9b1e5d4e2ce510c9d403cd28febe248032e0980e', 'apps/desktop/src/plugins/hermes-bots/group-turns.test.ts': '991a26be55d74dab8eea6592d9355d07700001c3', 'apps/desktop/src/plugins/hermes-bots/group-turns.ts': 'e630a52fd81dda610cac28d4dc68319987d537f8', 'apps/desktop/src/plugins/hermes-bots/plugin-panes.test.tsx': '3c6b5e1e3dd6972b425c9c400a9265dd1b895f59', 'apps/desktop/src/store/gateway.ts': '9faa15596fc213ee2868dda23b7f025f622ad1ad'}

def source_lines(ref: str, path: str) -> list[bytes]:
    return subprocess.check_output(['git', 'show', f'{ref}:{path}']).splitlines(keepends=True)

for path, operations in MANIFEST.items():
    sources: dict[str, list[bytes]] = {}
    output: list[bytes] = []
    for operation in operations:
        source = operation[0]
        if source == 'literal':
            output.append(operation[1].encode('utf-8'))
            continue
        if source not in sources:
            ref = 'refs/inputs/runtime' if source == 'runtime' else 'refs/inputs/lower'
            sources[source] = source_lines(ref, path)
        output.extend(sources[source][operation[1]:operation[2]])
    data = b''.join(output)
    blob = subprocess.check_output(['git', 'hash-object', '--stdin'], input=data).decode().strip()
    expected = EXPECTED_BLOBS[path]
    if blob != expected:
        raise SystemExit(f'{path}: expected composed blob {expected}, got {blob}')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
PY

semantic_paths=(
  apps/desktop/src/plugins/hermes-bots/bot-row.tsx
  apps/desktop/src/plugins/hermes-bots/create-dialog.tsx
  apps/desktop/src/plugins/hermes-bots/group-availability.test.tsx
  apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx
  apps/desktop/src/plugins/hermes-bots/group-chat.ts
  apps/desktop/src/plugins/hermes-bots/group-membership.ts
  apps/desktop/src/plugins/hermes-bots/group-round-members.ts
  apps/desktop/src/plugins/hermes-bots/group-rounds.ts
  apps/desktop/src/plugins/hermes-bots/group-speaker-display.test.tsx
  apps/desktop/src/plugins/hermes-bots/group-turns.test.ts
  apps/desktop/src/plugins/hermes-bots/group-turns.ts
  apps/desktop/src/plugins/hermes-bots/plugin-panes.test.tsx
  apps/desktop/src/store/gateway.ts
)
git add "${semantic_paths[@]}"
test -z "$(git diff --name-only --diff-filter=U)"
git diff --cached --check
git commit -q -m 'integration: compose runtime and lower owner pins'
LOWER_HEAD=$(git rev-parse HEAD)
require_tree "$EXPECTED_LOWER_TREE" 'lower substrate'

CANCELLATION_OVERLAY="$PWD/.git/desktop-cancellation-dependent-overlay.patch"
git show "refs/inputs/cancellation-owner:$CANCELLATION_OVERLAY_PATH" >"$CANCELLATION_OVERLAY"
test "$(sha256sum "$CANCELLATION_OVERLAY" | cut -d' ' -f1)" = "$CANCELLATION_OVERLAY_SHA"
expected_cancellation_paths=$(printf '%s\n' \
  apps/desktop/src/plugins/hermes-bots/desktop-room-command-runtime.ts \
  apps/desktop/src/plugins/hermes-bots/desktop-room-mailbox-integration.test.ts \
  apps/desktop/src/plugins/hermes-bots/group-chat-view.render.test.tsx \
  apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx \
  apps/desktop/src/plugins/hermes-bots/group-rounds.test.ts \
  apps/desktop/src/plugins/hermes-bots/group-rounds.ts | LC_ALL=C sort)
git apply --index "$CANCELLATION_OVERLAY"
actual_cancellation_paths=$(git diff --cached --name-only | LC_ALL=C sort)
require_paths "$expected_cancellation_paths" "$actual_cancellation_paths" 'cancellation correction'
git diff --cached --check
git commit -q -m 'integration: apply #97846 cancellation correction before Files'
CORRECTED_LOWER_HEAD=$(git rev-parse HEAD)
require_tree "$EXPECTED_CORRECTED_LOWER_TREE" 'corrected lower-only product'

if test "$VERIFY" = lower; then
  HOME=${HOME:-/tmp} npm ci --ignore-scripts --no-audit --no-fund
  (cd apps/desktop && verify_cancellation)
  test -z "$(git status --porcelain)"
  printf 'CORRECTED_LOWER_HEAD=%s\nCORRECTED_LOWER_TREE=%s\nVERIFY=lower\n' \
    "$CORRECTED_LOWER_HEAD" "$EXPECTED_CORRECTED_LOWER_TREE"
  exit 0
fi

DIRECT_OWNER_PATCH="$TARGET/.git/desktop-files-owner.patch"
git diff --binary refs/inputs/runtime refs/inputs/owner -- >"$DIRECT_OWNER_PATCH"
test "$(sha256sum "$DIRECT_OWNER_PATCH" | cut -d' ' -f1)" = "$DIRECT_OWNER_PATCH_SHA"
owner_paths=$(git diff --name-only refs/inputs/runtime refs/inputs/owner | LC_ALL=C sort)
test "$(printf '%s\n' "$owner_paths" | wc -l)" -eq 32
git apply --3way --index "$DIRECT_OWNER_PATCH"
test -z "$(git diff --name-only --diff-filter=U)"
actual_owner_paths=$(git diff --cached --name-only | LC_ALL=C sort)
require_paths "$owner_paths" "$actual_owner_paths" 'direct owner'
git diff --cached --check
git commit -q -m 'integration: apply direct Desktop Files owner delta'
DIRECT_OWNER_HEAD=$(git rev-parse HEAD)
require_tree "$EXPECTED_DIRECT_OWNER_TREE" 'direct owner'

expected_overlay_paths=$(printf '%s\n' \
  apps/desktop/src/plugins/hermes-bots/group-chat-view.render.test.tsx \
  apps/desktop/src/plugins/hermes-bots/group-chat-view.tsx \
  apps/desktop/src/plugins/hermes-bots/hosted-room-runtime.ts \
  apps/desktop/src/plugins/hermes-bots/types.ts | LC_ALL=C sort)
git apply --index "$DEPENDENT_OVERLAY"
test -z "$(git diff --name-only --diff-filter=U)"
actual_overlay_paths=$(git diff --cached --name-only | LC_ALL=C sort)
require_paths "$expected_overlay_paths" "$actual_overlay_paths" 'dependent overlay'
git diff --cached --check
git commit -q -m 'integration: apply Files dependent consumer overlay'
FILES_HEAD=$(git rev-parse HEAD)
require_tree "$EXPECTED_FILES_TREE" 'Files product'

FINAL_HEAD=$(git rev-parse HEAD)
require_tree "$EXPECTED_FINAL_TREE" 'corrected final product'
test -z "$(git status --porcelain)"

DEPENDENCY_RENDER_TEST_STATUS=not-run
if test "$VERIFY" = focused || test "$VERIFY" = full; then
  HOME=${HOME:-/tmp} npm ci --ignore-scripts --no-audit --no-fund
  (
    cd apps/desktop
    verify_cancellation
    if test "$VERIFY" = full; then
    dependency_render_log="$TARGET/.git/dependency-render.log"
    set +e
    ../../node_modules/.bin/vitest run --project ui \
      src/plugins/hermes-bots/group-availability.test.tsx \
      --maxWorkers=2 --retry=0 >"$dependency_render_log" 2>&1
    dependency_render_status=$?
    set -e
    test "$dependency_render_status" -eq 1
    python3 - "$dependency_render_log" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
print(text, end='')
required = (
    'retains classic availability in the chat header',
    '2 of 4 available',
    'Group driver unavailable. Update or reconnect the owning gateway.',
)
if any(token not in text for token in required):
    raise SystemExit('group-availability failure did not match the inherited exact-pin fixture')
if not re.search(r'Tests\s+1 failed\s+\|\s+11 passed\s+\(12\)', text):
    raise SystemExit('group-availability result was not exactly 11 pass / 1 known fail')
PY
    printf 'DEPENDENCY_RENDER_TEST_STATUS=%s (exact inherited 11 pass / 1 named fail; not hidden)\n' "$dependency_render_status"
    printf '%s\n' "$dependency_render_status" >"$TARGET/.git/dependency-render-status"
    fi
    npm run build
  )
  if test "$VERIFY" = full; then
    DEPENDENCY_RENDER_TEST_STATUS=$(<"$TARGET/.git/dependency-render-status")
  fi
fi

test "$(git rev-parse HEAD^{tree})" = "$EXPECTED_FINAL_TREE"
git diff --check refs/inputs/runtime..HEAD
printf 'LOWER_HEAD=%s\nCORRECTED_LOWER_HEAD=%s\nDIRECT_OWNER_HEAD=%s\nFILES_HEAD=%s\nFINAL_HEAD=%s\nFINAL_TREE=%s\nTESTED_OWNER_TREE=%s\nCANCELLATION_OWNER_SHA=%s\nDEPENDENCY_RENDER_TEST_STATUS=%s\nVERIFY=%s\n' \
  "$LOWER_HEAD" "$CORRECTED_LOWER_HEAD" "$DIRECT_OWNER_HEAD" "$FILES_HEAD" "$FINAL_HEAD" "$EXPECTED_FINAL_TREE" \
  "$TESTED_OWNER_TREE" "$CANCELLATION_OWNER_SHA" "$DEPENDENCY_RENDER_TEST_STATUS" "$VERIFY"
