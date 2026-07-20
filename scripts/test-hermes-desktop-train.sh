#!/usr/bin/env bash
# Hermetic safety regression test for scripts/hermes-desktop-train.
# It supplies fake git/gh/npm binaries and never touches real state/worktrees.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/scripts/hermes-desktop-train"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

HOME_DIR="$TMP/home"
SOURCE="$HOME_DIR/Projects/hermes-agent"
STATE="$HOME_DIR/.hermes/desktop-train"
WORKTREE="$HOME_DIR/Projects/.hermes-desktop-train"
FAKE_BIN="$TMP/bin"
CALLS="$TMP/calls"
mkdir -p "$FAKE_BIN" "$SOURCE/.git" "$CALLS"

cat >"$FAKE_BIN/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >>"$FAKE_CALLS/git.txt"; printf '\n' >>"$FAKE_CALLS/git.txt"
args="$*"
if [[ "$args" == *'clone --mirror '* ]]; then mkdir -p "${@: -1}/refs/heads"; exit 0; fi
if [[ "$args" == *'rev-parse --is-bare-repository'* ]]; then echo true; exit 0; fi
if [[ "$args" == *'fetch '* ]]; then exit 0; fi
if [[ "$args" == *'rev-parse refs/heads/main'* ]]; then echo cafebabecafebabecafebabecafebabecafebabe; exit 0; fi
if [[ "$args" == *'rev-parse refs/train/pr/101'* ]]; then
  [[ "${FAKE_PR_MISMATCH:-0}" == 1 ]] && echo 9999999999999999999999999999999999999999 || echo 1111111111111111111111111111111111111111
  exit 0
fi
if [[ "$args" == *'rev-parse refs/train/pr/102'* ]]; then echo 2222222222222222222222222222222222222222; exit 0; fi
if [[ "$args" == *'worktree add '* ]]; then
  path="${@: -2:1}"
  mkdir -p "$path/.git" "$path/apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS" "$path/apps/desktop/release/mac-arm64/Hermes.app/Contents/Resources"
  : >"$path/apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS/Hermes"; chmod +x "$path/apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS/Hermes"
  printf artifact >"$path/apps/desktop/release/mac-arm64/Hermes.app/Contents/Resources/app.asar"
  exit 0
fi
if [[ "$args" == *'worktree remove '* ]]; then rm -rf "${@: -1}"; exit 0; fi
if [[ "$args" == *'merge --no-edit --no-ff '* ]]; then [[ "${FAKE_MERGE_FAIL:-0}" == 1 ]] && exit 1; exit 0; fi
if [[ "$args" == *'rev-parse HEAD'* ]]; then echo deadbeefdeadbeefdeadbeefdeadbeefdeadbeef; exit 0; fi
exit 0
EOF

cat >"$FAKE_BIN/gh" <<'EOF'
#!/usr/bin/env bash
[[ "${FAKE_GH_FAIL:-0}" == 1 ]] && exit 1
printf '%q ' "$@" >>"$FAKE_CALLS/gh.txt"; printf '\n' >>"$FAKE_CALLS/gh.txt"
printf '101\t1111111111111111111111111111111111111111\tfix/a\tA\thttps://example/101\n'
printf '102\t2222222222222222222222222222222222222222\tfix/b\tB\thttps://example/102\n'
EOF

cat >"$FAKE_BIN/npm" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$FAKE_BIN/shasum" <<'EOF'
#!/usr/bin/env bash
# Hash is intentionally deterministic for launch-integrity testing.
printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  %s\n' "${@: -1}"
EOF

cat >"$FAKE_BIN/nohup" <<'EOF'
#!/usr/bin/env bash
touch "$FAKE_CALLS/nohup-ran"
exec "$@"
EOF
chmod +x "$FAKE_BIN"/*

run_train() {
  HOME="$HOME_DIR" PATH="$FAKE_BIN:$PATH" FAKE_CALLS="$CALLS" "$TRAIN" "$@"
}

# init creates only a fixed HOME-relative config, never caller-selected paths.
run_train init >/dev/null
test -f "$STATE/local-refs.txt"
chmod 600 "$STATE/local-refs.txt"
printf 'local/follow-up\n' >"$STATE/local-refs.txt"

# Discovery failure must fail closed before any train worktree is composed.
if FAKE_GH_FAIL=1 HOME="$HOME_DIR" PATH="$FAKE_BIN:$PATH" FAKE_CALLS="$CALLS" "$TRAIN" sync >/dev/null 2>&1; then
  echo 'FAIL: gh discovery failure was accepted' >&2; exit 1
fi
test ! -e "$WORKTREE"
test ! -e "$STATE/manifest.tsv"

# Exact discovery scope and normal sync.
run_train sync >/dev/null
test -d "$WORKTREE/.git"
test -f "$STATE/manifest.tsv"
grep -Fq -- '--repo NousResearch/hermes-agent --author Studio729 --state open' "$CALLS/gh.txt"
grep -Fq 'local/follow-up' "$STATE/manifest.tsv"

# A force-push between discovery and fetch must be rejected rather than merged.
if FAKE_PR_MISMATCH=1 HOME="$HOME_DIR" PATH="$FAKE_BIN:$PATH" FAKE_CALLS="$CALLS" "$TRAIN" sync >/dev/null 2>&1; then
  echo 'FAIL: fetched PR SHA mismatch was accepted' >&2; exit 1
fi
test ! -e "$WORKTREE"
test ! -e "$STATE/manifest.tsv"
run_train sync >/dev/null

# A symlink config is rejected rather than sourced/followed.
rm "$STATE/local-refs.txt"
ln -s /etc/hosts "$STATE/local-refs.txt"
if run_train sync >/dev/null 2>&1; then
  echo 'FAIL: symlink config was accepted' >&2; exit 1
fi
rm "$STATE/local-refs.txt"
printf 'local/follow-up\n' >"$STATE/local-refs.txt"; chmod 600 "$STATE/local-refs.txt"

# Failed merge invalidates the partial worktree and manifest.
if FAKE_MERGE_FAIL=1 HOME="$HOME_DIR" PATH="$FAKE_BIN:$PATH" FAKE_CALLS="$CALLS" "$TRAIN" sync >/dev/null 2>&1; then
  echo 'FAIL: merge failure was accepted' >&2; exit 1
fi
test ! -e "$WORKTREE"
test ! -e "$STATE/manifest.tsv"

# Recreate the worktree, then prove a hash-mismatched manifest refuses launch.
run_train sync >/dev/null
cat >"$STATE/manifest.tsv" <<'EOF'
version	1
train_head	deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
upstream_main	cafebabecafebabecafebabecafebabecafebabe
synced_at	2026-01-01T00:00:00Z
open_prs	101
local_refs	local/follow-up
built_at	2026-01-01T00:00:00Z
asar_sha256	bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
verified	true
EOF
chmod 600 "$STATE/manifest.tsv"
if run_train launch >/dev/null 2>&1; then
  echo 'FAIL: hash-mismatched artifact launched' >&2; exit 1
fi
test ! -e "$CALLS/nohup-ran"

printf 'PASS: hermes-desktop-train safety contract\n'
