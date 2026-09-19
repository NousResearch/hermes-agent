#!/usr/bin/env bash
# Smoke test for hermes-update-rehearsal.sh (this directory).
# Builds a synthetic install in a temp tree and drives pre/post/status for real.
# Runs on macOS/Linux directly, and under git-bash on Windows (native paths).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/hermes-update-rehearsal.sh"
[ -f "$SCRIPT" ] || { echo "missing $SCRIPT"; exit 1; }

echo "--- bash -n ---"
bash -n "$SCRIPT" && echo "syntax OK" || exit 1

# Native paths, so native git.exe/python.exe can reach the same files bash can.
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$(mktemp -d)")"; else ROOT="$(mktemp -d)"; fi
export HOME="$ROOT/home"; mkdir -p "$HOME"
export GIT_CONFIG_GLOBAL="$ROOT/gitconfig-test"; : > "$GIT_CONFIG_GLOBAL"
export HERMES_HOME="$ROOT/home/.hermes"
export HERMES_DESKTOP_USER_DATA_DIR="$ROOT/electron-user-data"
if tar --help 2>&1 | grep -q -- '--force-local'; then export TAR_OPTIONS=--force-local; fi

H="$HERMES_HOME"; INSTALL="$H/hermes-agent"

echo "--- fixture under $ROOT ---"
mkdir -p "$H/plugins/mnemosyne-wrapper" "$H/photon/sidecar/node_modules" "$H/memories"
mkdir -p "$H/skills/foo" "$H/cron" "$H/logs" "$ROOT/external-mnemosyne"
printf 'name: mnemosyne-wrapper\n'   > "$H/plugins/mnemosyne-wrapper/plugin.yaml"
printf '{"wrapper":true}\n'          > "$H/plugins/mnemosyne-wrapper/mnemosyne-wrapper.json"
printf 'witness\n'                   > "$ROOT/external-mnemosyne/witness.txt"
ln -sfn "$ROOT/external-mnemosyne"   "$H/plugins/mnemosyne-wrapper/runtime"
printf 'timezone: utc\n'             > "$H/config.yaml"
printf 'NOUS_API_KEY=xxx\n'          > "$H/.env"
printf '{"tokens":{}}\n'             > "$H/auth.json"
printf 'recall\n'                    > "$H/memories/note.md"
printf 'jobs\n'                      > "$H/cron/jobs.json"
printf 'console.log(1)\n'            > "$H/photon/sidecar/index.mjs"
printf '{"name":"sidecar"}\n'        > "$H/photon/sidecar/package.json"
printf '{"lockfileVersion":3}\n'     > "$H/photon/sidecar/package-lock.json"
printf '{"lockfileVersion":3}\n'     > "$H/photon/sidecar/node_modules/.package-lock.json"

python - "$H/state.db" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
con.execute("CREATE TABLE messages (id TEXT)")
con.executemany("INSERT INTO sessions VALUES (?)", [("s1",), ("s2",)])
con.commit(); con.close()
PY

# the install checkout: a real repo on branch `main`, plus untracked runtime
git init -q -b main "$INSTALL"
git -C "$INSTALL" config user.email t@example.com
git -C "$INSTALL" config user.name test
printf 'print(1)\n' > "$INSTALL/module_name.py"
git -C "$INSTALL" add -A
git -C "$INSTALL" -c commit.gpgsign=false commit -qm initial
git -C "$INSTALL" remote add origin https://github.com/NousResearch/hermes-agent.git
HEAD_SHA="$(git -C "$INSTALL" rev-parse HEAD)"
mkdir -p "$INSTALL/.hermes/bin" "$INSTALL/.hermes-runtime/python"
printf '#!/bin/sh\necho hermes 0.0.0\n' > "$INSTALL/.hermes/bin/hermes"; chmod +x "$INSTALL/.hermes/bin/hermes"
printf 'big' > "$INSTALL/.hermes-runtime/python/interpreter.bin"
mkdir -p "$HOME/.local/bin"
ln -sfn "$INSTALL/.hermes/bin/hermes" "$HOME/.local/bin/hermes"
SYMLINKS_OK=0
[ -L "$HOME/.local/bin/hermes" ] && SYMLINKS_OK=1
echo "host symlink support: $SYMLINKS_OK"

mkdir -p "$HERMES_DESKTOP_USER_DATA_DIR/Local Storage/leveldb" "$HERMES_DESKTOP_USER_DATA_DIR/Cache"
printf '{"window":{}}\n' > "$HERMES_DESKTOP_USER_DATA_DIR/Preferences"
printf '{"session":"t"}\n' > "$HERMES_DESKTOP_USER_DATA_DIR/connection.json"
printf 'x' > "$HERMES_DESKTOP_USER_DATA_DIR/Local Storage/leveldb/000001.ldb"
printf 'junk' > "$HERMES_DESKTOP_USER_DATA_DIR/Cache/data.bin"

BACKUPS="$ROOT/backups"
RUN=("$SCRIPT")
pass=0; fail=0
check() { if [ "$1" = 0 ]; then echo "  PASS $2"; pass=$((pass+1)); else echo "  FAIL $2"; fail=$((fail+1)); fi; }
# `cmd | grep -q` loses to SIGPIPE under pipefail: capture, then grep.
check_out() { local n="$1" m="$2"; shift 2; local out; out="$("$@" 2>&1 || true)"; grep -q "$n" <<< "$out"; check $? "$m"; }

echo
echo "--- pre (source = the fixture repo, so no network) ---"
STATUS_BEFORE="$(git -C "$INSTALL" status --porcelain)"
# A pristine copy of both trees: post is checked against it for exactness.
mkdir -p "$ROOT/pristine/home" "$ROOT/pristine/userdata"
cp -a "$H/." "$ROOT/pristine/home/"
cp -a "$HERMES_DESKTOP_USER_DATA_DIR/." "$ROOT/pristine/userdata/"
"${RUN[@]}" pre --source "$INSTALL" --ref main --backup-root "$BACKUPS" > "$ROOT/pre.log" 2>&1
check $? "pre exits 0"
SNAP="$(ls -1d "$BACKUPS"/*/ | head -1)"; SNAP="${SNAP%/}"
for f in hermes-home.tar electron-userdata.tar manifest.json hermes-home.txt target-sha; do
  [ -e "$SNAP/$f" ]; check $? "backup artifact $f"
done
TARLIST="$(tar -tf "$SNAP/hermes-home.tar" 2>&1 || true)"
grep -qE '^\./hermes-agent/\.git/config$' <<< "$TARLIST"; check $? "whole home: checkout .git in the tar"
grep -qE '^\./hermes-agent/\.hermes-runtime/python/interpreter\.bin$' <<< "$TARLIST"; check $? "whole home: PM store in the tar"
grep -qE '^\./config\.yaml$' <<< "$TARLIST"; check $? "whole home: config.yaml in the tar"
UDLIST="$(tar -tf "$SNAP/electron-userdata.tar" 2>&1 || true)"
grep -qE '^\./Cache/data\.bin$' <<< "$UDLIST"; check $? "whole userData: nothing filtered out of the tar"

echo
echo "--- pre points the install at the rehearsal copy ---"
[ "$(git -C "$SNAP/serve.git" rev-parse refs/heads/main)" = "$HEAD_SHA" ]; check $? "serve.git main is the custom ref"
[ "$(git -C "$INSTALL" config --local --get-regexp 'insteadOf' | wc -l | tr -d ' ')" = 2 ]; check $? "two insteadOf entries written (repo-local)"
[ -f "$H/.skip_upstream_prompt" ]; check $? "upstream-prompt marker created"
git -C "$INSTALL" remote get-url origin | grep -q 'serve.git'; check $? "remote get-url resolves to the rehearsal copy"
git -C "$INSTALL" config --get remote.origin.url | grep -q 'NousResearch'; check $? "config --get remote.origin.url stays official"

echo
echo "--- pre changed nothing else ---"
[ "$(git -C "$INSTALL" rev-parse HEAD)" = "$HEAD_SHA" ]; check $? "checkout untouched by pre"
[ "$(git -C "$INSTALL" status --porcelain)" = "$STATUS_BEFORE" ]; check $? "pre changed no files in the checkout"
grep -q 'nothing has been updated yet' "$ROOT/pre.log"; check $? "pre says it did not update anything"

echo
echo "--- status (read-only) ---"
check_out "$HEAD_SHA" "status reports what it prepared" "${RUN[@]}" status --backup-root "$BACKUPS"

echo
echo "--- post ---"
"${RUN[@]}" post --backup-root "$BACKUPS" --yes > "$ROOT/post.log" 2>&1
check $? "post exits 0"
[ -f "$H/config.yaml" ]; check $? "config.yaml restored"
[ -f "$H/.env" ]; check $? ".env restored"
[ -f "$H/memories/note.md" ]; check $? "memories restored"
[ -f "$H/plugins/mnemosyne-wrapper/mnemosyne-wrapper.json" ]; check $? "plugin marker restored"
[ -f "$H/photon/sidecar/node_modules/.package-lock.json" ]; check $? "photon sidecar marker restored"
[ -f "$INSTALL/.hermes-runtime/python/interpreter.bin" ]; check $? "PM store restored"
[ -d "$INSTALL/.git" ]; check $? "checkout restored"
[ "$(git -C "$INSTALL" rev-parse HEAD)" = "$HEAD_SHA" ]; check $? "checkout HEAD restored"
[ "$(git -C "$INSTALL" config --get remote.origin.url)" = "https://github.com/NousResearch/hermes-agent.git" ]; check $? "origin remote restored"
[ -f "$HERMES_DESKTOP_USER_DATA_DIR/connection.json" ]; check $? "userData connection.json restored"
[ "$(git -C "$INSTALL" config --local --get-regexp 'insteadOf' 2>/dev/null | wc -l | tr -d ' ')" = 0 ]; check $? "no stale insteadOf left in the checkout"
[ ! -f "$H/.skip_upstream_prompt" ]; check $? "upstream-prompt marker removed"
git -C "$INSTALL" remote get-url origin | grep -q 'NousResearch'; check $? "origin resolves officially again"
if [ "$SYMLINKS_OK" = 1 ]; then
  [ -L "$HOME/.local/bin/hermes" ]; check $? "shim outside the two trees left untouched"
else
  echo "  SKIP user-bin shim symlink (host cannot create symlinks)"
fi

echo
echo "--- the acceptance criterion: every file identical before/after ---"
diff -r --no-dereference "$ROOT/pristine/home" "$H" > "$ROOT/home.diff" 2>&1
check $? "HERMES_HOME identical to before pre"
[ -s "$ROOT/home.diff" ] && sed 's/^/    /' "$ROOT/home.diff" | head -30
diff -r --no-dereference "$ROOT/pristine/userdata" "$HERMES_DESKTOP_USER_DATA_DIR" > "$ROOT/userdata.diff" 2>&1
check $? "userData identical to before pre"
[ -s "$ROOT/userdata.diff" ] && sed 's/^/    /' "$ROOT/userdata.diff" | head -30

echo
echo "=== smoke: $pass passed, $fail failed ==="
if [ "${SMOKE_KEEP:-0}" = 1 ]; then echo "root kept: $ROOT"; else rm -rf "$ROOT"; fi
exit $(( fail > 0 ))
