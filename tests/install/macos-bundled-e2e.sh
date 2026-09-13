#!/usr/bin/env bash
# Prove the macOS packaged-app update route: a user on a REAL signed OLD
# bundle, installed in an isolated location, clicks About -> "Update now"
# and Squirrel.Mac swaps in a REAL signed NEW bundle and relaunches it —
# with no driver help and no source artifacts anywhere.
#
# Sibling of tests/install/macos-desktop-e2e.sh (the dmg/source arm). This
# arm has NO git redirect, NO staging, NO patching or re-signing: both
# bundles are the actual release zips resolved and downloaded by the
# PARENT-owned common resolver (tests/install/e2e-assets/bundle-inputs.mjs),
# which validates commits, versions, identities (CFBundleIdentifier +
# teamId), channels and sha256.
#
# Phases (state shared via the workroot):
#   install  resolve the bundle manifest with the common resolver, verify
#            artifact sha256, unzip the REAL signed OLD zip into an
#            isolated .app location, derive the executable from its
#            Info.plist (never a hard-coded binary name), verify codesign /
#            BOTH identifiers / version / arch / stamp, record the install
#            receipt for the update phase, seed isolated user state
#   update   build the loopback static feed from the REAL signed NEW zip in
#            the production update-feed contract on the tag's channel,
#            serve it on an ephemeral port (readiness JSON carries the
#            real port), configure updates.desktop_feed_base_url, start
#            the external relaunch watcher, click the real About ->
#            "Update now" under Playwright, then verify the automatic
#            relaunch (new pid/birth/path), the NEW bundle's codesign/
#            identifiers/version/arch/stamp, backend health, plugin and
#            user-state survival
#
# Usage:
#   tests/install/macos-bundled-e2e.sh --phase install|update|all
#     --manifest-url URL --arch arm64|x64
#
# CI-only native guard: macOS host + GITHUB_ACTIONS. Nothing here writes to
# a public feed, dispatches a release, or drives a local GUI outside the
# runner session.

set -euo pipefail

PHASE="all"
MANIFEST_URL=""
ARCH="arm64"
PLAYWRIGHT_VERSION="1.58.2"
UPDATE_WATCH_TIMEOUT_MS=900000

while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase)
      [ "$#" -ge 2 ] || { echo 'error: --phase needs a value' >&2; exit 1; }
      PHASE="$2"; shift 2 ;;
    --manifest-url)
      [ "$#" -ge 2 ] || { echo 'error: --manifest-url needs a value' >&2; exit 1; }
      MANIFEST_URL="$2"; shift 2 ;;
    --arch)
      [ "$#" -ge 2 ] || { echo 'error: --arch needs a value' >&2; exit 1; }
      ARCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$ARCH" in arm64|x64) ;; *) echo "error: --arch must be arm64 or x64" >&2; exit 1 ;; esac

# Native CI-only guard: this leg runs real signed bundles, real codesign
# and real Squirrel.Mac, so it is meaningful only on a macOS CI runner.
[ "$(uname -s)" = "Darwin" ] || { echo "error: this driver runs on macOS only" >&2; exit 1; }
[ "${GITHUB_ACTIONS:-}" = true ] || { echo "error: this driver is CI-only (GITHUB_ACTIONS required)" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS="$REPO_ROOT/tests/install/e2e-assets"
export TS_BASE=$SECONDS
NODE_BIN="$(command -v node)"

WORK_ROOT="${HERMES_E2E_WORKROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hermes-bundled-e2e}"
LOG_DIR="${HERMES_E2E_LOG_DIR:-$WORK_ROOT/logs}"
HOME_SANDBOX="$WORK_ROOT/home"
export HOME_SANDBOX
export HERMES_HOME="$HOME_SANDBOX/.hermes"
export HOME="$HOME_SANDBOX"   # the app must see the ISOLATED home, not the runner's
mkdir -p "$WORK_ROOT" "$LOG_DIR" "$HOME_SANDBOX" "$HERMES_HOME"

step() { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  OK %s\n' "$*"; }
fail() { printf 'E2E ASSERTION FAILED: %s\n' "$*" >&2; exit 1; }
# shellcheck source=../e2e-assets/ts-prefix.sh
source "$(dirname "$0")/e2e-assets/ts-prefix.sh" 2>/dev/null || ts_prefix() { cat; }
# shellcheck source=../e2e-assets/preserve-plugins.sh
source "$(dirname "$0")/e2e-assets/preserve-plugins.sh"
log_group() {
  printf '::group::%s\n' "$1"
  cat "$2"
  printf '::endgroup::\n'
}

MANIFEST="$WORK_ROOT/bundle-inputs.json"
INSTALL_RECEIPT="$WORK_ROOT/install-receipt.json"
FEED_DIR="$WORK_ROOT/feed"
# Port 0 = ephemeral: the server writes the ACTUAL port to a readiness
# JSON, and the driver configures the app from that (no fixed 8791).
FEED_READY="$LOG_DIR/feed-ready.json"

export E2E_ARCH="$ARCH"

# Background PIDs are GLOBALS so the EXIT trap can kill them from any
# scope, on success AND on failure paths (no watcher/server leak).
SERVE_PID=""
WATCHER_PID=""
cleanup() {
  if [ -n "$WATCHER_PID" ]; then kill "$WATCHER_PID" 2>/dev/null || true; fi
  if [ -n "$SERVE_PID" ]; then kill "$SERVE_PID" 2>/dev/null || true; fi
  if declare -F mock_stop >/dev/null 2>&1; then mock_stop 2>/dev/null || true; fi
}
trap cleanup EXIT

manifest_side() { # $1: old|new, $2: field
  "$NODE_BIN" -e '
    const fs = require("node:fs")
    const m = JSON.parse(fs.readFileSync(process.argv[1], "utf8"))
    const side = m[process.argv[2]]
    console.log(process.argv[3] === "artifact" ? side.artifact.path : side[process.argv[3]])
  ' "$MANIFEST" "$1" "$2"
}

require_manifest() {
  [ -f "$MANIFEST" ] || fail "no bundle manifest at $MANIFEST — run the install phase first"
  # Assert the common resolver's normalized output for this arm (schema 1,
  # platform macos, this arch, downloaded artifacts, old != new, SAME
  # CFBundleIdentifier, SAME teamId, same feed channel). The heavy schema/
  # semver/sha256 validation already happened inside bundle-inputs.mjs.
  "$NODE_BIN" -e '
    const fs = require("node:fs")
    const { validateBundleManifest } = require(process.argv[2])
    const m = JSON.parse(fs.readFileSync(process.argv[1], "utf8"))
    validateBundleManifest(m, { platform: "macos", arch: process.env.E2E_ARCH })
    console.log("bundle manifest valid: " + m.old.tag + " -> " + m.new.tag +
      " (identity " + m.old.identity + ", team " + m.old.teamId + ")")
  ' "$MANIFEST" "$ASSETS/mac-bundled-manifest.cjs" 2>&1 | ts_prefix
}

stage_bundle_inputs() {
  [ -n "$MANIFEST_URL" ] || fail "--manifest-url is required for the bundled arm"
  step "resolving the bundle manifest with the common resolver"
  # bundle-inputs.mjs imports semver (declared in the repo root
  # package.json); the checkout on the runner has node_modules via CI setup.
  rm -f "$MANIFEST"
  mkdir -p "$WORK_ROOT"
  "$NODE_BIN" "$ASSETS/bundle-inputs.mjs" \
    --manifest-url "$MANIFEST_URL" --platform macos --arch "$ARCH" \
    --out "$WORK_ROOT" 2>&1 | ts_prefix | tee "$LOG_DIR/bundle-inputs.log"
  ok "bundle inputs staged at $MANIFEST"
}

# Derive the executable from the bundle's own Info.plist — the product
# name is NOT assumed to be "Hermes" (packaged name: "Hermes Bundled") and
# nothing is renamed. Returns the absolute binary path.
derive_app_bin() { # $1: .app path
  local exec_name
  exec_name="$(plutil -extract CFBundleExecutable raw -o - "$1/Contents/Info.plist")"
  [ -n "$exec_name" ] || fail "CFBundleExecutable missing in $1/Contents/Info.plist"
  local bin="$1/Contents/MacOS/$exec_name"
  [ -x "$bin" ] || fail "no executable at $bin (CFBundleExecutable=$exec_name)"
  printf '%s' "$bin"
}

phase_install() {
  stage_bundle_inputs
  require_manifest
  local old_tag old_version old_commit old_identity old_team old_zip
  old_tag="$(manifest_side old tag)"; old_version="$(manifest_side old version)"
  old_commit="$(manifest_side old commit)"; old_identity="$(manifest_side old identity)"
  old_team="$(manifest_side old teamId)"
  old_zip="$(manifest_side old artifact)"
  step "installing OLD bundle $old_tag ($old_version, identity $old_identity, team $old_team)"

  # Artifact bytes: the common resolver verified the remote sha256; we
  # re-verify the local file so the unzipped bundle provably IS the release.
  local got
  got="$(shasum -a 256 "$old_zip" | awk '{print $1}')"
  local want
  want="$("$NODE_BIN" -e 'const m=require(process.argv[1]); console.log(m.old.artifact.sha256.toLowerCase())' "$MANIFEST")"
  [ "$got" = "$want" ] || fail "OLD zip sha256 $got != manifest $want"
  ok "OLD zip sha256 verified"

  # Install into an ISOLATED .app location (not /Applications): unzip the
  # real release zip. No patching, no re-signing, no fakes.
  rm -rf "$WORK_ROOT/apps" "$WORK_ROOT/install-old"
  mkdir -p "$WORK_ROOT/install-old" "$WORK_ROOT/apps"
  unzip -q "$old_zip" -d "$WORK_ROOT/install-old"
  local found
  found="$(find "$WORK_ROOT/install-old" -maxdepth 2 -name '*.app' -type d | head -1)"
  [ -n "$found" ] || fail "no .app inside the OLD release zip"
  local old_app="$WORK_ROOT/apps/Hermes.app"
  mv "$found" "$old_app"

  local old_app_bin
  old_app_bin="$(derive_app_bin "$old_app")"
  ok "OLD bundle installed at $old_app (executable: $(basename "$old_app_bin"))"

  "$NODE_BIN" "$ASSETS/mac-bundled-verify.mjs" verify-app \
    --app "$old_app" \
    --expect-version "$old_version" \
    --expect-commit "$old_commit" \
    --expect-tag "$old_tag" \
    --expect-identity "$old_identity" \
    --expect-team "$old_team" \
    --expect-arch "$ARCH" \
    --out "$LOG_DIR/old-bundle-verify.json" 2>&1 | ts_prefix | tee "$LOG_DIR/old-bundle-verify.log"
  ok "OLD bundle: codesign, identity, team, version, arch and stamp all match the manifest"

  # Persist the installed paths for the update phase: the executable name
  # comes from the bundle's own Info.plist and both phases must agree.
  "$NODE_BIN" -e '
    const fs = require("node:fs")
    fs.writeFileSync(process.argv[1], JSON.stringify({
      app: process.argv[2], bin: process.argv[3], arch: process.env.E2E_ARCH,
      recordedAt: new Date().toISOString(),
    }, null, 2) + "\n")
  ' "$INSTALL_RECEIPT" "$old_app" "$old_app_bin"
  ok "install receipt persisted at $INSTALL_RECEIPT"

  # Isolated user state: a private HOME/HERMES_HOME plus one profile marker
  # file the update must leave untouched.
  mkdir -p "$HERMES_HOME"
  printf 'user_state_marker=%s\n' "$old_commit" > "$HERMES_HOME/desktop-bundled-marker.txt"
  ok "isolated user state seeded at $HERMES_HOME"
}

ensure_playwright() {
  local pw_dir="$WORK_ROOT/playwright"
  [ -d "$pw_dir/node_modules/@playwright/test" ] && { printf '%s' "$pw_dir"; return 0; }
  mkdir -p "$pw_dir"
  (cd "$pw_dir" && npm install --no-save --no-audit --no-fund \
    "@playwright/test@$PLAYWRIGHT_VERSION" 2>&1 | ts_prefix > "$LOG_DIR/playwright-install.log") \
    || { log_group "playwright install transcript" "$LOG_DIR/playwright-install.log"; fail "playwright install failed"; }
  printf '%s' "$pw_dir"
}

phase_update() {
  require_manifest
  [ -f "$INSTALL_RECEIPT" ] || fail "no install receipt at $INSTALL_RECEIPT — run the install phase first"
  local old_app old_app_bin
  old_app="$("$NODE_BIN" -e 'console.log(require(process.argv[1]).app)' "$INSTALL_RECEIPT")"
  old_app_bin="$("$NODE_BIN" -e 'console.log(require(process.argv[1]).bin)' "$INSTALL_RECEIPT")"
  [ -d "$old_app" ] || fail "installed app $old_app (from the install receipt) is gone"
  [ -x "$old_app_bin" ] || fail "installed executable $old_app_bin (from the install receipt) is gone"
  ok "update phase using the receipt's installed executable: $old_app_bin"

  local new_tag new_version new_commit new_identity new_team new_zip
  new_tag="$(manifest_side new tag)"; new_version="$(manifest_side new version)"
  new_commit="$(manifest_side new commit)"; new_identity="$(manifest_side new identity)"
  new_team="$(manifest_side new teamId)"
  new_zip="$(manifest_side new artifact)"

  # Snapshot + seed plugin fixtures BEFORE anything moves (shared owner).
  preserve_before_upgrade

  # The app must boot configured or the onboarding overlay (a fullscreen
  # div) eats every click: configure the mock inference server exactly like
  # the dev:mock flow does (shared owner: mock-provider.sh). It writes
  # $HERMES_HOME/config.yaml; the feed base URL below is appended after.
  # shellcheck source=../e2e-assets/mock-provider.sh
  source "$ASSETS/mock-provider.sh"
  mock_start "$WORK_ROOT"

  # ── the controlled loopback feed ──────────────────────────────────────
  step "building the loopback feed from the REAL NEW signed zip"
  local got want
  got="$(shasum -a 256 "$new_zip" | awk '{print $1}')"
  want="$("$NODE_BIN" -e 'const m=require(process.argv[1]); console.log(m.new.artifact.sha256.toLowerCase())' "$MANIFEST")"
  [ "$got" = "$want" ] || fail "NEW zip sha256 $got != manifest $want"
  rm -rf "$FEED_DIR"
  "$NODE_BIN" "$ASSETS/mac-bundled-feed.mjs" \
    --out "$FEED_DIR" --zip "$new_zip" \
    --version "$new_version" --tag "$new_tag" --arch "$ARCH" \
    2>&1 | ts_prefix | tee "$LOG_DIR/feed-build.json"
  ok "feed materialized at $FEED_DIR"

  "$NODE_BIN" "$ASSETS/mac-bundled-serve.mjs" --dir "$FEED_DIR" --port 0 \
    --log "$LOG_DIR/feed-requests.jsonl" --ready "$FEED_READY" \
    > "$LOG_DIR/feed-server.log" 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 1 50); do
    [ -f "$FEED_READY" ] && break
    kill -0 "$SERVE_PID" 2>/dev/null || { cat "$LOG_DIR/feed-server.log" | ts_prefix; fail "feed server exited before listening"; }
    sleep 0.2
  done
  [ -f "$FEED_READY" ] || fail "feed server never reported its port ($FEED_READY)"
  local feed_port feed_url
  feed_port="$("$NODE_BIN" -e 'console.log(require(process.argv[1]).port)' "$FEED_READY")"
  feed_url="http://127.0.0.1:$feed_port"
  curl -fsS "$feed_url/__healthz" >/dev/null 2>&1 || fail "loopback feed did not answer on $feed_url"
  ok "loopback feed serving on $feed_url (ephemeral port $feed_port)"

  # ── configure the app like a user: config.yaml ────────────────────────
  # The production resolution order is config updates.desktop_feed_base_url
  # first (main.ts resolveDesktopFeedBaseUrl). Loopback HTTP is the one
  # non-HTTPS override the production client accepts (mac-client.ts).
  step "configuring updates.desktop_feed_base_url in the isolated config"
  local config_feed_line="  desktop_feed_base_url: $feed_url"
  grep -q '^updates:' "$HERMES_HOME/config.yaml" 2>/dev/null || printf '\nupdates:\n' >> "$HERMES_HOME/config.yaml"
  grep -qF "$config_feed_line" "$HERMES_HOME/config.yaml" || printf '%s\n' "$config_feed_line" >> "$HERMES_HOME/config.yaml"
  cat "$HERMES_HOME/config.yaml" | ts_prefix | tee "$LOG_DIR/config.yaml"
  ok "feed base URL configured: $feed_url"

  # ── the external relaunch watcher ─────────────────────────────────────
  # Started by THIS shell, detached from Playwright and from the app. It
  # owns the automatic-relaunch proof end to end.
  step "starting the external relaunch watcher"
  "$NODE_BIN" "$ASSETS/mac-bundled-relaunch-watch.cjs" \
    --app-bin "$old_app_bin" \
    --out "$LOG_DIR/relaunch-proof.json" \
    --old-pid-file "$LOG_DIR/old-pid" \
    --new-pid-file "$LOG_DIR/new-pid" \
    --timeout-ms "$UPDATE_WATCH_TIMEOUT_MS" > "$LOG_DIR/relaunch-watch.log" 2>&1 &
  WATCHER_PID=$!
  ok "watcher running (pid $WATCHER_PID)"

  # ── the real user trigger ─────────────────────────────────────────────
  step "launching the OLD app and clicking About -> Update now"
  local pw_dir
  pw_dir="$(ensure_playwright)"
  cp "$ASSETS/mac-bundled-update-driver.mjs" \
     "$ASSETS/process-close.cjs" \
     "$ASSETS/window-input.cjs" \
     "$pw_dir/"
  local rc=0
  (cd "$pw_dir" && node mac-bundled-update-driver.mjs \
    --app-bin "$old_app_bin" \
    --shots "$LOG_DIR/shots" \
    --close-timeout-ms 420000 2>&1 | ts_prefix | tee "$LOG_DIR/app-update.log") || rc=$?
  log_group "in-app update (Playwright) transcript" "$LOG_DIR/app-update.log"
  [ "$rc" -eq 0 ] || fail "in-app update driver exited $rc; transcript above"

  step "waiting for Squirrel.Mac to swap and relaunch (watcher owns the proof)"
  if ! wait "$WATCHER_PID"; then
    cat "$LOG_DIR/relaunch-proof.json" 2>/dev/null | ts_prefix || true
    fail "relaunch watcher did not observe the automatic relaunch within ${UPDATE_WATCH_TIMEOUT_MS}ms"
  fi
  WATCHER_PID=""
  log_group "relaunch watcher record" "$LOG_DIR/relaunch-proof.json"

  local old_pid new_pid
  old_pid="$(cat "$LOG_DIR/old-pid")"
  new_pid="$(cat "$LOG_DIR/new-pid")"
  [ "$old_pid" != "$new_pid" ] || fail "relaunch proof shows the SAME pid ($new_pid); no replacement happened"
  "$NODE_BIN" -e '
    const r = require(process.argv[1])
    if (r.newBirth && r.oldBirth && r.newBirth === r.oldBirth) process.exit(1)
  ' "$LOG_DIR/relaunch-proof.json" || fail "old and new processes report the same birth time"
  # The relaunch must be at the SAME installed path: Squirrel replaces the
  # bundle in place; a different path would mean something else started it.
  "$NODE_BIN" -e '
    const r = require(process.argv[1])
    if (!r.oldPid || !r.newPid || !r.oldExitedAt || !r.newSeenAt || r.appBin !== process.argv[2]) process.exit(1)
  ' "$LOG_DIR/relaunch-proof.json" "$old_app_bin" || fail "incomplete relaunch proof record"
  ok "automatic relaunch: old pid $old_pid exited, new pid $new_pid at $old_app_bin (no driver launch)"

  # ── the NEW bundle in place ───────────────────────────────────────────
  step "verifying the NEW bundle Squirrel installed"
  "$NODE_BIN" "$ASSETS/mac-bundled-verify.mjs" verify-app \
    --app "$old_app" \
    --expect-version "$new_version" \
    --expect-commit "$new_commit" \
    --expect-tag "$new_tag" \
    --expect-identity "$new_identity" \
    --expect-team "$new_team" \
    --expect-arch "$ARCH" \
    --out "$LOG_DIR/new-bundle-verify.json" 2>&1 | ts_prefix | tee "$LOG_DIR/new-bundle-verify.log"
  ok "NEW bundle: codesign, identity ($new_identity), team ($new_team), version $new_version, stamp commit $new_commit"

  # ── old processes are gone, new backend healthy ───────────────────────
  if pgrep -f "$old_app_bin" | grep -qvw "$new_pid"; then
    fail "old bundle processes still running after the update"
  fi
  ok "no stale bundle processes"

  step "probing the relaunched app's backend health"
  local backend_pid backend_port health
  backend_pid="$(ps -axo pid=,ppid=,command= | awk -v p="$new_pid" '$2==p && /serve/ && /hermes/ {print $1; exit}')"
  if [ -n "$backend_pid" ]; then
    backend_port="$(lsof -nP -a -p "$backend_pid" -iTCP -sTCP:LISTEN | awk 'NR>1{sub(".*:","",$9); print $9; exit}')"
  fi
  if [ -z "${backend_port:-}" ]; then
    # Fallback: the newest listening 127.0.0.1 port owned by any descendant
    # of the relaunched app (helper wrappers can reparent the backend).
    backend_port="$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '/127\.0\.0\.1/ {print $1, $2, $9}' \
      | while read -r _ pid addr; do
          if ps -o ppid= -p "$pid" 2>/dev/null | grep -qw "$new_pid" || [ "$pid" = "$new_pid" ]; then
            printf '%s\n' "${addr##*:}"; break
          fi
        done)"
  fi
  [ -n "$backend_port" ] || fail "could not discover the relaunched backend's listening port"
  for _ in $(seq 1 30); do
    health="$(curl -fsS "http://127.0.0.1:$backend_port/api/health" 2>/dev/null)" && break
    sleep 2
  done
  [ -n "${health:-}" ] || fail "backend /api/health never answered on 127.0.0.1:$backend_port"
  printf '%s\n' "$health" | ts_prefix | tee "$LOG_DIR/backend-health.json"
  ok "backend healthy on 127.0.0.1:$backend_port"

  # ── user state survived ───────────────────────────────────────────────
  preserve_after_upgrade
  grep -qF "user_state_marker=$(manifest_side old commit)" "$HERMES_HOME/desktop-bundled-marker.txt" \
    || fail "isolated user-state marker did not survive the update"
  ok "isolated user state survived"

  step "PASS: packaged $(manifest_side old tag) -> $new_tag via the real About -> Update now route"
}

case "$PHASE" in
  install) phase_install ;;
  update)  phase_update ;;
  all)     phase_install; phase_update ;;
  *) echo "error: --phase must be install, update or all" >&2; exit 1 ;;
esac
