#!/usr/bin/env bash
# Prove a user who installed OLD via the installer script can reach HEAD.
#
# The POSIX sibling of tests/install/windows-e2e.ps1, sharing its
# staging trick and replacing the old bubblewrap sandbox: instead of a fake
# Internet (MITM proxy + upload-pack shim), every git process is pointed at a
# local bare clone with url.<file://serve.git>.insteadOf rewrites for both
# canonical repo URLs in a driver-owned GIT_CONFIG_GLOBAL. The installer and
# updater run byte-for-byte against their real URLs and land on serve.git;
# `main` serves OLD during the install, then advances to HEAD for the update
# leg -- an update becomes available exactly the way it does for a real user.
# No bwrap, no slirp4netns, no TLS interception; the CI runner is disposable,
# so the host IS the sandbox.
#
# install.sh itself is not curl'd: the install leg runs the copy shipped AT
# the OLD ref (what a user who installed then actually executed), and the
# installer-script update leg runs HEAD's copy (what the website serves at
# update time).
#
# Phases (mirroring the windows driver):
#   stage      bare-clone this checkout to serve.git, park main at OLD
#   install    run OLD's scripts/install.sh under the redirect; assert the
#              install landed on OLD with a working `hermes`
#   update     advance served main to HEAD, apply ONE update method, assert
#              the checkout landed on HEAD with a working `hermes`
#
# Usage:
#   tests/install/installer-script-e2e.sh --update-method hermes-update|installer-script|installer-script+desktop
#                                         [--install-method installer-script|installer-script+desktop]
#                                         [--install-ref REF]
#
#   --install-method installer-script          the plain one-liner (default)
#                    installer-script+desktop  the one-liner with its desktop
#                                              stage opted in (--include-desktop)
#   --update-method  hermes-update      `hermes update`
#                    installer-script   re-run install.sh (HEAD's copy)
#                    installer-script+desktop  re-run with --include-desktop
#                    hermes-desktop-app-update  launch the app via `hermes
#                                       desktop` (spawn captured, Playwright
#                                       drives it) and click Update now
#   --install-ref    what to install first; anything git resolves. Default:
#                    the newest release tag in the checkout.
#   --update-ref     what to update TO. Default: HEAD. Pass the next release
#                    tag for a stable-to-stable leg; only label the leg
#                    stable-to-stable when BOTH refs are release tags.
#
# Requires a clean full-history checkout with release tags fetched.

set -euo pipefail

# One time base for every transcript in this leg: ts_prefix stamps lines
# relative to TS_BASE, so all logs share the driver's clock and a single
# playback.html offset slider aligns every file with the recording.
export TS_BASE=$SECONDS

INSTALL_METHOD="installer-script"
UPDATE_METHOD=""
INSTALL_REF=""
UPDATE_REF=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-method)
      [ "$#" -ge 2 ] || { echo 'error: --install-method needs a value' >&2; exit 1; }
      INSTALL_METHOD="$2"; shift 2 ;;
    --update-method)
      [ "$#" -ge 2 ] || { echo 'error: --update-method needs a value' >&2; exit 1; }
      UPDATE_METHOD="$2"; shift 2 ;;
    --install-ref)
      [ "$#" -ge 2 ] || { echo 'error: --install-ref needs a value' >&2; exit 1; }
      INSTALL_REF="$2"; shift 2 ;;
    --update-ref)
      [ "$#" -ge 2 ] || { echo 'error: --update-ref needs a value' >&2; exit 1; }
      UPDATE_REF="$2"; shift 2 ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$INSTALL_METHOD" in
  installer-script|installer-script+desktop) ;;
  *) echo "error: --install-method must be installer-script or installer-script+desktop, got '$INSTALL_METHOD'" >&2; exit 1 ;;
esac
case "$UPDATE_METHOD" in
  hermes-update|installer-script|installer-script+desktop|hermes-desktop-app-update) ;;
  *) echo "error: --update-method must be hermes-update, installer-script, installer-script+desktop or hermes-desktop-app-update, got '$UPDATE_METHOD'" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS="$REPO_ROOT/tests/install/e2e-assets"
# Pin driver tooling before an installer changes PATH. CI prepares locked deps.
export HERMES_E2E_NODE="${HERMES_E2E_NODE:-$(command -v node)}"

# Everything lives OUTSIDE the checkout; an untracked dir inside the repo
# would make later dirty-tree checks lie.
WORK_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hermes-installer-script-e2e"
LOG_DIR="${HERMES_E2E_LOG_DIR:-$WORK_ROOT/logs}"
SERVE_REPO="$WORK_ROOT/serve.git"

step() { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  OK %s\n' "$*"; }
fail() { printf 'E2E ASSERTION FAILED: %s\n' "$*" >&2; exit 1; }
# shellcheck source=../e2e-assets/ts-prefix.sh
source "$(dirname "$0")/e2e-assets/ts-prefix.sh" 2>/dev/null || ts_prefix() { cat; }
# shellcheck source=../e2e-assets/preserve-plugins.sh
source "$(dirname "$0")/e2e-assets/preserve-plugins.sh"
# shellcheck source=e2e-assets/source-driver.sh
source "$(dirname "$0")/e2e-assets/source-driver.sh"
# shellcheck source=e2e-assets/installer-common.sh
source "$(dirname "$0")/e2e-assets/installer-common.sh"
# shellcheck source=e2e-assets/source-build-env.sh
source "$ASSETS/source-build-env.sh"
# Full transcript in the job log, collapsed (GitHub renders ::group:: as a
# fold; plain text anywhere else). Win or lose -- a green install's log is
# how you diagnose the leg that fails next.
log_group() {
  printf '::group::%s\n' "$1"
  cat "$2"
  printf '::endgroup::\n'
}

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT" "$LOG_DIR"

# --- stage: serve.git with main parked at OLD --------------------------------

step "staging serve.git (main -> OLD)"
# Tracked changes only (-uno): the bare clone serves committed objects, so a
# modified tracked file means HEAD is not the code being reviewed -- but an
# untracked file (scratch notes, this driver before it lands) cannot leak
# into the clone at all.
[ -z "$(git -C "$REPO_ROOT" status --porcelain -uno)" ] \
  || fail "checkout has uncommitted tracked changes; the staged clone must be a reviewable commit"

if [ -z "$INSTALL_REF" ]; then
  INSTALL_REF="$(git -C "$REPO_ROOT" tag --list 'v[0-9]*' --sort=-creatordate | head -1)"
  [ -n "$INSTALL_REF" ] || fail "no release tags in the checkout to use as OLD"
fi
OLD_SHA="$(git -C "$REPO_ROOT" rev-parse "${INSTALL_REF}^{commit}")"
HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"

# The update target defaults to HEAD; --update-ref selects any other ref so
# a stable-to-stable leg can target the next release tag instead of the tip.
# Only call this leg stable-to-stable when BOTH refs are release tags.
TARGET_LABEL="HEAD"
TARGET_SHA="$HEAD_SHA"
if [ -n "$UPDATE_REF" ]; then
  TARGET_SHA="$(git -C "$REPO_ROOT" rev-parse "${UPDATE_REF}^{commit}")"
  TARGET_LABEL="$UPDATE_REF"
fi
[ "$OLD_SHA" != "$TARGET_SHA" ] || fail "OLD ($INSTALL_REF) IS the update target ($TARGET_LABEL); no update would be available"

git clone --bare --quiet "$REPO_ROOT" "$SERVE_REPO"
git -C "$SERVE_REPO" update-ref refs/heads/main "$OLD_SHA"
git -C "$SERVE_REPO" symbolic-ref HEAD refs/heads/main
# The installer may pin a commit that is reachable but not at a ref tip.
git -C "$SERVE_REPO" config uploadpack.allowAnySHA1InWant true
ok "serve.git main = $OLD_SHA ($INSTALL_REF), update target $HEAD_SHA"



arm_source_redirect "$REPO_ROOT" "$WORK_ROOT" "$SERVE_REPO"

# Isolated HOME: the runner's real one may carry a preinstalled hermes or a
# developer config, and old installer scripts hardcode $HOME/.hermes (the
# HERMES_HOME env override is newer than tags we sample). GIT_CONFIG_GLOBAL
# above keeps working -- an explicit path wins over $HOME/.gitconfig.
export HOME="$WORK_ROOT/home"
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
export HERMES_HOME="$HOME/.hermes"
export HERMES_DESKTOP_USER_DATA_DIR="$WORK_ROOT/electron-user-data"
mkdir -p "$HERMES_HOME"

INSTALL_DIR="$HERMES_HOME/hermes-agent"



assert_desktop_artifact() {
  # $1: label. After a +desktop install the built app must exist under the
  # checkout -- install.sh builds it there and registers no OS entry point.
  local release_dir="$INSTALL_DIR/apps/desktop/release"
  local found=""
  local cand
  for cand in \
    "$release_dir/linux-unpacked/Hermes" \
    "$release_dir/linux-unpacked/hermes" \
    "$release_dir/mac-arm64/Hermes.app" \
    "$release_dir/mac/Hermes.app"; do
    if [ -x "$cand" ] || [ -d "$cand" ]; then
      found="$cand"
      break
    fi
  done
  [ -n "$found" ] || fail "no desktop app under $release_dir after $1 (+desktop install)"
  ok "desktop app built by installer at $1: $found"
}

assert_checkout() {
  # $1: expected sha, $2: label
  local got
  got="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
  [ "$got" = "$1" ] || fail "installed checkout is $got, expected $2 ($1)"
  ok "checkout is $2 ($1)"
  local hermes
  hermes="$(source_hermes "$INSTALL_DIR")" || fail "no usable installed command at $2"
  python3 -B "$REPO_ROOT/tests/install/e2e-assets/source_driver.py" \
    --root "$INSTALL_DIR" --launcher "$hermes" --desktop "$EXPECT_DESKTOP" \
    || fail "read-only verification failed at $2; no repair was attempted"
  HERMES_DISABLE_LAZY_INSTALLS=1 PYTHONDONTWRITEBYTECODE=1 source_build_env "$hermes" --version 2>&1 | ts_prefix > "$LOG_DIR/version-$2.log" \
    || fail "hermes --version failed after $2; log in $LOG_DIR/version-$2.log"
  ok "hermes --version works: $(head -c 120 "$LOG_DIR/version-$2.log" | tr -d '\n')"
}

desktop_checkpoint() { # phase, expected commit, selected method
  source_build_env "$HERMES_E2E_NODE" "$ASSETS/source-desktop-smoke.mjs" \
    --root "$INSTALL_DIR" --home "$HERMES_HOME" --user-data "$HERMES_DESKTOP_USER_DATA_DIR" \
    --out "$LOG_DIR" --phase "$1" --expect-commit "$2" \
    --desktop "$EXPECT_DESKTOP" --method "$3"
}

# --- install OLD ---------------------------------------------------------------

step "installing OLD ($INSTALL_REF) via its own scripts/install.sh ($INSTALL_METHOD)"
EXPECT_DESKTOP=absent
if [ "$INSTALL_METHOD" = "installer-script+desktop" ]; then
  EXPECT_DESKTOP=present
  source_build_env run_source_installer "$REPO_ROOT" "$WORK_ROOT" "$LOG_DIR" "$OLD_SHA" old desktop
  assert_checkout "$OLD_SHA" OLD
  assert_desktop_artifact OLD
else
  source_build_env run_source_installer "$REPO_ROOT" "$WORK_ROOT" "$LOG_DIR" "$OLD_SHA" old
  assert_checkout "$OLD_SHA" OLD
fi
desktop_checkpoint old "$OLD_SHA" "$INSTALL_METHOD"
preserve_before_upgrade

# --- update OLD -> HEAD ----------------------------------------------------------

step "advancing served main to $TARGET_LABEL ($TARGET_SHA)"
git -C "$SERVE_REPO" update-ref refs/heads/main "$TARGET_SHA"
ok "serve.git main = $TARGET_SHA"

step "updating via $UPDATE_METHOD"
case "$UPDATE_METHOD" in
  hermes-update)
    # `--yes` reaches the update subcommand only in later releases, and
    # argparse rejects the whole invocation when it does not exist. Ask the
    # installed hermes; older ones read the prompt from stdin, so close it.
    HERMES="$(source_hermes "$INSTALL_DIR")" || fail "no installed update command"
    help="$(source_build_env "$HERMES" update --help 2>&1)" || fail "installed update --help failed: $help"
    if grep -qF -- --yes <<< "$help"; then
      update_cmd=("$HERMES" update --yes)
    else
      update_cmd=("$HERMES" update)
    fi
    rc=0
    (cd "$INSTALL_DIR" && source_build_env "${update_cmd[@]}" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/update.log") || rc=$?
    log_group "hermes update transcript" "$LOG_DIR/update.log"
    [ "$rc" -eq 0 ] || fail "hermes update exited $rc; transcript above, log at $LOG_DIR/update.log"
    ;;
  installer-script)
    # A user re-running the one-liner today gets the CURRENT script.
    source_build_env run_source_installer "$REPO_ROOT" "$WORK_ROOT" "$LOG_DIR" "$TARGET_SHA" "$TARGET_LABEL"
    ;;
  installer-script+desktop)
    EXPECT_DESKTOP=present
    source_build_env run_source_installer "$REPO_ROOT" "$WORK_ROOT" "$LOG_DIR" "$TARGET_SHA" "$TARGET_LABEL" desktop
    assert_desktop_artifact "$TARGET_LABEL"
    ;;
  hermes-desktop-app-update)
    # The real user surface: `hermes desktop` launches the app, the user
    # clicks Settings -> About -> Update now. Playwright must OWN the spawn
    # (it needs the inspection pipe), so the driver intercepts the product's
    # own launch call - argv/cwd/env captured at the spawn site by
    # e2e-assets/launch-capture/sitecustomize.py - and re-executes it under
    # _electron.launch. Everything before the spawn (build, stamps, sandbox
    # fixup) runs for real in the installed code.
    EXPECT_DESKTOP=present
    HERMES="$(source_hermes "$INSTALL_DIR")" || fail "no installed desktop command"
    ASSETS="$REPO_ROOT/tests/install/e2e-assets"
    SPEC="$WORK_ROOT/launch-spec.json"

    # A REAL configured provider: the mock inference server (the desktop E2E
    # suite's own) is configured into HERMES_HOME exactly like the dev:mock
    # flow does. The app then boots genuinely configured - no onboarding
    # overlay (a fullscreen div that intercepts every click) - and the chat
    # surface is real too.
    source "$ASSETS/mock-provider.sh"
    PATH="$(dirname "$HERMES_E2E_NODE"):$PATH" mock_start "$WORK_ROOT"
    trap mock_stop EXIT

    step "capturing the hermes desktop launch spec (build runs for real)"
    rc=0
    (cd "$INSTALL_DIR" && \
      PYTHONPATH="$ASSETS/launch-capture${PYTHONPATH:+:$PYTHONPATH}" \
      HERMES_E2E_CAPTURE_LAUNCH="$SPEC" \
      source_build_env "$HERMES" desktop < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/desktop-launch-capture.log") || rc=$?
    log_group "hermes desktop (launch capture) transcript" "$LOG_DIR/desktop-launch-capture.log"
    [ "$rc" -eq 0 ] || fail "hermes desktop exited $rc during launch capture; transcript above"
    # Exit 0 without a capture means a version that never reached its
    # launch - that must fail loudly, not pass as a no-op.
    [ -f "$SPEC.captured" ] || fail "hermes desktop exited 0 but no launch was captured at $SPEC"
    ok "captured $(cat "$SPEC.captured") launch spec"

    step "driving the app under Playwright: Settings -> About -> Update now"
    # Use the checkout module closure and current driver Node, not OLD tooling.
    rc=0
    (cd "$WORK_ROOT" && "$HERMES_E2E_NODE" "$ASSETS/launch-from-spec.mjs" \
      --spec "$SPEC" \
      --old-sha "$OLD_SHA" --chat-out "$LOG_DIR/update-window" --mock-url "$HERMES_E2E_MOCK_URL" \
      --result "$HERMES_HOME/.hermes-update-result.json" \
      --expect-sha "$TARGET_SHA" \
      --repo-dir "$INSTALL_DIR" 2>&1 \
      | ts_prefix > "$LOG_DIR/app-update.log") || rc=$?
    log_group "app update (Playwright) transcript" "$LOG_DIR/app-update.log"
    [ "$rc" -eq 0 ] || fail "app-driven update exited $rc; transcript above"
    ;;
esac

# Install-side state BEFORE the post-update assertions: on app-update legs
# the updater's transcript is streamed into the app UI (or runs detached)
# and is otherwise lost, so snapshot every place it also lands — product
# logs, update hand-off files, the venv's entry-point dir — while the
# install is still there to inspect. The assertions below can `fail` out
# of the driver; the evidence must already be on disk when they do.
ildest="$LOG_DIR/install-logs"
mkdir -p "$ildest"
cp -R "$HERMES_HOME/logs" "$ildest/hermes-logs" 2>/dev/null || true
if [ -n "${XDG_DATA_HOME:-}" ]; then
  cp -R "$XDG_DATA_HOME/hermes/logs" "$ildest/desktop-userdata-logs" 2>/dev/null || true
fi
cp "$HERMES_HOME/.hermes-update-result.json" "$ildest" 2>/dev/null || true
ls -la "$HERMES_HOME" > "$ildest/hermes-home-ls.txt" 2>/dev/null || true
ls -la "$INSTALL_DIR/venv/bin" > "$ildest/venv-bin-ls.txt" 2>/dev/null || true
ok "collected install-side logs to $ildest"

assert_checkout "$TARGET_SHA" "$TARGET_LABEL"

preserve_after_upgrade
desktop_checkpoint new "$TARGET_SHA" "$UPDATE_METHOD"

step "PASS: $INSTALL_REF -> $TARGET_LABEL via $UPDATE_METHOD"
