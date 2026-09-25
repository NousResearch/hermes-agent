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
#                    NEXT mints a synthetic child of --install-ref (the
#                    HEAD -> NEXT leg: install HEAD, update with HEAD's
#                    own updater).
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
# shellcheck source=../e2e-assets/preserve-user-state.sh
source "$(dirname "$0")/e2e-assets/preserve-user-state.sh"
# shellcheck source=../e2e-assets/user-state-actions.sh
source "$(dirname "$0")/e2e-assets/user-state-actions.sh"
# shellcheck source=../e2e-assets/mock-provider.sh
source "$(dirname "$0")/e2e-assets/mock-provider.sh"
# shellcheck source=e2e-assets/source-driver.sh
source "$(dirname "$0")/e2e-assets/source-driver.sh"
# shellcheck source=e2e-assets/installer-common.sh
source "$(dirname "$0")/e2e-assets/installer-common.sh"
# shellcheck source=e2e-assets/source-update-command.sh
source "$(dirname "$0")/e2e-assets/source-update-command.sh"
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

# The update target defaults to HEAD; --update-ref selects any other ref so
# a stable-to-stable leg can target the next release tag instead of the tip.
# Only call this leg stable-to-stable when BOTH refs are release tags.
# NEXT (the HEAD -> NEXT leg) is minted before the clone so it rides along.
TARGET_LABEL="${UPDATE_REF:-HEAD}"
TARGET_SHA="$(resolve_update_ref "$REPO_ROOT" "$OLD_SHA" "$TARGET_LABEL")" \
  || fail "cannot resolve update ref '$TARGET_LABEL'"
[ "$OLD_SHA" != "$TARGET_SHA" ] || fail "OLD ($INSTALL_REF) IS the update target ($TARGET_LABEL); no update would be available"

git clone --bare --quiet "$REPO_ROOT" "$SERVE_REPO"
git -C "$SERVE_REPO" cat-file -e "$TARGET_SHA^{commit}" \
  || fail "update target $TARGET_SHA ($TARGET_LABEL) did not reach serve.git"
git -C "$SERVE_REPO" update-ref refs/heads/main "$OLD_SHA"
git -C "$SERVE_REPO" symbolic-ref HEAD refs/heads/main
# The installer may pin a commit that is reachable but not at a ref tip.
git -C "$SERVE_REPO" config uploadpack.allowAnySHA1InWant true
ok "serve.git main = $OLD_SHA ($INSTALL_REF), update target $TARGET_SHA ($TARGET_LABEL)"



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



# The user-state verifier judges .env by content, so a lost provider write shows up
# only as "the upgrade changed the user's own state" minutes later. Print the KEY
# NAMES (never values) at each provider boundary: that is what names the step that
# clobbers them. Declared here with the other early helpers because bash resolves
# functions in execution order -- its first call is at the user-state snapshot,
# hundreds of lines above close_running_desktop.
env_key_names() { # label
  local label="$1"
  if [ ! -s "$HERMES_HOME/.env" ]; then
    printf '  [env] %s: (no .env)\n' "$label"
    return 0
  fi
  printf '  [env] %s: %s\n' "$label" \
    "$(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$HERMES_HOME/.env" | tr -d '=' | sort | tr '\n' ' ')"
}

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

# Each Playwright phase must own Electron's single-instance lock. Close all
# other app processes before a phase. Electron otherwise rejects the second
# instance before Playwright receives an app-ready event.
close_running_desktop() {
  local pattern="$INSTALL_DIR/apps/desktop/release"
  local pid waited=0
  for pid in $(pgrep -f "$pattern" 2>/dev/null); do
    kill "$pid" 2>/dev/null || true
  done
  while [ "$waited" -lt 30 ]; do
    pgrep -f "$pattern" >/dev/null 2>&1 || break
    sleep 0.5
    waited=$((waited + 1))
  done
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
      kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
  pgrep -f "$pattern" >/dev/null 2>&1 \
    && fail "a desktop instance from this install survived termination"

  # The install stamp is logged before requestSingleInstanceLock; a launch that
  # prints only that stamp is Electron's silent secondary-instance path. After
  # every matching process is gone, these isolated-route artifacts are stale,
  # not user data, and must not reject Playwright's lock-owning launch.
  rm -f "$HERMES_DESKTOP_USER_DATA_DIR/SingletonLock" \
    "$HERMES_DESKTOP_USER_DATA_DIR/SingletonSocket" \
    "$HERMES_DESKTOP_USER_DATA_DIR/SingletonCookie"
}

# The redirect must stay at TRANSPORT level. `hermes update` resolves its
# update channel from the release archive and validates the record against
# `git config --get remote.origin.url`; if the configured URL ever looked like
# the rehearsal source, channel resolution would fail outright and the leg
# would be testing a fork install instead of the real user path.
assert_redirect_is_transport_only() {
  # Either official form is valid: the installer clones over SSH or HTTPS
  # depending on the environment, and both are "the official URL" as far as
  # channel resolution is concerned.
  local official_https='https://github.com/NousResearch/hermes-agent.git'
  local official_ssh='git@github.com:NousResearch/hermes-agent.git'
  local configured observed
  configured="$(git -C "$INSTALL_DIR" config --get remote.origin.url)"
  case "$configured" in
    "$official_https"|"$official_ssh") ;;
    *) fail "origin is configured as '$configured', not an official URL — the redirect is not transport-only" ;;
  esac
  # `git` on PATH is the shim here (it reports the official origin so fork
  # detection sees it), so read the TRANSPORT url through the real git that
  # arm_source_redirect exported — otherwise `remote get-url origin` returns
  # the official URL and this check would always fail.
  local real="${HERMES_E2E_REAL_GIT:-git}"
  observed="$("$real" -C "$INSTALL_DIR" remote get-url origin)"
  case "$observed" in
    file://*|*serve.git*) ;;
    *) fail "origin transport '$observed' is not redirected to the staged repo" ;;
  esac
  ok "redirect is transport-only (configured: $configured, transport: $observed)"
}

# The user-visible launcher must survive the upgrade and still run. A launcher
# left pointing at a vanished tree is exactly the "update lost something" shape
# a checkout-hash assertion cannot see.
assert_user_shims() {
  local hermes user_shim
  hermes="$(source_hermes "$INSTALL_DIR")" || fail "no usable launcher after the upgrade"
  [ -x "$hermes" ] || fail "launcher is not executable: $hermes"
  user_shim="$HOME/.local/bin/hermes"
  if [ -e "$user_shim" ] || [ -L "$user_shim" ]; then
    HERMES_DISABLE_LAZY_INSTALLS=1 PYTHONDONTWRITEBYTECODE=1 \
      "$user_shim" --version > "$LOG_DIR/version-path-shim.log" 2>&1 \
      || fail "the PATH shim stopped working after the upgrade: $user_shim"
    ok "PATH shim still runs: $user_shim"
  else
    ok "no PATH shim at $user_shim (nothing to check there)"
  fi
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

# A real, chat-capable provider, started BEFORE the first desktop checkpoint. An
# existing user HAS one configured, the durability check needs a real turn (not a
# file we wrote ourselves), and the checkpoint's own chat smoke asserts against
# HERMES_E2E_MOCK_URL -- so starting this later left TWO mocks per leg: the
# checkpoint's (which the app's config pointed at and kept using) and the
# driver's, which the smoke then waited on. That is the "The mock must receive
# this checkpoint prompt after the send" timeout: the app was talking to 43475
# while the smoke asserted against 46723. One mock, started here, is also
# written into the provider config BEFORE preserve_before_upgrade snapshots the
# home, so nothing reconfigures provider state inside the verified window.
if [ -z "${HERMES_E2E_MOCK_URL:-}" ]; then
  PATH="$(dirname "$HERMES_E2E_NODE"):$PATH" mock_start "$WORK_ROOT"
  trap mock_stop EXIT
fi

desktop_checkpoint old "$OLD_SHA" "$INSTALL_METHOD"

# Produce the user's own state through the ordinary CLI, then snapshot what
# must survive. Done as late as possible before the update so the window
# verify() covers contains only the upgrade.
HERMES="$(source_hermes "$INSTALL_DIR")" || fail "no installed command to drive"
source_build_env user_state_produce "$HERMES"
user_state_before_upgrade
assert_redirect_is_transport_only
# Configure the provider LAST, immediately before the snapshot. The steps above
# drive the CLI and the app, and .env was not left where this run configured it by
# the time they finished -- the verifier caught OPENAI_API_KEY/OPENAI_BASE_URL as
# ADDITIONS after the snapshot, meaning the snapshot had missed them. Re-pointing
# here fixes what the upgrade starts from, whatever those steps did, and nothing
# rewrites provider state after this line.
mock_configure_provider "${HERMES_E2E_MOCK_URL:?HERMES_E2E_MOCK_URL must be set before snapshotting}"
env_key_names "after provider configure"
grep -q '^OPENAI_BASE_URL=' "$HERMES_HOME/.env" \
  || fail "provider configure did not reach $HERMES_HOME/.env"
preserve_before_upgrade
env_key_names "after snapshot"
grep -q '^OPENAI_BASE_URL=' "$HERMES_HOME/.env" \
  || fail "the snapshot phase cleared OPENAI_BASE_URL from $HERMES_HOME/.env"

# The verifier's OWN view of every .env it judges, printed right after its
# snapshot. When this disagrees with the probe above, the snapshot recorded a
# different file than the run wrote -- which is what "0 deleted, 1 modified ...
# OPENAI_BASE_URL ADDED" looked like while the probe saw the key present both
# immediately before and immediately after the snapshot.
env_verifier_view() {
  local py
  py="$(_user_state_python)"
  printf '  [env] verifier view:\n'
  "$py" "$USER_STATE_VERIFIER" env-keys --home "$HERMES_HOME" 2>&1 | sed 's/^/    /' || true
}
env_verifier_view

# --- update OLD -> HEAD ----------------------------------------------------------

step "advancing served main to $TARGET_LABEL ($TARGET_SHA)"
git -C "$SERVE_REPO" update-ref refs/heads/main "$TARGET_SHA"
ok "serve.git main = $TARGET_SHA"

step "updating via $UPDATE_METHOD"

# Install-side state capture, callable from inside the update branches: on
# app-update legs the updater's transcript is streamed into the app UI (or runs
# detached) and is otherwise lost, so snapshot every place it also lands --
# product logs, update hand-off files, the venv's entry-point dir -- while the
# install is still there to inspect. The assertions can `fail` out of the
# driver, so the evidence has to be on disk BEFORE they do: the app-update
# branch calls this the moment its observer returns, not after its rc check.
COLLECTED_INSTALL_LOGS=0
collect_install_side_logs() {
  [ "$COLLECTED_INSTALL_LOGS" -eq 0 ] || return 0
  COLLECTED_INSTALL_LOGS=1
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
}

case "$UPDATE_METHOD" in
  hermes-update)
    # `--yes` reaches the update subcommand only in later releases, and
    # argparse rejects the whole invocation when it does not exist. Ask the
    # installed hermes; older ones read the prompt from stdin, so close it.
    HERMES="$(source_hermes "$INSTALL_DIR")" || fail "no installed update command"
    help="$(source_build_env "$HERMES" update --help 2>&1)" || fail "installed update --help failed: $help"
    build_source_update_command "$HERMES" "$help"
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
    accept_installer_marker "$INSTALL_DIR" \
      || fail "installed source has changes other than the generated install marker"
    ASSETS="$REPO_ROOT/tests/install/e2e-assets"
    SPEC="$WORK_ROOT/launch-spec.json"

    # A REAL configured provider: the mock inference server (the desktop E2E
    # suite's own) is configured into HERMES_HOME exactly like the dev:mock
    # flow does. The app then boots genuinely configured - no onboarding
    # overlay (a fullscreen div that intercepts every click) - and the chat
    # surface is real too.
    #
    # The mock the app must talk to was started and configured ABOVE, and its URL
    # is what config.yaml/.env hold. Nothing here may touch provider state: this
    # point is INSIDE the window the user-state verifier judges, so a rewrite (or
    # a new mock on a new port) reads as the upgrade modifying .env. The app reads
    # config.yaml/.env, not HERMES_E2E_MOCK_URL.
    source "$ASSETS/mock-provider.sh"
    trap mock_stop EXIT

    step "capturing the hermes desktop launch spec (build runs for real)"
    rc=0
    if [ "$HERMES" = "$INSTALL_DIR/.hermes/bin/hermes" ]; then
      # The PM launcher uses -I: PYTHONPATH/sitecustomize cannot reach it.
      # Ask the installed launcher for its own isolated command, then inject
      # the driver hook into that command without changing product code.
      (cd "$INSTALL_DIR" && source_build_env python3 -I "$ASSETS/launch-capture/pm-launch.py" \
        "$HERMES" "$SPEC" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/desktop-launch-capture.log") || rc=$?
    else
      # Pre-PM console scripts load sitecustomize from PYTHONPATH.
      (cd "$INSTALL_DIR" && \
        PYTHONPATH="$ASSETS/launch-capture${PYTHONPATH:+:$PYTHONPATH}" \
        HERMES_E2E_CAPTURE_LAUNCH="$SPEC" \
        source_build_env "$HERMES" desktop < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/desktop-launch-capture.log") || rc=$?
    fi
    log_group "hermes desktop (launch capture) transcript" "$LOG_DIR/desktop-launch-capture.log"
    [ "$rc" -eq 0 ] || fail "hermes desktop exited $rc during launch capture; transcript above"
    # Exit 0 without a capture means a version that never reached its
    # launch - that must fail loudly, not pass as a no-op.
    [ -f "$SPEC.captured" ] || fail "hermes desktop exited 0 but no launch was captured at $SPEC"
    ok "captured $(cat "$SPEC.captured") launch spec"

    close_running_desktop
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
    # Evidence before the assertion: the hand-off transcript is exactly what a
    # failing app-update leg needs, and this branch used to collect it only
    # after `fail` had already exited the driver.
    collect_install_side_logs
    [ "$rc" -eq 0 ] || fail "app-driven update exited $rc; transcript above"
    ;;
esac

# Install-side state BEFORE the post-update assertions: the assertions below can
# `fail` out of the driver, so the evidence must already be on disk when they do.
# The app-update branch collects it earlier (its own rc check can fail first);
# the flag inside makes this second call a no-op on those legs.
collect_install_side_logs

# The update may publish a launcher before its installed dependency inputs are
# current. The next non-metadata startup then owns source completion, including
# rebuilding the packaged desktop app. Launching that app directly first lets
# its backend replace the live bundle and kills Playwright's renderer target.
# Desktop legs must therefore drive the ordinary CLI startup even when the
# launcher exists. No-desktop legs retain the legacy missing-launcher recovery.
if [ "$EXPECT_DESKTOP" = "present" ] || ! source_hermes "$INSTALL_DIR" >/dev/null 2>&1; then
  step "next ordinary startup after the update (completes deferred source-update work)"
  STARTUP_HERMES="$(source_hermes_for_startup "$INSTALL_DIR")" \
    || fail "no installed command to start after the update"
  startup_rc=0
  source_build_env "$STARTUP_HERMES" status > "$LOG_DIR/post-update-startup.log" 2>&1 || startup_rc=$?
  log_group "post-update startup" "$LOG_DIR/post-update-startup.log"
  ok "post-update startup ran (exit $startup_rc); the read-only checks below assert completion"
fi

assert_checkout "$TARGET_SHA" "$TARGET_LABEL"
assert_user_shims
user_state_after_upgrade

preserve_after_upgrade
env_key_names "after update"
close_running_desktop
desktop_checkpoint new "$TARGET_SHA" "$UPDATE_METHOD"

step "PASS: $INSTALL_REF -> $TARGET_LABEL via $UPDATE_METHOD"
