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
REPO_URL_SSH="git@github.com:NousResearch/hermes-agent.git"
REPO_URL_HTTPS="https://github.com/NousResearch/hermes-agent.git"

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
[ "$OLD_SHA" != "$HEAD_SHA" ] || fail "OLD ($INSTALL_REF) IS HEAD; no update would be available"

git clone --bare --quiet "$REPO_ROOT" "$SERVE_REPO"
git -C "$SERVE_REPO" update-ref refs/heads/main "$OLD_SHA"
git -C "$SERVE_REPO" symbolic-ref HEAD refs/heads/main
# The installer may pin a commit that is reachable but not at a ref tip.
git -C "$SERVE_REPO" config uploadpack.allowAnySHA1InWant true
ok "serve.git main = $OLD_SHA ($INSTALL_REF), update target $HEAD_SHA"

arm_redirect() {
  # --- the git URL redirect -----------------------------------------------------
  # we redirect to our own repo so we can play around with what commit hermes thinks we're on.
  # A driver-owned global gitconfig, NOT GIT_CONFIG_COUNT/KEY_n/VALUE_n env
  # config: install.sh sets those itself and would clobber ours.
  actual_git_url="$(git -C "$REPO_ROOT" remote get-url origin)"
  GIT_CFG="$WORK_ROOT/gitconfig"
  cat > "$GIT_CFG" <<EOF
[url "file://$SERVE_REPO"]
  insteadOf = $actual_git_url
  insteadOf = $REPO_URL_HTTPS
  insteadOf = $REPO_URL_SSH
EOF
  export GIT_CONFIG_GLOBAL="$GIT_CFG"

  # check it worked
  expected_git_url="file://$SERVE_REPO"
  actual_git_url="$(git -C "$REPO_ROOT" remote get-url origin)"
  if [[ "$actual_git_url" != "$expected_git_url" ]]; then
    fail "failed git remote get-url shim: origin resolves to '$actual_git_url', expected '$expected_git_url'"
  fi
  ok "git URL redirect via GIT_CONFIG_GLOBAL=$GIT_CFG"


  # shim git and make 'git remote get-url origin' report the actual HA upstream

  # insteadOf is transparent for transport but `git remote get-url origin` gives you the
  # replacement, so _get_origin_url() sees file://$SERVE_REPO and _is_fork() would return true.
  # we check for the arguments "remote get-url origin" in order in any position
  # to allow for e.g. -c with some config being passed.
  # if we didn't do this, we'd need the  .skip_upstream_prompt file to prevent a hang in headless,"add the
  # official repo as upstream?" prompt would hang a headless run. But we don't anymore :D
  REAL_GIT="$(command -v git)"
  REAL_GIT_QUOTED="$(printf '%q' "$REAL_GIT")"
  SHIM_DIR="$WORK_ROOT/shim"
  mkdir -p "$SHIM_DIR"
  cat > "$SHIM_DIR/git" <<EOF
#!/usr/bin/env bash
prev2=""
prev1=""
for arg in "\$@"; do
    if [ "\$prev2" = "remote" ] && [ "\$prev1" = "get-url" ] && [ "\$arg" = "origin" ]; then
        echo "$REPO_URL_HTTPS"
        exit 0
    fi
    prev2="\$prev1"
    prev1="\$arg"
done
exec "$REAL_GIT_QUOTED" "\$@"
EOF
  chmod +x "$SHIM_DIR/git"
  export PATH="$SHIM_DIR:$PATH"

  # check it worked
  observed_git_url="$(git -C "$REPO_ROOT" remote get-url origin)"
  if [[ "$observed_git_url" != "$REPO_URL_HTTPS" ]]; then
    fail "failed git remote get-url shim: origin resolves to '$observed_git_url', expected '$REPO_URL_HTTPS'"
  fi
  ok "git remote get-url shim: $SHIM_DIR/git -> $REAL_GIT (origin reports $REPO_URL_HTTPS)"
}

# later, we might factor this out into a separate step like the macos desktop one.
arm_redirect

# Isolated HOME: the runner's real one may carry a preinstalled hermes or a
# developer config, and old installer scripts hardcode $HOME/.hermes (the
# HERMES_HOME env override is newer than tags we sample). GIT_CONFIG_GLOBAL
# above keeps working -- an explicit path wins over $HOME/.gitconfig.
export HOME="$WORK_ROOT/home"
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
export HERMES_HOME="$HOME/.hermes"
mkdir -p "$HERMES_HOME"

INSTALL_DIR="$HERMES_HOME/hermes-agent"

# Does the installer script at REF accept FLAG? Read that ref's own
# install.sh rather than assuming this checkout's flag set: the point of the
# matrix is to install releases from months back, whose installers predate
# options we take for granted.
#
# Buffered through a variable, NOT `git show | grep -q`: under pipefail,
# grep -q exits at the first match (install.sh is ~140KB, the flags appear
# in the first few KB), git show takes SIGPIPE on its next write, and the
# pipeline reports 141 -- the probe answers NO for a flag the ref HAS.
installer_supports() {
  local text
  text="$(git -C "$REPO_ROOT" show "$1:scripts/install.sh")"
  grep -qF -- "$2" <<< "$text"
}

run_installer() {
  # $1: ref whose scripts/install.sh to run; $2: log name; $3: "desktop" to
  # opt the desktop stage in (--include-desktop)
  local script="$WORK_ROOT/install-$2.sh"
  git -C "$REPO_ROOT" show "$1:scripts/install.sh" > "$script"
  chmod +x "$script"
  # Installer flags have to match the installer being run, not this
  # checkout's: older releases reject options added later. --skip-setup goes
  # back further than any tag we sample; anything newer is probed for.
  local flags=(--skip-setup)
  if installer_supports "$1" "--skip-browser"; then
    flags+=(--skip-browser)
  fi
  if [ "${3:-}" = "desktop" ]; then
    # The desktop stage is the point of this leg, so a ref without the
    # flag is a hard failure, not a silent downgrade to a plain install.
    # (Releases that predate apps/desktop are already skipped upstream by
    # the tag-has-desktop gate; the flag shipped with the app.)
    installer_supports "$1" "--include-desktop" \
      || fail "ref $1 does not support --include-desktop; this leg cannot mean what it claims"
    flags+=(--include-desktop)
  fi
  # </dev/null: the script reads prompts from stdin when a tty is absent;
  # EOF makes every remaining prompt take its default.
  local rc=0
  bash "$script" "${flags[@]}" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/install-$2.log" || rc=$?
  log_group "install.sh ($2) transcript" "$LOG_DIR/install-$2.log"
  [ "$rc" -eq 0 ] || fail "install.sh ($2) exited $rc; transcript above, log at $LOG_DIR/install-$2.log"
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
  local hermes="$INSTALL_DIR/venv/bin/hermes"
  [ -x "$hermes" ] || fail "no hermes console script at $hermes"
  "$hermes" --version 2>&1 | ts_prefix > "$LOG_DIR/version-$2.log" \
    || fail "hermes --version failed after $2; log in $LOG_DIR/version-$2.log"
  ok "hermes --version works: $(head -c 120 "$LOG_DIR/version-$2.log" | tr -d '\n')"
}

smoke_desktop() {
  # $1: label (old|head). Prove the installed CLI can produce the desktop
  # app: `hermes desktop --build-only` runs the full desktop pipeline
  # (workspace install, renderer build, stamp write) and stops before the
  # launch -- the same call `hermes update` itself makes. Probe the
  # INSTALLED hermes for the flag rather than assuming this checkout's
  # surface: sampled OLD releases may predate `hermes desktop` or
  # --build-only entirely, and for them the phase skips, loudly.
  local hermes="$INSTALL_DIR/venv/bin/hermes"
  if ! "$hermes" desktop --help 2>/dev/null | grep -qF -- --build-only; then
    ok "hermes desktop --build-only not supported at $1; skipping desktop smoke"
    return 0
  fi
  local rc=0
  (cd "$INSTALL_DIR" && "$hermes" desktop --build-only < /dev/null 2>&1 \
    | ts_prefix > "$LOG_DIR/desktop-smoke-$1.log") || rc=$?
  log_group "hermes desktop --build-only ($1) transcript" "$LOG_DIR/desktop-smoke-$1.log"
  [ "$rc" -eq 0 ] || fail "hermes desktop --build-only ($1) exited $rc; transcript above"
  ok "hermes desktop --build-only works at $1"
  # TODO(launch): LAUNCH the built app and auto-close it. Mechanism when
  # the pieces land: driver-side spawn interception (a sitecustomize.py on
  # PYTHONPATH wraps subprocess.run under an env-var opt-in and captures
  # the real argv/cwd/env at the spawn site) + Playwright _electron.launch
  # on the captured spec; electronApp.close() is the auto-close. Blocked
  # on that asset and, for linux runners, on a virtual display (Xvfb).
}

# --- install OLD ---------------------------------------------------------------

step "installing OLD ($INSTALL_REF) via its own scripts/install.sh ($INSTALL_METHOD)"
if [ "$INSTALL_METHOD" = "installer-script+desktop" ]; then
  run_installer "$OLD_SHA" old desktop
  assert_checkout "$OLD_SHA" OLD
  assert_desktop_artifact OLD
else
  run_installer "$OLD_SHA" old
  assert_checkout "$OLD_SHA" OLD
fi
smoke_desktop old

# --- update OLD -> HEAD ----------------------------------------------------------

step "advancing served main to HEAD"
git -C "$SERVE_REPO" update-ref refs/heads/main "$HEAD_SHA"
ok "serve.git main = $HEAD_SHA"

step "updating via $UPDATE_METHOD"
case "$UPDATE_METHOD" in
  hermes-update)
    # `--yes` reaches the update subcommand only in later releases, and
    # argparse rejects the whole invocation when it does not exist. Ask the
    # installed hermes; older ones read the prompt from stdin, so close it.
    HERMES="$INSTALL_DIR/venv/bin/hermes"
    if "$HERMES" update --help 2>&1 | grep -qF -- --yes; then
      update_cmd=("$HERMES" update --yes)
    else
      update_cmd=("$HERMES" update)
    fi
    rc=0
    (cd "$INSTALL_DIR" && "${update_cmd[@]}" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/update.log") || rc=$?
    log_group "hermes update transcript" "$LOG_DIR/update.log"
    [ "$rc" -eq 0 ] || fail "hermes update exited $rc; transcript above, log at $LOG_DIR/update.log"
    ;;
  installer-script)
    # A user re-running the one-liner today gets the CURRENT script.
    run_installer "$HEAD_SHA" head
    ;;
  installer-script+desktop)
    run_installer "$HEAD_SHA" head desktop
    assert_desktop_artifact HEAD
    ;;
  hermes-desktop-app-update)
    # The real user surface: `hermes desktop` launches the app, the user
    # clicks Settings -> About -> Update now. Playwright must OWN the spawn
    # (it needs the inspection pipe), so the driver intercepts the product's
    # own launch call - argv/cwd/env captured at the spawn site by
    # e2e-assets/launch-capture/sitecustomize.py - and re-executes it under
    # _electron.launch. Everything before the spawn (build, stamps, sandbox
    # fixup) runs for real in the installed code.
    HERMES="$INSTALL_DIR/venv/bin/hermes"
    ASSETS="$REPO_ROOT/tests/install/e2e-assets"
    SPEC="$WORK_ROOT/launch-spec.json"

    # A REAL configured provider: the mock inference server (the desktop E2E
    # suite's own) is configured into HERMES_HOME exactly like the dev:mock
    # flow does. The app then boots genuinely configured - no onboarding
    # overlay (a fullscreen div that intercepts every click) - and the chat
    # surface is real too.
    source "$ASSETS/mock-provider.sh"
    mock_start "$WORK_ROOT"
    trap mock_stop EXIT

    step "capturing the hermes desktop launch spec (build runs for real)"
    rc=0
    (cd "$INSTALL_DIR" && \
      PYTHONPATH="$ASSETS/launch-capture${PYTHONPATH:+:$PYTHONPATH}" \
      HERMES_E2E_CAPTURE_LAUNCH="$SPEC" \
      "$HERMES" desktop < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/desktop-launch-capture.log") || rc=$?
    log_group "hermes desktop (launch capture) transcript" "$LOG_DIR/desktop-launch-capture.log"
    [ "$rc" -eq 0 ] || fail "hermes desktop exited $rc during launch capture; transcript above"
    # Exit 0 without a capture means a version that never reached its
    # launch - that must fail loudly, not pass as a no-op.
    [ -f "$SPEC.captured" ] || fail "hermes desktop exited 0 but no launch was captured at $SPEC"
    ok "captured $(cat "$SPEC.captured") launch spec"

    step "driving the app under Playwright: Settings -> About -> Update now"
    # Driver tooling comes from the driver: a scratch dir with our own
    # pinned @playwright/test, never resolved from the installed tree
    # (older OLD refs predate the dependency; hoisting moves it around).
    PW_DIR="$WORK_ROOT/playwright"
    mkdir -p "$PW_DIR"
    (cd "$PW_DIR" && npm install --no-save --no-audit --no-fund \
      "@playwright/test@1.58.2" 2>&1 | ts_prefix > "$LOG_DIR/playwright-install.log") \
      || { log_group "playwright install transcript" "$LOG_DIR/playwright-install.log"; fail "playwright install failed"; }
    cp "$ASSETS/launch-from-spec.mjs" "$PW_DIR/"
    rc=0
    (cd "$PW_DIR" && node launch-from-spec.mjs \
      --spec "$SPEC" \
      --result "$HERMES_HOME/.hermes-update-result.json" \
      --expect-sha "$HEAD_SHA" \
      --repo-dir "$INSTALL_DIR" 2>&1 \
      | ts_prefix > "$LOG_DIR/app-update.log") || rc=$?
    log_group "app update (Playwright) transcript" "$LOG_DIR/app-update.log"
    [ "$rc" -eq 0 ] || fail "app-driven update exited $rc; transcript above"
    # The in-app update spawns a DETACHED npm/updater whose parent chain does
    # not pass through the Electron root, so the driver's descendant sweep
    # cannot see it and a pre-clean can race a still-writing npm.
    # Deterministic quiesce instead: find processes whose cwd is inside
    # $INSTALL_DIR, wait for them to finish (they are the updater's tail),
    # then escalate TERM -> KILL. cwd matching is precise to this sandbox;
    # no name patterns.
    step "quiescing $INSTALL_DIR before the head desktop smoke"
    procs_in_install_dir() {
      # Linux: /proc cwd links (fast, no tools needed). Darwin has no /proc:
      # one lsof pass over ALL cwd descriptors, filtered by prefix in the
      # reader. Deliberately NOT `+D "$INSTALL_DIR"`: lsof exits 1 when a +D
      # match comes up empty, and under `set -euo pipefail` that non-zero
      # kills the leg at the assignment. The unanchored form always matches
      # other processes, so empty-for-OUR-dir is exit 0.
      if [ -d /proc ]; then
        local pid cwd
        for pid in /proc/[0-9]*; do
          cwd="$(readlink "$pid/cwd" 2>/dev/null)" || continue
          case "$cwd" in "$INSTALL_DIR"*) echo "${pid#/proc/}";; esac
        done
      else
        lsof -d cwd -F pn 2>/dev/null | awk -v dir="$INSTALL_DIR" '
          /^p/ { pid = substr($0, 2) }
          /^n/ { if (index(substr($0, 2), dir) == 1) print pid }'
      fi
    }
    # If the probe mechanism itself is broken (no lsof on the runner, output
    # shape surprise), say so and skip the wait... a blind quiesce must be
    # VISIBLE, not a vacuous "install dir quiet".
    if [ ! -d /proc ] && ! command -v lsof >/dev/null 2>&1; then
      echo "WARNING: no /proc and no lsof; quiesce is blind, proceeding on the pre-clean alone"
    else
    quiesce_deadline=$((SECONDS + 60))
    while :; do
      lingering="$(procs_in_install_dir || true)"
      [ -z "$lingering" ] && { ok "install dir quiet"; break; }
      if [ "$SECONDS" -ge "$quiesce_deadline" ]; then
        echo "install-dir processes still alive after 60s; terminating: $lingering"
        kill $lingering 2>/dev/null || true
        sleep 5
        lingering="$(procs_in_install_dir || true)"
        [ -n "$lingering" ] && kill -9 $lingering 2>/dev/null || true
        ok "install dir force-quieted"
        break
      fi
      sleep 2
    done
    fi
    # The smoke check rebuilds from scratch anyway; give it a pristine tree
    # rather than whatever the interrupted in-app update left behind.
    step "clearing node_modules after driver-killed in-app update"
    find "$INSTALL_DIR" -maxdepth 3 -name node_modules -type d -prune -print0 2>/dev/null \
      | xargs -0 rm -rf 2>/dev/null || true
    ok "node_modules cleared for the head desktop smoke"
    ;;
esac
assert_checkout "$HEAD_SHA" HEAD
smoke_desktop head

step "PASS: $INSTALL_REF -> HEAD via $UPDATE_METHOD"
