#!/usr/bin/env bash
# Prove a macOS user who installed OLD via the published desktop installer
# (Hermes-Setup.dmg from the website) can reach HEAD.
#
# The macOS sibling of tests/install/windows-e2e.ps1's desktop-installer
# arm, sharing the staging trick: every git process is pointed at a local
# bare clone via url.<file://serve.git>.insteadOf in a driver-owned
# GIT_CONFIG_GLOBAL. The published dmg carries no commit pin - it installs
# whatever `main` serves - so parking serve.git's main at OLD stages the
# "user on the current release" start, and advancing it to HEAD makes an
# update available exactly the way it does for a real user.
#
# Phases (state shared via the workroot, mirroring the windows driver):
#   stage    bare-clone this checkout to serve.git, park main at OLD
#   install  download the dmg, hdiutil attach, run the installer app's
#            binary DIRECTLY (env inheritance: an `open`-launched app sees
#            none of our redirect env), wait for the install to land
#   update   advance served main to HEAD, apply ONE update method:
#              open-app-update            launch the installed app binary
#                                         under Playwright, click Update now
#              hermes-desktop-app-update  capture `hermes desktop`'s spawn,
#                                         launch the spec under Playwright,
#                                         click Update now
#              hermes-update              CLI update from the installed venv
#              installer-script[+desktop] re-run the current install one-liner
#
# Usage:
#   tests/install/macos-desktop-e2e.sh --phase stage|install|update|all
#     --update-method open-app-update|hermes-desktop-app-update
#     [--install-ref REF] [--dmg-url URL]
#
# Requires a clean full-history checkout with release tags fetched, on a
# macOS host with a window server (the GitHub macos runners qualify).

set -euo pipefail

# One time base for every transcript in this leg: ts_prefix stamps lines
# relative to TS_BASE, so all logs share the driver's clock and a single
# playback.html offset slider aligns every file with the recording.
export TS_BASE=$SECONDS

PHASE="all"
UPDATE_METHOD=""
INSTALL_REF=""
DMG_URL="https://hermes-assets.nousresearch.com/Hermes-Setup.dmg"
PLAYWRIGHT_VERSION="1.58.2"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase)
      [ "$#" -ge 2 ] || { echo 'error: --phase needs a value' >&2; exit 1; }
      PHASE="$2"; shift 2 ;;
    --update-method)
      [ "$#" -ge 2 ] || { echo 'error: --update-method needs a value' >&2; exit 1; }
      UPDATE_METHOD="$2"; shift 2 ;;
    --install-ref)
      [ "$#" -ge 2 ] || { echo 'error: --install-ref needs a value' >&2; exit 1; }
      INSTALL_REF="$2"; shift 2 ;;
    --dmg-url)
      [ "$#" -ge 2 ] || { echo 'error: --dmg-url needs a value' >&2; exit 1; }
      DMG_URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$UPDATE_METHOD" in
  open-app-update|hermes-desktop-app-update|hermes-update|installer-script|installer-script+desktop) ;;
  *) echo "error: unsupported --update-method '$UPDATE_METHOD'" >&2; exit 1 ;;
esac
[ "$(uname -s)" = "Darwin" ] || { echo "error: this driver runs on macOS only" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_URL_SSH="git@github.com:NousResearch/hermes-agent.git"
REPO_URL_HTTPS="https://github.com/NousResearch/hermes-agent.git"
ASSETS="$REPO_ROOT/tests/install/e2e-assets"

WORK_ROOT="${HERMES_E2E_WORKROOT:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hermes-macos-desktop-e2e}"
LOG_DIR="${HERMES_E2E_LOG_DIR:-$WORK_ROOT/logs}"
SERVE_REPO="$WORK_ROOT/serve.git"
STATE="$WORK_ROOT/shas.env"
export HOME_SANDBOX="$WORK_ROOT/home"

step() { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  OK %s\n' "$*"; }
fail() { printf 'E2E ASSERTION FAILED: %s\n' "$*" >&2; exit 1; }
# shellcheck source=../e2e-assets/ts-prefix.sh
source "$(dirname "$0")/e2e-assets/ts-prefix.sh" 2>/dev/null || ts_prefix() { cat; }
log_group() {
  printf '::group::%s\n' "$1"
  cat "$2"
  printf '::endgroup::\n'
}

# Every phase runs in its own process (separate CI steps), so the redirect
# env is re-established here, not inherited.
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

  # -------
  export HOME="$HOME_SANDBOX"
  export PATH="$HOME/.local/bin:$PATH"
  export HERMES_HOME="$HOME/.hermes"
  export INSTALL_DIR="$HERMES_HOME/hermes-agent"
}

phase_stage() {
  step "staging serve.git (main -> OLD)"
  [ -z "$(git -C "$REPO_ROOT" status --porcelain -uno)" ] \
    || fail "checkout has uncommitted tracked changes; the staged clone must be a reviewable commit"

  rm -rf "$WORK_ROOT"
  mkdir -p "$WORK_ROOT" "$LOG_DIR" "$HOME_SANDBOX/.local/bin"

  local old_ref="$INSTALL_REF"
  if [ -z "$old_ref" ] || [ "$old_ref" = "auto" ]; then
    old_ref="$(git -C "$REPO_ROOT" tag --list 'v[0-9]*' --sort=-creatordate | head -1)"
    [ -n "$old_ref" ] || fail "no release tags in the checkout to use as OLD"
  fi
  local old_sha head_sha
  old_sha="$(git -C "$REPO_ROOT" rev-parse "${old_ref}^{commit}")"
  head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  [ "$old_sha" != "$head_sha" ] || fail "OLD ($old_ref) IS HEAD; no update would be available"

  git clone --bare --quiet "$REPO_ROOT" "$SERVE_REPO"
  git -C "$SERVE_REPO" update-ref refs/heads/main "$old_sha"
  git -C "$SERVE_REPO" symbolic-ref HEAD refs/heads/main
  git -C "$SERVE_REPO" config uploadpack.allowAnySHA1InWant true

  arm_redirect
  mkdir -p "$HERMES_HOME"
  touch "$HERMES_HOME/.skip_upstream_prompt"

  printf 'OLD_SHA=%s\nOLD_REF=%s\nHEAD_SHA=%s\n' "$old_sha" "$old_ref" "$head_sha" > "$STATE"
  ok "serve.git main = $old_sha ($old_ref), update target $head_sha"
}

find_installed_app() {
  # The bootstrap installs the packaged app; look where the product puts it
  # (the checkout's release dir), plus /Applications for a copied bundle.
  local cand
  for cand in \
    "$INSTALL_DIR/apps/desktop/release/mac-arm64/Hermes.app" \
    "$INSTALL_DIR/apps/desktop/release/mac/Hermes.app" \
    "/Applications/Hermes.app"; do
    [ -d "$cand" ] && { printf '%s' "$cand"; return 0; }
  done
  return 1
}

phase_install() {
  # shellcheck disable=SC1090
  . "$STATE"
  arm_redirect
  step "installing OLD ($OLD_REF) via the published Hermes-Setup.dmg"

  local dmg="$WORK_ROOT/Hermes-Setup.dmg"
  [ -f "$dmg" ] || curl -fsSL -o "$dmg" "$DMG_URL"
  [ "$(stat -f%z "$dmg")" -gt 1000000 ] || fail "dmg download too small: $(stat -f%z "$dmg") bytes"
  # curl'd files carry no quarantine attr, but belt and braces on a runner.
  xattr -dr com.apple.quarantine "$dmg" 2>/dev/null || true

  local mount
  mount="$(hdiutil attach -nobrowse -readonly "$dmg" | awk -F'\t' '/\/Volumes\//{print $NF; exit}')"
  [ -n "$mount" ] || fail "hdiutil attach produced no mount point"
  ok "dmg mounted at $mount"

  local app_bin=""
  local app
  app="$(find "$mount" -maxdepth 1 -name '*.app' | head -1)"
  [ -n "$app" ] || { hdiutil detach "$mount" >/dev/null 2>&1 || true; fail "no .app inside the dmg"; }
  app_bin="$(find "$app/Contents/MacOS" -type f -perm +111 | head -1)"
  [ -n "$app_bin" ] || fail "no executable inside $app/Contents/MacOS"

  # The Setup app is Tauri (Rust + system webview): Playwright/Electron
  # attach never works, and run bare it waits forever on its setup-choice
  # screen. Launch it in the background with our env (direct exec, not
  # `open`: launchd inherits NONE of the redirect env) and drive the
  # "Install Hermes" button with native input.
  local rc=0
  bash "$ASSETS/drive-dmg-install.sh" \
    --app-bin "$app_bin" \
    --install-dir "$INSTALL_DIR" \
    --proof-dir "$LOG_DIR" 2>&1 \
    | ts_prefix > "$LOG_DIR/bootstrap-install.log" || rc=$?
  log_group "Hermes-Setup (dmg bootstrap) transcript" "$LOG_DIR/bootstrap-install.log"
  hdiutil detach "$mount" >/dev/null 2>&1 || true
  [ "$rc" -eq 0 ] || fail "dmg bootstrap exited $rc; transcript above"

  [ -d "$INSTALL_DIR/.git" ] || fail "no checkout landed at $INSTALL_DIR"
  local got
  got="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
  [ "$got" = "$OLD_SHA" ] || fail "installed checkout is $got, expected OLD ($OLD_SHA)"
  ok "checkout is OLD ($OLD_SHA)"
  local hermes="$INSTALL_DIR/venv/bin/hermes"
  [ -x "$hermes" ] || fail "no hermes console script at $hermes"
  "$hermes" --version 2>&1 | ts_prefix > "$LOG_DIR/version-old.log" || fail "hermes --version failed after install"
  ok "hermes --version works: $(head -c 120 "$LOG_DIR/version-old.log" | tr -d '\n')"
  find_installed_app >/dev/null || fail "no installed Hermes.app after the dmg bootstrap"
  ok "installed app: $(find_installed_app)"
}

ensure_playwright() {
  # Install the driver's OWN pinned @playwright/test into a scratch dir
  # (never the installed tree's copy). Idempotent across phases.
  local pw_dir="$WORK_ROOT/playwright"
  [ -d "$pw_dir/node_modules/@playwright/test" ] && { printf '%s' "$pw_dir"; return 0; }
  mkdir -p "$pw_dir"
  (cd "$pw_dir" && npm install --no-save --no-audit --no-fund \
    "@playwright/test@$PLAYWRIGHT_VERSION" 2>&1 | ts_prefix > "$LOG_DIR/playwright-install.log") \
    || { log_group "playwright install transcript" "$LOG_DIR/playwright-install.log"; fail "playwright install failed"; }
  printf '%s' "$pw_dir"
}

installer_supports() {
  # $1: ref; $2: flag. Installer flags must match the installer being run,
  # not this checkout's: older releases reject options added later.
  # Capture before grepping: a `git show | grep -q` pipe takes SIGPIPE
  # under pipefail when grep exits at first match, so a supported flag
  # would read as unsupported.
  local text
  text="$(git -C "$REPO_ROOT" show "$1:scripts/install.sh")"
  grep -qF -- "$2" <<< "$text"
}

run_installer() {
  # $1: ref whose scripts/install.sh to run; $2: log name; $3: "desktop" to
  # opt the desktop stage in (--include-desktop). Mirrors the POSIX driver.
  local script="$WORK_ROOT/install-$2.sh"
  git -C "$REPO_ROOT" show "$1:scripts/install.sh" > "$script"
  chmod +x "$script"
  local flags=(--skip-setup)
  if installer_supports "$1" "--skip-browser"; then
    flags+=(--skip-browser)
  fi
  if [ "${3:-}" = "desktop" ]; then
    installer_supports "$1" "--include-desktop" \
      || fail "ref $1 does not support --include-desktop; this leg cannot mean what it claims"
    flags+=(--include-desktop)
  fi
  # </dev/null: EOF makes every prompt take its default.
  local rc=0
  bash "$script" "${flags[@]}" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/install-$2.log" || rc=$?
  log_group "installer ($2) transcript" "$LOG_DIR/install-$2.log"
  [ "$rc" -eq 0 ] || fail "installer ($2) exited $rc; transcript above"
}

run_playwright_update() {
  # $1: spec file to launch from.
  local spec="$1"
  local pw_dir
  pw_dir="$(ensure_playwright)"
  cp "$ASSETS/launch-from-spec.mjs" "$pw_dir/"
  local rc=0
  (cd "$pw_dir" && node launch-from-spec.mjs \
    --spec "$spec" \
    --result "$HERMES_HOME/.hermes-update-result.json" \
    --expect-sha "$HEAD_SHA" \
    --repo-dir "$INSTALL_DIR" 2>&1 \
    | ts_prefix > "$LOG_DIR/app-update.log") || rc=$?
  log_group "app update (Playwright) transcript" "$LOG_DIR/app-update.log"
  [ "$rc" -eq 0 ] || fail "app-driven update exited $rc; transcript above"
}

phase_update() {
  # shellcheck disable=SC1090
  . "$STATE"
  arm_redirect
  step "advancing served main to HEAD"
  git -C "$SERVE_REPO" update-ref refs/heads/main "$HEAD_SHA"
  ok "serve.git main = $HEAD_SHA"

  step "updating via $UPDATE_METHOD"
  # The app must boot configured or the onboarding overlay (a fullscreen
  # div) eats every click: configure the mock inference server exactly like
  # the dev:mock flow does, so the app is genuinely configured.
  # shellcheck source=../install/e2e-assets/mock-provider.sh
  source "$ASSETS/mock-provider.sh"
  mock_start "$WORK_ROOT"
  trap mock_stop EXIT
  case "$UPDATE_METHOD" in
    hermes-update)
      # The CLI route a dmg user takes from a terminal. `--yes` reaches the
      # update subcommand only in later releases; ask the installed hermes.
      local hermes="$INSTALL_DIR/venv/bin/hermes"
      local update_cmd=("$hermes" update)
      if "$hermes" update --help 2>&1 | grep -qF -- --yes; then
        update_cmd=("$hermes" update --yes)
      fi
      local rc=0
      (cd "$INSTALL_DIR" && "${update_cmd[@]}" < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/update.log") || rc=$?
      log_group "hermes update transcript" "$LOG_DIR/update.log"
      [ "$rc" -eq 0 ] || fail "hermes update exited $rc; transcript above"
      ;;
    installer-script)
      # A dmg user re-running today's install one-liner.
      run_installer "$HEAD_SHA" head
      ;;
    installer-script+desktop)
      run_installer "$HEAD_SHA" head desktop
      # The desktop stage is this leg's claim: the rebuilt app must exist.
      head_app=""
      for cand in \
        "$INSTALL_DIR/apps/desktop/release/mac-arm64/Hermes.app" \
        "$INSTALL_DIR/apps/desktop/release/mac/Hermes.app"; do
        [ -d "$cand" ] && { head_app="$cand"; break; }
      done
      [ -n "$head_app" ] || fail "no built Hermes.app under the checkout after the +desktop update"
      ok "rebuilt app present: $head_app"
      ;;
    open-app-update)
      # The installed app IS the user surface here (double-click the .app);
      # hand-build the spec Playwright launches from. Env: the redirect set,
      # which is exactly what the app's children (git, hermes update) need.
      local app app_bin
      app="$(find_installed_app)" || fail "no installed app to launch"
      app_bin="$(find "$app/Contents/MacOS" -type f -perm +111 | head -1)"
      python3 - "$app_bin" "$WORK_ROOT/launch-spec.json" <<'PYEOF'
import json, os, sys
spec = {
    "argv": [sys.argv[1]],
    "cwd": os.path.dirname(sys.argv[1]),
    "env": dict(os.environ),
    "matchedShape": "packaged",
}
with open(sys.argv[2], "w") as fh:
    json.dump(spec, fh, indent=2)
PYEOF
      run_playwright_update "$WORK_ROOT/launch-spec.json"
      ;;
    hermes-desktop-app-update)
      # The product's own launch, captured at its spawn site.
      local hermes="$INSTALL_DIR/venv/bin/hermes"
      local spec="$WORK_ROOT/launch-spec.json"
      local rc=0
      (cd "$INSTALL_DIR" && \
        PYTHONPATH="$ASSETS/launch-capture${PYTHONPATH:+:$PYTHONPATH}" \
        HERMES_E2E_CAPTURE_LAUNCH="$spec" \
        "$hermes" desktop < /dev/null 2>&1 | ts_prefix > "$LOG_DIR/desktop-launch-capture.log") || rc=$?
      log_group "hermes desktop (launch capture) transcript" "$LOG_DIR/desktop-launch-capture.log"
      [ "$rc" -eq 0 ] || fail "hermes desktop exited $rc during launch capture"
      [ -f "$spec.captured" ] || fail "hermes desktop exited 0 but no launch was captured"
      ok "captured $(cat "$spec.captured") launch spec"
      run_playwright_update "$spec"
      ;;
  esac

  local got
  got="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
  [ "$got" = "$HEAD_SHA" ] || fail "checkout is $got, expected HEAD ($HEAD_SHA)"
  ok "checkout landed on HEAD ($HEAD_SHA)"
  "$INSTALL_DIR/venv/bin/hermes" --version 2>&1 | ts_prefix > "$LOG_DIR/version-head.log" \
    || fail "hermes --version failed after update"
  ok "hermes --version works post-update"
  step "PASS: $OLD_REF -> HEAD via $UPDATE_METHOD"
}

case "$PHASE" in
  stage)   phase_stage ;;
  install) phase_install ;;
  update)  phase_update ;;
  all)     phase_stage; phase_install; phase_update ;;
  *) echo "error: --phase must be stage, install, update or all" >&2; exit 1 ;;
esac
