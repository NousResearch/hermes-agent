#!/usr/bin/env bash
# Prove a user on some earlier commit can reach this one.
#
# Installs a real, earlier Hermes the way a user does, applies ONE update route,
# and requires the checkout to land on this commit with a working `hermes`.
#
# Nothing here is mocked. scripts/dev-sandbox.sh provides the fake Internet --
# a bubblewrap sandbox with no writable host mounts, a MITM proxy serving the
# canonical install.sh URL, and a git-upload-pack shim standing in for
# github.com -- so `install.sh` really installs uv, a managed Python, Node and
# the venv, cloning "github.com" over the ssh-first path a user hits.
#
# One route per run, on a sandbox built from scratch, because the routes are only
# meaningful from a pristine install. Sharing one install across routes -- or
# rewinding the checkout with `git reset --hard` between them -- leaves the
# second route running against a tree the first already updated (same venv, same
# installed console script, same __pycache__), which is not the state any real
# user is in: a route can then pass only because its predecessor did the work,
# and a failure in the first leaves the second exercising something undefined.
# If you add a route, give it its own run.
#
# Usage:
#   tests/install/install-update-e2e.sh --route update|installer
#                                       [--install-ref REF]
#                                       [--include-browser] [--keep]
#
#   --route           which update path to exercise (required):
#                       update     `hermes update`
#                       installer  re-running the curl one-liner over the checkout
#   --install-ref     what to install first; anything git resolves (a branch, a
#                     tag like v2026.7.7, or a SHA reachable from main).
#                     Default: refs/heads/main.
#   --include-browser keep Node/npm/browser installation in scope even when the
#                     sampled installer supports --skip-browser. The specialized
#                     installer acceptance lanes use this to exercise npm trust.
#
# Requires a CLEAN worktree: every dev-sandbox invocation re-derives fake main
# from the working copy, so uncommitted changes move the update target between
# the call that installs and the call that verifies.

set -euo pipefail

ROUTE=""
INSTALL_REF="refs/heads/main"
INCLUDE_BROWSER=false
KEEP=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --route)
      [ "$#" -ge 2 ] || { echo 'error: --route needs a value' >&2; exit 1; }
      ROUTE="$2"; shift 2 ;;
    --install-ref)
      [ "$#" -ge 2 ] || { echo 'error: --install-ref needs a value' >&2; exit 1; }
      INSTALL_REF="$2"; shift 2 ;;
    --include-browser) INCLUDE_BROWSER=true; shift ;;
    --keep) KEEP=true; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$ROUTE" in
  update|installer) ;;
  '') echo 'error: --route is required (update or installer)' >&2; exit 1 ;;
  *) echo "error: unknown route: $ROUTE (want update or installer)" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Keep sandbox state out of the default .hermes-sandbox so a run never clobbers
# a developer's own sandbox, and scope it per route so two routes can run
# concurrently (CI runs them as parallel matrix legs). dev-sandbox.sh joins this
# onto the worktree root and feeds it to `tar --exclude`, so it MUST be a
# relative directory name.
SANDBOX_DIR_NAME=".hermes-sandbox-e2e-$ROUTE"
export HERMES_DEV_SANDBOX_DIR="$SANDBOX_DIR_NAME"

SANDBOX_ROOT="$REPO_ROOT/$SANDBOX_DIR_NAME"
INSTALL_DIR="/home/hermes/.hermes/hermes-agent"   # user-level layout (sandbox default)
FAKE_REMOTE="/work/repos/hermes-agent.git"
# Only used to fetch an old install.sh for the flag probe below; the sandbox does
# its own fetching. Same override dev-sandbox.sh honours, so a fork can retarget
# both together.
UPSTREAM_URL="${HERMES_DEV_SANDBOX_UPSTREAM:-https://github.com/NousResearch/hermes-agent.git}"

# Installer transcripts live outside the sandbox root: the sandbox is recreated
# and (unless --keep) deleted, and these logs are the most useful artifact when
# a real install breaks. Created after the dirty check below, so that a log dir
# pointed inside the repo cannot be the thing that makes the tree dirty.
LOG_DIR="${HERMES_E2E_LOG_DIR:-$(mktemp -d -t hermes-install-e2e-logs.XXXXXX)}"

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# The sandbox's internal logs (fake-internet proxy, slirp) explain failures that
# happen BEFORE install.sh gets to say anything -- a TLS handshake the proxy
# rejected looks like a bare `curl: (35)` from outside. Copy them out where a CI
# artifact upload can find them, and echo the proxy log since it is the usual
# culprit.
collect_sandbox_logs() {
  # Separate `local` statements on purpose: a single `local a=$1 b="$a"` does
  # NOT see the earlier assignment, so under `set -u` the second expansion dies
  # with "a: unbound variable".
  local tag="$1"
  local src="$SANDBOX_ROOT/root/logs"
  local dest="$LOG_DIR/sandbox-$tag"
  [ -d "$src" ] || return 0
  mkdir -p "$dest"
  cp -a "$src/." "$dest/" 2>/dev/null || true
  # Print it, not just archive it: a rejected TLS handshake here is the whole
  # explanation for a failure that otherwise reads as a bare `curl: (35)`, and
  # whoever is reading the job log should not have to download an artifact to
  # see it. In full, not tailed -- the file is short, and the useful line is not
  # reliably at the end.
  if [ -s "$dest/proxy.log" ]; then
    echo "--- sandbox proxy.log ---" >&2
    cat "$dest/proxy.log" >&2
    echo "--- end proxy.log ---" >&2
  fi
}

# ── preflight ──────────────────────────────────────────────────────────────
# Prefer the `sandbox` wrapper from the Nix devShell: it supplies both the PATH
# (bwrap, slirp4netns, openssl, ...) and the DEV_SANDBOX_* variables the script
# needs -- notably DEV_SANDBOX_DYNAMIC_LINKER, without which it cannot find a
# glibc loader on NixOS. Off Nix, the script is the entry point and finds its
# dependencies on the system PATH.
if command -v sandbox >/dev/null 2>&1; then
  SANDBOX=(sandbox)
elif command -v bwrap >/dev/null 2>&1; then
  SANDBOX=("$REPO_ROOT/scripts/dev-sandbox.sh")
else
  fail 'no usable sandbox: enter the Nix devShell (for `sandbox`) or install bubblewrap'
fi

if [ -n "$(git status --porcelain)" ]; then
  printf '\033[1;31m✗ working tree is dirty:\033[0m\n' >&2
  git status --porcelain | sed 's/^/    /' >&2
  fail 'Every sandbox invocation re-snapshots the working copy into a new
  fake-main commit, so the update target would move mid-run. Commit or stash
  first. (If a path above is build or log output, it needs gitignoring or to
  live outside the repo.)'
fi

mkdir -p "$LOG_DIR"

if [ "$KEEP" = false ]; then
  trap 'rm -rf -- "$SANDBOX_ROOT"' EXIT INT TERM
fi
rm -rf -- "$SANDBOX_ROOT"

# ── helpers ────────────────────────────────────────────────────────────────
# Does the INSTALLED hermes accept FLAG on `hermes update`?
#
# Asked of the installed binary rather than parsed out of a release's source:
# the update subcommand has lived in main.py, subcommands/update.py, and
# update_cmd.py across the releases we sample, so any static parse is a guess
# that silently rots. `hermes update --help` is the same surface a user meets,
# and argparse prints every option it accepts.
update_supports() {
  local flag="$1"
  local help=""
  help="$(in_sandbox "hermes update --help 2>&1")" || return 1
  # Do not pipe a producer into grep -q under pipefail: grep exits as soon as it
  # finds FLAG, the producer can receive SIGPIPE, and a supported flag is then
  # misreported as absent.
  grep -qF -- "$flag" <<<"$help"
}

# Does the installer at REF accept FLAG? Read it out of that ref's own
# install.sh rather than assuming this checkout's flag set: the point of the
# matrix is to install releases from months back, whose installers predate
# options we take for granted. (Unlike the updater, the installer runs before
# anything is installed, so there is no --help to ask yet.)
#
# The ref may not be local -- the sandbox does its own fetching -- so fall back
# to fetching just that blob. Unresolvable means "flag absent", which costs a
# more conservative invocation, never a wrong one.
installer_supports() {
  local ref="$1"
  local flag="$2"
  local script=""
  script="$(git show "$ref:scripts/install.sh" 2>/dev/null)" || {
    git fetch -q --depth 1 "$UPSTREAM_URL" "$ref" 2>/dev/null || return 1
    script="$(git show FETCH_HEAD:scripts/install.sh 2>/dev/null)" || return 1
  }
  # A here-string keeps grep's early success from SIGPIPE-ing a pipeline
  # producer under `set -o pipefail`.
  grep -qF -- "$flag" <<<"$script"
}

# Run the real install one-liner inside the sandbox. `ref` non-empty installs
# that upstream commit and promotes THIS checkout to fake main afterwards,
# leaving the state a user is in when an update is waiting; empty serves this
# worktree's own installer and points fake main here.
install_in_sandbox() {
  local what="$1"
  local ref="$2"
  local tag="$3"
  local log="$LOG_DIR/$tag.log"
  local args=(install --persistent)
  [ -n "$ref" ] && args+=(--install-ref "$ref")
  # astral.sh and its release mirrors intermittently return empty replies to
  # GitHub runner egress IPs. Serve a runner-prefetched fixture through the same
  # sandbox proxy so external bootstrap availability cannot prevent the test
  # from reaching the Hermes-owned npm trust and keep-alive boundary.
  if [ -n "$UV_INSTALLER_FIXTURE" ]; then
    args+=(--http-root "$UV_INSTALLER_FIXTURE")
  fi

  # Installer flags have to match the installer being run, not this checkout's.
  # Older releases reject options added later ("Unknown option: --skip-browser"),
  # and this test deliberately installs releases from months back. --skip-setup
  # goes back further than any tag we sample; anything newer is probed for.
  local installer_flags=(--skip-setup)
  if [ "$INCLUDE_BROWSER" = false ] \
      && { [ -z "$ref" ] || installer_supports "$ref" --skip-browser; }; then
    installer_flags+=(--skip-browser)
  fi
  # Sandbox flags must precede `--`; the rest goes to install.sh.
  args+=(-- "${installer_flags[@]}")

  # Stream the installer's output to stdout AND keep a copy on disk. It is the
  # substance of this test -- a real install of uv, a managed Python, Node and
  # the venv -- so it belongs in the job log where anyone reading the run can
  # see it, not only in an artifact they have to download. The file copy is what
  # the artifact upload keeps and what the failure paths grep.
  #
  # `set -o pipefail` is load-bearing here: without it the pipeline reports
  # tee's status and a failed install looks like a pass.
  local status=0
  "${SANDBOX[@]}" "${args[@]}" 2>&1 | tee "$log" || status=$?

  if [ "$status" -ne 0 ]; then
    collect_sandbox_logs "$tag"
    fail "$what failed (exit $status)"
  fi
  grep -q 'Installation Complete' "$log" \
    || { collect_sandbox_logs "$tag"; \
         fail "$what did not report a completed install"; }
  ok "$what completed (log: $log)"
}

in_sandbox() { "${SANDBOX[@]}" --persistent bash -lc "$1"; }

# fake main's SHA is read fresh whenever it is needed, never cached across a
# sandbox invocation: each invocation re-derives it from the worktree.
sandbox_target() { in_sandbox "git --git-dir=$FAKE_REMOTE rev-parse main" | tr -d '[:space:]'; }
sandbox_head()   { in_sandbox "cd $INSTALL_DIR && git rev-parse HEAD" | tr -d '[:space:]'; }

require_landed_on_target() {
  local what="$1" head target
  head="$(sandbox_head)"
  target="$(sandbox_target)"
  [ "$head" = "$target" ] || fail "$what left HEAD at $head, wanted $target"
  ok "$what landed on ${head:0:12}"
}

# The real smoke test: goes through the venv launcher and imports the app, so it
# fails if the venv, dependencies, or entry point are broken.
require_hermes_works() {
  local when="$1" out
  out="$(in_sandbox "hermes --version" 2>&1)" \
    || { printf '%s\n' "$out" >&2; fail "hermes --version failed $when"; }
  printf '%s\n' "$out" | sed 's/^/    /'
  ok "hermes runs $when"
}

# ── install the earlier Hermes ─────────────────────────────────────────────
# Prefetch the uv installer and binary so the sandbox can serve stable fixtures
# for third-party bootstrap bytes. This keeps the Hermes acceptance lane focused
# on Hermes-owned behavior while install.sh still performs a real uv/Python/Node
# and venv installation. Best-effort: if prefetching fails, the in-sandbox curl
# still attempts the public endpoints and the transcript remains authoritative.
UV_INSTALLER_FIXTURE=""
if command -v curl >/dev/null 2>&1; then
  UV_FIXTURE_DIR="$LOG_DIR/http-fixture"
  UV_SCRIPT="$UV_FIXTURE_DIR/astral.sh/uv/install.sh"
  mkdir -p "$UV_FIXTURE_DIR/astral.sh/uv"
  for _attempt in 1 2 3 4 5; do
    if curl -fsSL --retry 3 --retry-all-errors --retry-delay 2 \
        https://astral.sh/uv/install.sh -o "$UV_SCRIPT" 2>/dev/null \
        && [ -s "$UV_SCRIPT" ]; then
      UV_INSTALLER_FIXTURE="$UV_FIXTURE_DIR"
      break
    fi
    sleep 3
  done
fi

# Cache the architecture-matched uv tarball for both download hosts the
# installer tries. Unknown architectures retain only the script fixture and
# exercise the binary download upstream rather than serving the wrong artifact.
if [ -n "$UV_INSTALLER_FIXTURE" ]; then
  UV_VER="$(grep -om1 'releases/download/[0-9][0-9.]*' "$UV_SCRIPT" | cut -d/ -f3)"
  case "$(uname -m)" in
    x86_64|amd64) UV_TARGET="x86_64-unknown-linux-gnu" ;;
    aarch64|arm64) UV_TARGET="aarch64-unknown-linux-gnu" ;;
    *) UV_TARGET="" ;;
  esac
  if [ -n "$UV_VER" ] && [ -n "$UV_TARGET" ]; then
    _uv_rel="github/uv/releases/download/$UV_VER/uv-$UV_TARGET.tar.gz"
    mkdir -p "$UV_FIXTURE_DIR/releases.astral.sh/$(dirname "$_uv_rel")" \
             "$UV_FIXTURE_DIR/github.com/astral-sh/$(dirname "${_uv_rel#github/}")"
    if curl -fsSL --retry 3 --retry-all-errors --retry-delay 2 \
        "https://releases.astral.sh/$_uv_rel" \
        -o "$UV_FIXTURE_DIR/releases.astral.sh/$_uv_rel" 2>/dev/null; then
      cp "$UV_FIXTURE_DIR/releases.astral.sh/$_uv_rel" \
         "$UV_FIXTURE_DIR/github.com/astral-sh/${_uv_rel#github/}"
    else
      rm -f "$UV_FIXTURE_DIR/releases.astral.sh/$_uv_rel"
      echo "⚠ could not prefetch uv $UV_VER tarball; binary will fetch upstream" >&2
    fi
  fi
  ok "prefetched uv installer for sandbox fixture"
else
  echo "⚠ could not prefetch uv installer; install will fetch astral.sh directly" >&2
fi

step "installing upstream $INSTALL_REF (real curl | install.sh: uv, Python, Node, venv)"
install_in_sandbox "install of upstream $INSTALL_REF" "$INSTALL_REF" install

BASE="$(sandbox_head)"
TARGET="$(sandbox_target)"
[ -n "$BASE" ] || fail "could not read the installed commit"
[ "$BASE" != "$TARGET" ] \
  || fail "install landed on the update target ($BASE); base and target must differ"
ok "installed ${BASE:0:12}; update target is ${TARGET:0:12}"
require_hermes_works 'after install'

# ── apply exactly one update route ─────────────────────────────────────────
case "$ROUTE" in
  update)
    step 'ROUTE: hermes update'
    # `--yes` reaches the update subcommand only in later releases, and argparse
    # rejects the whole invocation when it does not exist. Ask the installed
    # hermes which it accepts; older ones read the prompt from stdin, so close it.
    if update_supports --yes; then
      update_cmd="hermes update --yes"
    else
      update_cmd="hermes update </dev/null"
    fi
    if ! in_sandbox "cd $INSTALL_DIR && $update_cmd"; then
      collect_sandbox_logs update
      fail "hermes update failed ($update_cmd)"
    fi
    require_landed_on_target 'hermes update'
    require_hermes_works 'after hermes update'
    ;;
  installer)
    step 'ROUTE: installer re-run over the existing checkout'
    # No ref: serves this worktree's installer and points fake main at this
    # checkout, which is what the re-run must land on.
    install_in_sandbox 'installer re-run' '' reinstall
    require_landed_on_target 'installer re-run'
    require_hermes_works 'after installer re-run'
    ;;
esac

printf '\n\033[1;32m✓ install/update E2E passed (route: %s, from: %s)\033[0m\n' \
  "$ROUTE" "$INSTALL_REF"
[ "$KEEP" = true ] && echo "  sandbox kept at $SANDBOX_ROOT"
exit 0
