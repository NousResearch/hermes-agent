#!/usr/bin/env bash
# Run a command in a disposable, network-isolated fake Internet.
#
# The command runs in private user, mount, PID, and network namespaces. This
# script is stage 1: it builds the sandbox tree, mints the fake CA, and creates
# the user+network namespaces with `unshare` (see the namespace plan further
# down), then re-execs into scripts/sandbox/stage2-run.sh, which adds the
# mount/pid namespaces with bubblewrap and runs the payload. Its only writable
# filesystem is SANDBOX_ROOT. HTTP(S) goes to a local static MITM proxy;
# github.com SSH uses a sandbox-local git-upload-pack shim; neither transport
# can reach the host network.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper files the sandbox needs: the stage-2 script it re-execs into, plus the
# files it copies in (the fake-internet proxy, the ssh shim, the openssl config).
# They sit next to this script in the repo, but the Nix wrapper installs the
# script into the store on its own, so it exports DEV_SANDBOX_ASSETS to point
# here.
SANDBOX_ASSETS="${DEV_SANDBOX_ASSETS:-$SCRIPT_DIR/sandbox}"
for asset in proxy.py ssh-shim.sh openssl.cnf stage2-run.sh; do
  [ -f "$SANDBOX_ASSETS/$asset" ] || {
    echo "error: missing sandbox asset: $SANDBOX_ASSETS/$asset" >&2
    exit 1
  }
done

print_help() {
  cat <<'EOF'
Usage: dev-sandbox.sh [options] [--] <command...>
       dev-sandbox.sh install [options] [--] [installer arguments...]

Run COMMAND in a throwaway chroot-like bubblewrap sandbox. The sandbox has no
writable host mounts: only its own root, mounted at /work, is writable.

Options:
  --persistent          Keep the whole sandbox under .hermes-sandbox/.
  --delete              Delete the persistent sandbox (asks first).
  --root                Install as uid 0 with the root FHS layout: code in
                        /usr/local/lib/hermes-agent, command in
                        /usr/local/bin. Default is the user-level layout.
  --from DIR            One-time copy of DIR into the sandbox's $HOME.
                        Existing persistent sandboxes are never overwritten.
  --http-root DIR       Copy DIR into the fake web server root for this run.
                        Requests map to DIR/<host>/<path>; no URL is forwarded.
  --installer PATH      With `install`, serve PATH at the canonical install.sh
                        URL. Default: scripts/install.sh in this worktree.
  --from-main           With `install`, fetch the real upstream main installer
                        and repository, then advance fake main to this folder
                        after a successful install for update testing.
                        Shorthand for --install-ref refs/heads/main.
  --install-ref REF     Like --from-main, but installs REF instead of main:
                        a branch, a tag (v2026.7.7), or a SHA reachable from main.
                        Use it to test updating from an older release, not just
                        from the tip.
  -h, --help            Show this help.

Option order matters: every option above is consumed by THIS script, and
parsing stops at the first argument it does not recognize. Everything from
that point on is passed through to the command (or, with `install`, to the
installer). Put sandbox options first and separate installer arguments with
`--`, otherwise they arrive here and fail:

  # WRONG — --from-main reaches install.sh, which rejects it
  scripts/dev-sandbox.sh install --skip-setup --from-main

  # RIGHT
  scripts/dev-sandbox.sh install --from-main -- --skip-setup

Install layout: `install.sh` picks its layout from `id -u` alone, so uid is what
separates the two real-world Linux installs. By default the sandbox runs as an
unprivileged `hermes` user, giving the layout most people have —
$HERMES_HOME/hermes-agent plus a ~/.local/bin launcher. Pass --root for the FHS
one. Both are worth testing; they differ in more than paths (root also relocates
uv's Python to /usr/local/share for world-readability).

The fake web server signs certificates with a CA trusted only inside this
sandbox. HTTP_PROXY/HTTPS_PROXY send fixture URLs there first; other HTTP(S)
requests pass through the sandbox's rootless outbound network. SSH to github.com
runs a sandbox-local upload-pack shim, never your SSH config, agent,
known-hosts file, or authorized keys.

Fake github main always comes from this folder. If it has staged, unstaged, or
non-ignored untracked changes, the sandbox warns and creates a temporary local
commit containing them; it never stages or commits the real worktree.

Environment:
  HERMES_DEV_SANDBOX_DIR    Sandbox directory name, relative to the repo root
                            (default: .hermes-sandbox).

Examples:
  # create a sandbox, install this branch as `main`, and then drop to a shell,
  # skipping `hermes setup` & the browser tools for speed.
  scripts/dev-sandbox.sh install --persistent -- --skip-setup --skip-browser

  # Install the official upstream main. You're dropped into a shell where
  # you can run `hermes update`.
  scripts/dev-sandbox.sh install --persistent --from-main

EOF
}

PERSISTENT=false
DELETE=false
RUN_AS_USER=true
SEED_DIR=""
HTTP_ROOT=""
INSTALL_SHORTCUT=false
INSTALLER_PATH=""
# Which upstream commit the sandbox installs before the update routes run.
# Empty means "install this worktree's own installer" (no upstream fetch); set,
# it is anything git can resolve -- a branch, a tag (v2026.7.7), or a SHA
# reachable from main -- so "can a user two releases back still update?" is
# expressible. --from-main is shorthand for refs/heads/main.
INSTALL_REF=""
UPSTREAM_URL="${HERMES_DEV_SANDBOX_UPSTREAM:-https://github.com/NousResearch/hermes-agent.git}"

if [ "${1:-}" = install ]; then
  INSTALL_SHORTCUT=true
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --persistent) PERSISTENT=true; shift ;;
    --delete) DELETE=true; shift ;;
    --root) RUN_AS_USER=false; shift ;;
    --user) RUN_AS_USER=true; shift ;;   # the default; accepted for symmetry
    --from)
      [ "$#" -ge 2 ] || { echo 'error: --from needs a directory' >&2; exit 1; }
      SEED_DIR="$2"; shift 2 ;;
    --http-root)
      [ "$#" -ge 2 ] || { echo 'error: --http-root needs a directory' >&2; exit 1; }
      HTTP_ROOT="$2"; shift 2 ;;
    --installer)
      [ "$#" -ge 2 ] || { echo 'error: --installer needs a file' >&2; exit 1; }
      INSTALLER_PATH="$2"; shift 2 ;;
    --from-main) INSTALL_REF="refs/heads/main"; shift ;;
    --install-ref)
      [ "$#" -ge 2 ] || { echo 'error: --install-ref needs a value' >&2; exit 1; }
      INSTALL_REF="$2"
      shift 2 ;;
    --from=*|--http-root=*|--installer=*|--install-ref=*)
      key="${1%%=*}"; value="${1#*=}"
      [ -n "$value" ] || { echo "error: $key needs a value" >&2; exit 1; }
      case "$key" in
        --from) SEED_DIR="$value" ;;
        --http-root) HTTP_ROOT="$value" ;;
        --installer) INSTALLER_PATH="$value" ;;
        --install-ref) INSTALL_REF="$value" ;;
      esac
      shift ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [ "$INSTALL_SHORTCUT" = false ] && [ "$#" -eq 0 ]; then
  print_help >&2
  exit 1
fi

if [ -n "$INSTALLER_PATH" ] && [ "$INSTALL_SHORTCUT" = false ]; then
  echo 'error: --installer is only valid with the install shortcut' >&2
  exit 1
fi
if [ -n "$INSTALL_REF" ] && [ "$INSTALL_SHORTCUT" = false ]; then
  echo 'error: --from-main / --install-ref are only valid with the install shortcut' >&2
  exit 1
fi
if [ -n "$INSTALL_REF" ] && [ -n "$INSTALLER_PATH" ]; then
  echo 'error: --from-main / --install-ref cannot be combined with --installer' >&2
  exit 1
fi

for dir in "$SEED_DIR" "$HTTP_ROOT"; do
  [ -z "$dir" ] || [ -d "$dir" ] || { echo "error: directory '$dir' does not exist" >&2; exit 1; }
done

GIT_ROOT="${HERMES_SANDBOX_SOURCE_ROOT:-$(git rev-parse --show-toplevel)}"
GIT_ROOT="$(cd "$GIT_ROOT" && pwd)"
if [ "$INSTALL_SHORTCUT" = true ] && [ -z "$INSTALL_REF" ] && [ -z "$INSTALLER_PATH" ]; then
  INSTALLER_PATH="$GIT_ROOT/scripts/install.sh"
fi
if [ -n "$INSTALLER_PATH" ] && [ ! -f "$INSTALLER_PATH" ]; then
  echo "error: installer '$INSTALLER_PATH' does not exist" >&2
  exit 1
fi
COMMIT="$(git -C "$GIT_ROOT" rev-parse --verify 'HEAD^{commit}')" || {
  echo "error: current folder has no HEAD commit" >&2
  exit 1
}
SANDBOX_DIR_NAME="${HERMES_DEV_SANDBOX_DIR:-.hermes-sandbox}"
PERSISTENT_ROOT="$GIT_ROOT/$SANDBOX_DIR_NAME"

if [ "$DELETE" = true ]; then
  if [ ! -d "$PERSISTENT_ROOT" ]; then
    echo "[sandbox] nothing to delete at $PERSISTENT_ROOT" >&2
    exit 0
  fi
  read -r -p "[sandbox] delete $PERSISTENT_ROOT? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) rm -rf -- "$PERSISTENT_ROOT" ;;
    *) echo '[sandbox] aborted' >&2; exit 1 ;;
  esac
  exit 0
fi

if [ "$PERSISTENT" = true ]; then
  SANDBOX_ROOT="$PERSISTENT_ROOT"
else
  SANDBOX_ROOT="$(mktemp -d -t hermes-sandbox.XXXXXX)"
  cleanup() { chmod -R u+w "$SANDBOX_ROOT"; rm -rf -- "$SANDBOX_ROOT"; }
  trap cleanup EXIT INT TERM
fi

mkdir -p "$SANDBOX_ROOT"/{root,home,etc}
UPSTREAM_REPO=""
UPSTREAM_COMMIT=""
if [ -n "$INSTALL_REF" ]; then
  echo "[sandbox] fetching upstream $INSTALL_REF for installer/update test" >&2
  UPSTREAM_REPO="$(mktemp -d -t hermes-sandbox-upstream.XXXXXX)"
  git -C "$UPSTREAM_REPO" init -q
  # Fetch the ref as given. A branch or tag name resolves on its own; a raw SHA
  # needs the remote to allow fetching it directly, so fall back to fetching
  # main and resolving the SHA locally (which works for any commit that is an
  # ancestor of main -- the interesting case for "update from N versions ago").
  #
  # Peel to ^{commit} in both cases: an annotated tag fetches as a tag OBJECT,
  # and using it directly fails later with "trying to write non-commit object
  # ... to branch 'refs/heads/main'".
  if git -C "$UPSTREAM_REPO" fetch -q "$UPSTREAM_URL" "$INSTALL_REF" 2>/dev/null; then
    UPSTREAM_COMMIT="$(git -C "$UPSTREAM_REPO" rev-parse "FETCH_HEAD^{commit}")"
  elif git -C "$UPSTREAM_REPO" fetch -q "$UPSTREAM_URL" refs/heads/main \
    && UPSTREAM_COMMIT="$(git -C "$UPSTREAM_REPO" rev-parse --verify -q "$INSTALL_REF^{commit}")"; then
    :
  else
    rm -rf -- "$UPSTREAM_REPO"
    echo "error: could not resolve upstream ref: $INSTALL_REF" >&2
    echo '       Use a branch (main), a tag (v2026.7.7), or a SHA reachable from main.' >&2
    exit 1
  fi
fi
if [ ! -e "$SANDBOX_ROOT/root/repo/.sandbox-source" ]; then
  mkdir -p "$SANDBOX_ROOT/root/repo"
  # Persistent roots live under the worktree, so copying with cp would recurse
  # into the sandbox itself. tar also lets us exclude a worktree's .git file,
  # which can point at the host's shared worktree metadata.
  tar -C "$GIT_ROOT" --exclude='./.git' --exclude="./$SANDBOX_DIR_NAME" -cf - . \
    | tar -C "$SANDBOX_ROOT/root/repo" -xf -
  : > "$SANDBOX_ROOT/root/repo/.sandbox-source"
fi

if [ -n "$SEED_DIR" ] && [ ! -e "$SANDBOX_ROOT/.seeded" ]; then
  echo "[sandbox] seeding home from $SEED_DIR" >&2
  cp -a "$SEED_DIR/." "$SANDBOX_ROOT/home/"
  : > "$SANDBOX_ROOT/.seeded"
fi

rm -rf "$SANDBOX_ROOT/root/http"
mkdir -p "$SANDBOX_ROOT/root/http"
if [ -n "$HTTP_ROOT" ]; then
  cp -a "$HTTP_ROOT/." "$SANDBOX_ROOT/root/http/"
fi
if [ "$INSTALL_SHORTCUT" = true ]; then
  mkdir -p "$SANDBOX_ROOT/root/http/hermes-agent.nousresearch.com"
  if [ -n "$INSTALL_REF" ]; then
    git -C "$UPSTREAM_REPO" show "$UPSTREAM_COMMIT:scripts/install.sh" \
      > "$SANDBOX_ROOT/root/http/hermes-agent.nousresearch.com/install.sh"
  else
    cp -a "$INSTALLER_PATH" "$SANDBOX_ROOT/root/http/hermes-agent.nousresearch.com/install.sh"
  fi
  set -- bash -c '
    set +e
    curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- "$@"
    install_status=$?
    if [ "$install_status" -eq 0 ] && [ -f /work/promote-main ]; then
      next_main=$(cat /work/promote-main)
      if git --git-dir=/work/repos/hermes-agent.git update-ref refs/heads/main "$next_main"; then
        rm -f /work/promote-main
        printf "[sandbox] fake main advanced to this folder for update testing\n" >&2
      else
        printf "[sandbox] failed to advance fake main after install\n" >&2
        install_status=1
      fi
    fi
    if [ "$DEV_SANDBOX_INTERACTIVE" = true ]; then
      printf "\n[sandbox] installer exited %s; entering sandbox shell\n" "$install_status" >&2
      exec </dev/tty >/dev/tty 2>&1
      exec bash -i
    fi
    exit "$install_status"
  ' sandbox-installer "$@"
fi

mkdir -p "$SANDBOX_ROOT/root"/{bin,certs,lib64,logs,repos,ssh,usr/bin,usr/local}
REAL_CA_CERT="${DEV_SANDBOX_REAL_CA_CERT:-}"
if [ -z "$REAL_CA_CERT" ]; then
  for candidate in /etc/ssl/certs/ca-certificates.crt /etc/ssl/cert.pem; do
    if [ -f "$candidate" ]; then
      REAL_CA_CERT="$candidate"
      break
    fi
  done
fi
if [ ! -f "$REAL_CA_CERT" ]; then
  echo 'error: no system CA bundle found for outbound sandbox HTTPS' >&2
  exit 1
fi
if [ ! -f "$SANDBOX_ROOT/root/certs/real-ca.pem" ]; then
  cp "$REAL_CA_CERT" "$SANDBOX_ROOT/root/certs/real-ca.pem"
fi
printf 'nameserver 10.0.2.3\n' > "$SANDBOX_ROOT/etc/resolv.conf"
SANDBOX_SHELL="$(command -v bash)"
DYNAMIC_LINKER="${DEV_SANDBOX_DYNAMIC_LINKER:-}"
if [ -z "$DYNAMIC_LINKER" ]; then
  # Nix store first: NixOS also ships a /lib64/ld-linux-x86-64.so.2 compat stub,
  # so probing FHS paths first would quietly switch which loader a bare script
  # invocation uses on this host. Globs that match nothing expand to themselves,
  # so every candidate is -f tested. The FHS paths cover Debian/Ubuntu (where
  # the loader is under /lib64 or a multiarch /lib dir), which is what CI runs.
  for candidate in \
    /nix/store/*-glibc-*/lib/ld-linux-*.so.* \
    /lib64/ld-linux-x86-64.so.2 \
    /lib/ld-linux-aarch64.so.1 \
    /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 \
    /lib/aarch64-linux-gnu/ld-linux-aarch64.so.1
  do
    if [ -f "$candidate" ]; then
      DYNAMIC_LINKER="$candidate"
      break
    fi
  done
fi
if [ ! -f "$DYNAMIC_LINKER" ]; then
  echo 'error: no glibc dynamic linker found for sandboxed release binaries' >&2
  echo '       Set DEV_SANDBOX_DYNAMIC_LINKER to its path.' >&2
  exit 1
fi
ln -sf "$SANDBOX_SHELL" "$SANDBOX_ROOT/root/bin/sh"
ln -sf "$(command -v ls)" "$SANDBOX_ROOT/root/bin/ls"
ln -sf "$(command -v env)" "$SANDBOX_ROOT/root/usr/bin/env"
ln -sf "$DYNAMIC_LINKER" "$SANDBOX_ROOT/root/lib64/$(basename "$DYNAMIC_LINKER")"
# Identity inside the sandbox. install.sh chooses its layout from `id -u`
# alone (see resolve_install_layout), so the uid here is what decides between
# the root FHS install and a user-level one.
if [ "$RUN_AS_USER" = true ]; then
  SANDBOX_UID=1000
  SANDBOX_GID=1000
  SANDBOX_USER=hermes
  SANDBOX_HOME=/home/hermes
else
  SANDBOX_UID=0
  SANDBOX_GID=0
  SANDBOX_USER=root
  SANDBOX_HOME=/root
fi
{
  printf 'root:x:0:0:Sandbox Root:/root:%s\n' "$SANDBOX_SHELL"
  if [ "$RUN_AS_USER" = true ]; then
    printf '%s:x:%s:%s:Sandbox User:%s:%s\n' \
      "$SANDBOX_USER" "$SANDBOX_UID" "$SANDBOX_GID" "$SANDBOX_HOME" "$SANDBOX_SHELL"
  fi
} > "$SANDBOX_ROOT/etc/passwd"
{
  printf 'root:x:0:\n'
  if [ "$RUN_AS_USER" = true ]; then
    printf '%s:x:%s:\n' "$SANDBOX_USER" "$SANDBOX_GID"
  fi
} > "$SANDBOX_ROOT/etc/group"
# A user-level install writes the `hermes` launcher to ~/.local/bin and the
# checkout to $HERMES_HOME; both live under the sandbox HOME, which is bound
# from $SANDBOX_ROOT/home. bwrap maps our real uid to $SANDBOX_UID, so the
# host-side ownership of that directory is what the sandbox sees as its own.
printf 'hosts: files dns\n' > "$SANDBOX_ROOT/etc/nsswitch.conf"
printf '127.0.0.1 localhost\n' > "$SANDBOX_ROOT/etc/hosts"

SOURCE_REPO="$GIT_ROOT"
SOURCE_REF="$COMMIT"
SNAPSHOT_REPO=""
FAKE_REPO="$SANDBOX_ROOT/root/repos/hermes-agent.git"
git -C "$SANDBOX_ROOT/root/repos" init --bare -q hermes-agent.git
if [ -n "$INSTALL_REF" ]; then
  git --git-dir="$FAKE_REPO" fetch -q --force "$UPSTREAM_REPO" \
    "$UPSTREAM_COMMIT:refs/heads/main"
fi
if [ -n "$(git -C "$GIT_ROOT" status --porcelain)" ]; then
  echo '[sandbox] warning: current folder is dirty; creating a temporary fake commit for main' >&2
  SNAPSHOT_REPO="$(mktemp -d -t hermes-sandbox-snapshot.XXXXXX)"
  git -C "$SNAPSHOT_REPO" init -q
  git -C "$SNAPSHOT_REPO" fetch -q "$GIT_ROOT" "$COMMIT"
  git -C "$SNAPSHOT_REPO" config user.name 'Hermes sandbox'
  git -C "$SNAPSHOT_REPO" config user.email 'sandbox@invalid'
  GIT_DIR="$SNAPSHOT_REPO/.git" GIT_WORK_TREE="$GIT_ROOT" git read-tree "$COMMIT"
  GIT_DIR="$SNAPSHOT_REPO/.git" GIT_WORK_TREE="$GIT_ROOT" \
    git add -A -- .
  SNAPSHOT_TREE="$(GIT_DIR="$SNAPSHOT_REPO/.git" git write-tree)"
  SNAPSHOT_PARENT="$COMMIT"
  if EXISTING_MAIN="$(git --git-dir="$FAKE_REPO" rev-parse --verify refs/heads/main 2>/dev/null)"; then
    git -C "$SNAPSHOT_REPO" fetch -q "$FAKE_REPO" "$EXISTING_MAIN"
    SNAPSHOT_PARENT="$EXISTING_MAIN"
  fi
  SOURCE_REF="$(GIT_DIR="$SNAPSHOT_REPO/.git" git commit-tree "$SNAPSHOT_TREE" -p "$SNAPSHOT_PARENT" \
    -m 'sandbox snapshot of dirty worktree')"
  SOURCE_REPO="$SNAPSHOT_REPO"
fi

if [ -n "$INSTALL_REF" ]; then
  git --git-dir="$FAKE_REPO" fetch -q --force "$SOURCE_REPO" \
    "$SOURCE_REF:refs/hermes-sandbox/next"
  printf '%s\n' "$SOURCE_REF" > "$SANDBOX_ROOT/root/promote-main"
else
  git --git-dir="$FAKE_REPO" fetch -q --force "$SOURCE_REPO" \
    "$SOURCE_REF:refs/heads/main"
fi
git --git-dir="$FAKE_REPO" symbolic-ref HEAD refs/heads/main
if [ -n "$SNAPSHOT_REPO" ]; then
  # Best-effort: it is a mktemp directory the OS will reap, and failing the whole
  # run over a leftover object file would be worse than leaking it. Concurrent
  # git activity in the worktree can still be writing here as we delete.
  rm -rf -- "$SNAPSHOT_REPO" 2>/dev/null || true
fi
if [ -n "$UPSTREAM_REPO" ]; then
  rm -rf -- "$UPSTREAM_REPO"
fi

# openssl reads a config even for `req -addext`, and its compiled-in path is a
# symlink into /etc/ssl on Debian/Ubuntu -- which the sandbox replaces. Ship our
# own and point OPENSSL_CONF at it, both here and inside the sandbox.
cp "$SANDBOX_ASSETS/openssl.cnf" "$SANDBOX_ROOT/root/certs/openssl.cnf"

if [ ! -f "$SANDBOX_ROOT/root/certs/ca.pem" ]; then
  if ! ca_error="$(OPENSSL_CONF="$SANDBOX_ROOT/root/certs/openssl.cnf" \
    openssl req -x509 -newkey rsa:2048 -nodes -days 2 \
    -subj '/CN=Hermes dev sandbox CA' \
    -extensions sandbox_ca_ext \
    -keyout "$SANDBOX_ROOT/root/certs/ca.key" \
    -out "$SANDBOX_ROOT/root/certs/ca.pem" 2>&1 >/dev/null)"; then
    echo 'error: could not create the sandbox CA:' >&2
    printf '%s\n' "$ca_error" >&2
    exit 1
  fi
fi
GIT_UPLOAD_PACK="$(command -v git-upload-pack)"
sed "s|@GIT_UPLOAD_PACK@|$GIT_UPLOAD_PACK|" "$SANDBOX_ASSETS/ssh-shim.sh" \
  > "$SANDBOX_ROOT/root/usr/bin/ssh"
chmod 700 "$SANDBOX_ROOT/root/usr/bin/ssh"

# The fake-internet proxy and the ssh shim are real files under
# scripts/sandbox/ rather than heredocs, so they can be linted, syntax-checked
# and diffed like any other source. Copy them into the sandbox tree.
cp "$SANDBOX_ASSETS/proxy.py" "$SANDBOX_ROOT/root/proxy.py"

if [ -n "$INSTALL_REF" ]; then
  echo "[sandbox] fake main: upstream $INSTALL_REF ($UPSTREAM_COMMIT)" >&2
  echo "[sandbox] prepared update: current folder ($SOURCE_REF)" >&2
else
  echo "[sandbox] fake main: current folder ($SOURCE_REF)" >&2
fi
echo "[sandbox] root: $SANDBOX_ROOT" >&2
echo "[sandbox] http root: $SANDBOX_ROOT/root/http" >&2
if [ "$RUN_AS_USER" = true ]; then
  echo "[sandbox] identity: $SANDBOX_USER (uid $SANDBOX_UID) — installs are user-level under $SANDBOX_HOME" >&2
else
  echo '[sandbox] identity: root (uid 0) — installs use the /usr/local FHS layout' >&2
fi
[ "$PERSISTENT" = true ] && echo '[sandbox] persistent' >&2 || echo '[sandbox] ephemeral' >&2

for command in awk bash bwrap curl git openssl python3 slirp4netns tar unshare; do
  command -v "$command" >/dev/null || {
    echo "error: missing required command: $command" >&2
    exit 1
  }
done

INTERACTIVE=false
if [ -t 0 ] && [ -t 1 ]; then
  INTERACTIVE=true
fi
NODE_DIR="${DEV_SANDBOX_NODE_DIR:-}"
if [ -z "$NODE_DIR" ] && command -v node >/dev/null; then
  NODE_DIR="$(dirname "$(dirname "$(command -v node)")")"
fi
WAYLAND_SOCKET=""
if [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -n "${WAYLAND_DISPLAY:-}" ] \
  && [ -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]; then
  WAYLAND_SOCKET="$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY"
fi

# Namespace plan (stage 1 -> stage 2).
#
# slirp4netns joins the target's userns and setuids to root before configuring
# the netns, so the userns MUST map a uid 0. bwrap's own --unshare-user maps
# exactly one uid, so it cannot both run the payload as uid 1000 and offer slirp
# a root to become: that combination fails with
# setns(CLONE_NEWNET): Operation not permitted.
#
# So stage 1 builds the namespaces here with two ranges:
#   inner 0    <- a subuid, unused by the payload, present only so slirp can
#                become root inside the namespace
#   inner $SANDBOX_UID <- our real host uid, so everything the sandbox writes
#                stays owned by us and `rm -rf` on a persistent sandbox needs
#                no privileges or chown dance
# The payload then runs in stage 2, where bwrap adds the mount/pid namespaces
# without creating a userns at all.
#
# The root layout needs no subuid at all: inner 0 IS the host uid there.
netns_args=(--user --net)
if [ "$RUN_AS_USER" = true ]; then
  host_user="$(id -un)"
  subuid_base="$(awk -F: -v u="$host_user" '$1 == u {print $2; exit}' /etc/subuid)"
  subgid_base="$(awk -F: -v u="$host_user" '$1 == u {print $2; exit}' /etc/subgid)"
  if [ -z "$subuid_base" ] || [ -z "$subgid_base" ]; then
    echo "error: no /etc/subuid or /etc/subgid range for $host_user" >&2
    echo '       A user-level sandbox needs one spare subordinate id to host' >&2
    echo "       its internal root. Add e.g. '$host_user:100000:65536' to both," >&2
    echo '       or use --root.' >&2
    exit 1
  fi
  netns_args+=(
    --map-users="0:$subuid_base:1" --map-users="$SANDBOX_UID:$(id -u):1"
    --map-groups="0:$subgid_base:1" --map-groups="$SANDBOX_GID:$(id -g):1"
  )
else
  netns_args+=(--map-root-user)
fi

sandbox_pid_file="$SANDBOX_ROOT/root/logs/sandbox.pid"
slirp_ready="$SANDBOX_ROOT/root/logs/slirp.ready"
slirp_log="$SANDBOX_ROOT/root/logs/slirp.log"
: > "$sandbox_pid_file"
: > "$slirp_ready"

env \
  DEV_SANDBOX_ROOT="$SANDBOX_ROOT" \
  DEV_SANDBOX_BASH="$(command -v bash)" \
  DEV_SANDBOX_REAL_CA_CERT="$REAL_CA_CERT" \
  DEV_SANDBOX_INTERACTIVE="$INTERACTIVE" \
  DEV_SANDBOX_USER="$SANDBOX_USER" \
  DEV_SANDBOX_HOME="$SANDBOX_HOME" \
  DEV_SANDBOX_NODE_DIR="$NODE_DIR" \
  DEV_SANDBOX_ELECTRON_LD_LIBRARY_PATH="${DEV_SANDBOX_ELECTRON_LD_LIBRARY_PATH:-}" \
  DEV_SANDBOX_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
  DEV_SANDBOX_WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  DEV_SANDBOX_WAYLAND_SOCKET="$WAYLAND_SOCKET" \
  unshare "${netns_args[@]}" \
    "$SANDBOX_ASSETS/stage2-run.sh" "$@" &
sandbox_launcher=$!

for _ in $(seq 1 200); do
  [ -s "$sandbox_pid_file" ] && break
  if ! kill -0 "$sandbox_launcher" 2>/dev/null; then
    wait "$sandbox_launcher"
    exit $?
  fi
  sleep 0.05
done
sandbox_pid="$(tr -dc '0-9' < "$sandbox_pid_file")"
if [ -z "$sandbox_pid" ]; then
  echo 'error: sandbox did not report its PID' >&2
  exit 1
fi

slirp4netns --configure --disable-host-loopback --ready-fd=3 \
  --userns-path="/proc/$sandbox_pid/ns/user" "$sandbox_pid" tap0 \
  3>"$slirp_ready" >"$slirp_log" 2>&1 &
slirp_pid=$!
cleanup_slirp() {
  kill "$slirp_pid" 2>/dev/null || true
  wait "$slirp_pid" 2>/dev/null || true
}
trap cleanup_slirp EXIT INT TERM

for _ in $(seq 1 200); do
  [ -s "$slirp_ready" ] && break
  if ! kill -0 "$slirp_pid" 2>/dev/null; then
    cat "$slirp_log" >&2 || true
    exit 1
  fi
  sleep 0.05
done
if [ ! -s "$slirp_ready" ]; then
  echo 'error: timed out waiting for sandbox network setup' >&2
  exit 1
fi

wait "$sandbox_launcher"
exit $?