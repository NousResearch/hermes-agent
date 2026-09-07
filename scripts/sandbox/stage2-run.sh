#!/usr/bin/env bash
# Stage 2 of the dev sandbox: build the mounts and run the payload.
#
# Not called directly. scripts/dev-sandbox.sh (stage 1) creates the user and
# network namespaces with `unshare` and re-execs into this script inside them,
# so by the time this runs we are already at the target uid with a private
# netns. bwrap therefore does NOT create a userns here -- it only adds the
# mount and pid namespaces. (`unshare --user` grants its creator full
# capabilities in the new userns regardless of which uid it maps, which is what
# lets bwrap mount as a non-root uid.)
#
# The whole interface with stage 1 is the DEV_SANDBOX_* environment, asserted
# below: there are no shared functions or variables between the two stages.
# Stage 1 locates this script alongside the other sandbox assets (see
# DEV_SANDBOX_ASSETS in dev-sandbox.sh), so the Nix wrapper's store copy and a
# plain repo checkout both work.

set -euo pipefail

: "${DEV_SANDBOX_ROOT:?missing DEV_SANDBOX_ROOT}"
: "${DEV_SANDBOX_BASH:?missing DEV_SANDBOX_BASH}"
: "${DEV_SANDBOX_INTERACTIVE:?missing DEV_SANDBOX_INTERACTIVE}"
: "${DEV_SANDBOX_USER:?missing DEV_SANDBOX_USER}"
: "${DEV_SANDBOX_HOME:?missing DEV_SANDBOX_HOME}"

# Announce our pid so stage 1 can point slirp4netns at these namespaces,
# then hold until it reports the network is up.
slirp_ready="$DEV_SANDBOX_ROOT/root/logs/slirp.ready"
printf '%s\n' "$$" > "$DEV_SANDBOX_ROOT/root/logs/sandbox.pid"
for _ in $(seq 1 200); do
  [ -s "$slirp_ready" ] && break
  sleep 0.05
done
if [ ! -s "$slirp_ready" ]; then
  echo 'error: timed out waiting for sandbox network setup' >&2
  cat "$DEV_SANDBOX_ROOT/root/logs/slirp.log" >&2 || true
  exit 1
fi

# The sandbox HOME is /root for a root install and /home/<user> for a
# user-level one. Only the latter needs its parent created first; --dir /
# is not a thing bwrap accepts.
home_mounts=()
home_parent="$(dirname "$DEV_SANDBOX_HOME")"
if [ "$home_parent" != / ]; then
  home_mounts+=(--dir "$home_parent")
fi
home_mounts+=(--bind "$DEV_SANDBOX_ROOT/home" "$DEV_SANDBOX_HOME")

node_env=()
if [ -n "${DEV_SANDBOX_NODE_DIR:-}" ]; then
  node_env+=(--setenv npm_config_nodedir "$DEV_SANDBOX_NODE_DIR")
fi
electron_env=()
if [ -n "${DEV_SANDBOX_ELECTRON_LD_LIBRARY_PATH:-}" ]; then
  electron_env+=(
    --setenv LD_LIBRARY_PATH "$DEV_SANDBOX_ELECTRON_LD_LIBRARY_PATH"
    --setenv HERMES_DESKTOP_DISABLE_GPU 1
  )
fi
gui_mounts=()
if [ -n "${DEV_SANDBOX_WAYLAND_SOCKET:-}" ]; then
  runtime_dir="${DEV_SANDBOX_XDG_RUNTIME_DIR:?missing DEV_SANDBOX_XDG_RUNTIME_DIR}"
  runtime_parent="$(dirname "$runtime_dir")"
  runtime_grandparent="$(dirname "$runtime_parent")"
  gui_mounts+=(
    --dir "$runtime_grandparent"
    --dir "$runtime_parent"
    --dir "$runtime_dir"
    --bind "$DEV_SANDBOX_WAYLAND_SOCKET" "$DEV_SANDBOX_WAYLAND_SOCKET"
    --setenv XDG_RUNTIME_DIR "$runtime_dir"
    --setenv WAYLAND_DISPLAY "${DEV_SANDBOX_WAYLAND_DISPLAY:?missing DEV_SANDBOX_WAYLAND_DISPLAY}"
  )
fi

# How the sandbox gets a usable runtime, and where its own shims go.
#
# On Nix, every binary lives under /nix/store, so the sandbox can own /bin,
# /lib64 and /usr/bin outright and fill them with symlinks into the store.
#
# Elsewhere the runtime IS /usr, /bin, /lib, /lib64 -- so binding the
# sandbox's near-empty versions over them hides the real thing, and bwrap
# dies with `execvp /usr/bin/bash: No such file or directory`. Keep the host
# directories read-only and override only the individual files we shim.
#
# The same answer decides how /etc is handled further down.
if [ -d /nix ] && [[ "$(readlink -f "$DEV_SANDBOX_BASH")" == /nix/* ]]; then
  USE_HOST_RUNTIME=false
else
  USE_HOST_RUNTIME=true
fi

runtime_mounts=()
shim_mounts=()
if [ "$USE_HOST_RUNTIME" = false ]; then
  runtime_mounts+=(--ro-bind /nix /nix)
  shim_mounts+=(
    --dir /usr
    --dir /bin
    --dir /lib64
    --bind "$DEV_SANDBOX_ROOT/root/bin" /bin
    --bind "$DEV_SANDBOX_ROOT/root/lib64" /lib64
    --bind "$DEV_SANDBOX_ROOT/root/usr/bin" /usr/bin
  )
else
  for path in /usr /bin /sbin /lib /lib64; do
    [ -e "$path" ] && runtime_mounts+=(--ro-bind "$path" "$path")
  done
  # The git-upload-pack shim standing in for github.com is the only file that
  # must beat the host's copy; sh/ls/env are already there for real.
  shim_mounts+=(--bind "$DEV_SANDBOX_ROOT/root/usr/bin/ssh" /usr/bin/ssh)
fi

# /etc: start from a copy of the host's and overwrite only the files we fake.
#
# Replacing the whole directory with a five-file one is the tempting shortcut
# and it is wrong: a distro puts things under /etc that binaries outside /etc
# depend on, so hiding all of it breaks tools that look fine on PATH. Two real
# examples, both Debian/Ubuntu: openssl's compiled-in openssl.cnf is a symlink
# into /etc/ssl, and /usr/bin/awk is a symlink to /etc/alternatives/awk -- with
# /etc replaced, openssl cannot mint a certificate and awk reports "not found".
# Those are two symptoms of one cause, and nothing says there are only two.
#
# Copying rather than mount-overlaying the individual files, because several of
# these are symlinks in the wild (resolv.conf -> ../run/systemd/... on Ubuntu,
# hosts and nsswitch.conf -> /etc/static/... on NixOS) and bwrap cannot bind a
# file onto a symlink whose target does not exist inside the sandbox.
#
# Symlinks are copied as symlinks, never dereferenced: on NixOS /etc/static
# points into the store and following it would copy gigabytes per sandbox. The
# store is already mounted at /nix on that path, and the host runtime dirs are
# mounted at their own paths, so absolute symlinks still resolve.
#
# The five we override, and why each must differ from the host's:
#   passwd, group     the sandbox identity, which does not exist on the host
#   resolv.conf       slirp4netns's DNS, not the host resolver
#   nsswitch.conf     files+dns only, so nothing consults host NSS modules
#   hosts             minimal, so no host entry leaks in
#
# os-release is removed rather than replaced. Installers branch on it to reach
# for a package manager -- `install.sh` reads ID from it and, on debian/ubuntu,
# offers to apt-get build tools, prompting on /dev/tty when sudo exists but is
# not passwordless. That prompt cannot be satisfied here (no terminal) and it is
# fatal under `set -e`. Inheriting the host's file would make the sandbox claim
# to be a distro whose package manager it cannot actually use; absent means
# DISTRO="unknown" and the apt path is skipped, which is the truth.
etc_mounts=()
if [ "$USE_HOST_RUNTIME" = true ] && [ -d /etc ]; then
  sandbox_etc="$DEV_SANDBOX_ROOT/etc-merged"
  rm -rf -- "$sandbox_etc"
  mkdir -p "$sandbox_etc"
  # -a keeps symlinks as symlinks; unreadable entries (shadow, sudoers) are
  # skipped rather than failing the run.
  cp -a /etc/. "$sandbox_etc/" 2>/dev/null || true
  for etc_file in passwd group resolv.conf nsswitch.conf hosts; do
    [ -f "$DEV_SANDBOX_ROOT/etc/$etc_file" ] || continue
    rm -f "$sandbox_etc/$etc_file"
    cp "$DEV_SANDBOX_ROOT/etc/$etc_file" "$sandbox_etc/$etc_file"
  done
  rm -f "$sandbox_etc/os-release" "$sandbox_etc/lsb-release"
  etc_mounts+=(--ro-bind "$sandbox_etc" /etc)
else
  etc_mounts+=(--bind "$DEV_SANDBOX_ROOT/etc" /etc)
fi

# /dev without a tty, so a script guarding on `[ -e /dev/tty ]` takes its
# no-terminal path.
#
# bwrap's --dev creates a /dev/tty NODE, but nothing in here has a controlling
# terminal, so opening it fails with "No such device or address". That is the
# worst of both: the guard passes and the read then fails. Under `set -e` --
# which install.sh uses -- a failed read inside a function aborts the whole
# installer, which is exactly how older releases died here while prompting for
# sudo to install ripgrep/ffmpeg.
#
# Making the tty real is not the fix: with an openable terminal that prompt
# blocks forever waiting for input nobody will type. Absent is what a headless
# machine looks like, and what every prompt in here should assume.
#
# --dev cannot be used with the node removed afterwards (bwrap refuses to mount
# a directory over a device node), so /dev is assembled explicitly.
dev_mounts=(
  --tmpfs /dev
  --dev-bind /dev/null /dev/null
  --dev-bind /dev/zero /dev/zero
  --dev-bind /dev/full /dev/full
  --dev-bind /dev/random /dev/random
  --dev-bind /dev/urandom /dev/urandom
  --symlink /proc/self/fd /dev/fd
  --symlink /proc/self/fd/0 /dev/stdin
  --symlink /proc/self/fd/1 /dev/stdout
  --symlink /proc/self/fd/2 /dev/stderr
)
if [ "$DEV_SANDBOX_INTERACTIVE" = true ]; then
  # An interactive shell is deliberately given a terminal; keep bwrap's /dev.
  dev_mounts=(--dev /dev)
fi

# Every payload TLS client must trust the SANDBOX CA: all HTTP(S) goes
# through the MITM proxy, which terminates each connection with a
# sandbox-CA-minted cert and re-validates upstream against the real CA
# bundle itself (real-ca.pem, passed to proxy.py below). Node reads
# NODE_EXTRA_CA_CERTS instead of SSL_CERT_FILE, so it gets ca.pem too --
# pointing it at the real bundle made npm reject the proxy's certs
# (UNABLE_TO_VERIFY_LEAF_SIGNATURE) and the proxy log the abort as an
# SSLEOFError. curl/git/openssl get the same ca.pem via their own vars.
# One asymmetry to note: the *_FILE vars replace the trust store, but
# NODE_EXTRA_CA_CERTS is additive -- Node keeps its built-in bundle --
# so a Node process that somehow bypassed the proxy could still reach
# public CAs directly. The *_PROXY vars are the enforcement; the CA
# vars are only the trust plumbing.

exec bwrap \
  --unshare-pid \
  --die-with-parent --proc /proc --tmpfs /tmp \
  "${dev_mounts[@]}" \
  "${gui_mounts[@]}" \
  "${runtime_mounts[@]}" \
  --bind "$DEV_SANDBOX_ROOT/root" /work \
  "${shim_mounts[@]}" \
  --bind "$DEV_SANDBOX_ROOT/root/usr/local" /usr/local \
  "${home_mounts[@]}" \
  "${etc_mounts[@]}" \
  --chdir /work/repo \
  --clearenv \
  --setenv PATH "$DEV_SANDBOX_HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH" \
  --setenv HOME "$DEV_SANDBOX_HOME" \
  --setenv USER "$DEV_SANDBOX_USER" \
  --setenv LOGNAME "$DEV_SANDBOX_USER" \
  --setenv CURL_CA_BUNDLE /work/certs/ca.pem \
  --setenv SSL_CERT_FILE /work/certs/ca.pem \
  --setenv GIT_SSL_CAINFO /work/certs/ca.pem \
  --setenv NODE_EXTRA_CA_CERTS /work/certs/ca.pem \
  --setenv OPENSSL_CONF /work/certs/openssl.cnf \
  --setenv HTTP_PROXY http://127.0.0.1:8080 \
  --setenv HTTPS_PROXY http://127.0.0.1:8080 \
  --setenv ALL_PROXY http://127.0.0.1:8080 \
  --setenv NO_PROXY '' \
  --setenv DEV_SANDBOX_INTERACTIVE "$DEV_SANDBOX_INTERACTIVE" \
  --setenv ELECTRON_DISABLE_SANDBOX 1 \
  "${node_env[@]}" \
  "${electron_env[@]}" \
  -- "$DEV_SANDBOX_BASH" -ceu '
    python3 /work/proxy.py /work/http /work/certs /work/certs/real-ca.pem >/work/logs/proxy.log 2>&1 &
    proxy_pid=$!
    cleanup() {
      kill "$proxy_pid" 2>/dev/null || true
      wait "$proxy_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    # Bash opens /dev/tcp itself, so the readiness probe needs no netcat --
    # one less binary the sandbox has to find on the host (GitHub runners
    # ship no `nc`).
    proxy_up() { (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null; }
    for _ in $(seq 1 100); do
      proxy_up && break
      sleep 0.05
    done
    if ! proxy_up; then
      echo "error: the sandbox fake-internet proxy never came up" >&2
      cat /work/logs/proxy.log >&2 || true
      exit 1
    fi
    "$@"
  ' sandbox-command "$@"
