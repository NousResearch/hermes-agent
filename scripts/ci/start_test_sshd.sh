#!/usr/bin/env bash
# Start a throwaway sshd container for the real SSH terminal-backend tests.
#
#   scripts/ci/start_test_sshd.sh <state-dir> [port]
#
# Generates (once per state dir) a client keypair and a host key, starts a
# Debian sshd on 127.0.0.1:<port> (default 2222) that accepts only that key for
# user `hermes`, waits until it answers, and prints the HERMES_TEST_SSH_*
# variables tests/tools/test_ssh_environment.py reads (appended to $GITHUB_ENV
# too when it is set). `docker rm -f hermes-test-sshd` stops it.
#
# The image is a plain Debian with GNU tar and bash, the shape of a real remote
# host (the backend's file sync uses `tar --no-overwrite-dir`). AcceptEnv lets
# the SendEnv passthrough test reach the remote shell.
set -euo pipefail

state_dir="${1:?usage: start_test_sshd.sh <state-dir> [port]}"
port="${2:-2222}"
image="${HERMES_TEST_SSHD_IMAGE:-debian:12-slim}"
name="hermes-test-sshd"

mkdir -p "$state_dir"
state_dir="$(cd "$state_dir" && pwd)"
[ -f "$state_dir/id_ed25519" ] || ssh-keygen -q -t ed25519 -N '' -C hermes-test -f "$state_dir/id_ed25519"
[ -f "$state_dir/host_ed25519" ] || ssh-keygen -q -t ed25519 -N '' -C hermes-test-host -f "$state_dir/host_ed25519"
cp "$state_dir/id_ed25519.pub" "$state_dir/authorized_keys"

docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" -p "127.0.0.1:${port}:22" \
  -v "$state_dir:/keys:ro" "$image" bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends openssh-server >/dev/null
    install -m 600 /keys/host_ed25519 /etc/ssh/ssh_host_ed25519_key
    install -m 644 /keys/host_ed25519.pub /etc/ssh/ssh_host_ed25519_key.pub
    useradd -m -s /bin/bash hermes
    usermod -p "*" hermes
    install -d -m 700 -o hermes -g hermes /home/hermes/.ssh
    install -m 600 -o hermes -g hermes /keys/authorized_keys /home/hermes/.ssh/authorized_keys
    printf "%s\n" "HostKey /etc/ssh/ssh_host_ed25519_key" "PasswordAuthentication no" \
      "AcceptEnv HERMES_RB_*" > /etc/ssh/sshd_config.d/hermes-test.conf
    mkdir -p /run/sshd
    exec /usr/sbin/sshd -D -e
  ' >/dev/null

for _ in $(seq 1 120); do
  if ssh -i "$state_dir/id_ed25519" -p "$port" -o BatchMode=yes -o ConnectTimeout=2 \
      -o StrictHostKeyChecking=accept-new hermes@127.0.0.1 true 2>/dev/null; then
    ready=1
    break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != true ]; then
    docker logs "$name" >&2 || true
    echo "sshd container exited before accepting connections" >&2
    exit 1
  fi
  sleep 1
done
if [ -z "${ready:-}" ]; then
  docker logs "$name" >&2 || true
  echo "sshd did not accept the test key within 120s" >&2
  exit 1
fi

env_lines="HERMES_TEST_SSH_HOST=127.0.0.1
HERMES_TEST_SSH_USER=hermes
HERMES_TEST_SSH_PORT=$port
HERMES_TEST_SSH_KEY=$state_dir/id_ed25519"
if [ -n "${GITHUB_ENV:-}" ]; then
  printf '%s\n' "$env_lines" >> "$GITHUB_ENV"
fi
printf '%s\n' "$env_lines"
