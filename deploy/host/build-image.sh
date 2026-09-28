#!/usr/bin/env bash
#
# build-image.sh: bake a versioned litco-agent matter-host snapshot on
# DigitalOcean. Runs on the operator's machine (doctl + ssh + scp), mirroring
# LitKit's deploy/fleet/build-worker-image.sh, but scripted end to end:
#
#   1. create a throwaway Ubuntu 24.04 builder droplet (with an account SSH key:
#      a droplet created without one gets an expired emailed root password that
#      blocks all non-interactive SSH)
#   2. copy deploy/host/install-host.sh up and run it with the pinned git ref
#   3. strip the builder's SSH key and run `cloud-init clean` (LOAD-BEARING: a
#      snapshot that keeps /var/lib/cloud first-boot state makes every droplet
#      created from it skip its user-data, so no env, no hostname, no service)
#   4. power off, snapshot as litco-agent-host-<version>, delete the builder
#
# The snapshot name IS the version, and the version is the git tag the image
# was built from (see deploy/host/README.md, "Versioning").
#
# Usage:
#   build-image.sh --version <tag> --ssh-key <fingerprint|id> [options]
#   build-image.sh --version <tag> --dry-run          # print the plan, touch nothing
#
# Options:
#   --version <tag>      required; snapshot is litco-agent-host-<tag>
#   --ref <git-ref>      litco-agent ref to install (default: the version tag)
#   --repo <url>         clone URL (default https://github.com/yavarb/litco-agent.git)
#   --region <slug>      default sfo3 (next to prod)
#   --size <slug>        builder size, default s-4vcpu-8gb
#   --base-image <slug>  default ubuntu-24-04-x64
#   --ssh-key <fp|id>    account SSH key for the builder (required unless --dry-run)
#   --tag <name>         droplet tag for the builder, default litco-host-builder; the
#                        cleanup trap finds a builder by this tag + name if the
#                        create call dies before it returns the droplet id
#   --dry-run            print every command in order and exit 0; runs nothing
#
# The builder is deleted on any failure after it is created (trap), so a failed
# build does not leave a billable droplet behind.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=""
REF=""
REPO="https://github.com/yavarb/litco-agent.git"
REGION="sfo3"
SIZE="s-4vcpu-8gb"
BASE_IMAGE="ubuntu-24-04-x64"
SSH_KEY=""
TAG="litco-host-builder"
DRY_RUN=false

usage() { sed -n '2,37p' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build-image: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "build-image: --version <tag> is required" >&2
  exit 2
fi
if [[ ! "$VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "build-image: version must match [A-Za-z0-9._-]+ (it becomes a snapshot name)" >&2
  exit 2
fi
REF="${REF:-$VERSION}"
SNAPSHOT="litco-agent-host-${VERSION}"
BUILDER="litco-agent-builder-${VERSION//./-}"
if [[ "$DRY_RUN" == false && -z "$SSH_KEY" ]]; then
  echo "build-image: --ssh-key is required (a keyless droplet cannot be driven over SSH)" >&2
  exit 2
fi
SSH_KEY_ARG="${SSH_KEY:-<ssh-key>}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes
  -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
DROPLET_ID="<droplet-id>"
DROPLET_IP="<droplet-ip>"
CREATE_ATTEMPTED=false
BUILDER_DELETED=false
STEP=0

# plan <description> -- <command...>: print in dry-run, run otherwise.
plan() {
  local desc="$1"; shift
  [[ "$1" == "--" ]] && shift
  STEP=$((STEP + 1))
  if [[ "$DRY_RUN" == true ]]; then
    printf 'PLAN %02d  %s\n         $ %s\n' "$STEP" "$desc" "$*"
  else
    echo "[build-image] step ${STEP}: ${desc}"
    "$@"
  fi
}

cleanup_builder() {
  local rc=$?
  set +e
  if [[ "$DRY_RUN" == true || "$BUILDER_DELETED" == true ]]; then
    exit "$rc"
  fi
  local ids="" id
  if [[ -n "$DROPLET_ID" && "$DROPLET_ID" != "<droplet-id>" ]]; then
    ids="$DROPLET_ID"
  elif [[ "$CREATE_ATTEMPTED" == true ]]; then
    # The create call died before returning an id (timeout, signal, parse
    # failure). Find every builder with this tag and exact name so none is
    # orphaned.
    ids="$(doctl compute droplet list --tag-name "$TAG" --format ID,Name --no-header 2>/dev/null \
      | awk -v n="$BUILDER" '$2 == n {print $1}')"
  fi
  for id in $ids; do
    echo "[build-image] deleting builder ${id} (exit ${rc})" >&2
    doctl compute droplet delete "$id" --force || \
      echo "[build-image] WARNING: could not delete builder ${id}; delete it by hand" >&2
  done
  exit "$rc"
}
trap cleanup_builder EXIT
# A signal must still reach the EXIT trap, so the builder is deleted on ^C too.
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

create_builder() {
  local out
  CREATE_ATTEMPTED=true
  out="$(doctl compute droplet create "$BUILDER" \
    --region "$REGION" --size "$SIZE" --image "$BASE_IMAGE" \
    --ssh-keys "$SSH_KEY" --enable-monitoring --tag-names "$TAG" \
    --wait --format ID,PublicIPv4 --no-header)"
  DROPLET_ID="$(awk '{print $1}' <<<"$out")"
  DROPLET_IP="$(awk '{print $2}' <<<"$out")"
  [[ -n "$DROPLET_ID" && -n "$DROPLET_IP" ]] || { echo "build-image: could not parse droplet create output: $out" >&2; return 1; }
  echo "[build-image] builder ${DROPLET_ID} at ${DROPLET_IP}"
}

wait_for_ssh() {
  local i
  for i in $(seq 1 60); do
    if ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" true 2>/dev/null; then return 0; fi
    sleep 5
  done
  echo "build-image: builder never accepted SSH" >&2
  return 1
}

snapshot_exists() {
  doctl compute snapshot list --resource droplet --format Name --no-header | grep -qx "$SNAPSHOT"
}

if [[ "$DRY_RUN" == true ]]; then
  echo "litco-agent host image build plan (dry run; nothing is created)"
  echo "  version=${VERSION} ref=${REF} snapshot=${SNAPSHOT} builder=${BUILDER}"
  echo "  region=${REGION} size=${SIZE} base=${BASE_IMAGE} repo=${REPO} tag=${TAG}"
fi

if [[ "$DRY_RUN" == true ]]; then
  plan "refuse to overwrite an existing snapshot" -- doctl compute snapshot list --resource droplet --format Name --no-header "| grep -qx ${SNAPSHOT} && exit 1"
elif snapshot_exists; then
  echo "build-image: snapshot ${SNAPSHOT} already exists; versions are immutable, pick a new tag" >&2
  exit 1
else
  STEP=$((STEP + 1))
fi

if [[ "$DRY_RUN" == true ]]; then
  plan "create builder droplet" -- doctl compute droplet create "$BUILDER" --region "$REGION" --size "$SIZE" \
    --image "$BASE_IMAGE" --ssh-keys "$SSH_KEY_ARG" --enable-monitoring --tag-names "$TAG" \
    --wait --format ID,PublicIPv4 --no-header
  plan "wait for SSH" -- ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" true
else
  plan "create builder droplet" -- create_builder
  plan "wait for SSH" -- wait_for_ssh
fi
plan "wait for the builder's own first boot (apt locks)" -- ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" "cloud-init status --wait >/dev/null || true"
plan "copy install script" -- scp "${SSH_OPTS[@]}" "${HERE}/install-host.sh" "root@${DROPLET_IP}:/root/install-host.sh"
plan "install host (packages, Tailscale, Chromium, Python 3.14, hermes user, litco-agent@${REF}, venvs, Node)" -- \
  ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" bash /root/install-host.sh --ref "$REF" --repo "$REPO"
plan "verify secret-free image" -- ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" \
  "test ! -e /etc/litco-agent/env && test ! -e /home/hermes/.hermes/.env"
plan "remove builder key and clean cloud-init state (load-bearing)" -- ssh "${SSH_OPTS[@]}" "root@${DROPLET_IP}" \
  "rm -f /root/.ssh/authorized_keys /root/install-host.sh && cloud-init clean --logs --machine-id"
plan "power off builder" -- doctl compute droplet-action shutdown "$DROPLET_ID" --wait
plan "snapshot as ${SNAPSHOT}" -- doctl compute droplet-action snapshot "$DROPLET_ID" --snapshot-name "$SNAPSHOT" --wait
plan "delete builder" -- doctl compute droplet delete "$DROPLET_ID" --force

if [[ "$DRY_RUN" == false ]]; then
  BUILDER_DELETED=true   # disarm the trap: a clean run neither re-lists nor re-deletes
  echo "[build-image] DONE: ${SNAPSHOT} (litco-agent@${REF})"
fi
