#!/usr/bin/env bash
#
# install-host.sh: turn a fresh Ubuntu 24.04 machine into a litco-agent host
# image. Runs as root ON THE MACHINE BEING BUILT. build-image.sh copies it
# to a throwaway builder droplet and runs it there; deploy/host/smoke/Dockerfile
# runs the same script inside a container with --container.
#
# The result is SECRET-FREE. It carries code, system packages, the service units
# and the profile templates. Every secret arrives at runtime: on a per-matter
# droplet through cloud-init user-data as the root-owned /etc/litco-agent/env;
# on a machine host through litco-supervisor, which writes each slot's
# /run/litco-agent/<slot>.env when it starts the slot.
#
# Usage:
#   install-host.sh --ref <git-ref> [--repo <url>] [--source <dir>] [--container]
#                   [--topology per_matter|machine] [--dry-run]
#
#   --ref        git ref (tag, branch or commit) of litco-agent to install. Required
#                unless --source is given.
#   --repo       clone URL (default https://github.com/yavarb/litco-agent.git).
#   --source     install from a local tree instead of cloning (the smoke image).
#   --container  skip what a container cannot or should not do: Tailscale,
#                ufw, fail2ban, unattended-upgrades, linger and nftables.
#   --topology   per_matter (default; also LITCO_TOPOLOGY): one matter per droplet,
#                litco-agent.service baked enabled. machine: one droplet per
#                deployment, one litco-agent@<slot> per matter under its own
#                Unix user, started by litco-supervisor (FIRM_AGENT_HOST
#                section 3).
#   --dry-run    print every step and command in order and exit 0; runs nothing
#                and needs no root.
#
# Layout, both topologies:
#   /opt/litco-agent/python     uv-managed CPython 3.14 (bundles a current SQLite)
#   /opt/litco-agent/tools      PM store: agent-browser + pinned Chromium
#   /opt/litco-agent/toolenv    venv the agent's own scripts use (duckdb, pandas, ...)
#   /opt/litco-agent/REF        the commit the host runs
#   /opt/litco-agent/TOPOLOGY   per_matter or machine
# per_matter:
#   /opt/litco-agent/app        the checkout (owned by root, read-only to hermes)
#   /opt/litco-agent/app/.venv  the Hermes venv (uv sync --frozen from uv.lock)
#   /home/hermes/.hermes        HERMES_HOME, rendered at each start by litco-agent-init
#   /etc/litco-agent/           env (0600 root, written by cloud-init), matter.json
#   /etc/systemd/system/litco-agent.service
# machine:
#   /opt/litco-agent/releases/<ref>        one checkout + venv per release
#   /opt/litco-agent/current -> releases/<ref>   flipped by litco-host-update
#   /srv/litco/m/<slot>                    each slot's home (made by the supervisor)
#   /etc/systemd/system/litco-agent@.service, litco-supervisor.service
#   /usr/local/sbin/litco-supervisor, /usr/local/sbin/litco-host-update
#   /etc/nftables.conf                     inbound only on tailscale0

set -euo pipefail

REF=""
REPO="https://github.com/yavarb/litco-agent.git"
SOURCE=""
CONTAINER=false
DRY_RUN=false
TOPOLOGY="${LITCO_TOPOLOGY:-per_matter}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --container) CONTAINER=true; shift ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "install-host: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TOPOLOGY" != per_matter && "$TOPOLOGY" != machine ]]; then
  echo "install-host: --topology must be per_matter or machine" >&2
  exit 2
fi
if [[ -z "$REF" && -z "$SOURCE" ]]; then
  echo "install-host: --ref <git-ref> is required (or --source <dir>)" >&2
  exit 2
fi
if [[ "$DRY_RUN" == false && $EUID -ne 0 ]]; then
  echo "install-host: run as root" >&2
  exit 1
fi

PREFIX=/opt/litco-agent
HERMES_UID=10000
PYTHON_MINOR=3.14
NODE_MAJOR=24          # Node LTS line as of 2026-09
export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy

# A release directory name: the ref with anything outside [A-Za-z0-9._-] made "-".
release_name() { local r="${1//[^A-Za-z0-9._-]/-}"; echo "${r:-local}"; }

if [[ "$TOPOLOGY" == machine ]]; then
  if [[ -n "$SOURCE" ]]; then
    RELEASE="$(release_name "$(git -C "$SOURCE" rev-parse --short=12 HEAD 2>/dev/null || echo local)")"
  else
    RELEASE="$(release_name "$REF")"
  fi
  if [[ "$RELEASE" == .* ]]; then
    echo "install-host: release name '$RELEASE' may not start with a dot" >&2
    exit 2
  fi
  APP="$PREFIX/releases/$RELEASE"
else
  APP="$PREFIX/app"
fi

STEP=0
step() {
  STEP=$((STEP + 1))
  if [[ "$DRY_RUN" == true ]]; then
    printf 'PLAN %02d  %s\n' "$STEP" "$*"
  else
    echo "[install-host] $*"
  fi
}
run() {
  if [[ "$DRY_RUN" == true ]]; then printf '         $ %s\n' "$*"; else "$@"; fi
}
run_sh() {
  if [[ "$DRY_RUN" == true ]]; then printf '         $ %s\n' "$1"; else bash -c "$1"; fi
}
# write_file <path> <mode>: the content comes on stdin.
write_file() {
  if [[ "$DRY_RUN" == true ]]; then
    printf '         > %s (%s)\n' "$1" "$2"
    sed 's/^/         | /'
  else
    (umask 077 && cat > "$1") && chmod "$2" "$1"
  fi
}
machine() { [[ "$TOPOLOGY" == machine ]]; }
droplet() { [[ "$CONTAINER" == false ]]; }
dry() { [[ "$DRY_RUN" == true ]]; }

if dry; then
  echo "litco-agent host install plan (dry run; nothing is run)"
  echo "  topology=${TOPOLOGY} ref=${REF:-<from source>} source=${SOURCE:-<clone ${REPO}>} container=${CONTAINER} app=${APP}"
fi

step "apt: base, document, archive and hardening packages"
run apt-get -o Acquire::Retries=3 update
run apt-get -o Acquire::Retries=3 upgrade -y
# 7zip is the Ubuntu 24.04 package that ships 7z/7zz; unrar is in multiverse.
if dry || ! grep -rqs "multiverse" /etc/apt/sources.list /etc/apt/sources.list.d/; then
  run apt-get install -y software-properties-common
  run add-apt-repository -y multiverse
  run apt-get -o Acquire::Retries=3 update
fi
run apt-get install -y --no-install-recommends \
  ca-certificates curl git jq ripgrep xz-utils unzip ffmpeg python3 \
  7zip unrar poppler-utils \
  libreoffice-core libreoffice-writer libreoffice-calc \
  fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea fonts-dejavu-core \
  build-essential libatomic1 sudo systemd dbus \
  ufw fail2ban unattended-upgrades
# Shared libraries the pinned Chromium links against (same set as the upstream
# Dockerfile, which stages Chromium through PM and declares its libs explicitly).
run apt-get install -y --no-install-recommends \
  libasound2t64 libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 libcairo2 \
  libcups2t64 libdbus-1-3 libgbm1 libglib2.0-0t64 libnspr4 libnss3 libpango-1.0-0 \
  libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2
if machine; then
  run apt-get install -y --no-install-recommends nftables
fi

step "Node.js ${NODE_MAJOR} (NodeSource)"
run_sh "curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash -"
run apt-get install -y nodejs

if droplet; then
  step "Tailscale (joined at first boot by cloud-init when a key is given)"
  run_sh "curl -fsSL https://tailscale.com/install.sh | sh"
  run systemctl enable tailscaled
fi

step "uv + CPython ${PYTHON_MINOR} (uv-managed; Ubuntu's SQLite 3.45 has the WAL-reset bug)"
run_sh "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh"
run mkdir -p "$PREFIX"
run uv python install "$PYTHON_MINOR"
if dry; then PY="<uv python find ${PYTHON_MINOR}>"; else PY="$(uv python find "$PYTHON_MINOR")"; fi
run "$PY" -c 'import sqlite3, sys
if sqlite3.sqlite_version_info < (3, 51, 3):
    sys.exit(f"linked SQLite {sqlite3.sqlite_version} still has the WAL-reset bug")
print("python", sys.version.split()[0], "sqlite", sqlite3.sqlite_version)'

# On a machine host no service runs as hermes; it owns the shared PM store.
step "hermes user (uid ${HERMES_UID})"
if dry || ! id hermes &>/dev/null; then
  run useradd --uid "$HERMES_UID" --create-home --home-dir /home/hermes --shell /bin/bash hermes
fi
run install -d -m 0700 -o hermes -g hermes /home/hermes/.hermes
if droplet && ! machine; then
  # Upstream's cron and Kanban workers cross `systemd-run --user`, which needs
  # hermes's own user manager to exist at boot. On a machine host the
  # supervisor enables linger for each slot's user instead.
  run loginctl enable-linger hermes
fi

step "litco-agent checkout -> $APP"
if [[ -n "$SOURCE" ]]; then
  run rm -rf "$APP"
  run mkdir -p "$APP"
  run cp -a "$SOURCE"/. "$APP"/
  if dry; then REF="<commit of ${SOURCE}>"; else REF="$(git -C "$APP" rev-parse HEAD 2>/dev/null || echo local)"; fi
else
  if dry || [[ ! -d "$APP/.git" ]]; then
    run git clone --filter=blob:none "$REPO" "$APP"
  fi
  run git -C "$APP" fetch --tags origin
  run git -C "$APP" checkout --detach "$REF"
fi
run git config --system --add safe.directory "$APP"
if dry; then
  run_sh "git -C $APP rev-parse HEAD > $PREFIX/REF"
else
  git -C "$APP" rev-parse HEAD > "$PREFIX/REF" 2>/dev/null || echo "$REF" > "$PREFIX/REF"
fi
echo "$TOPOLOGY" | write_file "$PREFIX/TOPOLOGY" 0644
if machine; then
  # Fail before the long venv and browser steps, not after them.
  for f in litco-supervisor litco-supervisor.service; do
    if ! dry && [[ ! -f "$APP/deploy/host/$f" ]]; then
      echo "install-host: $APP/deploy/host/$f is missing; this ref predates the supervisor" >&2
      exit 1
    fi
  done
fi

step "Hermes venv (uv sync --frozen, messaging extra for aiohttp/Slack/Telegram)"
run env -C "$APP" UV_PROJECT_ENVIRONMENT="$APP/.venv" uv sync --frozen --no-dev --extra messaging --python "$PY"
run "$APP/.venv/bin/python" -c "import litco.turn_server, gateway.run; print('hermes venv ok')"

step "tool venv for the agent's own scripts"
run uv venv --python "$PY" "$PREFIX/toolenv"
run uv pip install --python "$PREFIX/toolenv/bin/python" \
  duckdb pandas matplotlib python-docx pymupdf requests openpyxl
run "$PREFIX/toolenv/bin/python" -c "import duckdb, pandas, matplotlib, docx, fitz, requests, openpyxl; print('toolenv ok')"
run ln -sf "$PREFIX/toolenv/bin/python" /usr/local/bin/litco-python

step "browser: agent-browser + PM-pinned Chromium into $PREFIX/tools"
run install -d -o hermes -g hermes "$PREFIX/tools"
run env -C "$APP" sudo -u hermes -H env HERMES_RUNTIME_DIR="$PREFIX/tools" HERMES_HOME=/home/hermes/.hermes \
  "$APP/.venv/bin/python" -c 'from pm import ensure; [ensure(n, explicit=True) for n in ("agent-browser", "chromium")]'

if machine; then
  step "machine host: release ${RELEASE} behind $PREFIX/current, slot template unit, supervisor, host update"
  run chown -R root:root "$APP"
  run chmod -R go-w "$APP"
  echo "$REF" | write_file "$APP/.litco-release-ok" 0644
  # rename(2) replaces the old link in one step, as litco-host-update does.
  run ln -sfn "releases/$RELEASE" "$PREFIX/current.new"
  run python3 -c 'import os, sys; os.replace(sys.argv[1], sys.argv[2])' "$PREFIX/current.new" "$PREFIX/current"
  # No per-matter unit on a machine host: a baked litco-agent.service would
  # start one gateway for no matter.
  run rm -f /etc/systemd/system/litco-agent.service
  run install -m 0644 "$APP/deploy/host/litco-agent@.service" "/etc/systemd/system/litco-agent@.service"
  run install -m 0755 "$APP/deploy/host/litco-supervisor" /usr/local/sbin/litco-supervisor
  run install -m 0644 "$APP/deploy/host/litco-supervisor.service" /etc/systemd/system/litco-supervisor.service
  run install -m 0755 "$APP/deploy/host/litco-host-update" /usr/local/sbin/litco-host-update
  # Slot homes are created 0700 by the supervisor; the parent only lets owners in.
  run install -d -m 0711 -o root -g root /srv/litco /srv/litco/m
  run install -d -m 0700 -o root -g root /var/lib/litco-supervisor
  write_file /etc/tmpfiles.d/litco-agent.conf 0644 <<'TMPFILES'
# Per-slot secret env files, written by litco-supervisor (root 0600, tmpfs).
d /run/litco-agent 0700 root root -
TMPFILES
  run systemd-tmpfiles --create /etc/tmpfiles.d/litco-agent.conf
else
  step "service unit, init, drain and profile templates"
  run install -d -m 0755 /etc/litco-agent
  run install -m 0755 "$APP/deploy/host/litco-agent-init" /usr/local/bin/litco-agent-init
  run install -m 0755 "$APP/deploy/host/litco-agent-drain" /usr/local/bin/litco-agent-drain
  run install -m 0644 "$APP/deploy/host/litco-agent.service" /etc/systemd/system/litco-agent.service
  run chown -R root:root "$APP"
  run chmod -R go-w "$APP"
fi

if droplet; then
  step "hardening: fail2ban, unattended upgrades, ssh"
  run systemctl enable fail2ban
  run dpkg-reconfigure -f noninteractive unattended-upgrades
  write_file /etc/ssh/sshd_config.d/60-litco-agent.conf 0644 <<'SSHD'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
X11Forwarding no
AllowTcpForwarding no
SSHD
  run sshd -t
  # Per-matter droplet: ufw rules are applied at first boot by cloud-init (they
  # depend on the matter's egress policy); the image leaves ufw installed and
  # inactive.
  run mkdir -p /var/log/journal
  run_sh "systemd-tmpfiles --create --prefix /var/log/journal || true"
fi

if machine && droplet; then
  # Loaded at the next boot, not now: the builder is driven over its public
  # address, which this ruleset closes. Slot ports (8800+) and the supervisor
  # (8700) are reachable only over the tailnet.
  step "nftables: inbound only on tailscale0 (applied at next boot)"
  write_file /etc/nftables.conf 0644 <<'NFT'
#!/usr/sbin/nft -f
# litco-agent machine host (FIRM_AGENT_HOST section 3.3): inbound traffic only
# over the tailnet. Installed by deploy/host/install-host.sh. The table is
# replaced, not the whole ruleset, so Tailscale's own rules survive a reload.
table inet litco_host
delete table inet litco_host
table inet litco_host {
  chain input {
    type filter hook input priority filter; policy drop;
    iif "lo" accept
    ct state established,related accept
    ct state invalid drop
    iifname "tailscale0" accept
    # Tailscale's direct (peer-to-peer) UDP port.
    udp dport 41641 accept
    # What the link itself needs: IPv6 neighbour discovery and DHCP replies.
    icmpv6 type { nd-neighbor-solicit, nd-neighbor-advert, nd-router-advert } accept
    udp sport 67 udp dport 68 accept
  }
}
NFT
  run nft -c -f /etc/nftables.conf
  run systemctl enable nftables.service
fi

# Baked state, per_matter: ENABLED but STOPPED, with no env file. cloud-init
# writes /etc/litco-agent/env and then RESTARTS the unit (a `start` would be a
# no-op against a copy that auto-started before the env existed; see the LitKit
# fleet note of 2026-07-05). The unit also refuses to start without the env file.
# Baked state, machine: the supervisor enabled; no slot instance is enabled, and
# the supervisor starts slots on the app's request.
if droplet; then
  step "baked service state"
  run systemctl daemon-reload
  if machine; then
    run systemctl enable litco-supervisor.service
  else
    run systemctl enable litco-agent.service
    run_sh "systemctl stop litco-agent.service 2>/dev/null || true"
  fi
fi

step "verify the image is secret-free"
run_sh "test ! -e /etc/litco-agent/env || { echo 'install-host: /etc/litco-agent/env exists; image is not secret-free' >&2; exit 1; }"
if machine; then
  run_sh "test ! -e /etc/litco-supervisor.env || { echo 'install-host: /etc/litco-supervisor.env exists; image is not secret-free' >&2; exit 1; }"
  run_sh "! compgen -G '/run/litco-agent/*.env' >/dev/null || { echo 'install-host: a slot env file exists; image is not secret-free' >&2; exit 1; }"
fi

if dry; then
  echo "DONE (dry run)"
else
  log_ref="$(cat "$PREFIX/REF")"
  echo "[install-host] DONE at ${log_ref} (topology ${TOPOLOGY})"
fi
