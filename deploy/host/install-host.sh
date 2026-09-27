#!/usr/bin/env bash
#
# install-host.sh: turn a fresh Ubuntu 24.04 machine into a litco-agent matter
# host image. Runs as root ON THE MACHINE BEING BUILT. build-image.sh copies it
# to a throwaway builder droplet and runs it there; deploy/host/smoke/Dockerfile
# runs the same script inside a container with --container.
#
# The result is SECRET-FREE. It carries code, system packages, the service unit
# and the profile templates. Every secret arrives at first boot through
# cloud-init user-data (deploy/host/cloud-init.yaml.tmpl) as the root-owned
# /etc/litco-agent/env, which only systemd reads.
#
# Usage:
#   install-host.sh --ref <git-ref> [--repo <url>] [--source <dir>] [--container]
#
#   --ref        git ref (tag, branch or commit) of litco-agent to install. Required
#                unless --source is given.
#   --repo       clone URL (default https://github.com/yavarb/litco-agent.git).
#   --source     install from a local tree instead of cloning (the smoke image).
#   --container  skip what a container cannot or should not do: Tailscale,
#                ufw, fail2ban, unattended-upgrades and linger.
#
# Layout it produces:
#   /opt/litco-agent/app        the checkout (owned by root, read-only to hermes)
#   /opt/litco-agent/app/.venv  the Hermes venv (uv sync --frozen from uv.lock)
#   /opt/litco-agent/python     uv-managed CPython 3.14 (bundles a current SQLite)
#   /opt/litco-agent/tools      PM store: agent-browser + pinned Chromium
#   /opt/litco-agent/toolenv    venv the agent's own scripts use (duckdb, pandas, ...)
#   /home/hermes/.hermes        HERMES_HOME, rendered at each start by litco-agent-init
#   /etc/litco-agent/           env (0600 root, written by cloud-init), matter.json
#   /etc/systemd/system/litco-agent.service

set -euo pipefail

REF=""
REPO="https://github.com/yavarb/litco-agent.git"
SOURCE=""
CONTAINER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --container) CONTAINER=true; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "install-host: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$REF" && -z "$SOURCE" ]]; then
  echo "install-host: --ref <git-ref> is required (or --source <dir>)" >&2
  exit 2
fi
if [[ $EUID -ne 0 ]]; then
  echo "install-host: run as root" >&2
  exit 1
fi

PREFIX=/opt/litco-agent
APP="$PREFIX/app"
HERMES_UID=10000
PYTHON_MINOR=3.14
NODE_MAJOR=24          # Node LTS line as of 2026-09
export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy

log() { echo "[install-host] $*"; }

log "apt: base, document, archive and hardening packages"
apt-get -o Acquire::Retries=3 update
apt-get -o Acquire::Retries=3 upgrade -y
# 7zip is the Ubuntu 24.04 package that ships 7z/7zz; unrar is in multiverse.
if ! grep -rqs "multiverse" /etc/apt/sources.list /etc/apt/sources.list.d/; then
  apt-get install -y software-properties-common
  add-apt-repository -y multiverse
  apt-get -o Acquire::Retries=3 update
fi
apt-get install -y --no-install-recommends \
  ca-certificates curl git jq ripgrep xz-utils unzip ffmpeg python3 \
  7zip unrar poppler-utils \
  libreoffice-core libreoffice-writer libreoffice-calc \
  fonts-liberation2 fonts-crosextra-carlito fonts-crosextra-caladea fonts-dejavu-core \
  build-essential libatomic1 sudo systemd dbus \
  ufw fail2ban unattended-upgrades
# Shared libraries the pinned Chromium links against (same set as the upstream
# Dockerfile, which stages Chromium through PM and declares its libs explicitly).
apt-get install -y --no-install-recommends \
  libasound2t64 libatk-bridge2.0-0t64 libatk1.0-0t64 libatspi2.0-0t64 libcairo2 \
  libcups2t64 libdbus-1-3 libgbm1 libglib2.0-0t64 libnspr4 libnss3 libpango-1.0-0 \
  libx11-6 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2

log "Node.js ${NODE_MAJOR} (NodeSource)"
curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -
apt-get install -y nodejs

if [[ "$CONTAINER" == false ]]; then
  log "Tailscale (joined at first boot by cloud-init when a key is given)"
  curl -fsSL https://tailscale.com/install.sh | sh
  systemctl enable tailscaled
fi

log "uv + CPython ${PYTHON_MINOR} (uv-managed; Ubuntu's SQLite 3.45 has the WAL-reset bug)"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
mkdir -p "$PREFIX"
uv python install "$PYTHON_MINOR"
PY="$(uv python find "$PYTHON_MINOR")"
"$PY" - <<'PYCHECK'
import sqlite3, sys
if sqlite3.sqlite_version_info < (3, 51, 3):
    sys.exit(f"linked SQLite {sqlite3.sqlite_version} still has the WAL-reset bug")
print("python", sys.version.split()[0], "sqlite", sqlite3.sqlite_version)
PYCHECK

log "hermes service user (uid ${HERMES_UID})"
if ! id hermes &>/dev/null; then
  useradd --uid "$HERMES_UID" --create-home --home-dir /home/hermes --shell /bin/bash hermes
fi
install -d -m 0700 -o hermes -g hermes /home/hermes/.hermes
if [[ "$CONTAINER" == false ]]; then
  # Upstream's cron and Kanban workers cross `systemd-run --user`, which needs
  # hermes's own user manager to exist at boot.
  loginctl enable-linger hermes
fi

log "litco-agent checkout -> $APP"
if [[ -n "$SOURCE" ]]; then
  rm -rf "$APP"
  mkdir -p "$APP"
  cp -a "$SOURCE"/. "$APP"/
  REF="$(git -C "$APP" rev-parse HEAD 2>/dev/null || echo local)"
else
  if [[ ! -d "$APP/.git" ]]; then
    git clone --filter=blob:none "$REPO" "$APP"
  fi
  git -C "$APP" fetch --tags origin
  git -C "$APP" checkout --detach "$REF"
fi
git config --system --add safe.directory "$APP"
git -C "$APP" rev-parse HEAD > "$PREFIX/REF" 2>/dev/null || echo "$REF" > "$PREFIX/REF"

log "Hermes venv (uv sync --frozen, messaging extra for aiohttp/Slack/Telegram)"
(cd "$APP" && UV_PROJECT_ENVIRONMENT="$APP/.venv" uv sync --frozen --no-dev --extra messaging --python "$PY")
"$APP/.venv/bin/python" -c "import litco.turn_server, gateway.run; print('hermes venv ok')"

log "tool venv for the agent's own scripts"
uv venv --python "$PY" "$PREFIX/toolenv"
uv pip install --python "$PREFIX/toolenv/bin/python" \
  duckdb pandas matplotlib python-docx pymupdf requests openpyxl
"$PREFIX/toolenv/bin/python" -c "import duckdb, pandas, matplotlib, docx, fitz, requests, openpyxl; print('toolenv ok')"
ln -sf "$PREFIX/toolenv/bin/python" /usr/local/bin/litco-python

log "browser: agent-browser + PM-pinned Chromium into $PREFIX/tools"
install -d -o hermes -g hermes "$PREFIX/tools"
(cd "$APP" && sudo -u hermes -H env HERMES_RUNTIME_DIR="$PREFIX/tools" HERMES_HOME=/home/hermes/.hermes \
  "$APP/.venv/bin/python" -c 'from pm import ensure; [ensure(n, explicit=True) for n in ("agent-browser", "chromium")]')

log "service unit, init, drain and profile templates"
install -d -m 0755 /etc/litco-agent
install -m 0755 "$APP/deploy/host/litco-agent-init" /usr/local/bin/litco-agent-init
install -m 0755 "$APP/deploy/host/litco-agent-drain" /usr/local/bin/litco-agent-drain
install -m 0644 "$APP/deploy/host/litco-agent.service" /etc/systemd/system/litco-agent.service
chown -R root:root "$APP"
chmod -R go-w "$APP"

if [[ "$CONTAINER" == false ]]; then
  log "hardening: fail2ban, unattended upgrades, ssh"
  systemctl enable fail2ban
  dpkg-reconfigure -f noninteractive unattended-upgrades
  cat > /etc/ssh/sshd_config.d/60-litco-agent.conf <<'SSHD'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
X11Forwarding no
AllowTcpForwarding no
SSHD
  sshd -t
  # ufw rules are applied at first boot by cloud-init (they depend on the
  # matter's egress policy); the image leaves ufw installed and inactive.
  mkdir -p /var/log/journal
  systemd-tmpfiles --create --prefix /var/log/journal || true
fi

# Baked state: ENABLED but STOPPED, with no env file. cloud-init writes
# /etc/litco-agent/env and then RESTARTS the unit (a `start` would be a no-op
# against a copy that auto-started before the env existed; see the LitKit fleet
# note of 2026-07-05). The unit also refuses to start without the env file.
if [[ "$CONTAINER" == false ]]; then
  systemctl daemon-reload
  systemctl enable litco-agent.service
  systemctl stop litco-agent.service 2>/dev/null || true
fi
test ! -e /etc/litco-agent/env || { echo "install-host: /etc/litco-agent/env exists; image is not secret-free" >&2; exit 1; }

log "DONE at $(cat "$PREFIX/REF")"
