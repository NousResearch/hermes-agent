#!/bin/bash
# Hermes VIP — Passive Privilege Harness Installer
# Installs the daemon (systemd service) and plugin files on Linux.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Hermes VIP — Passive Privilege Harness Installer"
echo "================================================"
echo ""

# ── Platform check ──
case "$(uname)" in
    Linux)
        echo "✓ Linux detected"
        ;;
    Darwin)
        echo "✓ macOS detected"
        echo "  Note: macOS daemon via launchd is not included in this passive release."
        echo "  See contrib/privilege-harness/README.md for manual setup instructions."
        exit 0
        ;;
    *)
        echo "❌ Unsupported platform: $(uname)"
        exit 1
        ;;
esac

# ── Require root ──
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ Must run as root: sudo bash install.sh"
    exit 1
fi

# ── Check Python ──
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 not found. Install Python 3.9+ first."
    exit 1
fi
echo "✓ python3: $(python3 --version)"

# ── Create hermes-vip user + group ──
if ! getent group hermes-vip &>/dev/null; then
    echo "Creating hermes-vip group..."
    groupadd -r hermes-vip
    echo "✓ hermes-vip group created"
else
    echo "✓ hermes-vip group exists"
fi
if ! id hermes-vip &>/dev/null; then
    echo "Creating hermes-vip user..."
    useradd -r -s /sbin/nologin -d /nonexistent -g hermes-vip hermes-vip
    echo "✓ hermes-vip user created"
else
    echo "✓ hermes-vip user exists"
fi

# ── Install daemon ──
DAEMON_DIR="/usr/local/lib/hermes-vip"
echo "Installing daemon to $DAEMON_DIR..."
mkdir -p "$DAEMON_DIR"
cp -r "$DIR/daemon" "$DAEMON_DIR/"
chown -R root:root "$DAEMON_DIR"
chmod 755 "$DAEMON_DIR/daemon/"
echo "✓ daemon files installed"

# ── Create runtime directory ──
mkdir -p /var/run/hermes-vip
chown hermes-vip:hermes-vip /var/run/hermes-vip
# 0750: hermes-vip group can traverse; sockets are 0660 (daemon chmods)
chmod 750 /var/run/hermes-vip
echo "✓ /var/run/hermes-vip created"

# ── Create vipd launcher ──
cat > /usr/local/bin/hermes-vipd << 'LAUNCHER'
#!/bin/bash
cd /usr/local/lib/hermes-vip
exec python3 -m daemon.vipd "$@"
LAUNCHER
chmod 755 /usr/local/bin/hermes-vipd
echo "✓ hermes-vipd launcher created"

# ── Install systemd service ──
cat > /etc/systemd/system/hermes-vipd.service << SERVICE
[Unit]
Description=Hermes VIP Daemon
After=network.target

[Service]
Type=simple
User=hermes-vip
Group=hermes-vip
ExecStart=/usr/local/bin/hermes-vipd
Restart=always
RestartSec=5
RuntimeDirectory=hermes-vip
RuntimeDirectoryMode=0750

[Install]
WantedBy=multi-user.target
SERVICE
systemctl daemon-reload
echo "✓ systemd service installed"

# ── Generate daemon config ──
# trusted_user must be set: vipd.py reads config["trusted_user"] (top-level)
# or SUDO_USER env. Under systemd neither is auto-populated, and without it
# TRUSTED_UIDS={0} would reject the Hermes plugin's request.sock connection.
CONFIG_PATH="/etc/hermes-vip/config.yaml"
TRUSTED_USER="${SUDO_USER:-root}"
mkdir -p /etc/hermes-vip /var/log/hermes-vip
sed -e "s/__TRUSTED_USER__/${TRUSTED_USER}/" \
    "$DIR/../../config.yaml.tmpl" > "$CONFIG_PATH"
chown root:hermes-vip "$CONFIG_PATH"
chmod 640 "$CONFIG_PATH"
chown hermes-vip:hermes-vip /var/log/hermes-vip
echo "✓ daemon config generated (trusted_user=${TRUSTED_USER})"

# ── Add caller to hermes-vip group ──
if [ -n "${SUDO_USER:-}" ]; then
    if ! groups "$SUDO_USER" 2>/dev/null | grep -q hermes-vip; then
        usermod -a -G hermes-vip "$SUDO_USER"
        echo "✓ $SUDO_USER added to hermes-vip group (re-login required)"
    else
        echo "✓ $SUDO_USER already in hermes-vip group"
    fi
else
    echo "⚠ No SUDO_USER detected. Add your user to the hermes-vip group manually:"
    echo "   sudo usermod -a -G hermes-vip YOUR_USERNAME"
fi

# ── Start daemon ──
systemctl enable hermes-vipd
systemctl start hermes-vipd
echo "✓ hermes-vipd started"

# ── Auto-install plugin (system-wide hermes home, else user home) ──
PLUGIN_DIR="${HERMES_PLUGIN_DIR:-}"
if [ -z "$PLUGIN_DIR" ]; then
    if [ -n "${SUDO_USER:-}" ]; then
        SUDO_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        for cand in "/root/.hermes/plugins" "$SUDO_HOME/.hermes/plugins"; do
            if [ -d "$cand" ]; then
                PLUGIN_DIR="$cand"
                break
            fi
        done
    fi
    [ -z "$PLUGIN_DIR" ] && PLUGIN_DIR="/root/.hermes/plugins"
fi
mkdir -p "$PLUGIN_DIR/hermes-vip"
cp -R "$DIR/../../plugins/hermes-privilege-harness/." "$PLUGIN_DIR/hermes-vip/"
chmod -R a+rX "$PLUGIN_DIR/hermes-vip"
echo "✓ plugin installed to $PLUGIN_DIR/hermes-vip"

echo ""
echo "========================================"
echo "Installation complete."
echo ""
echo "Check status:"
echo "  systemctl status hermes-vipd"
echo "  ls -la /var/run/hermes-vip/"
echo ""
echo "IMPORTANT (security model):"
echo "  request.sock is 0660, group hermes-vip. The Hermes process must run"
echo "  as a user in the hermes-vip group (see usermod -a -G hermes-vip)."
echo "  The daemon rejects connections whose SO_PEERCRED uid is not trusted."
echo "========================================"

