#!/usr/bin/env bash
# Smoke test for the sandbox desktop image: what a terminal backend needs from it.
#
#   docker/sandbox-desktop-smoke.sh <image>
#
# Proves, as the unprivileged `pn` user the gateway will exec desktop processes as:
#   1. every binary tools/bot_desktop/launcher.sh and computer_use look for exists
#   2. the real launcher.sh brings up Xvnc + Xfce and publishes its env file
#   3. the RFB socket answers the protocol handshake when relayed over `docker exec -i`
#      stdio, which is exactly how the gateway bridge will reach it
#   4. headed Chromium starts on that display (cua-driver and agent-browser both need it)
set -euo pipefail

IMAGE="${1:?usage: sandbox-desktop-smoke.sh <image>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="sbx-smoke-$$"
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run -d --name "$NAME" --memory=4g "$IMAGE" >/dev/null

echo "== 1. binaries (as pn)"
docker exec -u pn "$NAME" bash -c '
  set -e
  for b in Xvnc xfwm4 xfce4-panel xfdesktop xfsettingsd dbus-run-session xauth xdpyinfo xprop \
           xfce4-terminal cua-driver agent-browser jq rg fd tmux rsync sudo python3 node; do
    command -v "$b" >/dev/null || { echo "missing: $b"; exit 1; }
  done
  cua-driver --version
  agent-browser --version
  ls /opt/playwright/chromium-*/chrome-linux*/chrome >/dev/null || { echo "missing headed chromium"; exit 1; }
  sudo -n true
  [ "$(id -u)" = 1000 ]'

echo "== 2. launcher.sh publishes a display"
docker cp "$HERE/tools/bot_desktop/launcher.sh" "$NAME:/tmp/launcher.sh"
docker cp "$HERE/tools/bot_desktop/wallpaper.png" "$NAME:/tmp/wallpaper.png"
docker exec -u pn -d "$NAME" bash -c '
  mkdir -p /tmp/bd && HERMES_BD_PROFILE=smoke HERMES_BD_DISPLAY_NUM=20 HERMES_BD_SOCKET=/tmp/bd/rfb.sock \
  HERMES_BD_XAUTH=/tmp/bd/Xauthority HERMES_BD_ENV_FILE=/tmp/bd/env HERMES_BD_CONFIG_HOME=/tmp/bd/xdg \
  HERMES_BD_WALLPAPER=/tmp/wallpaper.png bash /tmp/launcher.sh >/tmp/bd/launcher.log 2>&1'
for _ in $(seq 1 60); do
  docker exec -u pn "$NAME" test -S /tmp/bd/rfb.sock -a -f /tmp/bd/env 2>/dev/null && break
  sleep 0.5
done
docker exec -u pn "$NAME" bash -c '
  set -e
  grep -q "^DISPLAY=:20$" /tmp/bd/env
  [ "$(pgrep -c -f "xfwm4|xfce4-panel|xfdesktop")" -ge 3 ]' || { docker exec "$NAME" cat /tmp/bd/launcher.log; exit 1; }

echo "== 3. RFB handshake over docker exec -i stdio"
python3 - "$NAME" <<'EOF'
import subprocess, sys
relay = ("import socket,os,threading\n"
         "s=socket.socket(socket.AF_UNIX);s.connect('/tmp/bd/rfb.sock')\n"
         "def up():\n"
         "  while True:\n"
         "    d=os.read(0,65536)\n"
         "    if not d: break\n"
         "    s.sendall(d)\n"
         "threading.Thread(target=up,daemon=True).start()\n"
         "while True:\n"
         "  d=s.recv(65536)\n"
         "  if not d: break\n"
         "  os.write(1,d)\n")
p = subprocess.Popen(["docker", "exec", "-i", "-u", "pn", sys.argv[1], "python3", "-c", relay],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
banner = p.stdout.read(12)
assert banner == b"RFB 003.008\n", banner
p.stdin.write(b"RFB 003.008\n"); p.stdin.flush()
n = p.stdout.read(1)[0]
types = list(p.stdout.read(n))
assert 1 in types, types  # None: the bridge authenticates, Xvnc trusts its 0600 socket
p.kill()
print("   RFB 3.8, security types", types)
EOF

echo "== 4. headed Chromium opens on the display"
# Docker's default seccomp profile denies unprivileged user namespaces, so Chromium's own
# layer-2 sandbox cannot start in any container; the container is the sandbox. The gateway's
# spawn sites already inject this (browser_tool_session._needs_chromium_sandbox_bypass).
docker exec -u pn "$NAME" bash -c '
  set -e; set -a; . /tmp/bd/env; set +a
  chrome=$(ls /opt/playwright/chromium-*/chrome-linux*/chrome | head -1)
  timeout 40 "$chrome" --no-sandbox --disable-dev-shm-usage --no-first-run --disable-gpu \
      --user-data-dir=/tmp/bd/chrome about:blank >/tmp/bd/chrome.log 2>&1 &
  for _ in $(seq 1 60); do
    for w in $(xprop -root _NET_CLIENT_LIST 2>/dev/null | grep -o "0x[0-9a-f]*"); do
      xprop -id "$w" WM_CLASS 2>/dev/null | grep -qi chrom && exit 0
    done
    sleep 0.5
  done
  cat /tmp/bd/chrome.log; exit 1'
echo "OK: $IMAGE"
