#!/bin/sh
# install-watchdog.sh — install the bridge watchdog as a launchd job (macOS).
#
#   ./install-watchdog.sh [path/to/watchdog.env]
#
# What it does:
#   1. creates the guard dir (~/.hermes/agent-bridge-watchdog by default);
#   2. installs watchdog.sh + your watchdog.env into it, plus the current launcher shims as
#      *.bridge templates (the watchdog restores from these if something overwrites them);
#   3. writes and loads the launchd job with the app-bundle paths filled in;
#   4. creates the ENABLED marker and runs one check immediately.
#
# Opt out later:  rm ~/.hermes/agent-bridge-watchdog/ENABLED
# Remove entirely: launchctl bootout gui/$(id -u)/<label> && rm -rf ~/.hermes/agent-bridge-watchdog

set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
ENV_FILE=${1:-$HERE/watchdog.env}
GUARD="$HOME/.hermes/agent-bridge-watchdog"
LABEL="com.hermes.agent-bridge-watchdog"

[ -f "$ENV_FILE" ] || { echo "missing config: $ENV_FILE (copy scripts/watchdog.env.sample)"; exit 1; }
. "$ENV_FILE"

echo "guard dir: $GUARD"
mkdir -p "$GUARD"
cp "$HERE/watchdog.sh" "$GUARD/watchdog.sh"
chmod 755 "$GUARD/watchdog.sh"
cp "$ENV_FILE" "$GUARD/watchdog.env"
chmod 600 "$GUARD/watchdog.env"

# keep a restore copy of each launcher we own (these are what the watchdog repairs from)
for pair in $LAUNCHERS; do
  name="${pair%%:*}"
  if [ -f "$HOME/.local/bin/$name" ]; then
    cp "$HOME/.local/bin/$name" "$GUARD/$name.bridge"
  else
    echo "  note: ~/.local/bin/$name does not exist yet — install the bridge first (SKILL.md Step 1)"
  fi
done

# first CLI name drives the plist's watched launcher paths
set -- $LAUNCHERS
CLI1=$(basename "${1%%:*}")
CLI2=$(basename "${2:-${1}}" )
CLI2=${CLI2%%:*}

python3 - "$REPO/templates/com.example.agent-bridge-watchdog.plist" "$GUARD/$LABEL.plist" <<PY
import sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
for k, v in {"__LABEL__": "$LABEL", "__GUARD__": "$GUARD", "__APP_BUNDLE__": "$APP_BUNDLE",
             "__APP_SUPPORT__": "$APP_SUPPORT", "__HOME__": "$HOME",
             "__CLI1__": "$CLI1", "__CLI2__": "$CLI2"}.items():
    t = t.replace(k, v)
open(dst, "w").write(t)
PY

plutil -lint "$GUARD/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$GUARD/$LABEL.plist" "$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/$LABEL.plist"
touch "$GUARD/ENABLED"

echo "loaded: $LABEL (event-driven on the watched paths + at login)"
echo "running one check now..."
GUARD="$GUARD" sh "$GUARD/watchdog.sh"
echo
echo "status:"; cat "$GUARD/status.json" 2>/dev/null || echo "  (none — check $GUARD/status.log)"
echo
echo "self-test the alert path (delivers a real alert):  SELFTEST=1 sh $GUARD/watchdog.sh"
