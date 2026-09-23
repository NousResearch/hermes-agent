#!/bin/sh
# watchdog.sh — verify the app->remote-agent bridge still holds, and alert if it doesn't.
#
# Triggered by launchd WatchPaths (event-driven, no polling): fires when the app bundle, its agent
# registry, or our launcher shims change — i.e. when an app update or a runtime install happens.
# It re-verifies the WHOLE contract, because an update can break any part of it independently:
#
#   1. restore the launcher bridges if something overwrote them;
#   2. re-read the app's own bundle and assert it still launches a binary we intercept with the
#      expected argv (an update can switch to a bundled runtime without touching our files);
#   3. flag a bundled agent runtime that would bypass the bridge;
#   4. ask the app for its tool list and fail if a tool the workflow needs disappeared (an update can
#      keep the launcher identical while changing the tool surface);
#   5. check the remote profile is still reachable;
#   6. write status + alert the user where they look.
#
# Opt out: rm "$GUARD/ENABLED"
# Self-test the alert path: SELFTEST=1 watchdog.sh
#
# Install with scripts/install-watchdog.sh.

set -u

GUARD="${GUARD:-$HOME/.hermes/agent-bridge-watchdog}"
[ -f "$GUARD/watchdog.env" ] && . "$GUARD/watchdog.env"
[ -f "$GUARD/ENABLED" ] || exit 0

LOG="$GUARD/status.log"
STATUS="$GUARD/status.json"
FINDINGS=""
ANOMALY=0
PY=/usr/bin/python3

say() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG"; }
flag() { ANOMALY=1; FINDINGS="$FINDINGS$1
"; say "ANOMALY: $1"; }
ok() { say "ok: $1"; }

# ── 1. our launcher shims intact ───────────────────────────────────────────────
for pair in $LAUNCHERS; do
  name="${pair%%:*}"; template="$GUARD/${pair##*:}"
  target="$HOME/.local/bin/$name"
  [ -f "$template" ] || { flag "bridge template missing: $template"; continue; }
  if [ ! -f "$target" ]; then
    cp "$template" "$target" && chmod 755 "$target" && ok "recreated $target"
  elif ! grep -q "$BRIDGE_MARKER" "$target" 2>/dev/null; then
    cp "$template" "$target" && chmod 755 "$target" && ok "restored bridge -> $target (had been overwritten)"
  fi
done

# ── 2+3. what does the app launch now? ─────────────────────────────────────────
APPCHECK=""
if [ -f "$ASAR" ]; then
  APPCHECK=$("$PY" - "$ASAR" "$AGENT_ID_KEY" "$ARGV_IDENTIFIER" <<'PY' 2>/dev/null
import json, os, re, struct, sys
asar, key, argv_id = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(asar, 'rb').read()
_, hdr_size, _, json_size = struct.unpack('<IIII', data[:16])
header = json.loads(data[16:16 + json_size])
base = 16 + hdr_size
out = {"agent_map": {}, "acp_args": None, "asar_mtime": os.path.getmtime(asar)}
def walk(node, path=""):
    if isinstance(node, dict) and 'files' in node:
        for n, c in node['files'].items():
            p = path + "/" + n
            if 'files' in c:
                yield from walk(c, p)
            else:
                yield p, c
for p, e in walk(header):
    if not p.endswith('.js') or 'offset' not in e:
        continue
    size = int(e['size'])
    if size > 12_000_000:
        continue
    body = data[base + int(e['offset']): base + int(e['offset']) + size].decode('utf-8', 'replace')
    m = re.search(r'CLI_BY_AGENT_ID\s*=\s*\{(.*?)\}', body, re.S)
    if m:
        for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)):
            out["agent_map"][k] = v
    # the argv may live in a different chunk than the map: search a window around its identifier
    if argv_id and argv_id in body and out["acp_args"] is None:
        i = body.find(argv_id)
        win = body[max(0, i - 500): i + 1500]
        am = re.search(r'args:\s*\[([^\]]*)\]', win)
        if am:
            out["acp_args"] = [a.strip().strip('"') for a in am.group(1).split(',') if a.strip()]
print(json.dumps(out))
PY
)
  if [ -n "$APPCHECK" ]; then
    printf '%s\n' "$APPCHECK" > "$GUARD/app_launch_check.json"
    BAD=$(printf '%s' "$APPCHECK" | KEY="$AGENT_ID_KEY" WANT_BIN="$EXPECTED_BINARY" WANT_ARGV="$EXPECTED_ARGV" "$PY" -c '
import json, os, sys
d = json.load(sys.stdin)
key, want_bin, want_argv = os.environ["KEY"], os.environ["WANT_BIN"].split(), os.environ["WANT_ARGV"]
bad = []
name = d.get("agent_map", {}).get(key)
if name not in want_bin:
    bad.append("app now launches an agent binary we do not intercept: %r (map=%s)" % (name, d.get("agent_map")))
args = d.get("acp_args")
if args is None:
    bad.append("could not read the argv the app passes (bundle layout changed?)")
elif want_argv not in args:
    bad.append("app now calls the agent with argv %r, expected %r among them" % (args, want_argv))
print("; ".join(bad))')
    [ -n "$BAD" ] && flag "$BAD" || ok "app still launches the intercepted agent"
  else
    say "warn: could not parse the app bundle's agent map (bundle layout changed?)"
  fi
else
  say "note: $APP_BUNDLE not found at $ASAR"
fi

# a bundled runtime of our own name would bypass the bridge entirely
if [ -d "$APP_SUPPORT" ]; then
  for d in "$APP_SUPPORT"/acp-agents/installed-agents/*; do
    [ -e "$d" ] || continue
    case "$(basename "$d")" in $BUNDLED_RUNTIME_PATTERN) flag "the app installed its own bundled runtime: $d" ;; esac
  done
fi
if [ -d "$APP_BUNDLE" ] && find "$APP_BUNDLE" -maxdepth 5 -name "$BUNDLED_RUNTIME_PATTERN" 2>/dev/null | grep -q .; then
  flag "the app bundle now ships its own runtime matching $BUNDLED_RUNTIME_PATTERN (would bypass the bridge)"
fi

# ── 4. the tool surface the app exposes to the bridge ──────────────────────────
# The surface is not stable across app versions: it can narrow, widen, or lose the wrapper tool the
# profile's instructions name. Assert the tools this workflow needs are still present.
TOOLS_JSON=""
if [ -n "$APP_MCP_WRAPPER" ] && [ -x "$APP_MCP_WRAPPER" ]; then
  TOOLS_JSON=$("$PY" - "$APP_MCP_WRAPPER" <<'PY' 2>/dev/null
import json, select, subprocess, sys
p = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
def send(o):
    p.stdin.write(json.dumps(o) + "\n"); p.stdin.flush()
def recv(t=45):
    if not select.select([p.stdout], [], [], t)[0]: return None
    try: return json.loads(p.stdout.readline())
    except Exception: return None
send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "bridge-watchdog", "version": "1"}}})
init = recv()
send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
tl = recv()
tools = sorted(t["name"] for t in ((tl or {}).get("result") or {}).get("tools", []))
print(json.dumps({"server": ((init or {}).get("result") or {}).get("serverInfo", {}).get("version"),
                  "tool_count": len(tools), "tools": tools}))
p.terminate()
PY
)
fi
if [ -n "$TOOLS_JSON" ]; then
  printf '%s\n' "$TOOLS_JSON" > "$GUARD/app_tools_check.json"
  MISSING=$(printf '%s' "$TOOLS_JSON" | "$PY" -c "
import json, sys
d = json.load(sys.stdin); tools = d.get('tools') or []
if not tools:
    print('SKIP')
else:
    print(' '.join(t for t in '$REQUIRED_TOOLS'.split() if t not in tools))")
  case "$MISSING" in
    # an app with no project open exposes nothing — that is a note, never an anomaly
    SKIP) say "note: app not reachable (no project open?) — tool surface not checked" ;;
    "")   ok "tool surface ok ($(printf '%s' "$TOOLS_JSON" | "$PY" -c 'import json,sys;d=json.load(sys.stdin);print(str(d["tool_count"])+" tools, server "+str(d["server"]))'))" ;;
    *)    flag "tool surface changed: now missing $MISSING — see app_tools_check.json" ;;
  esac
else
  [ -n "$APP_MCP_WRAPPER" ] && say "note: app MCP wrapper not queryable ($APP_MCP_WRAPPER)" || say "note: no app MCP wrapper configured — tool surface not checked"
fi

# ── 5. the remote side ─────────────────────────────────────────────────────────
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
     "test -f '$REMOTE_PROFILE_FILE' && echo OK" 2>/dev/null | grep -q OK; then
  ok "remote profile reachable and present"
else
  flag "cannot reach the remote profile over ssh ($REMOTE_HOST:$REMOTE_PROFILE_FILE)"
fi

# ── 6. status + alert ──────────────────────────────────────────────────────────
[ "${SELFTEST:-}" = "1" ] && flag "SELFTEST: forced anomaly, verifying the alert path"

"$PY" - "$STATUS" "$ANOMALY" "$FINDINGS" "$APPCHECK" "$TOOLS_JSON" <<'PY' 2>/dev/null
import json, sys, time
status, anomaly, findings, appcheck, tools_json = sys.argv[1:6]
def load(s):
    try: return json.loads(s) if s else {}
    except Exception: return {}
json.dump({"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "ok": anomaly != "1",
           "findings": [f for f in findings.strip().split("\n") if f],
           "app_launch_check": load(appcheck),
           "app_tool_surface": load(tools_json)}, open(status, "w"), indent=2)
PY

if [ "$ANOMALY" = "1" ]; then
  MSG="$APP_NAME bridge check FAILED on $(hostname -s)

${FINDINGS}The agent may be running locally again instead of on $REMOTE_HOST.
Details: $GUARD/status.json"
  if printf '%s' "$MSG" | ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
       "$ALERT_CMD send --to '$ALERT_TARGET' --file -" >/dev/null 2>&1; then
    say "alert sent to $ALERT_TARGET"
  else
    say "ALERT DELIVERY FAILED — see status.json"
    /usr/bin/osascript -e "display notification \"$APP_NAME bridge check failed - see $GUARD/status.json\" with title \"agent bridge\"" >/dev/null 2>&1 || true
  fi
fi

exit 0
