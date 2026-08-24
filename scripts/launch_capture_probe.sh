#!/usr/bin/env bash
# Probe-the-probe for launch-capture/sitecustomize.py, no real install needed.
# Control rows FIRST: without the opt-in env var, and for non-launch argv
# shapes, subprocess.run must behave untouched. Then treatment rows: both
# launch shapes must be captured without spawning.
set -euo pipefail
CAP_DIR="$(cd "$(dirname "$0")" && pwd)/../tests/install/e2e-assets/launch-capture"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

run_py() {
  # $1: with_var (yes/no); rest: python -c payload
  local with_var="$1"; shift
  if [ "$with_var" = yes ]; then
    PYTHONPATH="$CAP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      HERMES_E2E_CAPTURE_LAUNCH="$WORK/spec.json" python3 "$@"
  else
    PYTHONPATH="$CAP_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
  fi
}

fail() { echo "PROBE FAILED: $*" >&2; exit 1; }

echo "--- control 1: no env var -> run() untouched, real spawn happens"
out="$(run_py no -c 'import subprocess; print(subprocess.run(["echo","real-spawn"],capture_output=True,text=True).stdout.strip())')"
[ "$out" = "real-spawn" ] || fail "control 1: expected real-spawn, got '$out'"
[ ! -e "$WORK/spec.json" ] || fail "control 1: spec written without opt-in"
echo "OK"

echo "--- control 2: env var set, NON-launch argv (npm run pack) -> passthrough"
rm -f "$WORK"/spec.json*
out="$(run_py yes -c 'import subprocess; r=subprocess.run(["echo","npm-build-ran"],capture_output=True,text=True); print(r.stdout.strip())')"
[ "$out" = "npm-build-ran" ] || fail "control 2: echo did not run"
# and an actual npm-shaped BUILD argv (argv[0]=npm but no electron token):
run_py yes -c 'import subprocess,sys; r=subprocess.run(["npm","run","pack"],capture_output=True); sys.exit(0)' 2>/dev/null || true
[ ! -e "$WORK/spec.json" ] || fail "control 2: build argv was wrongly captured"
echo "OK"

echo "--- treatment 1: source shape (npm exec -- electron .) captured, not spawned"
rm -f "$WORK"/spec.json*
run_py yes -c '
import subprocess
r = subprocess.run(["npm", "exec", "--", "electron", "."], cwd="/tmp", env={"HERMES_DESKTOP_CWD": "/tmp", "PATH": "/usr/bin"})
assert r.returncode == 0, r
'
[ -e "$WORK/spec.json" ] || fail "treatment 1: no spec written"
[ "$(cat "$WORK/spec.json.captured")" = "source" ] || fail "treatment 1: wrong shape"
python3 - "$WORK/spec.json" <<'EOF'
import json, sys
spec = json.load(open(sys.argv[1]))
assert spec["argv"] == ["npm", "exec", "--", "electron", "."], spec["argv"]
assert spec["cwd"] == "/tmp", spec["cwd"]
assert spec["env"]["HERMES_DESKTOP_CWD"] == "/tmp", "env= kwarg not captured"
assert spec["matchedShape"] == "source"
print("spec contents OK")
EOF
echo "OK"

echo "--- treatment 2: packaged shape captured, not spawned"
rm -f "$WORK"/spec.json*
run_py yes -c '
import subprocess
exe = "/x/apps/desktop/release/linux-unpacked/Hermes"
r = subprocess.run([exe, "--no-sandbox"], cwd="/tmp", env={"PATH": "/usr/bin"})
assert r.returncode == 0, r   # a real spawn of this path would ENOENT
'
[ "$(cat "$WORK/spec.json.captured")" = "packaged" ] || fail "treatment 2: wrong shape"
echo "OK"

echo "--- treatment 3: windows-style packaged argv matches too"
rm -f "$WORK"/spec.json*
run_py yes -c '
import subprocess
r = subprocess.run(["C:\\x\\apps\\desktop\\release\\win-unpacked\\Hermes.exe"], env={})
assert r.returncode == 0
'
[ "$(cat "$WORK/spec.json.captured")" = "packaged" ] || fail "treatment 3: wrong shape"
echo "OK"

echo "ALL PROBES PASSED"
