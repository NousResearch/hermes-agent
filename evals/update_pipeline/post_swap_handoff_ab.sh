#!/usr/bin/env bash
# Live A/B for the post-swap hand-off (hermes_cli/update_handoff.py).
#
# Installs a real clone at <installed-sha>, publishes a target commit on its origin that
# (a) adds a symbol to hermes_logging (purge-protected on the old updater),
# (b) imports it at module level from hermes_cli/config.py (re-imported by the old updater's
#     config-migration step → stale-symbol ImportError class, #111271/#112604), and
# (c) tags the completion line from update_cmd_maint.py with which code printed it,
# then runs `hermes update` from the installed clone and reads the receipt + output.
#
#   evals/update_pipeline/post_swap_handoff_ab.sh <repo> <installed-sha> <label>
#
# Phase A (installed-sha = merge-base): the tail runs in the pre-pull interpreter — old
# completion line, "Could not check config version". Phase B (head): tag present, no warning.
# Needs: uv on PATH, a Python for the venv. Runs with a disposable HOME/HERMES_HOME, no npm
# on PATH (skips Node/web/Desktop work) and --no-gateway-restart (never touches your fleet).
set -euo pipefail
REPO=$1; SHA=$2; LABEL=$3
ROOT=$(mktemp -d -t hermes-post-swap-ab.XXXXXX)
echo "== [$LABEL] scratch: $ROOT (installed at ${SHA:0:10})"
git clone -q --bare --shared "$REPO" "$ROOT/origin.git"
git --git-dir="$ROOT/origin.git" update-ref refs/heads/main "$SHA"
git clone -q --shared -b main "$ROOT/origin.git" "$ROOT/install"

# Target commit: what the NEXT update pulls.
git clone -q --shared -b main "$ROOT/origin.git" "$ROOT/author"
(
  cd "$ROOT/author"
  printf '\n\ndef post_swap_probe() -> str:\n    """Added by the update under test."""\n    return "pulled-code"\n' >> hermes_logging.py
  python3 - <<'PY'
import re
p = "hermes_cli/config.py"; s = open(p).read()
s = s.replace("import logging\n", "import logging\nfrom hermes_logging import post_swap_probe  # noqa: F401  (A/B probe: new symbol on a cached module)\n", 1)
open(p, "w").write(s)
p = "hermes_cli/update_cmd_maint.py"; s = open(p).read()
needle = 'def _update_complete_message('
i = s.index(needle); j = s.index('\n', s.index('"""', s.index('"""', i) + 3)) + 1  # end of docstring
s = s[:j] + '    from hermes_logging import post_swap_probe\n    print(f"  [completion printed by: {post_swap_probe()}]")\n' + s[j:]
open(p, "w").write(s)
PY
  git -c user.name=ab -c user.email=ab@example.invalid commit -qam "probe: new hermes_logging symbol used by config.py and the completion line"
  git push -q origin main
)

# Venv for the installed clone (deps unchanged between the two commits → the updater skips the sync).
uv venv -q --python "${HERMES_AB_PYTHON:-3.11}" "$ROOT/install/.venv"
uv pip install -q --python "$ROOT/install/.venv/bin/python" -e "$ROOT/install"

export HOME="$ROOT/home" HERMES_HOME="$ROOT/home/.hermes"
mkdir -p "$HERMES_HOME"
export PATH="$ROOT/install/.venv/bin:/usr/bin:/bin"
set +e
( cd "$ROOT/install" && python -m hermes_cli.main update --yes --no-backup --no-gateway-restart ) > "$ROOT/update.out" 2>&1
CODE=$?
set -e
echo "== [$LABEL] exit=$CODE  head=$(git -C "$ROOT/install" rev-parse --short HEAD)"
grep -nE "completion printed by|Could not check config version|Update complete|Code updated|Traceback|Error" "$ROOT/update.out" | head -20 || true
python3 - "$HERMES_HOME/logs/update_receipts/latest.json" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as exc:
    print("receipt: NONE", exc); sys.exit(0)
print("receipt:", r.get("outcome"), "exit", r.get("exit_code"), "pid", r.get("pid"), "post_swap_pid", r.get("post_swap_pid"),
      "steps", [s["name"] for s in r.get("steps", [])], "stop", (r.get("stop_reason") or "")[:60])
PY
if grep -q "completion printed by: pulled-code" "$ROOT/update.out" && ! grep -q "Could not check config version" "$ROOT/update.out"; then
  echo "VERDICT: [$LABEL] tail ran on the PULLED code"
else
  echo "VERDICT: [$LABEL] tail ran in the PRE-PULL interpreter (stale-symbol class reachable)"
fi
echo "   log: $ROOT/update.out"
