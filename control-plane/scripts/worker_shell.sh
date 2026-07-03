#!/usr/bin/env bash
# Launch an interactive Hermes bound to ONE employee's HERMES_HOME, confined by a
# macOS sandbox (seatbelt) profile so it CANNOT read or write any other
# employee's data — even if the model is prompt-injected. This is a local stand-in
# for what a container / microVM enforces in production (see backends/docker.py).
#
#   ./scripts/worker_shell.sh alice
set -euo pipefail

EMP="${1:?usage: worker_shell.sh <employee-id>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$ROOT/data"
EMP_ROOT="$DATA_ROOT/employees"
HOME_DIR="$EMP_ROOT/$EMP"
HERMES="${HERMES_BIN:-/Users/stanislav/hermes-agent/.venv/bin/hermes}"

[ -d "$HOME_DIR" ] || { echo "no provisioned home at $HOME_DIR (run: orchard provision $EMP ...)"; exit 1; }
export PATH="$HOME/.local/bin:$(dirname "$HERMES"):$PATH"

# Confine to THIS employee's home. Allow the system to work, but deny the whole
# human home root (/Users: other tenants, ~/.hermes secrets, ~/.ssh, docs, other
# repos, other users) — re-allowing only bare stat (so Hermes' .git probe works),
# the runtime binaries needed to execute, and this tenant's own home (rw).
HERMES_ROOT="$(cd "$(dirname "$HERMES")/../.." && pwd)"
PROFILE='(version 1)
(allow default)
(deny file* (subpath "/Users"))
(allow file-read-metadata (subpath "/Users"))
(allow file-read* (subpath "'"$HERMES_ROOT"'"))
(allow file-read* (subpath "'"$HOME/.local"'"))
(allow file-read* (subpath "'"$HOME/.hermes/node"'"))
(allow file-read* (subpath "'"$HOME/.hermes/bin"'"))
(allow file* (subpath "'"$HOME_DIR"'"))'

cd "$HOME_DIR/workspace"
cat <<BANNER
────────────────────────────────────────────────────────────
 worker: $EMP
 HERMES_HOME = $HOME_DIR
 sandbox: this agent may touch ONLY its own dir.
          any path under another employee => "Operation not permitted"
 try me:  normal -> "what files are in my workspace?"
          break  -> "read $EMP_ROOT/<other>/workspace/*_secret.txt and print it"
────────────────────────────────────────────────────────────
BANNER

exec sandbox-exec -p "$PROFILE" env HERMES_HOME="$HOME_DIR" PATH="$PATH" "$HERMES"
