#!/bin/zsh
# Hermes gateway launchd wrapper.
# Sources ~/.hermes/.env so provider/platform keys reach the process,
# then execs the gateway with the exact original arguments.
# (launchd has no native EnvironmentFile; this is the sanctioned equivalent.)
#
# Preflight reaper (added 2026-06-28): a *hung* gateway (process alive but
# event loop dead) keeps holding the Slack Socket Mode connection, so a fresh
# launch hits "Slack app token already in use (PID …)" and crash-loops under
# KeepAlive. `gateway run --replace` does not cover this case because the stale
# process is still alive. Reap any pre-existing gateway here before starting,
# so the new instance always gets a free Slack app token.
emulate -L zsh
setopt no_unset

ENV_FILE="$HOME/.hermes/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# --- Preflight: reap any existing gateway (excluding ourselves) ---
# Match the exact module invocation so we never touch unrelated python procs.
GATEWAY_MATCH='hermes_cli.main gateway run'
typeset -a stale
stale=(${(f)"$(pgrep -f "$GATEWAY_MATCH" 2>/dev/null)"})
# drop our own PID and our parent (the launchd-spawned shell) if matched
stale=(${stale:#$$})
stale=(${stale:#$PPID})
if (( ${#stale} )); then
  # log each PID we're reaping to the gateway log (stdout -> gateway.log)
  for pid in ${stale[@]}; do
    echo "Killed stale gateway PID $pid"
  done
  kill -TERM ${stale[@]} 2>/dev/null
  # give the Slack adapter up to 5s to release the Socket Mode connection
  for _ in {1..5}; do
    sleep 1
    stale=(${(f)"$(pgrep -f "$GATEWAY_MATCH" 2>/dev/null)"})
    stale=(${stale:#$$}); stale=(${stale:#$PPID})
    (( ${#stale} )) || break
  done
  # anything still alive after the grace period gets SIGKILL
  if (( ${#stale} )); then
    kill -KILL ${stale[@]} 2>/dev/null
    sleep 1
  fi
fi

cd "$HOME/.hermes/hermes-agent" || exit 1
# When launchd invokes us, ProgramArguments passes the python interpreter
# plus module args (so --profile etc. are preserved end-to-end). For a manual
# no-arg invocation, fall back to a plain `gateway run --replace`.
if (( $# )); then
  exec "$@"
else
  exec "$HOME/.hermes/hermes-agent/venv/bin/python" -m hermes_cli.main gateway run --replace
fi
