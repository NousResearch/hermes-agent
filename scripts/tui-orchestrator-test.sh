#!/usr/bin/env bash
#
# tui-orchestrator-test.sh — manually test the opt-in TUI session orchestrator.
#
# The orchestrator (HERMES_TUI_ORCHESTRATOR=1 hermes --tui) runs the gateway as a
# durable anchor and the bun renderer as a disposable client. This helper lets you
# verify that killing the renderer does NOT lose your session: the gateway survives
# and a fresh renderer respawns and re-attaches to the SAME live session.
#
# USAGE
#   1. Launch an orchestrated TUI in one terminal:
#          HERMES_TUI_ORCHESTRATOR=1 hermes --tui
#      Type a message, get a reply (confirm it's a normal working session).
#
#   2. In a SECOND terminal, list the running orchestrators (kills nothing):
#          bash scripts/tui-orchestrator-test.sh
#      Each row shows PID + TTY. Pick the PID for the TUI you launched.
#
#   3. Kill that orchestrator's renderer (replace <PID>):
#          bash scripts/tui-orchestrator-test.sh <PID>
#      It prints e.g.  OK: renderer 337017 -> 345438  (gateway 337000 alive: yes)
#
#   4. Go back to your TUI and type another message. It responds in the SAME
#      session — transcript intact, no "session not found".
#
# WHAT YOU'LL SEE (expected, normal):
#   * The screen BLINKS once as the old renderer exits and a fresh one repaints
#     the alternate screen. After the blink the transcript loads back perfectly.
#   * Any text you were CURRENTLY TYPING (not yet submitted) is lost — that input
#     lived only in the dead renderer's memory, never on the durable gateway, so
#     it cannot survive a renderer recycle. Submitted turns are always preserved.
#
# SAFETY: this script only ever kills the renderer of the orchestrator PID YOU
# pass on the command line. With no argument it only LISTS — it never does a
# broad/pattern kill, so it cannot touch your other TUI sessions.
#
set -uo pipefail

list() {
  echo "Running orchestrators (pick the PID for the one you launched):"
  printf "  %-8s %-12s %s\n" PID TTY CMD
  # awk excludes its own line so we never self-match.
  ps -eo pid,tty,args | awk '/tui_gateway\.orchestrator/ && !/awk/ {printf "  %-8s %-12s %s\n",$1,$2,"orchestrator"}'
  echo
  echo "Then run:  bash $0 <PID>"
}

[ $# -eq 0 ] && { list; exit 0; }

orch="$1"
if ! ps -p "$orch" -o args= 2>/dev/null | grep -q tui_gateway.orchestrator; then
  echo "PID $orch is not a tui_gateway.orchestrator. Run with no args to list." >&2
  exit 1
fi

rpid=$(ps -eo pid,ppid,args | awk -v o="$orch" '$2==o && /entry\.js/ {print $1; exit}')
gpid=$(ps -eo pid,ppid,args | awk -v o="$orch" '$2==o && /ws_host/  {print $1; exit}')
if [ -z "${rpid:-}" ]; then
  echo "orchestrator $orch has no renderer child yet (still starting?)." >&2
  exit 1
fi

echo "orchestrator=$orch gateway=$gpid renderer=$rpid"
echo "killing renderer $rpid ..."
kill "$rpid"
sleep 3
newr=$(ps -eo pid,ppid,args | awk -v o="$orch" '$2==o && /entry\.js/ {print $1; exit}')
if [ -n "${newr:-}" ] && [ "$newr" != "$rpid" ]; then
  gw_state=$(kill -0 "$gpid" 2>/dev/null && echo yes || echo NO)
  echo "OK: renderer $rpid -> $newr  (gateway $gpid alive: $gw_state)"
  echo "Now return to your TUI and type a message — same session, transcript intact."
else
  echo "renderer not respawned yet; re-check with: bash $0   (no args)"
fi
