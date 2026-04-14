#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMUX_BOOTSTRAP="$SCRIPT_DIR/tmux_bootstrap.sh"

have() { command -v "$1" >/dev/null 2>&1; }

tailscale_ssh_state() {
  if ! have tailscale; then
    echo "not-installed"
    return 0
  fi

  local runssh=""
  if have jq; then
    runssh="$(tailscale debug prefs 2>/dev/null | jq -r 'if has("RunSSH") then (.RunSSH|tostring) else "" end' || true)"
  else
    runssh="$(tailscale debug prefs 2>/dev/null | grep -Eo '"RunSSH":(true|false)' | cut -d: -f2 || true)"
  fi

  case "$runssh" in
    true) echo "enabled" ;;
    false) echo "disabled" ;;
    *) echo "unknown" ;;
  esac
}

tailscale_dns_name() {
  if ! have tailscale; then
    return 0
  fi

  local dns=""
  if have jq; then
    dns="$(tailscale status --json 2>/dev/null | jq -r '.Self.DNSName // empty' || true)"
  fi

  dns="${dns%.}"
  [[ -n "$dns" ]] && echo "$dns"
}

usage() {
  cat <<'EOF'
Usage:
  remote_ops.sh status [--compact]
  remote_ops.sh check
  remote_ops.sh restart tmux [session]
  remote_ops.sh restart openclaw
  remote_ops.sh attach [session]

Examples:
  ./scripts/holiday_remote/remote_ops.sh status
  ./scripts/holiday_remote/remote_ops.sh check
  ./scripts/holiday_remote/remote_ops.sh restart tmux holiday
  ./scripts/holiday_remote/remote_ops.sh attach holiday
EOF
}

print_status() {
  local compact="${1:-false}"
  local now host user shell_name

  now="$(date '+%F %T %Z')"
  host="$(hostname)"
  user="$(whoami)"
  shell_name="${SHELL:-unknown}"

  if [[ "$compact" == "true" ]]; then
    printf "time=%s host=%s user=%s shell=%s\n" "$now" "$host" "$user" "$shell_name"
    if have tailscale; then
      printf "tailscale_ipv4=%s\n" "$(tailscale ip -4 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo unavailable)"
      printf "tailscale_ssh=%s\n" "$(tailscale_ssh_state)"
      local dns_compact
      dns_compact="$(tailscale_dns_name || true)"
      [[ -n "$dns_compact" ]] && printf "tailscale_dns=%s\n" "$dns_compact"
    else
      printf "tailscale=not-installed\n"
    fi
    if have tmux; then
      printf "tmux_sessions=%s\n" "$(tmux ls 2>/dev/null | wc -l | tr -d ' ')"
    else
      printf "tmux=not-installed\n"
    fi
    printf "uptime=%s\n" "$(uptime -p 2>/dev/null || echo unavailable)"
    return 0
  fi

  echo "== Remote Ops Status =="
  echo "time     : $now"
  echo "host     : $host"
  echo "user     : $user"
  echo "shell    : $shell_name"
  echo "repo     : $REPO_ROOT"

  echo
  echo "-- SSH --"
  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    echo "session  : remote (SSH_CONNECTION set)"
    echo "from/to  : ${SSH_CONNECTION}"
  else
    echo "session  : local shell"
  fi
  if pgrep -x sshd >/dev/null 2>&1; then
    echo "sshd     : running"
  else
    echo "sshd     : not detected (expected if using Tailscale SSH)"
  fi

  echo
  echo "-- Tailscale --"
  if have tailscale; then
    echo "ipv4     : $(tailscale ip -4 2>/dev/null | tr '\n' ' ' || echo unavailable)"
    echo "ipv6     : $(tailscale ip -6 2>/dev/null | tr '\n' ' ' || echo unavailable)"
    echo "ssh      : $(tailscale_ssh_state)"
    local dns
    dns="$(tailscale_dns_name || true)"
    [[ -n "$dns" ]] && echo "dns      : $dns"
    echo "status   :"
    tailscale status 2>/dev/null | sed -n '1,8p' || echo "  unable to read tailscale status"
  else
    echo "tailscale: command not found"
  fi

  echo
  echo "-- tmux --"
  if have tmux; then
    if tmux ls >/dev/null 2>&1; then
      tmux ls
    else
      echo "no tmux server/sessions"
    fi
  else
    echo "tmux: command not found"
  fi

  echo
  echo "-- Host --"
  echo "uptime   : $(uptime -p 2>/dev/null || echo unavailable)"
  echo "load     : $(uptime 2>/dev/null | awk -F'load average:' '{print $2}' | xargs || echo unavailable)"
  echo "disk     :"
  df -h "$REPO_ROOT" | sed -n '1,2p'
  echo "memory   :"
  free -h 2>/dev/null | sed -n '1,2p' || echo "free command unavailable"
}

run_check() {
  local fail=0

  echo "== Holiday Remote Preflight Check =="

  if [[ -d "$REPO_ROOT" ]]; then
    echo "[ok] workspace exists: $REPO_ROOT"
  else
    echo "[fail] workspace missing: $REPO_ROOT"
    fail=1
  fi

  if have tmux; then
    echo "[ok] tmux installed"
  else
    echo "[fail] tmux command not found"
    fail=1
  fi

  if have tailscale; then
    echo "[ok] tailscale command available"
    if tailscale status >/dev/null 2>&1; then
      echo "[ok] tailscale status readable"
    else
      echo "[warn] tailscale installed but status unavailable (service may be down/login required)"
    fi

    local ts_ssh dns
    ts_ssh="$(tailscale_ssh_state)"
    dns="$(tailscale_dns_name || true)"
    case "$ts_ssh" in
      enabled)
        echo "[ok] tailscale ssh enabled"
        [[ -n "$dns" ]] && echo "[ok] tailscale ssh target: $(whoami)@$dns"
        ;;
      disabled)
        echo "[warn] tailscale ssh disabled (recommended: enable before travel)"
        ;;
      *)
        echo "[warn] tailscale ssh state unknown"
        ;;
    esac
  else
    echo "[warn] tailscale command not found"
  fi

  if pgrep -x sshd >/dev/null 2>&1; then
    echo "[info] openssh sshd is running"
  else
    echo "[info] openssh sshd not detected (acceptable when using tailscale ssh)"
  fi

  if [[ -x "$TMUX_BOOTSTRAP" ]]; then
    echo "[ok] tmux bootstrap script executable"
  else
    echo "[warn] tmux bootstrap script not executable: $TMUX_BOOTSTRAP"
  fi

  if have openclaw; then
    echo "[ok] openclaw CLI available"
  else
    echo "[warn] openclaw CLI not found"
  fi

  if [[ "$fail" -eq 0 ]]; then
    echo "[pass] critical checks passed"
  else
    echo "[fail] critical checks failed"
  fi

  return "$fail"
}

restart_cmd() {
  local target="${1:-}"
  case "$target" in
    tmux)
      local session="${2:-holiday}"
      "$TMUX_BOOTSTRAP" "$session" "$REPO_ROOT" --force-reset
      echo "tmux session reset: $session"
      ;;
    openclaw)
      if ! have openclaw; then
        echo "openclaw CLI not found"
        return 1
      fi
      openclaw gateway restart
      ;;
    *)
      echo "Unknown restart target: ${target:-<empty>}"
      usage
      return 1
      ;;
  esac
}

attach_cmd() {
  local session="${1:-holiday}"
  if ! have tmux; then
    echo "tmux command not found"
    return 1
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    exec tmux attach -t "$session"
  fi

  "$TMUX_BOOTSTRAP" "$session" "$REPO_ROOT"
  exec tmux attach -t "$session"
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "$cmd" in
    status)
      local compact="false"
      if [[ "${1:-}" == "--compact" ]]; then
        compact="true"
      fi
      print_status "$compact"
      ;;
    check)
      run_check
      ;;
    restart)
      restart_cmd "$@"
      ;;
    attach)
      attach_cmd "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown command: $cmd"
      usage
      return 1
      ;;
  esac
}

main "$@"
