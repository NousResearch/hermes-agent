#!/usr/bin/env bash
# Hermes diagnostics — one-shot system health check
# Uses $HERMES_HOME (defaults to ~/.hermes)
set -euo pipefail

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'; CYA='\033[0;36m'; NC='\033[0m'
errors=0; warnings=0
ok()   { echo -e "  ${GRN}✅${NC} $1"; }
warn() { echo -e "  ${YEL}⚠️ ${NC} $1"; warnings=$((warnings+1)); }
err()  { echo -e "  ${RED}❌${NC} $1"; errors=$((errors+1)); }
sec()  { echo -e "\n${CYA}── $1 ──${NC}"; }

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

echo -e "${CYA}══════════════════════════════════════${NC}"
echo -e "${CYA}  Hermes Diagnostics — $(hostname)${NC}"
echo -e "${CYA}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${CYA}══════════════════════════════════════${NC}"

sec "Gateway States"
for pdir in "$HERMES_HOME"/profiles/*/; do
  [ -d "$pdir" ] || continue
  name=$(basename "$pdir"); sf="${pdir}gateway_state.json"
  [ -f "$sf" ] || { err "${name}: no state"; continue; }
  state=$(python3 -c "import json; print(json.load(open('$sf')).get('gateway_state','unknown'))")
  case "$state" in running) ok "${name}: ${state}";; *) warn "${name}: ${state}";; esac
done

sec "Config"
VALIDATOR="${HERMES_HOME}/scripts/validate-config.sh"
if [ -f "$VALIDATOR" ]; then
  bash "$VALIDATOR" >/dev/null 2>&1 && ok "All configs valid" || err "Config issues (run validate-config.sh)"
else
  warn "Validator script not found at $VALIDATOR"
fi

sec "Cron Jobs"
echo "    Use cronjob(action='list') in a Hermes session to inspect active jobs."

sec "Disk Usage"
for mp in /; do
  pct=$(df -h "$mp" 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%')
  [ "$pct" -gt 85 ] && err "${mp}: ${pct}% used" || [ "$pct" -gt 70 ] && warn "${mp}: ${pct}% used" || ok "${mp}: ${pct}% used"
done

sec "Memory"
mem=$(free -m | awk '/Mem:/{printf "%.0f", $3/$2*100}')
swap=$(free -m | awk '/Swap:/{if($2>0) printf "%.0f", $3/$2*100; else print "0"}')
[ "$mem" -gt 85 ] && err "Memory: ${mem}% used" || [ "$mem" -gt 70 ] && warn "Memory: ${mem}% used" || ok "Memory: ${mem}% used"
[ "$swap" -gt 20 ] && warn "Swap: ${swap}% used" || ok "Swap: ${swap}% used"

sec "Uptime"
ok "$(uptime -p)"
ok "Load: $(cat /proc/loadavg | cut -d' ' -f1-3)"

sec "Recent Errors (last 24h)"
log_files=""
for pdir in "$HERMES_HOME"/profiles/*/; do
  [ -d "$pdir" ] || continue
  found=$(find "${pdir}logs/" -name "gateway*" -mtime -1 2>/dev/null || true)
  [ -n "$found" ] && log_files="$log_files$found"$'\n'
done
if [ -z "$log_files" ]; then
  ok "No log files found"
else
  err_count=$(echo "$log_files" | xargs grep -l "error\|traceback\|exception" 2>/dev/null | wc -l || true)
  [ "$err_count" -gt 0 ] && warn "${err_count} files with errors" || ok "No recent errors"
fi

sec "Skills"
total=$(ls -d "$HERMES_HOME"/skills/*/*/SKILL.md 2>/dev/null | wc -l)
ok "${total} skills installed"

echo ""
echo -e "${CYA}══════════════════════════════════════${NC}"
[ $errors -gt 0 -a $warnings -gt 0 ] && { echo -e "  ${RED}${errors} errors, ${warnings} warnings${NC}"; exit 2; }
[ $errors -gt 0 ] && { echo -e "  ${RED}${errors} errors${NC}"; exit 2; }
[ $warnings -gt 0 ] && { echo -e "  ${YEL}${warnings} warnings${NC}"; exit 1; }
echo -e "  ${GRN}All clear ✅${NC}"
