#!/usr/bin/env bash
# PreToolUse hook for the Bash tool. Blocks a fixed list of destructive
# command patterns before they execute. Registered on the "Bash" matcher
# in .claude/settings.local.json.
#
# Input:  hook payload JSON on stdin, e.g. {"tool_name":"Bash","tool_input":{"command":"..."}}
# Output: on a match, JSON on stdout telling Claude Code to deny the call.
#         On no match, prints nothing and exits 0 so the command proceeds.
#
# This is a pattern-based heuristic, not a shell parser — it can be evaded
# by sufficiently obfuscated input and can over-block safe commands that
# merely contain a matching substring (e.g. inside a quoted string). Treat
# it as a safety net, not a security boundary.

set -euo pipefail

input="$(cat)"
command="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"

if [ -z "$command" ]; then
  exit 0
fi

deny() {
  # $1 = human-readable reason shown to the user/model.
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 0
}

# 1. rm -rf and equivalents: combined flag cluster in either order
#    (-rf, -fr, -Rf, -rfv, -vrf, ...) plus the long-form pair split across
#    separate flags (-r --force, --recursive -f, --recursive --force, ...).
if printf '%s' "$command" | grep -Eqi -- 'rm[[:space:]]+(-[a-z]*r[a-z]*f[a-z]*|-[a-z]*f[a-z]*r[a-z]*)\b'; then
  deny "재귀+강제 삭제(rm -rf 계열) 명령은 차단됩니다: $command"
fi
if printf '%s' "$command" | grep -Eqi -- 'rm\b.*(--recursive\b|-[a-z]*r[a-z]*\b)' \
  && printf '%s' "$command" | grep -Eqi -- 'rm\b.*(--force\b|-[a-z]*f[a-z]*\b)'; then
  deny "재귀+강제 삭제(rm -rf 계열) 명령은 차단됩니다: $command"
fi

# 2. git push --force / -f / --force-with-lease / --force-if-includes
if printf '%s' "$command" | grep -Eqi -- 'git[[:space:]]+push\b.*(--force([a-z-]*)?\b|[[:space:]]-f([[:space:]]|$))'; then
  deny "강제 푸시(git push --force 계열) 명령은 차단됩니다: $command"
fi

# 3. DROP TABLE (SQL)
if printf '%s' "$command" | grep -Eqi -- '\bdrop[[:space:]]+table\b'; then
  deny "DROP TABLE 명령은 차단됩니다: $command"
fi

# 4. dd if=... (raw disk/device writes)
if printf '%s' "$command" | grep -Eqi -- '\bdd\b.*\bif='; then
  deny "dd if= 명령은 차단됩니다: $command"
fi

exit 0
