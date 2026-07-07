#!/usr/bin/env bash
# Self-contained test matrix for BU-1 Hermes guardrail hooks.

set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WRITE_FENCE="$ROOT_DIR/bop/agent-hooks/write-fence.sh"
REPO_GUARD="$ROOT_DIR/bop/agent-hooks/repo-guard.sh"
PARSER_FILE="$ROOT_DIR/bop/agent-hooks/patch_parser.py"
PARSER_ABSENT="$ROOT_DIR/bop/agent-hooks/patch_parser.py.absent"

TMP_DIR=$(mktemp -d)
ORIG_HOME=${HOME:-}
export HOME="$TMP_DIR/home"

pass_count=0
fail_count=0

cleanup() {
  if [ -f "$PARSER_ABSENT" ] && [ ! -f "$PARSER_FILE" ]; then
    mv "$PARSER_ABSENT" "$PARSER_FILE"
  fi
  if [ -n "$ORIG_HOME" ]; then
    export HOME="$ORIG_HOME"
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p \
  "$HOME/assistant" \
  "$HOME/brain/raw" \
  "$HOME/brain/wiki/entities" \
  "$HOME/brain/wiki/concepts" \
  "$HOME/brain/wiki/synthesis" \
  "$HOME/brain/_meta" \
  "$HOME/osr/_intake" \
  "$HOME/.hermes/workspace" \
  "$HOME/.hermes/outbox" \
  "$HOME/.hermes/agent-hooks" \
  "$HOME/ds-max" \
  "$HOME/HERK-2/docs" \
  "$HOME/Documents"

ln -s "$HOME/ds-max" "$HOME/assistant/link-to-ds-max"

json_payload() {
  local tool_name=$1
  local cwd=$2
  local tool_json=$3
  python3 -c '
import json
import sys

tool_name = sys.argv[1]
cwd = sys.argv[2]
tool_input = json.loads(sys.stdin.read())
print(json.dumps({
    "hook_event_name": "pre_tool_call",
    "tool_name": tool_name,
    "tool_input": tool_input,
    "session_id": "test-session",
    "cwd": cwd,
    "extra": {},
}, separators=(",", ":")))
' "$tool_name" "$cwd" <<< "$tool_json"
}

run_case() {
  local name=$1
  local script=$2
  local payload=$3
  local expected=$4
  local actual

  actual=$(printf '%s' "$payload" | "$script")
  if [ "$actual" = "$expected" ]; then
    printf 'PASS %s\n' "$name"
    pass_count=$((pass_count + 1))
  else
    printf 'FAIL %s\n' "$name"
    printf '  expected: %s\n' "${expected:-<empty>}"
    printf '  actual:   %s\n' "${actual:-<empty>}"
    fail_count=$((fail_count + 1))
  fi
}

wf_payload() {
  json_payload "$1" "$HOME" "$2"
}

terminal_payload() {
  local command=$1
  local cwd=$2
  python3 -c '
import json
import sys

print(json.dumps({
    "hook_event_name": "pre_tool_call",
    "tool_name": "terminal",
    "tool_input": {"command": sys.argv[1]},
    "session_id": "test-session",
    "cwd": sys.argv[2],
    "extra": {},
}, separators=(",", ":")))
' "$command" "$cwd"
}

expect_path_denied='{"decision":"block","reason":"write-fence: path denied"}'
expect_outside='{"decision":"block","reason":"write-fence: path outside allowlist"}'
expect_missing='{"decision":"block","reason":"write-fence: missing target path"}'
expect_parser='{"decision":"block","reason":"write-fence: invalid patch parser"}'
expect_invalid_wf='{"decision":"block","reason":"write-fence: invalid hook input"}'
expect_invalid_rg='{"decision":"block","reason":"repo-guard: invalid hook input"}'
expect_ssn='{"decision":"block","reason":"write-fence: NPI pattern detected: ssn"}'
expect_fico='{"decision":"block","reason":"write-fence: NPI pattern detected: fico"}'
expect_income='{"decision":"block","reason":"write-fence: NPI pattern detected: income"}'
expect_git='{"decision":"block","reason":"repo-guard: protected repo git/gh mutation"}'
expect_file='{"decision":"block","reason":"repo-guard: protected repo file mutation"}'

run_case "write allowed assistant" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/ledger.md","content":"ok"}')" ""
run_case "write allowed hermes workspace" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/workspace/r.md","content":"ok"}')" ""
run_case "write allowed hermes outbox" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/outbox/r.md","content":"ok"}')" ""
run_case "write allowed hermes MEMORY" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/MEMORY.md","content":"ok"}')" ""
run_case "write allowed raw" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/raw/x.md","content":"ok"}')" ""
run_case "write allowed intake" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/osr/_intake/y.md","content":"ok"}')" ""
run_case "patch replace allowed" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"replace","path":"~/assistant/a.md","old_string":"a","new_string":"b"}')" ""
run_case "patch v4a allowed" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n*** Update File: ~/assistant/a.md\n@@\n-a\n+b\n*** End Patch"}')" ""

run_case "write blocked ds-max" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/ds-max/x.py","content":"ok"}')" "$expect_path_denied"
run_case "write blocked HERK-2" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/HERK-2/a.md","content":"ok"}')" "$expect_path_denied"
run_case "write blocked synthesis" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/wiki/synthesis/s.md","content":"ok"}')" "$expect_path_denied"
run_case "write blocked meta" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/_meta/rules.md","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes agent-hooks" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/agent-hooks/write-fence.sh","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes config" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/config.yaml","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes env" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/.env","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes auth" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/auth.json","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes SOUL" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/SOUL.md","content":"ok"}')" "$expect_path_denied"
run_case "write blocked hermes USER" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/.hermes/USER.md","content":"ok"}')" "$expect_path_denied"
run_case "write blocked Documents" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/Documents/z.txt","content":"ok"}')" "$expect_outside"
run_case "patch replace blocked Documents" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"replace","path":"~/Documents/z.txt","old_string":"a","new_string":"b"}')" "$expect_outside"
run_case "patch v4a blocked ds-max" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n*** Update File: ~/ds-max/x.py\n@@\n-a\n+b\n*** End Patch"}')" "$expect_path_denied"
run_case "patch v4a move ds-max blocked" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n*** Move File: ~/assistant/old.md -> ~/ds-max/new.md\n*** End Patch"}')" "$expect_path_denied"
run_case "patch v4a no-space add HERK-2 blocked" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n***Add File: ~/HERK-2/no.md\n+hello\n*** End Patch"}')" "$expect_path_denied"
run_case "traversal blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/../ds-max/x.py","content":"ok"}')" "$expect_path_denied"
run_case "symlink blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/link-to-ds-max/x.py","content":"ok"}')" "$expect_path_denied"

run_case "wiki entities toggle off" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/wiki/entities/e.md","content":"ok"}')" "$expect_outside"
touch "$HOME/.hermes/fence-wiki-enabled"
run_case "wiki entities marker on" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/wiki/entities/e.md","content":"ok"}')" ""
run_case "wiki synthesis marker denied" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/brain/wiki/synthesis/s.md","content":"ok"}')" "$expect_path_denied"
rm -f "$HOME/.hermes/fence-wiki-enabled"

run_case "NPI ssn blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/npi.md","content":"ssn 123-45-6789 FICO 742 income $85,000"}')" "$expect_ssn"
run_case "NPI spaced ssn blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/npi.md","content":"ssn 123 45 6789"}')" "$expect_ssn"
run_case "NPI fico blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/npi.md","content":"FICO 742"}')" "$expect_fico"
run_case "NPI income blocked" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/npi.md","content":"income $85,000"}')" "$expect_income"
run_case "NPI clean allowed" "$WRITE_FENCE" "$(wf_payload write_file '{"path":"~/assistant/clean.md","content":"loan stage: submitted, next action: call"}')" ""
run_case "patch missing target" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n*** End Patch"}')" "$expect_missing"
mv "$PARSER_FILE" "$PARSER_ABSENT"
run_case "patch parser missing blocked" "$WRITE_FENCE" "$(wf_payload patch '{"mode":"patch","patch":"*** Begin Patch\n*** Update File: ~/assistant/a.md\n@@\n-a\n+b\n*** End Patch"}')" "$expect_parser"
mv "$PARSER_ABSENT" "$PARSER_FILE"
run_case "malformed write fence" "$WRITE_FENCE" "not-json" "$expect_invalid_wf"

run_case "terminal git ds-max blocked" "$REPO_GUARD" "$(terminal_payload 'git -C ~/ds-max commit -m x' "$HOME")" "$expect_git"
run_case "terminal gh pr cwd blocked" "$REPO_GUARD" "$(terminal_payload 'gh pr create' "$HOME/ds-max")" "$expect_git"
run_case "terminal redirect HERK-2 blocked" "$REPO_GUARD" "$(terminal_payload 'echo hi > ~/HERK-2/f' "$HOME")" "$expect_file"
run_case "terminal sed ds-max blocked" "$REPO_GUARD" "$(terminal_payload "sed -i '' s/a/b/ ~/ds-max/f.py" "$HOME")" "$expect_file"
run_case "terminal git hermes allowed" "$REPO_GUARD" "$(terminal_payload 'git -C ~/.hermes/workspace commit -m receipts' "$HOME")" ""
run_case "terminal ls ds-max allowed" "$REPO_GUARD" "$(terminal_payload 'ls ~/ds-max' "$HOME")" ""
run_case "terminal cat HERK-2 allowed" "$REPO_GUARD" "$(terminal_payload 'cat ~/HERK-2/docs/x.md' "$HOME")" ""
run_case "malformed repo guard" "$REPO_GUARD" "not-json" "$expect_invalid_rg"

if [ "$fail_count" -ne 0 ]; then
  printf 'hook-matrix: %d passed, %d failed\n' "$pass_count" "$fail_count"
  exit 1
fi

printf 'hook-matrix: %d passed, 0 failed\n' "$pass_count"
