#!/usr/bin/env bash
# Hermes pre_tool_call write fence.
#
# Best-effort defense-in-depth for write_file and patch tool calls. The hook
# fails closed on malformed input and only allows writes under the approved
# roots below. Paths are resolved through realpath before allow/deny checks so
# ../ traversal and symlinks cannot escape the fence.
#
# Wiki entity/concept writes are disabled by default. Enable them by creating
# ~/.hermes/fence-wiki-enabled; HERMES_FENCE_WIKI=1 is also honored for tests
# and ad-hoc runs. The synthesis and _meta wiki roots are always denied.
#
# V4A patch-mode paths are parsed with the vendored Hermes patch_parser.py
# beside this hook. Parser import or parse failures block. The NPI scan is
# best-effort pattern matching only; it is not a sound detector.

set -u

_block() {
  printf '{"decision":"block","reason":"%s"}\n' "$1"
}

payload=$(cat)
python_bin=$(command -v python3 || true)
script_path=${BASH_SOURCE[0]:-$0}

if command -v realpath >/dev/null 2>&1; then
  script_path=$(realpath "$script_path" 2>/dev/null || printf '%s' "$script_path")
fi

hook_dir=$(cd "$(dirname "$script_path")" 2>/dev/null && pwd -P || true)

if [ -z "$python_bin" ]; then
  _block "write-fence: python3 unavailable"
  exit 0
fi

output=$(HERMES_WRITE_FENCE_HOOK_DIR="$hook_dir" "$python_bin" -c '
import importlib.util
import json
import os
import re
import sys


def emit_block(reason):
    sys.stdout.write(json.dumps(
        {"decision": "block", "reason": reason},
        separators=(",", ":"),
    ) + "\n")


def fail(reason):
    emit_block(reason)
    sys.exit(0)


def truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def home_path(*parts):
    return os.path.realpath(os.path.join(os.path.expanduser("~"), *parts))


def resolve_path(raw_path, cwd):
    if not isinstance(raw_path, str) or not raw_path.strip():
        fail("write-fence: missing target path")
    expanded = os.path.expanduser(raw_path)
    if not os.path.isabs(expanded):
        base = cwd if isinstance(cwd, str) and cwd else os.getcwd()
        expanded = os.path.join(base, expanded)
    return os.path.realpath(os.path.abspath(expanded))


def is_inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def extract_v4a_paths(patch_text):
    if not isinstance(patch_text, str) or not patch_text:
        return []

    hook_dir = os.environ.get("HERMES_WRITE_FENCE_HOOK_DIR", "")
    parser_path = os.path.join(hook_dir, "patch_parser.py") if hook_dir else ""
    if not hook_dir or not os.path.isfile(parser_path):
        fail("write-fence: invalid patch parser")

    try:
        spec = importlib.util.spec_from_file_location("patch_parser", parser_path)
        if spec is None or spec.loader is None:
            fail("write-fence: invalid patch parser")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parse_v4a_patch = module.parse_v4a_patch
    except Exception:
        fail("write-fence: invalid patch parser")

    try:
        operations, error = parse_v4a_patch(patch_text)
    except Exception:
        fail("write-fence: invalid patch input")
    if error is not None:
        fail("write-fence: invalid patch input")

    paths = []
    for op in operations:
        file_path = getattr(op, "file_path", None)
        new_path = getattr(op, "new_path", None)
        if file_path:
            paths.append(file_path)
        if new_path:
            paths.append(new_path)
    return paths


def collect_targets(tool_name, tool_input):
    if tool_name == "write_file":
        return [tool_input.get("path")]
    if tool_name != "patch":
        fail("write-fence: invalid hook input")

    targets = []
    mode = tool_input.get("mode", "replace")
    path = tool_input.get("path")
    if path:
        targets.append(path)
    if mode == "patch":
        targets.extend(extract_v4a_paths(tool_input.get("patch")))
    return targets


def npi_reason(tool_input):
    fields = []
    for key in ("content", "old_string", "new_string", "patch"):
        value = tool_input.get(key)
        if isinstance(value, str):
            fields.append(value)
    text = "\n".join(fields)
    if not text:
        return None

    checks = [
        ("ssn", re.compile(r"\b(?:\d{3}[- .]\d{2}[- .]\d{4}|\d{9})\b")),
        ("fico", re.compile(r"\b(fico|credit\s*score)\b\D{0,24}\d{3}\b", re.IGNORECASE)),
        ("income", re.compile(r"\b(income|salary|agi)\b\D{0,24}\$?\d{2,3}[,.]?\d{3}\b", re.IGNORECASE)),
        ("acct", re.compile(r"\b(account|acct|routing)\s*(number|no\.?|#)?\s*[:=]?\s*\d{6,17}\b", re.IGNORECASE)),  # bare 9-digit hits ssn first; acct catches 6-17 digit account/routing forms the ssn shape misses
    ]
    for label, pattern in checks:
        if pattern.search(text):
            return label
    return None


try:
    payload = sys.stdin.read()
    data = json.loads(payload)
except Exception:
    fail("write-fence: invalid hook input")

if not isinstance(data, dict):
    fail("write-fence: invalid hook input")

tool_name = data.get("tool_name")
tool_input = data.get("tool_input")
cwd = data.get("cwd")

if tool_name not in {"write_file", "patch"} or not isinstance(tool_input, dict):
    fail("write-fence: invalid hook input")

hit = npi_reason(tool_input)
if hit:
    fail(f"write-fence: NPI pattern detected: {hit}")

deny_roots = [
    home_path("ds-max"),
    home_path("HERK-2"),
    home_path("brain", "wiki", "synthesis"),
    home_path("brain", "_meta"),
    home_path(".hermes", "agent-hooks"),
    home_path(".hermes", "config.yaml"),
    home_path(".hermes", ".env"),
    home_path(".hermes", "auth.json"),
    home_path(".hermes", "SOUL.md"),
    home_path(".hermes", "USER.md"),
]

allow_roots = [
    home_path("assistant"),
    home_path("brain", "raw"),
    home_path("osr", "_intake"),
    home_path(".hermes", "workspace"),
    home_path(".hermes", "outbox"),
]

allow_paths = [
    home_path(".hermes", "MEMORY.md"),
]

wiki_marker = home_path(".hermes", "fence-wiki-enabled")
if os.path.exists(wiki_marker) or truthy(os.environ.get("HERMES_FENCE_WIKI")):
    allow_roots.extend([
        home_path("brain", "wiki", "entities"),
        home_path("brain", "wiki", "concepts"),
    ])

crm_marker = home_path(".hermes", "fence-crm-enabled")
if os.path.exists(crm_marker) or truthy(os.environ.get("HERMES_FENCE_CRM")):
    allow_roots.extend([
        home_path("ai-agency", "_intake"),
    ])

targets = collect_targets(tool_name, tool_input)
if not targets:
    fail("write-fence: missing target path")

for raw_target in targets:
    resolved = resolve_path(raw_target, cwd)
    if any(is_inside(resolved, root) for root in deny_roots):
        fail("write-fence: path denied")
    if resolved not in allow_paths and not any(is_inside(resolved, root) for root in allow_roots):
        fail("write-fence: path outside allowlist")
' <<< "$payload" 2>/dev/null)
status=$?

if [ "$status" -ne 0 ]; then
  _block "write-fence: internal error"
  exit 0
fi

printf '%s' "$output"
