#!/usr/bin/env bash
# Hermes pre_tool_call terminal repo guard.
#
# This hook is a best-effort, trivially evadable string classifier for
# protected repository mutations through the terminal tool. Shell aliases,
# wrappers, escaped strings, encoded payloads, and equivalent command forms can
# bypass it. It must never be the sole control: keep approvals.mode: manual and
# run Hermes without push credentials for ds-max/HERK-2. This hook only adds
# friction for common direct git/gh and file-mutation command shapes.

set -u

_block() {
  printf '{"decision":"block","reason":"%s"}\n' "$1"
}

payload=$(cat)
python_bin=$(command -v python3 || true)

if [ -z "$python_bin" ]; then
  _block "repo-guard: python3 unavailable"
  exit 0
fi

output=$("$python_bin" -c '
import json
import os
import re
import shlex
import sys


def emit_block(reason):
    sys.stdout.write(json.dumps(
        {"decision": "block", "reason": reason},
        separators=(",", ":"),
    ) + "\n")


def fail(reason):
    emit_block(reason)
    sys.exit(0)


def home_path(*parts):
    return os.path.realpath(os.path.join(os.path.expanduser("~"), *parts))


def resolve_cwd(raw_cwd):
    if isinstance(raw_cwd, str) and raw_cwd:
        return os.path.realpath(os.path.abspath(os.path.expanduser(raw_cwd)))
    return os.path.realpath(os.getcwd())


def is_inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def command_mentions_protected_repo(command, protected_roots):
    home = os.path.expanduser("~")
    separator = r"(^|[\s=:;|&<>\"" + chr(39) + r"])"
    textual_patterns = [
        separator + r"(~?/)?ds-max(/|\s|$)",
        separator + r"(~?/)?HERK-2(/|\s|$)",
        re.escape(os.path.join(home, "ds-max")),
        re.escape(os.path.join(home, "HERK-2")),
    ]
    for pattern in textual_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    for root in protected_roots:
        if root in command:
            return True
    return False


def protected_context(command, cwd):
    protected_roots = [home_path("ds-max"), home_path("HERK-2")]
    if any(is_inside(cwd, root) for root in protected_roots):
        return True
    return command_mentions_protected_repo(command, protected_roots)


def has_git_or_gh_mutation(command):
    lowered = command.lower()
    git_pattern = re.compile(r"(?<![\w.-])git(?:\s+(?:-[A-Za-z]\s+\S+|--[A-Za-z0-9_-]+(?:=\S+)?))*\s+(push|commit|merge|rebase|reset|worktree)\b", re.IGNORECASE)
    gh_pattern = re.compile(r"(?<![\w.-])gh\s+pr\b", re.IGNORECASE)
    if git_pattern.search(command) or gh_pattern.search(command):
        return True

    try:
        tokens = shlex.split(command)
    except ValueError:
        return any(word in lowered for word in ("git ", "gh pr"))

    for i, token in enumerate(tokens):
        base = os.path.basename(token).lower()
        if base == "git":
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if t == "-C" and j + 1 < len(tokens):
                    j += 2
                    continue
                if t.startswith("-"):
                    j += 2 if j + 1 < len(tokens) and not tokens[j + 1].startswith("-") else 1
                    continue
                return t.lower() in {"push", "commit", "merge", "rebase", "reset", "worktree"}
        if base == "gh" and i + 1 < len(tokens) and tokens[i + 1].lower() == "pr":
            return True
    return False


def has_file_mutation(command):
    if re.search(r"(^|[^<])>>?($|[^>])", command):
        return True
    patterns = [
        r"(?<![\w.-])tee\b",
        r"(?<![\w.-])sed\s+-i(?:\b|\s)",
        r"(?<![\w.-])(rm|mv|cp|mkdir|touch|ln)\b",
    ]
    return any(re.search(pattern, command, re.IGNORECASE) for pattern in patterns)


try:
    payload = sys.stdin.read()
    data = json.loads(payload)
except Exception:
    fail("repo-guard: invalid hook input")

if not isinstance(data, dict):
    fail("repo-guard: invalid hook input")

tool_input = data.get("tool_input")
if data.get("tool_name") != "terminal" or not isinstance(tool_input, dict):
    fail("repo-guard: invalid hook input")

command = tool_input.get("command")
if not isinstance(command, str) or not command.strip():
    fail("repo-guard: invalid hook input")

cwd = resolve_cwd(data.get("cwd"))
if not protected_context(command, cwd):
    sys.exit(0)

if has_git_or_gh_mutation(command):
    fail("repo-guard: protected repo git/gh mutation")

if has_file_mutation(command):
    fail("repo-guard: protected repo file mutation")
' <<< "$payload" 2>/dev/null)
status=$?

if [ "$status" -ne 0 ]; then
  _block "repo-guard: internal error"
  exit 0
fi

printf '%s' "$output"
