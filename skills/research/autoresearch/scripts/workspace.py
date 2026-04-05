#!/usr/bin/env python3
"""Autoresearch git workspace management. Outputs commands for agent to execute.

Usage:
    python workspace.py init <workspace_dir>
    python workspace.py branch <workspace_dir> <exp_id> <short_description>
    python workspace.py branch-name <exp_id> <description>
    python workspace.py diff <workspace_dir>
    python workspace.py merge <workspace_dir> <exp_id> <short_description> <commit_message>
    python workspace.py revert <workspace_dir> <exp_id> <short_description>
    python workspace.py log <workspace_dir> [--oneline]
    python workspace.py current-branch <workspace_dir>
"""
import json, os, re, shlex, sys

def _safe_branch_name(exp_id, desc):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', desc.lower())
    safe = re.sub(r'_+', '_', safe)[:40].rstrip('_')
    return f"exp_{exp_id}_{safe}"

def _cmds(workspace_dir, commands):
    print(json.dumps({"workspace": workspace_dir, "commands": commands}, indent=2))

def init(d):
    qd = shlex.quote(d)
    os.makedirs(d, exist_ok=True)
    _cmds(d, [
        f"cd {qd} && git init --initial-branch=main 2>/dev/null || (cd {qd} && git init && cd {qd} && git checkout -b main)",
        f"cd {qd} && git config user.email 'autoresearch@hermes'",
        f"cd {qd} && git config user.name 'autoresearch'",
        f"cd {qd} && git commit --allow-empty -m 'init autoresearch workspace'",
    ])

def branch(d, eid, desc):
    qd = shlex.quote(d)
    b = _safe_branch_name(eid, desc)
    _cmds(d, [f"cd {qd} && git checkout main", f"cd {qd} && git checkout -b {b}"])

def diff(d):
    qd = shlex.quote(d)
    _cmds(d, [f"cd {qd} && git diff main"])

def merge(d, eid, desc, msg):
    qd = shlex.quote(d)
    b = _safe_branch_name(eid, desc)
    qmsg = shlex.quote(msg)
    _cmds(d, [
        f"cd {qd} && git add -A",
        f"cd {qd} && git commit -m {qmsg}",
        f"cd {qd} && git checkout main",
        f"cd {qd} && git merge {b} --no-edit",
        f"cd {qd} && git branch -d {b}",
    ])

def revert(d, eid, desc):
    qd = shlex.quote(d)
    b = _safe_branch_name(eid, desc)
    _cmds(d, [f"cd {qd} && git checkout -f main", f"cd {qd} && git branch -D {b}"])

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args: print(__doc__); sys.exit(1)
    cmd = args[0]
    if cmd == "init": init(args[1])
    elif cmd == "branch": branch(args[1], int(args[2]), args[3])
    elif cmd == "branch-name": print(_safe_branch_name(int(args[1]), args[2]))
    elif cmd == "diff": diff(args[1])
    elif cmd == "merge": merge(args[1], int(args[2]), args[3], args[4])
    elif cmd == "revert": revert(args[1], int(args[2]), args[3])
    elif cmd == "log":
        qd = shlex.quote(args[1])
        _cmds(args[1], [f"cd {qd} && git log{' --oneline' if '--oneline' in args else ''} -20"])
    elif cmd == "current-branch":
        qd = shlex.quote(args[1])
        _cmds(args[1], [f"cd {qd} && git branch --show-current"])
    else: print(f"Unknown: {cmd}"); sys.exit(1)
