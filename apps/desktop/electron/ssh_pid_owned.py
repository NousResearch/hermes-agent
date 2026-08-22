"""Classify Desktop SSH dashboard argv. Stdlib only. Embedded by remote-lifecycle.ts."""

from __future__ import annotations

import os
import shlex


def _wrapper_exec_targets(hermes_path):
    try:
        with open(hermes_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return []
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#!"):
        return []
    interp = lines[0][2:].strip()
    if not interp:
        return []
    name = os.path.basename(interp.split()[-1])
    if name not in {"sh", "bash"}:
        return []
    exec_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("exec"):
            exec_line = stripped
    if not exec_line:
        return []
    try:
        parts = shlex.split(exec_line)
    except ValueError:
        return []
    if len(parts) != 4 or parts[0] != "exec" or parts[-1] != "$@":
        return []
    return [os.path.expanduser(parts[1]), os.path.expanduser(parts[2])]


def classify_dashboard_argv(
    args,
    expected,
    nonce,
    hermes_home="",
    token_path="",
    profile="",
    allow_spawn_proof=True,
    include_repo_hermes_entry=True,
    resolve_wrapper=True,
):
    expected = os.path.expanduser(expected or "")
    hermes_home = os.path.expanduser(hermes_home) if hermes_home else ""
    expected_token = os.path.expanduser(token_path) if token_path else ""
    expected_profile = profile or ""
    expected_entries = {expected} if expected else set()
    if hermes_home:
        expected_entries.add(
            os.path.join(hermes_home, "hermes-agent", "venv", "bin", "hermes")
        )
        if include_repo_hermes_entry:
            expected_entries.add(os.path.join(hermes_home, "hermes-agent", "hermes"))
    if resolve_wrapper and expected:
        expected_entries.update(_wrapper_exec_targets(expected))
    try:
        serve = args.index("serve")
        owner = args.index("--ssh-owner-nonce", serve + 1)
        token = args.index("--ssh-session-token-file", serve + 1) if expected_token else -1
        isolated = args.index("--isolated", serve + 1)
        profile_arg = args.index("--profile") if expected_profile else -1
        serve_count = args.count("serve")
        owner_count = args.count("--ssh-owner-nonce")
        token_count = args.count("--ssh-session-token-file")
        isolated_count = args.count("--isolated")
        profile_count = args.count("--profile")
        direct = bool(args) and args[0] in expected_entries
        python_entry = (
            len(args) > 1
            and args[1] in expected_entries
            and os.path.basename(args[0]).startswith("python")
        )
        token_ok = (not expected_token) or args[token + 1] == expected_token
        isolated_ok = isolated_count == 1 and isolated > serve
        if expected_profile:
            profile_ok = (
                profile_count == 1
                and profile_arg < serve
                and args[profile_arg + 1] == expected_profile
            )
        else:
            profile_ok = profile_count == 0
        spawn_proof = (
            bool(allow_spawn_proof)
            and bool(expected_token)
            and owner_count == 1
            and token_count == 1
            and token_ok
            and profile_ok
        )
        ok = (
            (direct or python_entry or spawn_proof)
            and serve_count == 1
            and isolated_ok
            and owner_count == 1
            and args[owner + 1] == nonce
            and token_ok
            and profile_ok
        )
    except (ValueError, IndexError):
        ok = False
    return "OWNED" if ok else "FOREIGN"
