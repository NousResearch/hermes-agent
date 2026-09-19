#!/usr/bin/env python3
"""Apply the kanban dispatch fix by rewriting the file section."""

import re

file_path = "/Users/mikedemott/hermes-fork-work/fork-repo/hermes_cli/kanban_db_dispatch.py"

with open(file_path, "r") as f:
    content = f.read()

# Find and replace the specific section
old_section = '''    env = build_subprocess_env(
        scrub_secrets=is_multiplex_active(),
        inherit_profile_home=True,
    )
    # Keep the assigned repository cwd from shadowing Hermes runtime imports.
    env["PYTHONSAFEPATH"] = "1"
    # The dispatcher is detached from every conversation; its worker must never
    # inherit routing mirrored by a previous gateway turn.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)
    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml:
    # without it the child's get_hermes_home() falls back to the DEFAULT
    # profile root because `hermes -p` applies its override before
    # hermes_constants is imported.
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
        strip_launch_profile_env(env, env["HERMES_HOME"])
        from hermes_cli.kanban_worker_environment import validate_profile_config
        validate_profile_config(env["HERMES_HOME"])
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        pass'''

new_section = '''    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml:
    # without it the child's get_hermes_home() falls back to the DEFAULT
    # profile root because `hermes -p` applies its override before
    # hermes_constants is imported.
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
        strip_launch_profile_env(env, env["HERMES_HOME"])
        from hermes_cli.kanban_worker_environment import validate_profile_config
        validate_profile_config(env["HERMES_HOME"])
        # Build subprocess env with profile's secret scope active for passthrough resolution
        if is_multiplex_active():
            from pathlib import Path
            profile_hermes_home = Path(env["HERMES_HOME"])
            profile_secrets = build_profile_secret_scope(profile_hermes_home)
            scope_token = set_secret_scope(profile_secrets)
            try:
                env = build_subprocess_env(
                    scrub_secrets=is_multiplex_active(),
                    inherit_profile_home=True,
                )
            finally:
                reset_secret_scope(scope_token)
        else:
            env = build_subprocess_env(
                scrub_secrets=is_multiplex_active(),
                inherit_profile_home=True,
            )
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        pass
    # Keep the assigned repository cwd from shadowing Hermes runtime imports.
    env["PYTHONSAFEPATH"] = "1"
    # The dispatcher is detached from every conversation; its worker must never
    # inherit routing mirrored by a previous gateway turn.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open(file_path, "w") as f:
        f.write(content)
    print("SUCCESS: File updated")
else:
    print("FAILURE: Could not find old_section in file")
    # Debug: find similar text
    import sys
    sys.exit(1)