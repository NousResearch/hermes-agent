"""Windows SSH spawn must pin HERMES_HOME from the Desktop payload.

The helper does not re-resolve the profile directory: the caller already
computed the profile-specific home and sends it as payload['hermesHome'].
"""

from hermes_cli.windows_ssh_runtime import apply_spawn_hermes_home


def test_apply_spawn_hermes_home_sets_named_profile_home():
    env = {"PATH": "C:\\Windows"}
    apply_spawn_hermes_home(
        env,
        {"hermesHome": r"C:\Users\alice\AppData\Local\hermes\profiles\homelab-delegator"},
    )
    assert env["HERMES_HOME"] == r"C:\Users\alice\AppData\Local\hermes\profiles\homelab-delegator"
    assert env["PATH"] == "C:\\Windows"


def test_apply_spawn_hermes_home_sets_default_root():
    env = {}
    apply_spawn_hermes_home(env, {"hermesHome": r"C:\Users\alice\AppData\Local\hermes"})
    assert env["HERMES_HOME"] == r"C:\Users\alice\AppData\Local\hermes"


def test_apply_spawn_hermes_home_skips_when_absent():
    env = {"PATH": "x"}
    apply_spawn_hermes_home(env, {"profile": "homelab-delegator"})
    assert "HERMES_HOME" not in env
