#!/usr/bin/env python3
"""Guard the clean-room env contract in run_evals.py and run_triggers.py.

The eval result is only honest if nothing from the host shell leaks into the
subprocess. Both scripts carry their own build_case_env (they are
deliberately self-contained); this test pins the contract on both copies:
exactly PATH + fresh HOME + CLAUDE_CONFIG_DIR + auth-var-only-when-set +
declared passthrough keys, nothing else.
Run with: python3 -m pytest test_env_isolation.py
(or plain `python3 test_env_isolation.py` for a lightweight self-check).
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import run_evals  # noqa: E402
import run_triggers  # noqa: E402

BUILDERS = [run_evals.build_case_env, run_triggers.build_case_env]

HOST_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/Users/host",
    "ANTHROPIC_API_KEY": "sk-test-123",
    "AWS_SECRET_ACCESS_KEY": "host-secret-must-not-leak",
    "CLAUDE_CONFIG_DIR": "/Users/host/.claude",
    "EXTRA_VAR": "extra",
}

HOME = Path("/tmp/eval-case/.home")


def test_minimal_env_keys():
    adapter = {"auth_env": "ANTHROPIC_API_KEY"}
    for build in BUILDERS:
        env = build(adapter, HOME, HOST_ENV)
        assert set(env) == {"PATH", "HOME", "CLAUDE_CONFIG_DIR",
                            "ANTHROPIC_API_KEY"}, (build.__module__, env)
        assert env["PATH"] == HOST_ENV["PATH"]
        assert env["HOME"] == str(HOME), "HOME must be the fresh case home"
        assert env["CLAUDE_CONFIG_DIR"] == str(HOME / ".claude")
        assert env["ANTHROPIC_API_KEY"] == "sk-test-123"
        assert "AWS_SECRET_ACCESS_KEY" not in env, "host secrets leaked"


def test_auth_var_absent_when_unset():
    # Setting auth to "" breaks the runtime's OAuth fallback — the key must
    # be absent, never empty.
    adapter = {"auth_env": "ANTHROPIC_API_KEY"}
    for host in ({}, {"ANTHROPIC_API_KEY": ""}):
        for build in BUILDERS:
            env = build(adapter, HOME, {"PATH": "/bin", **host})
            assert "ANTHROPIC_API_KEY" not in env, (build.__module__, env)


def test_no_adapter_still_minimal():
    for build in BUILDERS:
        env = build(None, HOME, HOST_ENV)
        assert set(env) == {"PATH", "HOME", "CLAUDE_CONFIG_DIR"}, env


def test_env_passthrough_only_declared_and_present():
    adapter = {"auth_env": "ANTHROPIC_API_KEY",
               "env_passthrough": ["EXTRA_VAR", "NOT_SET_ON_HOST"]}
    for build in BUILDERS:
        env = build(adapter, HOME, HOST_ENV)
        assert env.get("EXTRA_VAR") == "extra"
        assert "NOT_SET_ON_HOST" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env


if __name__ == "__main__":
    test_minimal_env_keys()
    test_auth_var_absent_when_unset()
    test_no_adapter_still_minimal()
    test_env_passthrough_only_declared_and_present()
    print("ok: build_case_env contract holds in run_evals and run_triggers")
