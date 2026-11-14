"""The 1Password (`op` CLI) child env built from a staged source environment.

Startup resolves secret sources against a plain ``dict`` copy of the process
environment. On Windows ``os.environ`` upper-cases its keys, so that copy holds
``SYSTEMROOT``; without it the ``op`` child cannot resolve DNS (getaddrinfow).
"""

from __future__ import annotations

from agent.secret_sources import onepassword as op
from agent.secret_sources.base import reset_source_environment, set_source_environment


def test_op_child_env_keeps_systemroot_from_staged_environment():
    token = set_source_environment(
        {"SYSTEMROOT": "C:\\Windows", "UNRELATED_PROVIDER_KEY": "not-forwarded"}
    )
    try:
        env = op._op_child_env("")
    finally:
        reset_source_environment(token)

    assert env.get("SYSTEMROOT") == "C:\\Windows"
    assert "UNRELATED_PROVIDER_KEY" not in env


def test_op_child_env_reads_the_staged_view_not_the_process_env(monkeypatch):
    """The staged view, not os.environ, is the source of the child env."""
    monkeypatch.setenv("SYSTEMROOT", "C:\\Ambient")
    monkeypatch.setenv("PATH", "/ambient/bin")
    token = set_source_environment({"PATH": "/staged/bin"})
    try:
        env = op._op_child_env("")
    finally:
        reset_source_environment(token)

    assert env == {"PATH": "/staged/bin", "NO_COLOR": "1"}
