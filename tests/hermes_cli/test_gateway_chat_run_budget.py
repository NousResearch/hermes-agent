"""Regression for #143735: `hermes chat --run-budget` on the gateway path.

The flag was refused up front (exit 2, "Unsupported gateway CLI options") even
though it was still advertised in `hermes chat --help` and the whole wire for
honoring it already existed: the launch dict carries it, `_classic_only_policy`
maps it into the session.create `run_budget` policy, and the gateway freezes it
as the session's `agent.run_budget_seconds` (gateway/session_policy.py). The
refusal was the only blocker, so it is removed.
"""
import argparse


def test_run_budget_passes_validation():
    """Pre-fix this raised GatewayClientError (the exit-2 refusal)."""
    from hermes_cli import gateway_chat
    gateway_chat.validate_options(argparse.Namespace(run_budget=30.0))


def test_run_budget_reaches_session_create_policy():
    """The accepted flag folds into the session.create policy, like the config key."""
    from hermes_cli.gateway_session_store import _creation_policy
    assert _creation_policy({"run_budget": 30.0})["run_budget"] == 30.0
    assert "run_budget" not in _creation_policy({"run_budget": None})
