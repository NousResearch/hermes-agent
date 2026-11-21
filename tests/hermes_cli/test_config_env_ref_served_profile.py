"""Unscoped ``${VAR}`` config refs for a served profile must not read the launch
profile's ``os.environ`` values on a multi-profile host (#133044).

``hermes serve`` arms multiplexing process-wide and serves several profile homes from
one process; ``os.environ`` keeps holding the LAUNCH profile's credentials. An
unscoped ``load_config()`` for a served profile (home override set, no secret scope
bound) used to expand its refs against that env — an expanded value has no ref left
for the connect path to re-render under the owner's scope, so the launch profile's
secret leaked into the served profile's config and ``mcp-tokens`` files.
"""

import os

import agent.secret_scope as ss
import hermes_cli.config as cfg
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _arm_multiplex_host(monkeypatch, *names):
    """Flip the process multiplex latch like ``hermes serve`` does (conftest resets it)."""
    monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(ss, "_AUTO_PINNED_HOME", None)
    for name in names:
        monkeypatch.setenv(name, "launch-value")


def test_served_profile_ref_stays_unexpanded(monkeypatch, tmp_path):
    _arm_multiplex_host(monkeypatch, "SLACK_MCP_CLIENT_SECRET")
    token = set_hermes_home_override(str(tmp_path / "profiles" / "b"))
    try:
        assert (
            cfg._expand_env_vars("${SLACK_MCP_CLIENT_SECRET}")
            == "${SLACK_MCP_CLIENT_SECRET}"
        )
        assert cfg._env_ref_lookup("SLACK_MCP_CLIENT_SECRET") is None
    finally:
        reset_hermes_home_override(token)


def test_served_profile_env_ref_form_stays_unexpanded(monkeypatch, tmp_path):
    _arm_multiplex_host(monkeypatch, "SLACK_MCP_CLIENT_SECRET")
    token = set_hermes_home_override(str(tmp_path / "profiles" / "b"))
    try:
        assert (
            cfg._expand_env_vars("${env:SLACK_MCP_CLIENT_SECRET}")
            == "${env:SLACK_MCP_CLIENT_SECRET}"
        )
    finally:
        reset_hermes_home_override(token)


def test_global_names_still_expand_for_served_profile(monkeypatch, tmp_path):
    _arm_multiplex_host(monkeypatch, "HERMES_KANBAN_DB")
    token = set_hermes_home_override(str(tmp_path / "profiles" / "b"))
    try:
        assert cfg._expand_env_vars("${HERMES_KANBAN_DB}") == "launch-value"
    finally:
        reset_hermes_home_override(token)


def test_launch_profile_unscoped_read_keeps_environ(monkeypatch):
    """The launch profile's own unscoped reads (no override) keep os.environ."""
    _arm_multiplex_host(monkeypatch, "SLACK_MCP_CLIENT_SECRET")
    assert cfg._expand_env_vars("${SLACK_MCP_CLIENT_SECRET}") == "launch-value"


def test_single_profile_foreign_override_keeps_environ(monkeypatch, tmp_path):
    """Without multiplexing the legacy os.environ read stands (both conditions required)."""
    monkeypatch.setenv("SLACK_MCP_CLIENT_SECRET", "launch-value")
    token = set_hermes_home_override(str(tmp_path / "profiles" / "b"))
    try:
        assert cfg._expand_env_vars("${SLACK_MCP_CLIENT_SECRET}") == "launch-value"
    finally:
        reset_hermes_home_override(token)


def test_scoped_read_unaffected(monkeypatch, tmp_path):
    """A bound scope keeps resolving against that profile's mapping."""
    _arm_multiplex_host(monkeypatch, "SLACK_MCP_CLIENT_SECRET", "SERVED_PROFILE_VALUE")
    served_val = os.environ["SERVED_PROFILE_VALUE"]
    token = set_hermes_home_override(str(tmp_path / "profiles" / "b"))
    scope_token = ss.set_secret_scope({"SLACK_MCP_CLIENT_SECRET": served_val})
    try:
        assert cfg._env_ref_lookup("SLACK_MCP_CLIENT_SECRET") == served_val
    finally:
        ss.reset_secret_scope(scope_token)
        reset_hermes_home_override(token)
