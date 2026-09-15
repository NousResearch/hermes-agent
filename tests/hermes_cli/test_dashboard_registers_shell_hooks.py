"""``hermes serve`` / ``hermes dashboard`` must register config-owned shell hooks.

``serve`` reaches neither ``main()``'s dispatch (``_try_fast_serve_launch`` calls
``cmd_dashboard`` directly) nor ``_prepare_agent_startup`` (``serve`` is not in
``_AGENT_COMMANDS``), so before this fix every ``hooks:`` entry in ``config.yaml``
was silently inert in that process while the same entries worked under
``hermes chat``. Desktop starts one ``serve`` backend per profile, so a hook
configured as a safety or policy boundary did not run for any turn served there.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


class _Stop(Exception):
    """Sentinel: end cmd_dashboard right after the registration point."""


def _ns(**kw):
    defaults = dict(
        port=9119, host="127.0.0.1", no_open=False, insecure=False,
        stop=False, status=False, headless_backend=True,
        ssh_session_token_file=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _run_cmd_dashboard(register_mock, load_config_mock):
    """Call cmd_dashboard with everything around the registration point stubbed."""
    from hermes_cli.main import cmd_dashboard

    with patch("hermes_cli.main._dashboard_lifecycle_flags"), \
         patch("hermes_cli.main._dashboard_validate_serve_args", return_value=None), \
         patch("hermes_cli.main._dashboard_sanitize_desktop_env"), \
         patch("hermes_cli.main._route_named_profile_dashboard"), \
         patch("hermes_cli.resource_limits.apply_nofile_soft_limit"), \
         patch("hermes_cli.config.load_config", load_config_mock), \
         patch("agent.shell_hooks.register_from_config", register_mock), \
         patch("hermes_cli.main._dashboard_prepare_runtime", side_effect=_Stop):
        with pytest.raises(_Stop):
            cmd_dashboard(_ns())


def test_serve_registers_configured_shell_hooks():
    register = MagicMock(return_value=[])
    load_config = MagicMock(return_value={"hooks": {"pre_tool_call": [{"command": "x"}]}})
    _run_cmd_dashboard(register, load_config)
    assert register.call_count == 1, "serve must register config shell hooks exactly once"
    cfg = register.call_args.args[0]
    assert cfg == {"hooks": {"pre_tool_call": [{"command": "x"}]}}
    # accept_hooks=False defers to hooks_auto_accept / the per-profile allowlist
    # rather than prompting on a headless backend.
    assert register.call_args.kwargs.get("accept_hooks") is False


def test_registration_failure_does_not_break_serve():
    """A broken hooks block must not take the backend down with it."""
    register = MagicMock(side_effect=RuntimeError("bad hooks block"))
    load_config = MagicMock(return_value={})
    # _Stop still propagates, i.e. startup continued past the registration point.
    _run_cmd_dashboard(register, load_config)
    assert register.call_count == 1
