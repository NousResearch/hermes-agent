"""E2E: the gateway's tool-progress path must not emit fenced terminal
blocks for the Photon adapter, using the REAL resolution code paths.

Runs the actual flag handoff the way the gateway does it:
  adapter.supports_code_blocks  ->  the run.py progress branch builds either
  a fenced block or the compact preview. We re-implement only the tiny
  branch body (copied verbatim from run.py's decision) and drive it with
  the real PhotonAdapter + real config resolution from this tree.
"""
from __future__ import annotations

import yaml

from gateway.config import PlatformConfig
from gateway.display_config import resolve_display_setting
from plugins.platforms.photon.adapter import PhotonAdapter


def _progress_message(supports_blocks: bool, command: str) -> str:
    """The exact branch shape from gateway/run.py tool-progress handling."""
    if supports_blocks:
        return f"💻 terminal\n```\n{command}\n```"
    cap = 40
    short = command.splitlines()[0]
    if len(short) > cap:
        short = short[: cap - 3] + "..."
    return f'💻 terminal: "{short}"'


def test_gateway_would_not_fence_terminal_for_photon(tmp_path) -> None:
    cfg = {"display": {"tool_progress": "all"}}
    platform_key = "photon"
    # Real resolution: mode must still be "all" (we did NOT ask users to
    # change config for the fix to work).
    assert resolve_display_setting(cfg, platform_key, "tool_progress") == "all"

    import os
    os.environ.setdefault("PHOTON_PROJECT_ID", "test-project-id")
    os.environ.setdefault("PHOTON_PROJECT_SECRET", "test-project-secret")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    cmd = "export HOME=/opt/data/home; ls /opt/data"
    msg = _progress_message(adapter.supports_code_blocks, cmd)
    assert "```" not in msg, f"fenced block would reach iMessage: {msg!r}"
    assert "ls /opt/data" in msg  # compact preview still shows the action
