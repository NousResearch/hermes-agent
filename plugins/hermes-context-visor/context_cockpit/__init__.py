"""Hermes Context Cockpit — read-only context health dashboard package.

Layers:
  metrics   — pure read-only collection from Hermes profile state
  status    — classification (ALL GOOD / QUIET / … / OLD NUMBERS / HERMES OFFLINE)
  render    — Rich Flight Deck fallback UI
  web       — graphical localhost HTML/CSS/JS cockpit
  launcher  — fixed-argv browser/server launcher (no shell interpolation)
"""

from .metrics import collect_metrics
from .status import classify_status
from .controls import build_action_controls
from .live_lcm import (
    read_live_lcm_snapshot,
    snapshot_is_bound,
    snapshot_matches_conversation,
    snapshot_path,
    write_live_lcm_snapshot_for_engine,
)
from .launcher import launch_context_visor, build_visor_argv, build_visor_url, platform_fallback_instructions
from .web import stream_interval_ms

__all__ = [
    "collect_metrics",
    "classify_status",
    "build_action_controls",
    "read_live_lcm_snapshot",
    "write_live_lcm_snapshot_for_engine",
    "snapshot_is_bound",
    "snapshot_matches_conversation",
    "snapshot_path",
    "launch_context_visor",
    "build_visor_argv",
    "build_visor_url",
    "platform_fallback_instructions",
    "stream_interval_ms",
]
