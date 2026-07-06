from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from torben_live_profile_investigate import attach_live_profile_tool_policy


def test_live_profile_investigation_forbids_execute_code() -> None:
    payload = attach_live_profile_tool_policy({"wakeAgent": True})
    policy = payload["tool_policy"]

    assert "execute_code" in policy["blocked_tools"]
    assert "terminal" in policy["allowed_tools"]
    assert "read_file" in policy["allowed_tools"]
    assert "Never call execute_code" in policy["agent_instruction"]
    assert "/opt/homebrew/bin/uv run python" in " ".join(policy["terminal_rules"])
    assert "bare python" in " ".join(policy["terminal_rules"])
    assert payload["contracts"]["tool_policy"] == policy["agent_instruction"]
