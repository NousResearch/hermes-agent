"""Auto-loaded skills retain a real load failure instead of calling it missing."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

from gateway.run_turn import GatewayTurnMixin


def test_auto_skill_logs_hard_load_failure(caplog):
    event = SimpleNamespace(text="original request")
    response = json.dumps({"success": False, "error": "Object of type date is not JSON serializable"})
    with patch("tools.skills_tool.skill_view", return_value=response), caplog.at_level(logging.WARNING):
        GatewayTurnMixin._hmwa_auto_load_skills(object(), event, "broken-skill", "task", "session")

    assert event.text == "original request"
    assert "Auto-skill 'broken-skill' failed to load" in caplog.text
    assert "Object of type date is not JSON serializable" in caplog.text
    assert "not found" not in caplog.text
