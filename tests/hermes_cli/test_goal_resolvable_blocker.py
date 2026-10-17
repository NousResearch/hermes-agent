"""A recoverable dependency must not be reported as an impossible goal."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_external_input_blocks_without_declaring_goal_unachievable(tmp_path, monkeypatch):
    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    manager = goals.GoalManager("external-blocker")
    manager.set("Publish the verified patch")
    captured = {}

    def reply(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"verdict":"blocked","reason":"Waiting for the user to approve publication"}'
        ))])

    with patch("agent.auxiliary_client.call_llm", side_effect=reply):
        decision = manager.evaluate_after_turn("All local checks passed. Publication requires user approval.")

    assert decision["verdict"] == "blocked"
    assert decision["status"] == "paused"
    assert "unachievable" not in decision["message"].lower()
    assert "unachievable" not in manager.status_line().lower()
    assert "independent authorized work" in captured["prompt"].lower()
    stored = goals.load_goal("external-blocker")
    assert stored is not None and stored.status == "paused"
    goals._DB_CACHE.clear()
