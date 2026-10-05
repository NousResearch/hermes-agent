"""Automatic review consent stays with the selected profile (#117294)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent import secret_scope
from agent.background_review import load_background_review_settings
from hermes_cli.web_server_profiles import _hermes_home_scope
from run_agent import AIAgent


@pytest.mark.parametrize("launch_profile", ["default", "beta"])
def test_review_consent_round_trips_without_changing_other_profiles(tmp_path, monkeypatch, launch_profile):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    alpha = home / "profiles" / "alpha"
    beta = home / "profiles" / "beta"
    for directory in (home, alpha, beta):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(
            "auxiliary:\n  background_review:\n    provider: auto\n    model: ''\n    defer: never\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(home if launch_profile == "default" else beta))
    from hermes_cli import web_server

    spawns = []
    agent = SimpleNamespace(_delegate_depth=0, _spawn_background_review_now=lambda **kw: spawns.append(kw))
    snapshot = [{"role": "user", "content": "Remember my preference"}]
    secret_scope.set_multiplex_active(True)
    try:
        with TestClient(web_server.app) as client:
            client.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
            for profile, directory in (("alpha", alpha), ("beta", beta), ("alpha", alpha)):
                with _hermes_home_scope(directory):
                    assert load_background_review_settings()[0] is False
                    AIAgent._spawn_background_review(agent, snapshot, review_memory=True)
                response = client.get("/api/model/auxiliary", params={"profile": profile})
                assert response.status_code == 200, response.text
                task = next(row for row in response.json()["tasks"] if row["task"] == "background_review")
                assert task["enabled"] is False
            assert spawns == []
            response = client.put("/api/config", params={"profile": "alpha"}, json={
                "config": {"auxiliary": {"background_review": {"enabled": True}}},
            })
            assert response.status_code == 200, response.text
            for directory, expected in ((alpha, True), (beta, False), (alpha, True)):
                with _hermes_home_scope(directory):
                    assert load_background_review_settings()[0] is expected
                    AIAgent._spawn_background_review(agent, snapshot, review_memory=True)
            assert len(spawns) == 2
            assert all(spawn["task_cfg"]["model"] == "" for spawn in spawns)
            with _hermes_home_scope(beta):
                AIAgent._spawn_background_review(agent, snapshot, review_memory=True, focus="", explicit=True)
                assert len(spawns) == 3
                (beta / "config.yaml").write_text("auxiliary: [unclosed", encoding="utf-8")
                assert load_background_review_settings()[0] is False
                AIAgent._spawn_background_review(agent, snapshot, review_memory=True)
                assert len(spawns) == 3
                import hermes_cli.config as config_module
                def unreadable():
                    raise OSError("unreadable configuration")
                monkeypatch.setattr(config_module, "load_config_readonly", unreadable)
                assert load_background_review_settings()[0] is False
                AIAgent._spawn_background_review(agent, snapshot, review_memory=True)
                assert len(spawns) == 3
    finally:
        secret_scope.set_multiplex_active(False)
