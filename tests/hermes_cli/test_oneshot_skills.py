from __future__ import annotations

import sys
import types

from hermes_cli import oneshot


def test_run_oneshot_forwards_preloaded_skills(monkeypatch, capsys):
    captured = {}

    def fake_run_agent(prompt, **kwargs):
        captured.update({"prompt": prompt, **kwargs})
        return "ok", {}

    monkeypatch.setattr(oneshot, "_run_agent", fake_run_agent)

    rc = oneshot.run_oneshot("hello", skills=["saju-social-content"])

    assert rc == 0
    assert capsys.readouterr().out == "ok\n"
    assert captured["skills"] == ["saju-social-content"]


def test_run_agent_appends_preloaded_skill_to_system_prompt(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["ephemeral_system_prompt"] = kwargs.get("ephemeral_system_prompt")
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "test-model", "provider": "test"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": None,
            "base_url": None,
            "provider": "test",
            "requested_provider": "test",
            "api_mode": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr("hermes_cli.oneshot.get_fallback_chain", lambda _cfg: [])
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))
    monkeypatch.setattr(
        "agent.skill_commands.build_preloaded_skills_prompt",
        lambda identifiers, task_id=None: (
            "loaded saju canonical text",
            ["saju-social-content"],
            [],
        ),
    )

    response, _result = oneshot._run_agent(
        "answer from canon",
        skills=["saju-social-content"],
    )

    assert response == "ok"
    assert captured == {
        "prompt": "answer from canon",
        "ephemeral_system_prompt": "loaded saju canonical text",
    }
