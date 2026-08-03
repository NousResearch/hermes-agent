from types import SimpleNamespace

import batch_runner


def test_batch_image_override_registers_apple_container_image(monkeypatch):
    registered = []

    class FakeAgent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, _prompt, task_id=None):
            return {"messages": [], "completed": True, "api_calls": 0}

        def _convert_to_trajectory_format(self, *_args):
            return []

    monkeypatch.setattr(batch_runner, "AIAgent", FakeAgent)
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda _dist: [])
    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: registered.append((task_id, overrides)),
    )

    result = batch_runner._process_single_prompt(
        3,
        {"prompt": "hello", "image": "python:3.13-slim"},
        1,
        {"distribution": {}, "model": "test", "max_iterations": 1},
    )

    assert result["success"] is True
    assert registered == [
        (
            "task_3",
            {
                "docker_image": "python:3.13-slim",
                "modal_image": "python:3.13-slim",
                "singularity_image": "docker://python:3.13-slim",
                "daytona_image": "python:3.13-slim",
                "apple_container_image": "python:3.13-slim",
            },
        )
    ]
