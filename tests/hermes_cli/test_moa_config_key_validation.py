"""Supported MoA preset keys survive CLI writes and reach the executing advisor loop."""

import json
from types import SimpleNamespace

import pytest

from hermes_cli.config import get_config_value, load_config, set_config_value
from hermes_cli.moa_config import resolve_moa_preset


@pytest.fixture
def moa_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").touch()
    return tmp_path


@pytest.mark.parametrize("name", ["default", "review", "review.v2", "code review"])
def test_configured_cadence_and_timeout_reach_moa(name, moa_home, monkeypatch, capsys):
    (moa_home / "config.yaml").write_text(
        json.dumps(
            {
                "moa": {
                    "presets": {
                        name: {
                            "reference_models": [
                                {"provider": "openrouter", "model": "test/advisor"}
                            ],
                            "aggregator": {
                                "provider": "openrouter",
                                "model": "test/actor",
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prefix = "moa.presets." + name.replace(".", r"\.")
    set_config_value(f"{prefix}.fanout", "every_n:5")
    set_config_value(f"{prefix}.reference_timeout", "900")
    written = capsys.readouterr()
    preset = resolve_moa_preset(load_config()["moa"], name)
    assert preset["fanout"] == "every_n:5"
    assert preset["reference_timeout"] == 900

    reads = []
    for field, expected in (("fanout", "every_n:5"), ("reference_timeout", 900)):
        get_config_value(f"{prefix}.{field}", as_json=True)
        captured = capsys.readouterr()
        assert json.loads(captured.out) == expected
        reads.append(captured.err)

    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="advice", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
            model="test/model",
        )

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call)
    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions(name)
    messages = [{"role": "user", "content": "Review the task"}]
    reference_counts = []
    for iteration in range(6):
        facade.create(messages=list(messages), tools=[])
        reference_counts.append(sum(call["task"] == "moa_reference" for call in calls))
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"c{iteration}",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"c{iteration}",
                    "content": f"result {iteration}",
                },
            ]
        )
    assert reference_counts == [1, 1, 1, 1, 1, 2]
    assert all(
        call["timeout"] == 900 for call in calls if call["task"] == "moa_reference"
    )
    assert "not a recognized config key" not in written.out + written.err + "".join(
        reads
    )


@pytest.mark.parametrize("name", ["default", "review", "review.v2"])
@pytest.mark.parametrize(
    "field", ["fanoutt", "reference_timeot", "unexpected.fanout", "fanout.unexpected"]
)
def test_preset_typos_still_warn(name, field, moa_home, capsys):
    prefix = "moa.presets." + name.replace(".", r"\.")
    set_config_value(f"{prefix}.{field}", "5")
    assert "not a recognized config key" in capsys.readouterr().out
