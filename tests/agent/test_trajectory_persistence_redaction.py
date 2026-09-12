import json

from agent.trajectory import save_trajectory


def test_save_trajectory_redacts_secrets_in_temp_file(tmp_path):
    secret = "sk-proj-trajectory-secret-1234567890"
    path = tmp_path / "trajectory.jsonl"

    save_trajectory(
        [{"from": "human", "value": f"use api_key={secret}"}],
        "test-model",
        True,
        filename=str(path),
    )

    raw = path.read_text(encoding="utf-8")
    assert secret not in raw
    entry = json.loads(raw)
    assert entry["conversations"][0]["from"] == "human"
    assert secret not in entry["conversations"][0]["value"]
    assert entry["conversations"][0]["value"] != f"use api_key={secret}"


def test_save_trajectory_preserves_sharegpt_shape_while_redacting_tool_arguments(tmp_path):
    secret = "sk-proj-trajectory-secret-1234567890"
    path = tmp_path / "trajectory.jsonl"
    trajectory = [
        {"from": "human", "value": "Run the tool"},
        {"from": "gpt", "value": '{"name":"terminal","arguments":"token=' + secret + '"}'},
        {"from": "tool", "value": "command completed"},
    ]

    save_trajectory(trajectory, "test-model", False, filename=str(path))

    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["model"] == "test-model"
    assert entry["completed"] is False
    assert [item["from"] for item in entry["conversations"]] == ["human", "gpt", "tool"]
    assert secret not in json.dumps(entry)
