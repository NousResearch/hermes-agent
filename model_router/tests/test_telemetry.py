import json
from pathlib import Path

from telemetry import append_jsonl, build_event
from model_router import (
    Mode,
    Model,
    Privacy,
    Priority,
    Quota,
    RouterDecision,
    RouterInput,
    TaskType,
)


def test_build_event_generates_request_id_when_missing():
    event = build_event(
        router_version="0.3",
        config_path="/tmp/router_config.yaml",
        request_input=RouterInput(
            task_type=TaskType.CHAT,
            mode=Mode.DRAFT,
            priority=Priority.MEDIUM,
            privacy=Privacy.NORMAL,
            quota=Quota.NORMAL,
        ),
        decision=RouterDecision(
            primary_model=Model.CLAUDE,
            fallback_models=[Model.GPT, Model.DEEPSEEK],
            reviewer=None,
            reason="demo",
            trace=["x"],
        ),
    )

    assert event["event_type"] == "decision"
    assert event["request_id"]
    assert event["input"]["task_type"] == "chat"
    assert event["decision"]["primary_model"] == "claude-sonnet-4.6"


def test_append_jsonl_writes_one_line(tmp_path: Path):
    path = tmp_path / "router.jsonl"
    append_jsonl(path, {"request_id": "abc", "event_type": "decision"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "abc"
