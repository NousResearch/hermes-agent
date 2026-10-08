"""Capture preserves persisted identities without repeated history serialization."""
import hashlib
import json
from datetime import datetime, timezone

import pytest

from agent import exploration_policy, trajectory as trajectory_io


@pytest.mark.parametrize("turns", [32, 128])
def test_capture_preserves_prefix_identities_with_linear_serialization(monkeypatch, turns):
    messages = [
        {"from": "gpt", "value": "你好", "metadata": {"z": 1, "a": [None, True]}},
        {"from": "human", "value": datetime(2026, 1, 1, tzinfo=timezone.utc)},
    ]
    messages.extend(
        {"from": "gpt" if index % 2 else "tool", "value": "résultat " * 20}
        for index in range(turns)
    )
    dumps = json.dumps
    options = dict(ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    expected = [
        hashlib.sha256(dumps(messages[:index], **options).encode("utf-8")).hexdigest()
        for index, message in enumerate(messages) if message["from"] == "gpt"
    ]
    tree_identity = hashlib.sha256(dumps(
        {"policy_version": exploration_policy.DEFAULT_POLICY_VERSION,
         "trajectory": messages, "completed": True}, **options
    ).encode("utf-8")).hexdigest()
    serialized_size = len(dumps(messages, **options).encode("utf-8"))
    serialized_bytes = 0

    def counted_dumps(*args, **kwargs):
        nonlocal serialized_bytes
        result = dumps(*args, **kwargs)
        serialized_bytes += len(result.encode("utf-8"))
        return result

    monkeypatch.setattr(exploration_policy.json, "dumps", counted_dumps)
    tree = exploration_policy.build_exploration_tree(messages, completed=True)
    assert [event.state_digest for event in tree.events] == expected
    assert tree.tree_id == tree_identity
    assert serialized_bytes <= 3 * serialized_size + 1024


@pytest.mark.parametrize("completed", [True, False])
def test_secondary_capture_failure_keeps_primary_trajectory(tmp_path, monkeypatch, caplog, completed):
    primary = tmp_path / "trajectory.jsonl"
    messages = [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "done"}]

    def broken_capture(*args, **kwargs):
        raise ValueError("invalid secondary capture")

    monkeypatch.setattr(trajectory_io, "build_exploration_tree", broken_capture)
    trajectory_io.save_trajectory(messages, "test-model", completed, filename=str(primary))
    saved = json.loads(primary.read_text(encoding="utf-8"))
    assert saved["conversations"] == messages
    assert saved["completed"] is completed
    assert saved["model"] == "test-model"
    assert "invalid secondary capture" in caplog.text
