from gateway.config import ChannelOverride
from gateway.platforms.base import resolve_channel_project


def test_channel_override_project_round_trip():
    override = ChannelOverride(project="worktree")
    assert override.to_dict() == {"project": "worktree"}
    assert ChannelOverride.from_dict({"project": "worktree"}).project == "worktree"


def test_channel_project_prefers_exact_override_and_inherits_parent():
    config = {
        "channel_overrides": {
            "parent": {"project": "parent-project"},
            "thread": {"project": "thread-project"},
        }
    }
    assert resolve_channel_project(config, "thread", "parent") == "thread-project"
    assert resolve_channel_project(config, "unknown", "parent") == "parent-project"


def test_channel_project_resolves_topic_and_fails_open():
    config = {
        "group_topics": [
            {"chat_id": "chat", "topics": [{"thread_id": 20, "project": "android"}]}
        ]
    }
    assert resolve_channel_project(config, "20", "chat") == "android"
    assert resolve_channel_project({"group_topics": "invalid"}, "20") is None
    assert resolve_channel_project(config, "missing") is None
