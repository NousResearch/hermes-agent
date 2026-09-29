"""Channel-scoped post-reply idle compaction policy contracts."""

from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity
from gateway.post_reply_idle_policy import resolve_post_reply_idle_policy
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.config_effective import load_user_config_effective


def policy(*rules, enabled=True):
    return {"compression": {"enabled": enabled, "post_reply_idle": {"channels": list(rules)}}}


def rule(platform="signal", chat_id="group-a", after_seconds=300, **extra):
    return {"platform": platform, "chat_id": chat_id, "after_seconds": after_seconds, **extra}


def source(platform=Platform.SIGNAL, chat_id="group-a", **extra):
    return SessionSource(platform=platform, chat_id=chat_id, **extra)


def identity(runtime="default", transport="default"):
    return RoutingIdentity(transport_profile=transport, runtime_profile=runtime,
                           authorization_home=Path("/unused"), runtime_home=Path("/unused"))


def resolve(config, src=None, ident=None):
    return resolve_post_reply_idle_policy(config, src or source(), ident or identity())


def test_default_is_disabled_and_does_not_change_resume_idle_setting():
    assert DEFAULT_CONFIG["compression"]["post_reply_idle"] == {"channels": []}
    assert DEFAULT_CONFIG["compression"]["idle_compact_after_seconds"] == 0
    assert resolve(DEFAULT_CONFIG) is None


def test_signal_group_matches_only_exact_platform_and_chat_id_not_name_or_dm():
    config = policy(rule())
    assert resolve(config, source(chat_name="renamed group", chat_type="group")) == 300
    assert resolve(config, source(chat_id="group-b", chat_name="group-a")) is None
    assert resolve(config, source(chat_id="user-1", chat_type="dm", chat_name="group-a")) is None
    assert resolve(config, source(platform=Platform.DISCORD, chat_name="group-a")) is None


def test_matching_chat_covers_all_user_isolated_sessions_and_threads():
    config = policy(rule())
    assert resolve(config, source(user_id="alice", thread_id="topic-1")) == 300
    assert resolve(config, source(user_id="bob", thread_id="topic-2")) == 300


def test_thread_rule_overrides_chat_and_zero_disables_only_that_thread():
    config = policy(rule(), rule(thread_id="topic-1", after_seconds=0))
    assert resolve(config, source(thread_id="topic-1")) is None
    assert resolve(config, source(thread_id="topic-2")) == 300
    assert resolve(config, source()) == 300


def test_parent_chat_id_is_matched_for_thread_source():
    config = policy(rule())
    assert resolve(config, source(chat_id="thread-1", parent_chat_id="group-a", thread_id="thread-1")) == 300


def test_profile_and_transport_are_resolved_from_identity_not_source_hint():
    config = policy(rule(profile="work", transport_profile="bot-a"))
    src = source(profile="work")
    assert resolve(config, src, identity("work", "bot-a")) == 300
    assert resolve(config, src, identity("default", "bot-a")) is None
    assert resolve(config, src, identity("work", "bot-b")) is None


def test_scope_distinguishes_same_chat_id_between_workspaces():
    config = policy(rule(platform="discord", chat_id="42", scope_id="guild-1"))
    assert resolve(config, source(Platform.DISCORD, "42", scope_id="guild-1")) == 300
    assert resolve(config, source(Platform.DISCORD, "42", scope_id="guild-2")) is None
    assert resolve(config, source(Platform.SLACK, "42", scope_id="guild-1")) is None


def test_disabled_compression_and_non_gateway_sources_do_not_schedule():
    assert resolve(policy(rule(), enabled=False)) is None
    assert resolve(policy(rule()), source(Platform.LOCAL)) is None
    assert resolve_post_reply_idle_policy(policy(rule()), source(), None) is None


@pytest.mark.parametrize("bad", [
    {"platform": "signal", "chat_id": "", "after_seconds": 300},
    {"platform": "signal", "chat_id": 123, "after_seconds": 300},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": True},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": -1},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": 1.5},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": "300"},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": 300, "chat_name": "unsafe"},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": 300, "thread_id": ""},
    {"platform": "signal", "chat_id": "group-a", "after_seconds": 300, "profile": " "},
    {"platform": "signal", "chat_id": "group-a"},
    {"chat_id": "group-a", "after_seconds": 300},
    {"platform": "unknown-platform", "chat_id": "group-a", "after_seconds": 300},
])
def test_invalid_rule_is_rejected_even_when_it_does_not_match(bad):
    with pytest.raises(ValueError, match=r"post_reply_idle.channels\[0\]"):
        resolve(policy(bad), source(chat_id="other"))


def test_duplicate_rule_rejected_including_zero_duration():
    with pytest.raises(ValueError, match="duplicate.*post_reply_idle.channels"):
        resolve(policy(rule(), rule(after_seconds=0)))


def test_effective_raw_config_resolves_rule_without_inheriting_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("compression:\n  post_reply_idle:\n    channels:\n      - platform: signal\n        chat_id: group-a\n        after_seconds: 300\n")
    effective = load_user_config_effective(path)
    assert effective["compression"]["post_reply_idle"]["channels"][0]["chat_id"] == "group-a"
    assert "idle_compact_after_seconds" not in effective["compression"]
    assert resolve(effective) == 300


def test_thread_source_without_explicit_thread_id_uses_its_chat_id():
    config = policy(rule(), rule(thread_id="thread-1", after_seconds=0))
    assert resolve(config, source(chat_id="thread-1", parent_chat_id="group-a")) is None


def test_thread_scoped_rule_inherits_parent_scope_and_overrides_chat():
    config = policy(rule(after_seconds=120), rule(thread_id="t", scope_id="guild-1", after_seconds=60))
    assert resolve(config, source(thread_id="t", scope_id="guild-1")) == 60
    assert resolve(config, source(thread_id="t", scope_id="guild-2")) == 120


def test_global_duration_is_rejected():
    config = {"compression": {"post_reply_idle": {"after_seconds": 300, "channels": []}}}
    with pytest.raises(ValueError, match="post_reply_idle"):
        resolve(config)


@pytest.mark.parametrize("bad", ["local", "webhook", "api_server"])
def test_non_chat_rule_is_rejected(bad):
    with pytest.raises(ValueError, match="messaging platform"):
        resolve(policy(rule(platform=bad)))


@pytest.mark.parametrize("channels", [None, {}, "signal"])
def test_invalid_channels_container_rejected(channels):
    with pytest.raises(ValueError, match="post_reply_idle.channels"):
        resolve({"compression": {"post_reply_idle": {"channels": channels}}})
