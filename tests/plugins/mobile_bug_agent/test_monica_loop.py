from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

from plugins.mobile_bug_agent.config import (
    LinearConfig,
    LoopConfig,
    MonicaConfig,
    ProofConfig,
    RepoConfig,
    RuntimeConfig,
    SlackConfig,
    VerificationConfig,
)
from plugins.mobile_bug_agent.loop import MonicaLoop, MonicaLoopSkills
from plugins.mobile_bug_agent.slack_flow import MonicaSlackFlow
from plugins.mobile_bug_agent.state import MonicaState


def _slack_event(
    text: str,
    *,
    raw_text: str | None = None,
    raw_type: str = "message",
    raw_channel_type: str | None = None,
    raw_files: list[dict[str, Any]] | None = None,
    channel_id: str = "C_MOBILE",
    chat_type: str = "channel",
    user_id: str = "U_TAGGER",
    thread_ts: str = "1710000000.000100",
    message_ts: str = "1710000000.000200",
) -> MessageEvent:
    raw_message = {
        "type": raw_type,
        "text": raw_text if raw_text is not None else text,
        "channel": channel_id,
        "user": user_id,
        "thread_ts": thread_ts,
        "ts": message_ts,
        "permalink": "https://example.slack.com/archives/C_MOBILE/p1710000000000200",
        "files": raw_files or [],
    }
    if raw_channel_type is not None:
        raw_message["channel_type"] = raw_channel_type

    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=user_id,
            thread_id=thread_ts,
            message_id=message_ts,
        ),
        raw_message=raw_message,
        message_id=message_ts,
    )


def test_slack_flow_ignores_unmentioned_messages(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crashes on Android",
            raw_text="checkout crashes on Android",
        )
    )

    assert result is None
    assert state.list_runs() == []


def test_slack_flow_swallow_tagged_monica_message_when_config_disabled(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=False,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crashes on Android",
            raw_text="<@BMONICA> checkout crashes on Android",
        )
    )

    assert result == {"action": "skip", "reason": "monica_disabled"}
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_treats_tagged_natural_language_as_agentic_work(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "this checkout crash looks like the same Android issue from last week, can you clean it up?",
            raw_text="<@BMONICA> this checkout crash looks like the same Android issue from last week, can you clean it up?",
        )
    )

    assert result == {"action": "skip", "reason": "monica_loop_queued"}
    runs = state.list_runs()
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert runs[0].status == "queued"
    assert runs[0].intent == "agentic_triage"
    assert "checkout crash" in runs[0].request_text
    assert runs[0].raw_event is not None
    assert runs[0].raw_event["permalink"].startswith("https://example.slack.com/")


def test_slack_flow_strips_monica_mention_from_normalized_text(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "<@BMONICA> checkout crashes after promo on Android",
            raw_text="<@BMONICA> checkout crashes after promo on Android",
        )
    )

    runs = state.list_runs()
    assert result == {"action": "skip", "reason": "monica_loop_queued"}
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert runs[0].request_text == "checkout crashes after promo on Android"


def test_slack_flow_accepts_and_strips_display_label_mention(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("UMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "<@UMONICA|monica> checkout crashes after promo on Android",
            raw_text="<@UMONICA|monica> checkout crashes after promo on Android",
        )
    )

    runs = state.list_runs()
    assert result == {"action": "skip", "reason": "monica_loop_queued"}
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert runs[0].request_text == "checkout crashes after promo on Android"


def test_slack_flow_without_bot_user_ids_does_not_read_generic_channel_messages(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(allowed_channels=("C_MOBILE",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crashes on Android",
            raw_text="checkout crashes on Android",
            raw_type="message",
        )
    )

    assert result is None
    assert state.list_runs() == []


def test_slack_flow_treats_direct_message_as_explicit_monica_intent(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(bot_user_ids=("U_MONICA",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crashes on Android after applying a promo code",
            raw_text="checkout crashes on Android after applying a promo code",
            raw_channel_type="im",
            channel_id="D_MONICA",
            chat_type="dm",
        )
    )

    runs = state.list_runs()
    assert result == {"action": "skip", "reason": "monica_loop_queued"}
    assert len(runs) == 1
    assert launched == [runs[0].id]
    assert runs[0].channel_id == "D_MONICA"
    assert runs[0].request_text == "checkout crashes on Android after applying a promo code"


def test_slack_flow_ignores_unmentioned_group_dm_messages(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(bot_user_ids=("U_MONICA",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crashes on Android after applying a promo code",
            raw_text="checkout crashes on Android after applying a promo code",
            raw_channel_type="mpim",
            channel_id="G_MONICA",
            chat_type="dm",
        )
    )

    assert result is None
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_allows_app_mention_event_before_bot_user_id_is_configured(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(allowed_channels=("C_MOBILE",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@BMONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    runs = state.list_runs()
    assert result == {"action": "skip", "reason": "monica_loop_queued"}
    assert len(runs) == 1
    assert launched == [runs[0].id]


def test_slack_flow_dry_run_ignores_app_mention_for_different_configured_bot_user_id(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_STALE",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_ACTUAL> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result is None
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_ignores_app_mention_for_different_bot_user_id(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_CHANDLER> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result is None
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_requires_configured_bot_user_id(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(allowed_channels=("C_MOBILE",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_bot_user_ids_required",
        "text": (
            "I cannot start Monica here yet. "
            "Configure mobile_bug_agent.slack.bot_user_ids with Monica's Slack user ID "
            "before enabling real side effects."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_rejects_bot_id_values_as_mentions(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@BMONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_bot_user_ids_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.bot_user_ids must contain Slack mention user IDs "
            "like U123, not bot_id values like BMONICA."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_rejects_bot_id_config_on_real_app_mention(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_bot_user_ids_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.bot_user_ids must contain Slack mention user IDs "
            "like U123, not bot_id values like BMONICA."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_rejects_malformed_bot_user_ids_on_app_mention(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("monica",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_bot_user_ids_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.bot_user_ids must contain Slack mention user IDs "
            "like U123, not invalid values like monica."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_side_effect_mode_rejects_handle_style_bot_user_ids_on_app_mention(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("@monica",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_bot_user_ids_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.bot_user_ids must contain Slack mention user IDs "
            "like U123, not handles like @monica."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_refuses_side_effect_rollout_when_allowed_channels_are_empty(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(bot_user_ids=("BMONICA",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@BMONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_allowed_channels_required",
        "text": (
            "I cannot start Monica here yet. "
            "Configure mobile_bug_agent.slack.allowed_channels before enabling real side effects."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_empty_allowlist_guard_still_ignores_unmentioned_messages(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(bot_user_ids=("BMONICA",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="checkout crash on Android",
            raw_type="message",
        )
    )

    assert result is None
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_refuses_side_effect_rollout_with_channel_names_in_allowlist(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("#mobile-bugs",),
            bot_user_ids=("U_MONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_allowed_channels_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.allowed_channels must contain Slack channel IDs "
            "like C123 or G123, not names like #mobile-bugs."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_refuses_dry_run_with_channel_names_in_allowlist(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="dry_run",
        slack=SlackConfig(
            allowed_channels=("#mobile-bugs",),
            bot_user_ids=("U_MONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@U_MONICA> checkout crash on Android",
            raw_type="app_mention",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_allowed_channels_invalid",
        "text": (
            "I cannot start Monica here yet. "
            "mobile_bug_agent.slack.allowed_channels must contain Slack channel IDs "
            "like C123 or G123, not names like #mobile-bugs."
        ),
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_tagged_message_in_disallowed_channel_does_not_fall_through(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@BMONICA> checkout crash on Android",
            raw_type="app_mention",
            channel_id="C_OTHER",
        )
    )

    assert result == {
        "action": "skip_reply",
        "reason": "monica_channel_not_allowed",
        "text": "I cannot run Monica in this channel. Ask an admin to add this channel to mobile_bug_agent.slack.allowed_channels.",
    }
    assert state.list_runs() == []
    assert launched == []


def test_slack_flow_does_not_relaunch_existing_active_thread(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)
    event = _slack_event(
        "this checkout crash looks mobile, please clean it up",
        raw_text="<@BMONICA> this checkout crash looks mobile, please clean it up",
    )

    first = flow.handle_gateway_event(event)
    second = flow.handle_gateway_event(event)

    runs = state.list_runs()
    assert first == {"action": "skip", "reason": "monica_loop_queued"}
    assert second == {"action": "skip", "reason": "monica_loop_already_active"}
    assert len(runs) == 1
    assert launched == [runs[0].id]


def test_slack_flow_does_not_launch_when_create_discovers_existing_run(tmp_path, monkeypatch):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    existing = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="checkout crash on Android",
    )
    monkeypatch.setattr(state, "find_run", lambda **_: None)
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "checkout crash on Android",
            raw_text="<@BMONICA> checkout crash on Android",
            thread_ts="1710000000.000100",
        )
    )

    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert state.get_run(existing.id) == existing
    assert launched == []


def test_slack_flow_tagged_approval_resumes_waiting_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@BMONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == [run.id]


def test_slack_flow_tagged_approval_logs_breadcrumbs(tmp_path, caplog):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.slack_flow"):
        result = flow.handle_gateway_event(
            _slack_event(
                "approved, fix it",
                raw_text="<@BMONICA> approved, fix it",
                user_id="U_APPROVER",
                thread_ts="1710000000.000100",
            )
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert "event=approved" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs
    assert "approved_by=U_APPROVER" in logs
    assert launched == [run.id]


def test_slack_flow_does_not_launch_when_approval_discovers_already_approved(tmp_path, monkeypatch):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    stale = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
        status="awaiting_fix_approval",
    )
    state.approve_fix(stale.id, approved_by_user_id="U_FIRST")
    monkeypatch.setattr(state, "find_run", lambda **_: stale)
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@BMONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(stale.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_FIRST"
    assert launched == []


def test_slack_flow_accepts_take_the_fix_as_tagged_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "yes, take the fix",
            raw_text="<@BMONICA> yes, take the fix",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == [run.id]


def test_slack_flow_accepts_yes_fix_it_as_tagged_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "yes, fix it",
            raw_text="<@BMONICA> yes, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert launched == [run.id]


def test_slack_flow_fix_request_is_not_approval(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "can you fix it?",
            raw_text="<@BMONICA> can you fix it?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""


def test_slack_flow_new_context_requeues_waiting_run_for_linear_update(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        approved_by_user_id="",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "new reproduction detail: this also happens on iOS after Apple Pay",
            raw_text="<@BMONICA> new reproduction detail: this also happens on iOS after Apple Pay",
            user_id="U_REPRODUCER",
            thread_ts="1710000000.000100",
            message_ts="1710000002.000300",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_requeued"}
    assert updated is not None
    assert updated.status == "queued"
    assert updated.linear_identifier == "MOB-123"
    assert updated.linear_issue_id == "issue-id"
    assert updated.linear_url == "https://linear.app/acme/issue/MOB-123"
    assert updated.message_ts == "1710000002.000300"
    assert updated.user_id == "U_REPRODUCER"
    assert "new reproduction detail" in updated.request_text
    assert updated.approved_by_user_id == ""
    assert launched == [run.id]


def test_slack_flow_question_shaped_approval_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "should I approve this after QA is done?",
            raw_text="<@BMONICA> should I approve this after QA is done?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_bare_approved_question_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved?",
            raw_text="<@BMONICA> approved?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_approve_this_question_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approve this?",
            raw_text="<@BMONICA> approve this?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_ship_it_question_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "ship it?",
            raw_text="<@BMONICA> ship it?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_can_you_approve_question_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "can you approve this once QA is done?",
            raw_text="<@BMONICA> can you approve this once QA is done?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_negated_approval_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "not approved yet, still checking QA",
            raw_text="<@BMONICA> not approved yet, still checking QA",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_negated_go_ahead_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "don't go ahead yet, waiting for QA",
            raw_text="<@BMONICA> don't go ahead yet, waiting for QA",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_negated_ship_it_is_not_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "don't ship it yet, QA is still checking",
            raw_text="<@BMONICA> don't ship it yet, QA is still checking",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_cancel_blocks_waiting_run(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "do not fix this",
            raw_text="<@BMONICA> do not fix this",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_cancelled"}
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"


def test_slack_flow_stop_question_does_not_cancel_waiting_run(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "stop?",
            raw_text="<@BMONICA> stop?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.failure_reason == ""


def test_slack_flow_do_not_fix_question_does_not_cancel_waiting_run(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "do not fix?",
            raw_text="<@BMONICA> do not fix?",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.failure_reason == ""


def test_slack_flow_tagged_cancel_logs_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.slack_flow"):
        result = flow.handle_gateway_event(
            _slack_event(
                "do not fix this",
                raw_text="<@BMONICA> do not fix this",
                user_id="U_APPROVER",
                thread_ts="1710000000.000100",
            )
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result == {"action": "skip", "reason": "monica_loop_cancelled"}
    assert "event=cancelled" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs
    assert "cancelled_by=U_APPROVER" in logs


def test_slack_flow_tagged_cancel_blocks_active_run_before_approval(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="triaging")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "cancel",
            raw_text="<@BMONICA> cancel",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_cancelled"}
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"


def test_slack_flow_tagged_cancel_does_not_requeue_blocked_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="blocked", failure_reason="cancelled by U_APPROVER")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "cancel",
            raw_text="<@BMONICA> cancel",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_cancelled"}
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"
    assert launched == []


def test_slack_flow_tagged_cancel_does_not_requeue_completed_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        pr_url="https://github.com/acme/mobile/pull/123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "stop",
            raw_text="<@BMONICA> stop",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_cancelled"}
    assert updated is not None
    assert updated.status == "done"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert launched == []


def test_slack_flow_negated_cancel_does_not_block_waiting_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "don't stop yet, still checking QA",
            raw_text="<@BMONICA> don't stop yet, still checking QA",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.failure_reason == ""
    assert launched == []


def test_slack_flow_bug_context_with_stop_is_not_cancel(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "can you stop this Android checkout crash from happening?",
            raw_text="<@BMONICA> can you stop this Android checkout crash from happening?",
            user_id="U_TAGGER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_active"}
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.failure_reason == ""
    assert launched == []


def test_slack_flow_untagged_approval_is_ignored(tmp_path):
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=lambda run_id: None)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result is None
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"


def test_slack_flow_tagged_approval_from_unauthorized_user_is_explicitly_denied(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@BMONICA> approved, fix it",
            user_id="U_RANDOM",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_denied",
        "text": (
            "I cannot start the fix from this approval. "
            "A configured Monica approver must tag me to approve code changes. "
            "Allowed approvers: <@U_APPROVER>."
        ),
    }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_approval_refuses_when_readiness_check_fails(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (False, "repo.url is missing"),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_not_ready",
        "text": (
            "I cannot start the fix from this approval because Monica is not ready for approved-PR mode: "
            "repo.url is missing"
        ),
    }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_approval_refuses_when_linear_api_key_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
        linear=LinearConfig(team_id="team-id"),
        repo=RepoConfig(url="git@github.com:acme/mobile.git"),
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_not_ready",
        "text": (
            "I cannot start the fix from this approval because Monica is not ready for approved-PR mode: "
            "LINEAR_API_KEY is missing"
        ),
    }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_approval_refuses_when_slack_bot_token_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("MONICA_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.setenv("LINEAR_API_KEY", "lin-key")
    monkeypatch.setattr(
        "plugins.mobile_bug_agent.slack_flow.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
        linear=LinearConfig(team_id="team-id"),
        repo=RepoConfig(url="git@github.com:acme/mobile.git"),
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_not_ready",
            "text": (
                "I cannot start the fix from this approval because Monica is not ready for approved-PR mode: "
                "MONICA_SLACK_BOT_TOKEN is missing"
            ),
        }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_approval_refuses_chandler_worker_session_prefix(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LINEAR_API_KEY", "lin-key")
    monkeypatch.setattr(
        "plugins.mobile_bug_agent.slack_flow.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
        linear=LinearConfig(team_id="team-id"),
        repo=RepoConfig(url="git@github.com:acme/mobile.git"),
        runtime=RuntimeConfig(worker_session_prefix="chandler"),
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_not_ready",
        "text": (
            "I cannot start the fix from this approval because Monica is not ready for approved-PR mode: "
            "runtime.worker_session_prefix must include `monica` to keep worker sessions segregated"
        ),
    }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_tagged_approval_is_denied_when_no_approvers_are_configured(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_RANDOM",
            thread_ts="1710000000.000100",
        )
    )

    updated = state.get_run(run.id)
    assert result == {
        "action": "skip_reply",
        "reason": "monica_loop_approval_denied",
        "text": (
            "I cannot start the fix from this approval. "
            "No Monica approver is configured for code changes."
        ),
    }
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert launched == []


def test_slack_flow_requeues_completed_thread_for_linear_update(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        pr_url="https://github.com/acme/mobile/pull/123",
        failure_reason="old failure",
        approved_by_user_id="U_OLD_APPROVER",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "new reproduction detail: happens only after promo code",
            raw_text="<@BMONICA> new reproduction detail: happens only after promo code",
            user_id="U_REPRODUCER",
            thread_ts="1710000000.000100",
            message_ts="1710000002.000300",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_requeued"}
    assert updated is not None
    assert updated.status == "queued"
    assert updated.linear_issue_id == "issue-id"
    assert updated.linear_identifier == "MOB-123"
    assert updated.linear_url == "https://linear.app/acme/issue/MOB-123"
    assert updated.branch_name == ""
    assert updated.pr_url == ""
    assert updated.failure_reason == ""
    assert updated.approved_by_user_id == ""
    assert updated.user_id == "U_REPRODUCER"
    assert updated.message_ts == "1710000002.000300"
    assert "new reproduction detail" in updated.request_text
    assert launched == [run.id]


def test_slack_flow_requeue_logs_breadcrumbs(tmp_path, caplog):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        pr_url="https://github.com/acme/mobile/pull/123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.slack_flow"):
        result = flow.handle_gateway_event(
            _slack_event(
                "new reproduction detail: happens only after promo code",
                raw_text="<@BMONICA> new reproduction detail: happens only after promo code",
                user_id="U_REPRODUCER",
                thread_ts="1710000000.000100",
                message_ts="1710000002.000300",
            )
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result == {"action": "skip", "reason": "monica_loop_requeued"}
    assert "event=requeued" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs
    assert "requeued_by=U_REPRODUCER" in logs
    assert launched == [run.id]


def test_slack_flow_does_not_requeue_completed_thread_for_tagged_thanks(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        pr_url="https://github.com/acme/mobile/pull/123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "thanks, this is fixed now",
            raw_text="<@BMONICA> thanks, this is fixed now",
            user_id="U_TAGGER",
            thread_ts="1710000000.000100",
            message_ts="1710000003.000400",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_noop"}
    assert updated is not None
    assert updated.status == "done"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert updated.message_ts == "1710000000.000100"
    assert updated.request_text == "original Android checkout crash"
    assert launched == []


def test_slack_flow_does_not_requeue_completed_thread_when_thanks_mentions_bug(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        pr_url="https://github.com/acme/mobile/pull/123",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "thanks, the Android checkout crash is fixed now",
            raw_text="<@BMONICA> thanks, the Android checkout crash is fixed now",
            user_id="U_TAGGER",
            thread_ts="1710000000.000100",
            message_ts="1710000003.000400",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_noop"}
    assert updated is not None
    assert updated.status == "done"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert updated.message_ts == "1710000000.000100"
    assert updated.request_text == "original Android checkout crash"
    assert launched == []


def test_slack_flow_late_approval_resumes_completed_ticket_only_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
            message_ts="1710000004.000500",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert updated.linear_identifier == "MOB-123"
    assert updated.linear_issue_id == "issue-id"
    assert updated.linear_url == "https://linear.app/acme/issue/MOB-123"
    assert updated.request_text == "original Android checkout crash"
    assert launched == [run.id]


def test_slack_flow_late_approval_logs_breadcrumbs(tmp_path, caplog):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.slack_flow"):
        result = flow.handle_gateway_event(
            _slack_event(
                "approved, fix it",
                raw_text="<@U_MONICA> approved, fix it",
                user_id="U_APPROVER",
                thread_ts="1710000000.000100",
                message_ts="1710000004.000500",
            )
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert "event=approved" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs
    assert "approved_by=U_APPROVER" in logs
    assert launched == [run.id]


def test_slack_flow_tagged_approval_resumes_blocked_linear_run(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="blocked",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        failure_reason="draft_pr_url_missing",
        approved_by_user_id="U_APPROVER",
    )
    flow = MonicaSlackFlow(
        config=config,
        state=state,
        loop_launcher=launched.append,
        approval_readiness_checker=lambda: (True, ""),
    )

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, try again",
            raw_text="<@U_MONICA> approved, try again",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
            message_ts="1710000005.000600",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_approved"}
    assert updated is not None
    assert updated.status == "approved"
    assert updated.linear_identifier == "MOB-123"
    assert updated.linear_issue_id == "issue-id"
    assert updated.linear_url == "https://linear.app/acme/issue/MOB-123"
    assert updated.failure_reason == ""
    assert updated.approved_by_user_id == "U_APPROVER"
    assert updated.pr_url == ""
    assert updated.request_text == "original Android checkout crash"
    assert launched == [run.id]


def test_slack_flow_does_not_requeue_completed_pr_for_late_approval(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("BMONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="done",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        pr_url="https://github.com/acme/mobile/pull/123",
        approved_by_user_id="U_APPROVER",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@BMONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
            message_ts="1710000004.000500",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_done"}
    assert updated is not None
    assert updated.status == "done"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert updated.message_ts == "1710000000.000100"
    assert updated.request_text == "original Android checkout crash"
    assert launched == []


def test_slack_flow_does_not_requeue_blocked_run_that_already_has_pr(tmp_path):
    launched: list[str] = []
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        slack=SlackConfig(
            allowed_channels=("C_MOBILE",),
            bot_user_ids=("U_MONICA",),
            approver_user_ids=("U_APPROVER",),
        ),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000100",
        user_id="U_TAGGER",
        request_text="original Android checkout crash",
    )
    state.update_run(
        run.id,
        status="blocked",
        linear_identifier="MOB-123",
        linear_issue_id="issue-id",
        linear_url="https://linear.app/acme/issue/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        pr_url="https://github.com/acme/mobile/pull/123",
        failure_reason="cancelled by U_APPROVER",
        approved_by_user_id="U_APPROVER",
    )
    flow = MonicaSlackFlow(config=config, state=state, loop_launcher=launched.append)

    result = flow.handle_gateway_event(
        _slack_event(
            "approved, fix it",
            raw_text="<@U_MONICA> approved, fix it",
            user_id="U_APPROVER",
            thread_ts="1710000000.000100",
            message_ts="1710000006.000700",
        )
    )

    updated = state.get_run(run.id)
    assert result == {"action": "skip", "reason": "monica_loop_already_done"}
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == "https://github.com/acme/mobile/pull/123"
    assert updated.failure_reason == "cancelled by U_APPROVER"
    assert updated.approved_by_user_id == "U_APPROVER"
    assert updated.message_ts == "1710000000.000100"
    assert updated.request_text == "original Android checkout crash"
    assert launched == []


@dataclass
class FakeSkills(MonicaLoopSkills):
    calls: list[str] = field(default_factory=list)
    status_posts: list[str] = field(default_factory=list)

    def read_slack_thread(self, run: Any) -> dict[str, Any]:
        self.calls.append("read_slack_thread")
        return {
            "permalink": "https://example.slack.com/archives/C_MOBILE/p1710000000000200",
            "messages": [
                "Alice: Android checkout crashes after applying a promo code.",
                "Bob: I reproduced on 2.14.1.",
            ],
            "attachments": ["screenshot.png"],
        }

    def infer_user_intent(self, run: Any, thread: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("infer_user_intent")
        return {
            "is_mobile_bug": True,
            "confidence": 0.91,
            "wants_fix": True,
            "needs_clarification": False,
            "summary": "Android checkout crashes after applying a promo code.",
        }

    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        return {
            "identifier": "DRY-RUN",
            "url": "",
            "dry_run": True,
            "title": "[Mobile] Android checkout crashes after promo code",
            "description": "## Summary\nAndroid checkout crashes after applying a promo code.",
        }

    def ask_fix_approval(self, run: Any, issue: dict[str, Any]) -> None:
        self.calls.append("ask_fix_approval")

    def post_status(self, run: Any, text: str) -> None:
        self.status_posts.append(text)

    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        self.calls.append("run_internal_codex_worker")
        raise AssertionError("Monica must not write code before approval")

    def run_verification(self, run: Any, worker_result: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("run_verification")
        return {"passed": True}

    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("open_draft_pr")
        return {"url": "https://github.com/example/mobile/pull/123"}


def test_dry_run_finishes_without_waiting_for_approval_or_code(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="dry_run")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FakeSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.linear_identifier == "DRY-RUN"
    assert skills.calls == [
        "read_slack_thread",
        "infer_user_intent",
        "create_or_update_linear",
    ]
    assert "Dry run" in skills.status_posts[0]
    assert "Preview:" in skills.status_posts[0]
    assert "## Summary" in skills.status_posts[0]


def test_dry_run_logs_ticket_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="dry_run")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FakeSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=done" in logs
    assert "stage=dry_run" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "DRY-RUN" in logs


@dataclass
class CancellingReadSkills(FakeSkills):
    state: MonicaState | None = None

    def read_slack_thread(self, run: Any) -> dict[str, Any]:
        self.calls.append("read_slack_thread")
        assert self.state is not None
        self.state.update_run(
            run.id,
            status="blocked",
            failure_reason="cancelled by U_APPROVER",
        )
        return {"permalink": "https://example.slack.com/thread", "messages": [run.request_text]}


def test_loop_stops_when_run_is_cancelled_during_triage(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = CancellingReadSkills(state=state)

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"
    assert skills.calls == ["read_slack_thread"]


def test_unknown_rollout_mode_blocks_before_side_effects(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="typo")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FakeSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "unknown_rollout_mode: typo"
    assert skills.calls == []
    assert "known rollout mode" in skills.status_posts[0]


def test_unknown_rollout_mode_logs_blocked_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="typo")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FakeSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=blocked" in logs
    assert "stage=preflight" in logs
    assert "failure_reason=unknown_rollout_mode: typo" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs


def test_linear_only_blocks_when_linear_creation_is_disabled(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="linear_only",
        loop=LoopConfig(create_linear=False),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FakeSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "linear_creation_disabled_in_rollout"
    assert skills.calls == []
    assert "Linear creation is disabled" in skills.status_posts[0]


@dataclass
class ClarificationNeededSkills(FakeSkills):
    def infer_user_intent(self, run: Any, thread: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("infer_user_intent")
        return {
            "is_mobile_bug": True,
            "wants_linear": False,
            "wants_fix": False,
            "needs_clarification": True,
            "confidence": 0.52,
            "summary": "Android checkout report is missing reproduction details",
            "missing_questions": ["Which app build and device reproduced this?"],
        }

    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        raise AssertionError("Monica must not file Linear before clarification")


def test_clarification_needed_logs_triage_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you look into this checkout thing?",
    )
    skills = ClarificationNeededSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=needs_clarification" in logs
    assert "stage=triaging" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs


@dataclass
class NonMobileBugSkills(FakeSkills):
    def infer_user_intent(self, run: Any, thread: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("infer_user_intent")
        return {
            "is_mobile_bug": False,
            "wants_linear": False,
            "wants_fix": False,
            "needs_clarification": False,
            "confidence": 0.24,
            "summary": "Backend webhook discussion",
            "reason": "The thread is about backend webhook delivery rather than the mobile app.",
        }

    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        raise AssertionError("Monica must not file Linear for non-mobile bugs")


def test_not_mobile_bug_logs_blocked_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean up this webhook delivery bug?",
    )
    skills = NonMobileBugSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=blocked" in logs
    assert "stage=triaging" in logs
    assert "failure_reason=not_a_mobile_bug" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs


@dataclass
class QuestionOnlyBugSkills(FakeSkills):
    def infer_user_intent(self, run: Any, thread: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("infer_user_intent")
        return {
            "is_mobile_bug": True,
            "wants_linear": False,
            "wants_fix": False,
            "needs_clarification": False,
            "confidence": 0.86,
            "summary": "Android checkout crash discussion",
            "reason": "The thread is about a mobile bug, but the tag asks for thoughts only.",
        }

    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        raise AssertionError("Monica must not file Linear when the classifier says no action was requested")


def test_question_only_mobile_bug_tag_asks_for_clarification_without_filing(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="any thoughts on this Android checkout crash?",
    )
    skills = QuestionOnlyBugSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "needs_clarification"
    assert updated.linear_identifier == ""
    assert skills.calls == ["read_slack_thread", "infer_user_intent"]
    assert "file a Linear issue or prepare a fix" in skills.status_posts[0]


@dataclass
class LinearOnlySkills(FakeSkills):
    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        return {
            "id": "issue-id",
            "identifier": "MOB-123",
            "url": "https://linear.app/acme/issue/MOB-123",
            "dry_run": False,
            "title": "[Mobile] Android checkout crashes after promo code",
        }


def test_linear_only_creates_ticket_without_waiting_for_code_approval(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = LinearOnlySkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.linear_identifier == "MOB-123"
    assert skills.calls == [
        "read_slack_thread",
        "infer_user_intent",
        "create_or_update_linear",
    ]
    assert "Created Linear issue" in skills.status_posts[0]
    assert "Code fixes are disabled" in skills.status_posts[0]


def test_linear_only_logs_ticket_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = LinearOnlySkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=done" in logs
    assert "stage=linear_only" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs


def test_approved_pr_creates_ticket_then_waits_for_approval_before_code(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="approved_pr")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = LinearOnlySkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.linear_identifier == "MOB-123"
    assert skills.calls == [
        "read_slack_thread",
        "infer_user_intent",
        "create_or_update_linear",
        "ask_fix_approval",
    ]


def test_approved_pr_logs_awaiting_approval_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="approved_pr")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = LinearOnlySkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=awaiting_fix_approval" in logs
    assert "stage=linear_created" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs


def test_approved_pr_still_requires_tagged_approval_when_config_disables_gate(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        loop=LoopConfig(require_fix_approval=False),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = LinearOnlySkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.approved_by_user_id == ""
    assert skills.calls == [
        "read_slack_thread",
        "infer_user_intent",
        "create_or_update_linear",
        "ask_fix_approval",
    ]


@dataclass
class FailingLinearSkills(FakeSkills):
    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        raise RuntimeError("Linear write failed")


@dataclass
class SensitiveFailingLinearSkills(FakeSkills):
    def create_or_update_linear(self, run: Any, thread: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("create_or_update_linear")
        raise RuntimeError("Linear write failed at /Users/ritik/.hermes/secrets with xoxb-secret-token")


def test_loop_marks_unexpected_linear_failure_with_stage(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = FailingLinearSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.failure_reason == "creating_linear_failed: Linear write failed"
    assert "Stage: creating_linear" in skills.status_posts[0]


def test_loop_failure_status_does_not_post_sensitive_exception_details(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you clean this Android checkout crash up?",
    )
    skills = SensitiveFailingLinearSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.failure_reason == (
        "creating_linear_failed: Linear write failed at /Users/ritik/.hermes/secrets "
        "with xoxb-secret-token"
    )
    assert "Stage: creating_linear" in skills.status_posts[0]
    assert "/Users/ritik/.hermes/secrets" not in skills.status_posts[0]
    assert "xoxb-secret-token" not in skills.status_posts[0]
    assert "Check Monica logs or `hermes mobile-bug-agent show" in skills.status_posts[0]


@dataclass
class ApprovedFixSkills(FakeSkills):
    verification_passed: bool = True
    proof_passed: bool = True
    proof_artifacts: tuple[str, ...] = ("/tmp/monica-proof/screenshot.png",)

    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        self.calls.append("run_internal_codex_worker")
        return {"branch_name": "monica/MOB-123-checkout-crash", "changed": True}

    def run_verification(self, run: Any, worker_result: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("run_verification")
        return {"passed": self.verification_passed, "summary": "npm test"}

    def run_proof(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("run_proof")
        return {
            "passed": self.proof_passed,
            "summary": "Simulator proof captured." if self.proof_passed else "simctl is unavailable",
            "artifacts": list(self.proof_artifacts),
            "output": "",
        }

    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("open_draft_pr")
        return {"url": "https://github.com/example/mobile/pull/123"}


def test_approved_run_writes_code_then_opens_draft_pr(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == "https://github.com/example/mobile/pull/123"
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "open_draft_pr"]


def test_local_fix_only_run_writes_code_and_stops_before_pr(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="local_fix_only",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker", "run_verification"]
    assert "Local fix is ready" in skills.status_posts[0]
    assert "not pushed" in skills.status_posts[0]


def test_local_fix_only_requires_proof_before_done_when_configured(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="local_fix_only",
        verification=VerificationConfig(commands=("npm test",)),
        proof=ProofConfig(enabled=True, required_for_done=True),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills(proof_passed=False, proof_artifacts=())

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "proof_blocked"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == ""
    assert updated.failure_reason == "proof_unavailable: simctl is unavailable"
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "run_proof"]
    assert "Verification passed" in skills.status_posts[0]
    assert "proof is unavailable" in skills.status_posts[0]
    assert "not mark this run done" in skills.status_posts[0]


def test_proof_blocked_run_resumes_existing_branch_without_worker(tmp_path):
    runtime = tmp_path / "runtime"
    worktree = runtime / "workspace" / "worktrees" / "monica-MOB-123-checkout-crash"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /tmp/fake\n", encoding="utf-8")
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        repo=RepoConfig(branch_prefix="monica"),
        runtime=RuntimeConfig(home_subdir=str(runtime)),
        verification=VerificationConfig(commands=("npm test",)),
        proof=ProofConfig(enabled=True, required_for_done=True),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
        raw_event={"permalink": "https://slack.example/archives/C_MOBILE/p1710000000000100"},
    )
    state.update_run(
        run.id,
        status="proof_blocked",
        linear_identifier="MOB-123",
        linear_url="https://linear.example/MOB-123",
        branch_name="monica/MOB-123-checkout-crash",
        failure_reason="proof_unavailable: simulator not configured",
        approved_by_user_id="U_TAGGER",
    )
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == "https://github.com/example/mobile/pull/123"
    assert skills.calls == ["run_verification", "run_proof", "open_draft_pr"]


def test_local_fix_only_marks_done_after_required_proof_artifact(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="local_fix_only",
        verification=VerificationConfig(commands=("npm test",)),
        proof=ProofConfig(enabled=True, required_for_done=True),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "done"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "run_proof"]
    assert "Proof captured" in skills.status_posts[0]
    assert "/tmp/monica-proof/screenshot.png" in skills.status_posts[0]


def test_approved_pr_requires_proof_before_opening_pr_when_configured(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
        proof=ProofConfig(enabled=True, required_for_done=True),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills(proof_passed=True, proof_artifacts=())

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "proof_blocked"
    assert updated.pr_url == ""
    assert updated.failure_reason == "proof_unavailable: Simulator proof captured."
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "run_proof"]


def test_local_fix_only_creates_ticket_then_waits_for_approval_before_code(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="local_fix_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    skills = LinearOnlySkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "awaiting_fix_approval"
    assert updated.linear_identifier == "MOB-123"
    assert skills.calls == ["read_slack_thread", "infer_user_intent", "create_or_update_linear", "ask_fix_approval"]


def test_approved_run_logs_operator_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "monica/MOB-123-checkout-crash" in logs
    assert "https://github.com/example/mobile/pull/123" in logs


def test_approved_run_does_not_write_code_outside_approved_pr_rollout(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "approved_pr_rollout_not_enabled"
    assert skills.calls == []


def test_approved_run_outside_approved_pr_rollout_logs_blocked_breadcrumbs(tmp_path, caplog):
    config = MonicaConfig(enabled=True, rollout_mode="linear_only")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=blocked" in logs
    assert "stage=preflight" in logs
    assert "failure_reason=approved_pr_rollout_not_enabled" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs


def test_approved_run_does_not_write_code_without_linear_issue(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "linear_issue_missing_before_fix"
    assert skills.calls == []
    assert "Linear issue" in skills.status_posts[0]


def test_approved_run_does_not_write_code_without_verification_commands(tmp_path):
    config = MonicaConfig(enabled=True, rollout_mode="approved_pr")
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "verification_commands_missing"
    assert skills.calls == []
    assert "verification.commands" in skills.status_posts[0]


@dataclass
class MissingBranchFixSkills(ApprovedFixSkills):
    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        self.calls.append("run_internal_codex_worker")
        return {"changed": True, "summary": "Patched checkout crash"}


def test_approved_run_blocks_when_worker_returns_no_branch(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = MissingBranchFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "worker_branch_missing"
    assert updated.branch_name == ""
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker"]
    assert "could not identify a branch" in skills.status_posts[0]


@dataclass
class MismatchedBranchFixSkills(ApprovedFixSkills):
    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        self.calls.append("run_internal_codex_worker")
        return {
            "branch_name": "chandler/MOB-123-checkout-crash",
            "changed": True,
            "summary": "Patched checkout crash",
        }


def test_approved_run_blocks_when_worker_returns_mismatched_branch(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = MismatchedBranchFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "worker_branch_mismatch"
    assert updated.branch_name == ""
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker"]
    assert "unexpected branch" in skills.status_posts[0]


@dataclass
class NoChangeFixSkills(ApprovedFixSkills):
    def run_internal_codex_worker(self, run: Any) -> dict[str, Any]:
        self.calls.append("run_internal_codex_worker")
        return {
            "branch_name": "monica/MOB-123-checkout-crash",
            "changed": False,
            "summary": "The worker did not make code changes.",
        }


def test_approved_run_blocks_when_worker_reports_no_changes(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = NoChangeFixSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "worker_no_changes"
    assert updated.branch_name == "monica/MOB-123-checkout-crash"
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker"]
    assert "did not report any code changes" in skills.status_posts[0]


def test_failed_verification_blocks_draft_pr(tmp_path, caplog):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(
        run.id,
        status="awaiting_fix_approval",
        linear_identifier="MOB-123",
        linear_url="https://linear.app/acme/issue/MOB-123",
    )
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = ApprovedFixSkills(verification_passed=False)

    with caplog.at_level(logging.INFO, logger="plugins.mobile_bug_agent.loop"):
        MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.pr_url == ""
    assert "verification_failed" in updated.failure_reason
    assert skills.calls == ["run_internal_codex_worker", "run_verification"]
    assert "Verification failed, so I did not open a PR." in skills.status_posts[0]
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "event=blocked" in logs
    assert "stage=verifying" in logs
    assert run.id in logs
    assert "C_MOBILE" in logs
    assert "1710000000.000100" in logs
    assert "MOB-123" in logs
    assert "https://linear.app/acme/issue/MOB-123" in logs
    assert "monica/MOB-123-checkout-crash" in logs
    assert "verification_failed: npm test" in logs


@dataclass
class FailingPrSkills(ApprovedFixSkills):
    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("open_draft_pr")
        raise RuntimeError("gh pr create failed")


@dataclass
class MissingPrUrlSkills(ApprovedFixSkills):
    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("open_draft_pr")
        return {"url": ""}


@dataclass
class CancellingPrSkills(ApprovedFixSkills):
    state: MonicaState | None = None

    def open_draft_pr(
        self,
        run: Any,
        worker_result: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("open_draft_pr")
        assert self.state is not None
        self.state.update_run(
            run.id,
            status="blocked",
            failure_reason="cancelled by U_APPROVER",
        )
        return {"url": "https://github.com/example/mobile/pull/123"}


def test_loop_does_not_overwrite_cancellation_during_pr_opening(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = CancellingPrSkills(state=state)

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "cancelled by U_APPROVER"
    assert updated.pr_url == "https://github.com/example/mobile/pull/123"
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "open_draft_pr"]
    assert not any("Draft PR is ready" in post for post in skills.status_posts)


def test_loop_blocks_when_pr_publisher_returns_no_url(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = MissingPrUrlSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.failure_reason == "draft_pr_url_missing"
    assert updated.pr_url == ""
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "open_draft_pr"]
    assert "did not return a draft PR URL" in skills.status_posts[0]


def test_loop_marks_unexpected_pr_failure_with_stage(tmp_path):
    config = MonicaConfig(
        enabled=True,
        rollout_mode="approved_pr",
        verification=VerificationConfig(commands=("npm test",)),
    )
    state = MonicaState.open(tmp_path / "monica.sqlite")
    run = state.create_run(
        platform="slack",
        channel_id="C_MOBILE",
        thread_ts="1710000000.000100",
        message_ts="1710000000.000200",
        user_id="U_TAGGER",
        request_text="can you fix this Android checkout crash?",
    )
    state.update_run(run.id, status="awaiting_fix_approval", linear_identifier="MOB-123")
    state.approve_fix(run.id, approved_by_user_id="U_TAGGER")
    skills = FailingPrSkills()

    MonicaLoop(config=config, state=state, skills=skills).run(run.id)

    updated = state.get_run(run.id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.failure_reason == "opening_pr_failed: gh pr create failed"
    assert skills.calls == ["run_internal_codex_worker", "run_verification", "open_draft_pr"]
    assert "Stage: opening_pr" in skills.status_posts[0]
