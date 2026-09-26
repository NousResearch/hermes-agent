"""Public Codex commentary is a display projection, never a replay mutation."""

from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages
from agent.history_commentary import visible_commentary
from hermes_constants import get_hermes_home_override

import asyncio
import pytest
from fastapi import Request


def _row(text="I will inspect the files."):
    return {
        "role": "assistant",
        "content": "",
        "reasoning": f"Private summary.\n\n{text}",
        "reasoning_content": f"Private summary.\n\n{text}",
        "codex_message_items": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": text}],
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "analysis",
                "content": [{"type": "output_text", "text": "private analysis"}],
            },
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
        "timestamp": 1,
    }


def test_rest_history_separates_public_commentary_without_changing_source():
    row = _row()
    projected = _project_for_display([row])[0]

    assert projected["display_commentary"] == ["I will inspect the files."]
    assert projected["display_reasoning"] == "Private summary."
    assert projected["codex_message_items"] == row["codex_message_items"]
    assert projected["tool_calls"] == row["tool_calls"]
    assert row["reasoning"] == "Private summary.\n\nI will inspect the files."


def test_gateway_history_uses_same_display_projection():
    row = _row()
    projected = _history_to_messages([row])[0]

    assert projected["display_commentary"] == ["I will inspect the files."]
    assert projected["display_reasoning"] == "Private summary."
    assert projected["codex_message_items"] == row["codex_message_items"]
    assert row["reasoning"].endswith("I will inspect the files.")


def test_history_applies_live_stripping_and_redaction_without_changing_raw_items(
    monkeypatch,
):
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)
    raw = (
        "Start. <think>private scratchpad</think> Key=" + "sk-" + "demo0123456789abcdef"
    )
    row = _row(raw)

    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == [visible_commentary(raw)]
        assert "<think>" not in projected["display_commentary"][0]
        assert "sk-demo0123456789abcdef" not in projected["display_commentary"][0]
        assert projected["display_reasoning"] == "Private summary."
        assert projected["codex_message_items"] == row["codex_message_items"]


@pytest.mark.parametrize(
    "disabled",
    [
        {"show_commentary": False},
        {"show_commentary": "false"},
        {"interim_assistant_messages": False},
    ],
)
def test_owner_profile_settings_apply_to_rest_and_rpc(monkeypatch, tmp_path, disabled):
    from hermes_cli import config as config_mod

    enabled_home, disabled_home = tmp_path / "enabled", tmp_path / "disabled"
    observed = []

    def config():
        home = get_hermes_home_override()
        observed.append(str(home))
        return {"display": {} if str(home) == str(enabled_home) else disabled}

    monkeypatch.setattr(config_mod, "load_config", config)
    row = _row()
    for adapter in (
        lambda home: _project_for_display([row], home=home)[0],
        lambda home: _history_to_messages([row], profile_home=home)[0],
    ):
        assert adapter(enabled_home)["display_commentary"] == [
            "I will inspect the files."
        ]
        hidden = adapter(disabled_home)
        assert hidden["display_commentary"] == []
        assert hidden["display_reasoning"] == "Private summary."
        assert hidden["reasoning"] == row["reasoning"]
        assert adapter(enabled_home)["display_reasoning"] == "Private summary."
    assert observed == [str(enabled_home), str(disabled_home), str(enabled_home)] * 2


def test_rest_pages_bind_the_history_owner_for_messages_and_around(
    monkeypatch, tmp_path
):
    from hermes_cli import config as config_mod
    from hermes_cli.web_routers import sessions
    import hermes_state_timeline

    row = _row()
    homes = {name: tmp_path / name for name in ("visible", "hidden")}
    monkeypatch.setattr(
        sessions, "_cron_profile_home", lambda profile: (profile, homes[profile])
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {
                "show_commentary": str(get_hermes_home_override())
                == str(homes["visible"])
            }
        },
    )

    class FakeDB:
        def resolve_session_id(self, sid):
            return sid

        def resolve_resume_session_id(self, sid):
            return sid

        def get_messages(self, *args, **kwargs):
            return [row]

    monkeypatch.setattr(
        sessions, "_with_db", lambda profile, fn, read_only: fn(FakeDB())
    )
    monkeypatch.setattr(sessions, "_timeline_session_id", lambda db, sid, owner: sid)
    monkeypatch.setattr(
        hermes_state_timeline,
        "get_session_messages_around",
        lambda *args, **kwargs: {"messages": [row], "pagination": {}},
    )

    async def exercise():
        for profile, expected in (
            ("visible", ["I will inspect the files."]),
            ("hidden", []),
        ):
            page = await sessions.get_session_messages(
                Request(scope={"type": "http", "headers": []}), "sid",
                profile=profile,
                limit=120,
                offset=0,
                order="latest",
                include_compacted=True,
            )
            around = await sessions.get_session_messages_around(
                "sid", row_id=1, profile=profile, limit=120
            )
            assert page["messages"][0]["display_commentary"] == expected
            assert around["messages"][0]["display_commentary"] == expected
            assert page["profile"] == around["profile"] == profile

    asyncio.run(exercise())


def test_unscoped_rest_history_uses_custom_home_of_its_database(monkeypatch, tmp_path):
    from hermes_cli import config as config_mod
    from hermes_cli.web_routers import sessions
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import hermes_state_timeline

    custom, default = tmp_path / "custom", tmp_path / "default"
    monkeypatch.setattr(
        sessions, "_cron_profile_home", lambda profile: ("default", default)
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {
                "show_commentary": str(get_hermes_home_override()) != str(custom)
            }
        },
    )
    row = _row()

    class FakeDB:
        def resolve_session_id(self, sid):
            return sid

        def resolve_resume_session_id(self, sid):
            return sid

        def get_messages(self, *args, **kwargs):
            return [row]

    monkeypatch.setattr(
        sessions, "_with_db", lambda profile, fn, read_only: fn(FakeDB())
    )
    monkeypatch.setattr(sessions, "_timeline_session_id", lambda db, sid, owner: sid)
    monkeypatch.setattr(
        hermes_state_timeline,
        "get_session_messages_around",
        lambda *args, **kwargs: {"messages": [row], "pagination": {}},
    )

    async def exercise():
        page = await sessions.get_session_messages(
            Request(scope={"type": "http", "headers": []}), "sid",
            profile=None,
            limit=120,
            offset=0,
            order="latest",
            include_compacted=True,
        )
        around = await sessions.get_session_messages_around(
            "sid",
            row_id=1,
            profile=None,
            limit=120,
        )
        assert page["messages"][0]["display_commentary"] == []
        assert around["messages"][0]["display_commentary"] == []

    token = set_hermes_home_override(str(custom))
    try:
        asyncio.run(exercise())
    finally:
        reset_hermes_home_override(token)


def test_cold_resume_uses_the_resumed_session_home(monkeypatch, tmp_path):
    from hermes_cli import config as config_mod
    from tui_gateway.server import _Resume

    home = tmp_path / "disabled"
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {"show_commentary": str(get_hermes_home_override()) != str(home)}
        },
    )
    resume = _Resume.__new__(_Resume)
    resume.omit_messages = False
    resume.profile_home = home

    assert resume.messages([_row()])[0]["display_commentary"] == []


def test_joined_canonical_commentary_uses_sanitized_display_copy(monkeypatch):
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)
    raw = "Checking. Key=" + "sk-" + "demo0123456789abcdef"
    row = _row(raw)
    row["content"] = raw

    visible = _project_for_display([row])[0]
    assert visible["display_content"] == visible_commentary(raw)
    assert visible["content"] == raw
    assert visible["display_reasoning"] == "Private summary."

    from agent.history_commentary import project_history_commentary
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": False}}
    )
    hidden = project_history_commentary([row])[0]
    assert hidden["display_content"] == visible_commentary(raw)
    assert hidden["display_commentary"] == []


def test_only_exact_delimiter_bounded_copy_is_removed():
    row = _row("Same phrase")
    row["reasoning"] = "Summary mentioning Same phrase as an incidental substring."
    projected = _project_for_display([row])[0]
    assert projected["display_reasoning"] == row["reasoning"]

    row["reasoning"] = "Before.\n\nSame phrase\n\nAfter."
    assert _project_for_display([row])[0]["display_reasoning"] == "Before.\n\nAfter."


def test_foreign_and_malformed_items_are_not_public():
    row = _row()
    row["codex_message_items"][0]["role"] = "user"
    assert "display_commentary" not in _project_for_display([row])[0]
    row["codex_message_items"] = "{invalid JSON"
    assert "display_commentary" not in _project_for_display([row])[0]


def test_mixed_canonical_content_is_sanitized_without_truncating_possible_final(
    monkeypatch,
):
    raw = "Checking. <think>hidden</think> key=" + "sk-" + "demo0123456789abcdef"
    row = _row(raw)
    row["content"] = raw + "\n\nFinal result."
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)

    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == []
        assert projected["display_content"] == visible_commentary(row["content"])
        assert raw not in projected["display_reasoning"]

    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": False}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == []
        assert projected["display_content"] == visible_commentary(row["content"])
    assert row["content"].startswith(raw)


@pytest.mark.parametrize("final", ["Checking. The answer is 42.", "Checking."])
def test_stream_recovered_final_without_final_sidecar_is_authoritative(
    monkeypatch, final
):
    from types import SimpleNamespace
    from agent.turn_finalizer import _close_transcript_tail
    from hermes_cli import config as config_mod

    row = _row("Checking.")
    agent = SimpleNamespace(_db_flush_scan_prefix=None)
    _close_transcript_tail(agent, [row], final, False, True)
    assert row["content"] == final
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert (
                projected.get(
                    "display_content", projected.get("content", projected.get("text"))
                )
                == final
            )


@pytest.mark.parametrize("final", ["Checking. The answer is 42.", "Checking."])
@pytest.mark.parametrize("phase", ["final", "final_answer", None])
def test_canonical_content_matching_final_item_is_never_stripped(
    monkeypatch, final, phase
):
    from hermes_cli import config as config_mod

    row = _row("Checking.")
    row["content"] = final
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": final}],
    }
    if phase is not None:
        item["phase"] = phase
    row["codex_message_items"].append(item)
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert (
                projected.get(
                    "display_content", projected.get("content", projected.get("text"))
                )
                == final
            )
            assert projected["display_commentary"] == (["Checking."] if visible else [])


def test_mixed_canonical_preserves_final_repetition_of_a_commentary_phrase(monkeypatch):
    from hermes_cli import config as config_mod

    row = _row("Checking")
    row["content"] = "Checking\n\nFinal: Checking completed."
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == row["content"]
            assert projected["display_commentary"] == []

    row["codex_message_items"].append({
        "type": "message",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "Second update"}],
    })
    row["content"] = "Checking\n\nSecond update\n\nFinal: Checking completed."
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": True}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_content"] == row["content"]
        assert projected["display_commentary"] == []


def test_legacy_newline_and_ambiguous_repeats_do_not_leak_public_text_into_thinking():
    row = _row("Public update.")
    row["reasoning"] = "Private before.\nPublic update.\nPrivate after."
    assert (
        _project_for_display([row])[0]["display_reasoning"]
        == "Private before.\nPrivate after."
    )

    row["reasoning"] = (
        "Private before.\n\nPublic update.\n\nPrivate after.\n\nPublic update."
    )
    assert (
        _project_for_display([row])[0]["display_reasoning"]
        == "Private before.\n\nPrivate after."
    )
    assert row["reasoning"].endswith("Public update.")


def test_overlapping_public_items_do_not_leave_partial_text_in_final_or_thinking(
    monkeypatch,
):
    from hermes_cli import config as config_mod

    row = _row("Start")
    row["codex_message_items"].insert(
        1,
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Start\nTail public"}],
        },
    )
    row["reasoning"] = "Private.\n\nStart\nTail public"
    row["content"] = "Start\nTail public\n\nFinal answer."

    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == row["content"]
            assert projected["display_reasoning"] == "Private."
            assert projected["display_commentary"] == []

    # Without an attributable leading commentary span, this belongs to the final.
    row["content"] = "Leading Start\nTail public trailing"
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == "Leading Start\nTail public trailing"
            assert projected["display_commentary"] == (
                ["Start", "Start\nTail public"] if visible else []
            )

    row["content"] = "Start\nTail publicFinal answer."
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": True}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_content"] == row["content"]
        assert projected["display_commentary"] == []
