import copy
from unittest.mock import patch

import pytest

import run_agent
from run_agent import AIAgent
from agent import pii_redaction
from hermes_constants import get_hermes_home


ENABLED_CFG = {
    "enabled": True,
    "provider": "rampart",
    "hosted_only": True,
    "fail_closed": True,
    "timeout_seconds": 10,
    "rampart": {
        "command": "",
        "model": "nationaldesignstudio/rampart",
        "heuristics_only": False,
    },
}


def _fake_redact(texts, _cfg):
    return [
        text.replace("alice@example.com", "[EMAIL]")
        .replace("555-1212", "[PHONE]")
        .replace("secret-token", "[TOKEN]")
        for text in texts
    ]


def test_load_pii_redaction_config_defaults_from_partial_config():
    cfg = pii_redaction.load_pii_redaction_config({"security": {"pii_redaction": {"enabled": True}}})

    assert cfg == {
        "enabled": True,
        "provider": "rampart",
        "hosted_only": True,
        "fail_closed": True,
        "timeout_seconds": 10.0,
        "rampart": {
            "command": "",
            "model": "nationaldesignstudio/rampart",
            "heuristics_only": False,
        },
    }


def test_should_redact_skips_local_endpoints_when_hosted_only():
    assert pii_redaction.should_redact_for_llm("https://api.openai.com/v1", pii_config=ENABLED_CFG)
    assert not pii_redaction.should_redact_for_llm("http://localhost:11434/v1", pii_config=ENABLED_CFG)

    cfg = {**ENABLED_CFG, "hosted_only": False}
    assert pii_redaction.should_redact_for_llm("http://127.0.0.1:8000/v1", pii_config=cfg)


def test_redact_text_for_llm_changes_text_and_reports_disabled_skip(monkeypatch):
    monkeypatch.setattr(pii_redaction, "_redact_texts", _fake_redact)

    redacted, stats = pii_redaction.redact_text_for_llm(
        "Email alice@example.com before calling 555-1212.",
        pii_config=ENABLED_CFG,
    )
    assert redacted == "Email [EMAIL] before calling [PHONE]."
    assert stats["redacted"] is True
    assert stats["texts_changed"] == 1
    assert stats["replacement_count"] == 1

    original, skipped = pii_redaction.redact_text_for_llm(
        "Email alice@example.com",
        pii_config={**ENABLED_CFG, "enabled": False},
    )
    assert original == "Email alice@example.com"
    assert skipped["skipped"] is True
    assert skipped["skipped_reason"] == "disabled"


def test_redact_messages_deep_copies_and_does_not_mutate(monkeypatch):
    monkeypatch.setattr(pii_redaction, "_redact_texts", _fake_redact)
    messages = [{"role": "user", "content": "Contact alice@example.com"}]
    before = copy.deepcopy(messages)

    redacted, stats = pii_redaction.redact_messages_for_llm(messages, pii_config=ENABLED_CFG)

    assert redacted is not messages
    assert redacted[0] is not messages[0]
    assert messages == before
    assert redacted[0]["content"] == "Contact [EMAIL]"
    assert stats["texts_scanned"] == 1


def test_redact_payload_traverses_text_content_and_tool_call_arguments_only(monkeypatch):
    seen = []

    def fake(texts, cfg):
        seen.extend(texts)
        return _fake_redact(texts, cfg)

    monkeypatch.setattr(pii_redaction, "_redact_texts", fake)
    payload = {
        "model": "gpt-alice@example.com",
        "messages": [
            {
                "role": "user",
                "name": "alice@example.com",
                "content": "My email is alice@example.com",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_alice@example.com",
                        "type": "function",
                        "function": {
                            "name": "lookup_alice@example.com",
                            "arguments": '{"email":"alice@example.com","phone":"555-1212"}',
                        },
                    }
                ],
            },
        ],
    }

    redacted, stats = pii_redaction.maybe_redact_api_kwargs(payload, pii_config=ENABLED_CFG)

    assert seen == [
        "My email is alice@example.com",
        '{"email":"alice@example.com","phone":"555-1212"}',
    ]
    assert redacted["model"] == "gpt-alice@example.com"
    assert redacted["messages"][0]["name"] == "alice@example.com"
    assert redacted["messages"][0]["content"] == "My email is [EMAIL]"
    function = redacted["messages"][1]["tool_calls"][0]["function"]
    assert function["name"] == "lookup_alice@example.com"
    assert function["arguments"] == '{"email":"[EMAIL]","phone":"[PHONE]"}'
    assert stats["texts_scanned"] == 2
    assert stats["texts_changed"] == 2


def test_encrypted_content_and_multimodal_image_file_parts_are_skipped(monkeypatch):
    seen = []

    def fake(texts, cfg):
        seen.extend(texts)
        return _fake_redact(texts, cfg)

    monkeypatch.setattr(pii_redaction, "_redact_texts", fake)
    messages = [
        {
            "role": "user",
            "encrypted_content": "alice@example.com",
            "content": [
                {"type": "input_text", "text": "Visible alice@example.com"},
                {"type": "input_image", "image_url": "https://example.test/alice@example.com.png"},
                {"type": "input_file", "filename": "alice@example.com.pdf", "file_data": "alice@example.com"},
                {"type": "image_url", "url": "data:image/png;base64,alice@example.com"},
            ],
        }
    ]

    redacted, stats = pii_redaction.redact_messages_for_llm(messages, pii_config=ENABLED_CFG)

    assert seen == ["Visible alice@example.com"]
    assert redacted[0]["encrypted_content"] == "alice@example.com"
    assert redacted[0]["content"][0]["text"] == "Visible [EMAIL]"
    assert redacted[0]["content"][1]["image_url"] == "https://example.test/alice@example.com.png"
    assert redacted[0]["content"][2]["filename"] == "alice@example.com.pdf"
    assert redacted[0]["content"][2]["file_data"] == "alice@example.com"
    assert redacted[0]["content"][3]["url"] == "data:image/png;base64,alice@example.com"
    assert stats["texts_scanned"] == 1


def test_redaction_fail_closed_raises(monkeypatch):
    def fail(_texts, _cfg):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(pii_redaction, "_redact_texts", fail)

    with pytest.raises(RuntimeError, match="refusing to send unredacted payload"):
        pii_redaction.redact_messages_for_llm(
            [{"role": "user", "content": "alice@example.com"}],
            pii_config=ENABLED_CFG,
        )


def test_redaction_fail_open_returns_original_payload(monkeypatch):
    def fail(_texts, _cfg):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(pii_redaction, "_redact_texts", fail)
    payload = [{"role": "user", "content": "alice@example.com"}]

    redacted, stats = pii_redaction.redact_messages_for_llm(
        payload,
        pii_config={**ENABLED_CFG, "fail_closed": False},
    )

    assert redacted is payload
    assert redacted[0]["content"] == "alice@example.com"
    assert stats["failure"] == "RuntimeError"
    assert stats["redacted"] is True


def test_rampart_worker_uses_sanitized_environment(monkeypatch):
    captured = {}

    class Completed:
        returncode = 0
        stdout = '{"texts":["redacted"]}'

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Completed()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setattr(pii_redaction.subprocess, "run", fake_run)

    assert pii_redaction._redact_with_rampart(["raw"], ENABLED_CFG) == ["redacted"]
    env = captured["env"]
    assert env["PATH"] == "/bin"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_agent_build_api_kwargs_redacts_before_provider_dispatch(monkeypatch):
    monkeypatch.setattr(pii_redaction, "_redact_texts", _fake_redact)
    run_agent._hermes_home = get_hermes_home()
    config = {"security": {"pii_redaction": ENABLED_CFG}}
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="custom",
            model="gpt-test",
            base_url="https://api.example.test/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        messages = [{"role": "user", "content": "Email alice@example.com"}]
        kwargs = agent._build_api_kwargs(messages)

    assert kwargs["messages"][0]["content"] == "Email [EMAIL]"
    assert messages[0]["content"] == "Email alice@example.com"
    assert agent._last_pii_redaction_stats["texts_changed"] == 1
