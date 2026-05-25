import hashlib
import hmac
import json
from argparse import Namespace

import pytest

from hermes_cli import linear_aig


SECRET = "linear-signing-secret"
NOW_MS = 1_770_000_000_000


def _body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signature(raw_body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def _payload(**overrides):
    payload = {
        "action": "created",
        "webhookTimestamp": NOW_MS,
        "agentSession": {
            "id": "ags_123",
            "promptContext": "Implement WHO-192",
            "issue": {
                "identifier": "WHO-192",
                "title": "Implement Hermes Linear AIG webhook receiver",
            },
        },
    }
    payload.update(overrides)
    return payload


def _args(**overrides):
    defaults = {
        "linear_aig_action": "check-config",
        "access_token": "",
        "webhook_secret": "",
        "host": "",
        "port": None,
        "graphql_url": "",
        "ack_only": False,
        "task_mode": "",
        "model": "",
        "provider": "",
        "toolsets": "",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_verify_linear_signature_accepts_valid_signature():
    payload = _payload()
    raw_body = _body(payload)

    assert linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_load_runtime_config_uses_environment_defaults():
    config = linear_aig.load_runtime_config(
        _args(),
        {
            linear_aig.ACCESS_TOKEN_ENV: "lin_access_token",
            linear_aig.WEBHOOK_SECRET_ENV: "signing_secret",
            linear_aig.HOST_ENV: "127.0.0.1",
            linear_aig.PORT_ENV: "9999",
            linear_aig.GRAPHQL_URL_ENV: "https://linear.test/graphql",
            linear_aig.TASK_MODE_ENV: "oneshot",
            linear_aig.MODEL_ENV: "model-a",
            linear_aig.PROVIDER_ENV: "provider-a",
            linear_aig.TOOLSETS_ENV: "linear,git",
        },
    )

    assert config.access_token == "lin_access_token"
    assert config.webhook_secret == "signing_secret"
    assert config.host == "127.0.0.1"
    assert config.port == 9999
    assert config.graphql_url == "https://linear.test/graphql"
    assert config.task_mode == "oneshot"
    assert config.model == "model-a"
    assert config.provider == "provider-a"
    assert config.toolsets == "linear,git"


def test_load_runtime_config_cli_args_override_environment():
    config = linear_aig.load_runtime_config(
        _args(
            access_token="arg_token",
            webhook_secret="arg_secret",
            host="localhost",
            port=8668,
            graphql_url="https://override.test/graphql",
            ack_only=True,
            task_mode="bridge",
            model="arg-model",
            provider="arg-provider",
            toolsets="arg-tools",
        ),
        {
            linear_aig.ACCESS_TOKEN_ENV: "env_token",
            linear_aig.WEBHOOK_SECRET_ENV: "env_secret",
            linear_aig.PORT_ENV: "9999",
            linear_aig.TASK_MODE_ENV: "oneshot",
        },
    )

    assert config.access_token == "arg_token"
    assert config.webhook_secret == "arg_secret"
    assert config.host == "localhost"
    assert config.port == 8668
    assert config.graphql_url == "https://override.test/graphql"
    assert config.ack_only is True
    assert config.task_mode == "bridge"
    assert config.model == "arg-model"
    assert config.provider == "arg-provider"
    assert config.toolsets == "arg-tools"


def test_load_runtime_config_requires_token_and_webhook_secret():
    with pytest.raises(ValueError, match=linear_aig.ACCESS_TOKEN_ENV):
        linear_aig.load_runtime_config(_args(), {})


def test_load_runtime_config_rejects_unknown_task_mode():
    with pytest.raises(ValueError, match="task mode"):
        linear_aig.load_runtime_config(
            _args(task_mode="launch"),
            {
                linear_aig.ACCESS_TOKEN_ENV: "lin_access_token",
                linear_aig.WEBHOOK_SECRET_ENV: "signing_secret",
            },
        )


def test_describe_runtime_config_redacts_sensitive_values():
    config = linear_aig.LinearAIGRuntimeConfig(
        access_token="lin_very_secret_token",
        webhook_secret="webhook_secret_value",
        host="127.0.0.1",
        port=8667,
    )

    description = linear_aig.describe_runtime_config(config)

    assert "lin_very_secret_token" not in description
    assert "webhook_secret_value" not in description
    assert "lin_...oken" in description
    assert "webh...alue" in description
    assert "task_mode: bridge" in description


def test_linear_aig_command_check_config_prints_redacted_config(monkeypatch, capsys):
    monkeypatch.setenv(linear_aig.ACCESS_TOKEN_ENV, "lin_very_secret_token")
    monkeypatch.setenv(linear_aig.WEBHOOK_SECRET_ENV, "webhook_secret_value")

    rc = linear_aig.linear_aig_command(_args(linear_aig_action="check-config"))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Linear AIG runtime config" in out
    assert "lin_very_secret_token" not in out
    assert "webhook_secret_value" not in out


def test_linear_aig_command_fails_when_config_missing(monkeypatch, capsys):
    monkeypatch.delenv(linear_aig.ACCESS_TOKEN_ENV, raising=False)
    monkeypatch.delenv(linear_aig.WEBHOOK_SECRET_ENV, raising=False)

    rc = linear_aig.linear_aig_command(_args(linear_aig_action="check-config"))

    out = capsys.readouterr().out
    assert rc == 1
    assert linear_aig.ACCESS_TOKEN_ENV in out
    assert linear_aig.WEBHOOK_SECRET_ENV in out


def test_run_server_builds_receiver_and_starts_aiohttp(monkeypatch):
    calls = []

    class FakeWeb:
        @staticmethod
        def run_app(app, *, host, port):
            calls.append({"app": app, "host": host, "port": port})

    app = object()

    monkeypatch.setattr(linear_aig, "AIOHTTP_AVAILABLE", True)
    monkeypatch.setattr(linear_aig, "web", FakeWeb)
    monkeypatch.setattr(linear_aig, "create_app", lambda receiver: app)

    linear_aig.run_server(
        linear_aig.LinearAIGRuntimeConfig(
            access_token="lin_token",
            webhook_secret="webhook_secret",
            host="127.0.0.1",
            port=8668,
            graphql_url="https://linear.test/graphql",
            ack_only=True,
        )
    )

    assert calls
    assert calls[0]["app"] is app
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8668


def test_build_oneshot_prompt_includes_linear_context():
    prompt = linear_aig.build_oneshot_prompt(linear_aig.parse_agent_session_event(_payload()))

    assert "Linear Agent Session" in prompt
    assert "Action: created" in prompt
    assert "Issue: WHO-192" in prompt
    assert "Issue title: Implement Hermes Linear AIG webhook receiver" in prompt
    assert "Implement WHO-192" in prompt


@pytest.mark.asyncio
async def test_oneshot_dispatcher_runs_hermes_agent(monkeypatch):
    calls = []

    def fake_run(prompt, *, model, provider, toolsets):
        calls.append(
            {
                "prompt": prompt,
                "model": model,
                "provider": provider,
                "toolsets": toolsets,
            }
        )
        return "Hermes result"

    monkeypatch.setattr(linear_aig, "run_hermes_oneshot", fake_run)
    dispatcher = linear_aig.build_task_dispatcher(
        linear_aig.LinearAIGRuntimeConfig(
            access_token="lin_token",
            webhook_secret="webhook_secret",
            task_mode="oneshot",
            model="model-a",
            provider="provider-a",
            toolsets="linear,git",
        )
    )

    result = await dispatcher(linear_aig.parse_agent_session_event(_payload()))

    assert result == "Hermes result"
    assert calls[0]["model"] == "model-a"
    assert calls[0]["provider"] == "provider-a"
    assert calls[0]["toolsets"] == "linear,git"
    assert "Implement WHO-192" in calls[0]["prompt"]


def test_build_task_dispatcher_respects_ack_only():
    dispatcher = linear_aig.build_task_dispatcher(
        linear_aig.LinearAIGRuntimeConfig(
            access_token="lin_token",
            webhook_secret="webhook_secret",
            ack_only=True,
            task_mode="oneshot",
        )
    )

    assert dispatcher is None


def test_verify_linear_signature_accepts_sha256_prefix():
    payload = _payload()
    raw_body = _body(payload)

    assert linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"linear-signature": f"sha256={_signature(raw_body)}"},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_bad_signature():
    payload = _payload()

    assert not linear_aig.verify_linear_signature(
        raw_body=_body(payload),
        headers={"Linear-Signature": "0" * 64},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_stale_timestamp():
    payload = _payload(webhookTimestamp=NOW_MS - 90_000)
    raw_body = _body(payload)

    assert not linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_missing_timestamp():
    payload = _payload()
    payload.pop("webhookTimestamp")
    raw_body = _body(payload)

    assert not linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_parse_agent_session_event_extracts_issue_and_prompt_context():
    event = linear_aig.parse_agent_session_event(_payload())

    assert event.action == "created"
    assert event.agent_session_id == "ags_123"
    assert event.prompt == "Implement WHO-192"
    assert event.issue_identifier == "WHO-192"
    assert event.issue_title == "Implement Hermes Linear AIG webhook receiver"


def test_parse_agent_session_event_accepts_nested_data_shape():
    event = linear_aig.parse_agent_session_event(
        {
            "action": "created",
            "data": {
                "promptContext": "Nested prompt",
                "agentSession": {"id": "ags_nested"},
            },
        }
    )

    assert event.agent_session_id == "ags_nested"
    assert event.prompt_context == "Nested prompt"


def test_parse_agent_session_event_requires_action_and_session_id():
    with pytest.raises(ValueError, match="missing action"):
        linear_aig.parse_agent_session_event({"agentSession": {"id": "ags_123"}})

    with pytest.raises(ValueError, match="missing agentSession.id"):
        linear_aig.parse_agent_session_event({"action": "created"})


def test_build_activity_content_supports_linear_content_types():
    assert linear_aig.build_activity_content("thought", "Starting") == {
        "type": "thought",
        "body": "Starting",
    }
    assert linear_aig.build_activity_content("action", "checkout", url=None) == {
        "type": "action",
        "action": "checkout",
    }

    with pytest.raises(ValueError, match="Unsupported"):
        linear_aig.build_activity_content("note", "Nope")


@pytest.mark.asyncio
async def test_process_event_emits_start_and_response_activity():
    sent = []

    async def sender(agent_session_id, content):
        sent.append((agent_session_id, content))

    async def dispatcher(event):
        assert event.issue_identifier == "WHO-192"
        return "Ready for review."

    receiver = linear_aig.LinearAIGReceiver(
        signing_secret=SECRET,
        activity_sender=sender,
        task_dispatcher=dispatcher,
    )

    await receiver.process_event(linear_aig.parse_agent_session_event(_payload()))

    assert sent == [
        (
            "ags_123",
            {
                "type": "thought",
                "body": "Hermes received Linear Agent Session `created` and is starting.",
            },
        ),
        ("ags_123", {"type": "response", "body": "Ready for review."}),
    ]


@pytest.mark.asyncio
async def test_process_event_emits_error_activity_on_dispatch_failure():
    sent = []

    async def sender(agent_session_id, content):
        sent.append((agent_session_id, content))

    async def dispatcher(_event):
        raise RuntimeError("boom")

    receiver = linear_aig.LinearAIGReceiver(
        signing_secret=SECRET,
        activity_sender=sender,
        task_dispatcher=dispatcher,
    )

    await receiver.process_event(linear_aig.parse_agent_session_event(_payload()))

    assert sent[-1] == ("ags_123", {"type": "error", "body": "Hermes failed: boom"})


@pytest.mark.asyncio
async def test_send_agent_activity_posts_graphql_payload(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"agentActivityCreate": {"success": True}}}

    def post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return Response()

    monkeypatch.setattr(linear_aig.httpx, "post", post)

    await linear_aig.send_agent_activity(
        api_key="lin_api_key",
        agent_session_id="ags_123",
        content={"type": "response", "body": "Done"},
        graphql_url="https://linear.test/graphql",
        timeout_seconds=3,
    )

    assert calls[0]["url"] == "https://linear.test/graphql"
    assert calls[0]["headers"] == {
        "Authorization": "Bearer lin_api_key",
        "Content-Type": "application/json",
    }
    assert "agentActivityCreate" in calls[0]["json"]["query"]
    assert calls[0]["json"]["variables"] == {
        "input": {
            "agentSessionId": "ags_123",
            "content": {"type": "response", "body": "Done"},
        },
    }
    assert calls[0]["timeout"] == 3


@pytest.mark.asyncio
async def test_send_agent_activity_raises_on_graphql_error(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"errors": [{"message": "bad input"}]}

    monkeypatch.setattr(linear_aig.httpx, "post", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="Linear GraphQL error"):
        await linear_aig.send_agent_activity(
            api_key="lin_api_key",
            agent_session_id="ags_123",
            content={"type": "response", "body": "Done"},
        )
