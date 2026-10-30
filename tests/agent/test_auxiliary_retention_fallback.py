"""Regression for #62965: model retention rejection uses the configured aux route."""
import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

from agent import auxiliary_client as ac


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("task", ["vision", "compression"])
@pytest.mark.parametrize("scenario", ["retention", "normal", "validation", "no-fallback", "empty", "fallback-error"])
def test_model_retention_rejection_reaches_configured_fallback(monkeypatch, async_mode, task, scenario):
    import openai

    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"] == "rejected-model" and scenario != "normal":
            message = ("invalid data retention mode parameter" if scenario == "validation" else
                       "data retention mode 'default' is not available for this model")
            return httpx.Response(400, json={"error": {"message": message,
                "type": "invalid_request_error"}})
        if scenario == "fallback-error":
            return httpx.Response(400, json={"error": {"message": "invalid request fixture"}})
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion",
            "created": 0, "model": body["model"], "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "" if scenario == "empty" else "fallback answer"}, "finish_reason": "stop"}]})

    transport = httpx.MockTransport(respond)
    sync_init, async_init = openai.OpenAI.__init__, openai.AsyncOpenAI.__init__
    monkeypatch.setattr(openai.OpenAI, "__init__", lambda self, **kw: sync_init(
        self, **{**kw, "http_client": httpx.Client(transport=transport)}))
    monkeypatch.setattr(openai.AsyncOpenAI, "__init__", lambda self, **kw: async_init(
        self, **{**kw, "http_client": httpx.AsyncClient(transport=transport)}))
    config = {"model": {"provider": "custom", "model": "rejected-model",
                       "base_url": "http://fixture.invalid/v1"},
              "auxiliary": {task: {"provider": "custom", "model": "rejected-model",
                "base_url": "http://fixture.invalid/v1", "api_key": "fixture-only",
                "fallback_chain": [{"provider": "custom", "model": "working-model",
                    "base_url": "http://fixture.invalid/v1", "api_key": "fixture-only"}]}}}
    if scenario == "no-fallback":
        config["auxiliary"][task]["fallback_chain"] = []
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(json.dumps(config), encoding="utf8")
    ac._reset_aux_unhealthy_cache()
    kwargs = {"task": task, "messages": [{"role": "user", "content": "describe"}], "max_tokens": 32}
    try:
        result = asyncio.run(ac.async_call_llm(**kwargs)) if async_mode else ac.call_llm(**kwargs)
        outcome = result.choices[0].message.content
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    if scenario in {"retention", "normal"}:
        assert outcome == "fallback answer", outcome
    elif scenario == "empty":
        # Existing auxiliary validation accepts a well-shaped empty completion;
        # callers own the task-specific empty-summary/vision policy.
        assert outcome == ""
    else:
        assert outcome != "fallback answer", outcome
        assert {"validation": "invalid data retention mode parameter",
                "no-fallback": "is not available for this model", "empty": "empty",
                "fallback-error": "invalid request fixture"}[scenario] in outcome.lower()
    expected = ["rejected-model"]
    if scenario in {"retention", "empty", "fallback-error"}:
        expected.append("working-model")
    assert calls == expected
    assert not ac._is_provider_unhealthy("custom", "http://fixture.invalid/v1")


def test_retention_rejection_is_model_scoped_not_generic_validation():
    from botocore.exceptions import ClientError

    error = ClientError({"Error": {"Code": "ValidationException", "Message":
        "data retention mode 'default' is not available for this model"},
        "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse")
    assert ac._is_model_incompatible_error(error)
    for message, status in [
        ("invalid data retention mode parameter", 400),
        ("data retention mode 'default' is not available for this model", 401),
        ("data retention mode 'default' is not available for this model", 500),
        ("billing: data retention mode 'default' is not available for this model", 400),
    ]:
        exc = RuntimeError(message)
        exc.status_code = status
        assert not ac._is_model_incompatible_error(exc)
