"""Named custom credentials stay at their configured origin on both surfaces."""

import json
import shlex
import sys

import httpx
import pytest
import yaml


@pytest.mark.parametrize("surface", ["runtime", "auxiliary"])
@pytest.mark.parametrize("provider", ["review-relay", "custom:review-relay", "Review Relay"])
@pytest.mark.parametrize("credential", ["inline", "env", "command", "pool"])
@pytest.mark.parametrize("target,foreign", [
    ("https://configured.invalid/v1", False),
    ("https://configured.invalid:443/v2", False),
    ("https://foreign.invalid/v1", True),
    ("http://configured.invalid/v1", True),
    ("https://configured.invalid:8443/v1", True),
])
def test_saved_credentials_are_origin_bound(
    tmp_path, monkeypatch, surface, provider, credential, target, foreign,
):
    sentinel = "SENTINEL_SAVED_" + credential.upper()
    entry = {"name": "Review Relay", "base_url": "https://configured.invalid/v1"}
    marker = tmp_path / "command-ran"
    if credential == "inline":
        entry["api_key"] = sentinel
    elif credential == "env":
        entry["key_env"] = "ORIGIN_TEST_KEY"
        monkeypatch.setenv("ORIGIN_TEST_KEY", sentinel)
    elif credential == "command":
        script = tmp_path / "token.py"
        script.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint({sentinel!r})\n",
            encoding="utf-8",
        )
        entry["key_cmd"] = shlex.join([sys.executable, str(script)])
    else:
        (tmp_path / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {}, "credential_pool": {
                "review-relay": [{
                    "id": "sentinel", "label": "sentinel", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": sentinel,
                }],
            },
        }), encoding="utf-8")
    configure(tmp_path, monkeypatch, entry)
    auth = request(surface, provider, target, monkeypatch)
    assert auth == "Bearer " + ("no-key-required" if foreign else sentinel)
    if credential == "command":
        assert marker.exists() is not foreign


@pytest.mark.parametrize("surface", ["runtime", "auxiliary"])
@pytest.mark.parametrize("provider", ["review-relay", "custom:review-relay", "Review Relay"])
@pytest.mark.parametrize("explicit_key", [None, "SENTINEL_EXPLICIT"])
@pytest.mark.parametrize("vendor,target", [
    ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
])
def test_foreign_host_requires_explicit_credentials(
    tmp_path, monkeypatch, surface, provider, explicit_key, vendor, target,
):
    monkeypatch.setenv(vendor, "SENTINEL_AMBIENT")
    configure(tmp_path, monkeypatch, {
        "name": "Review Relay", "base_url": "https://configured.invalid/v1",
        "api_key": "SENTINEL_SAVED",
    })
    auth = request(surface, provider, target, monkeypatch, explicit_key)
    assert auth == "Bearer " + (explicit_key or "no-key-required")


def configure(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "review-model"}, "providers": {"review-relay": entry},
    }), encoding="utf-8")


def request(surface, provider, target, monkeypatch, explicit_key=None):
    recorded = []

    def send(self, req, **kwargs):
        recorded.append((str(req.url), req.headers.get("authorization", "")))
        return httpx.Response(200, request=req, json={
            "id": "sentinel", "object": "chat.completion", "created": 0,
            "model": "review-model", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        })

    # Intercept the final constructed request: no external HTTP or real keys.
    monkeypatch.setattr(httpx.Client, "send", send)
    if surface == "runtime":
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from openai import OpenAI
        runtime = resolve_runtime_provider(
            requested=provider, explicit_base_url=target,
            explicit_api_key=explicit_key, target_model="review-model",
        )
        client = OpenAI(base_url=runtime["base_url"], api_key=runtime["api_key"])
    else:
        from agent.auxiliary_client import resolve_provider_client
        client, _ = resolve_provider_client(
            provider, "review-model", explicit_base_url=target,
            explicit_api_key=explicit_key,
        )
    assert client is not None
    try:
        client.chat.completions.create(
            model="review-model", messages=[{"role": "user", "content": "sentinel probe"}],
        )
    finally:
        client.close()
    assert len(recorded) == 1
    url, auth = recorded[0]
    assert httpx.URL(url) == httpx.URL(target.rstrip("/") + "/chat/completions")
    return auth
