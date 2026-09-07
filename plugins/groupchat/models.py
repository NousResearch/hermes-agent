"""One filter-model chain shared by relevance scoring and pingpong decisions."""
from __future__ import annotations

import json
import os
import urllib.request

from hermes_constants import get_hermes_home


def credential(*names):
    from agent.secret_scope import current_secret_scope
    scope = current_secret_scope()
    if scope is not None:
        return next((scope.get(name, "") for name in names if scope.get(name)), "")
    for name in names:
        if os.getenv(name):
            return os.environ[name]
    if os.getenv("GROUPCHAT_CREDENTIALS_RESOLVED") == "1":
        return ""
    from dotenv import dotenv_values
    values = dotenv_values(get_hermes_home() / ".env")
    return next((values.get(name, "") for name in names if values.get(name)), "")


def profile_codex_model():
    import yaml
    path = get_hermes_home() / "config.yaml"
    if not path.exists():
        return ""
    model = (yaml.safe_load(path.read_text()) or {}).get("model", {})
    if isinstance(model, dict) and model.get("provider") == "openai-codex":
        return str(model.get("default") or "")
    return ""


def complete(choice, prompt, max_tokens):
    provider = choice["provider"]
    model = choice["model"]
    if provider == "openai-codex":
        from agent.auxiliary_client import _build_codex_client
        model = model or profile_codex_model()
        if not model:
            raise RuntimeError("No Codex model configured")
        client, resolved = _build_codex_client(model)
        if client is None or not resolved:
            raise RuntimeError("Codex credentials unavailable")
        result = client.chat.completions.create(
            model=resolved, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, reasoning_effort="low",
        )
        return result.choices[0].message.content or ""
    if provider == "mistral":
        endpoint = "https://api.mistral.ai/v1/chat/completions"
        key = credential("MISTRAL_API_KEY")
    elif provider == "openrouter":
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        key = credential("OPENROUTER_API_KEY", "OPENROUTER_KEY")
    else:
        raise ValueError("Unsupported filter provider")
    if not key:
        raise RuntimeError("Filter credentials unavailable")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read())
    return result.get("choices", [{}])[0].get("message", {}).get("content") or ""


def run_chain(config, prompt, max_tokens=120):
    choices = [{"provider": config["provider"], "model": config["model"]}, *config.get("fallbacks", [])]
    for index, choice in enumerate(choices):
        try:
            raw = complete(choice, prompt, max_tokens)
            if not raw.strip():
                raise ValueError("Empty model result")
            return raw, {"provider": choice["provider"], "model": choice["model"],
                         "fallback_used": index > 0, "fail_open": False}
        except Exception:
            # Never include provider exceptions or credentials in model prompts/logs.
            continue
    raise RuntimeError("All configured Groupchat filter models failed")
