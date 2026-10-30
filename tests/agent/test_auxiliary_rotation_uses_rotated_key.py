"""After auxiliary credential-pool rotation, the retry must send the rotated entry's key.

Compression passes the main agent's api_key in ``main_runtime``. When that key 429s and the
pool rotates, the same-provider retry rebuilt the client from the same explicit key, so the
"rotated" retry re-sent the capped account's token. Its 429 (with that account's reset) was
then recorded against the entry the pool had rotated to, benching a healthy account.
"""

import json

import anthropic
import httpx
import pytest

import agent.auxiliary_client as ac
from hermes_constants import get_hermes_home

CAPPED = "sk-ant-oat01-" + "a" * 95
HEALTHY = "sk-ant-oat01-" + "b" * 95


def _anthropic_429():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, headers={"retry-after": "420000"}, request=request)
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}}
    return anthropic.RateLimitError("rate limited", response=response, body=body)


def _entry(id_, label, token, priority):
    return {"id": id_, "label": label, "auth_type": "oauth", "priority": priority, "source": "manual",
            "access_token": token, "base_url": "https://api.anthropic.com"}


@pytest.fixture
def pool_home(monkeypatch):
    home = get_hermes_home()
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {
        "anthropic": [_entry("aaa", "capped", CAPPED, 0), _entry("bbb", "healthy", HEALTHY, 1)]}}))
    # Never adopt (or refresh) a real Claude Code login from the developer's machine.
    (home / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ac, "_client_cache", {})
    return home


def test_compression_retry_after_rotation_sends_the_rotated_key(pool_home, monkeypatch):
    sent = []

    class _Messages:
        def __init__(self, key):
            self.key = key

        def create(self, **kwargs):
            sent.append(self.key)
            raise _anthropic_429()

    class _FakeClient:
        def __init__(self, key):
            self.api_key = key
            self.messages = _Messages(key)
            self.base_url = "https://api.anthropic.com"

        def close(self):
            pass

    import agent.anthropic_adapter as adapter
    monkeypatch.setattr(adapter, "build_anthropic_client", lambda api_key, *a, **k: _FakeClient(api_key))

    with pytest.raises(Exception):
        ac.call_llm(
            "compression",
            main_runtime={"model": "claude-sonnet-5-5", "provider": "anthropic",
                          "base_url": "https://api.anthropic.com", "api_key": CAPPED, "api_mode": "anthropic_messages"},
            messages=[{"role": "user", "content": "summarise"}],
        )

    assert sent and sent[0] == CAPPED
    assert HEALTHY in sent, f"no request ever used the rotated-to key; sent={['capped' if k == CAPPED else 'healthy' for k in sent]}"
    first_healthy = sent.index(HEALTHY)
    assert all(k == CAPPED for k in sent[:first_healthy])
