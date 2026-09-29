"""Async-transport regression for the commandcode-oauth auxiliary client.

Regression from the executed review of head f663e29: ``_to_async_client`` rebuilt
an ``AsyncOpenAI`` against ``/provider/v1`` for ``CommandCodeOAuthAuxiliaryClient``,
so every async aux task (vision_analyze et al.) posted the CLI-tier OAuth bearer
to an endpoint that cannot accept it. The async counterpart must keep the
/alpha/generate transport.
"""

import asyncio
import io
import json
import urllib.request

import pytest

from agent.auxiliary_client import (
    AsyncCommandCodeOAuthAuxiliaryClient,
    CommandCodeOAuthAuxiliaryClient,
    _to_async_client,
)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_to_async_client_preserves_alpha_transport(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        lines = [
            {"type": "text-delta", "text": "async ok"},
            {"type": "finish", "finishReason": "stop", "totalUsage": {}},
        ]
        return _FakeResponse(b"".join(json.dumps(e).encode() + b"\n" for e in lines))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    sync = CommandCodeOAuthAuxiliaryClient("tok-123", "https://api.commandcode.ai", "meta/muse-spark-1.3-contributor")
    async_client, model = _to_async_client(sync, "meta/muse-spark-1.3-contributor")

    assert isinstance(async_client, AsyncCommandCodeOAuthAuxiliaryClient)
    assert async_client.base_url == "https://api.commandcode.ai"

    resp = asyncio.run(async_client.chat.completions.create(
        model="meta/muse-spark-1.3-contributor",
        messages=[{"role": "user", "content": "ping"}],
    ))
    assert resp.choices[0].message.content == "async ok"
    assert captured["url"].endswith("/alpha/generate"), captured["url"]
    assert captured["auth"] == "Bearer tok-123"
