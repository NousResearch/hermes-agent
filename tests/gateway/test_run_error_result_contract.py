"""Gateway error returns must be marked failed, not delivered as answers (#57899).

Consumers gate on ``result.get("failed")`` — see ``cli.py`` and
``run_agent.py`` — so an error dict without the flag is treated as a
successful turn whose answer happens to be an error message. That makes
provider/proxy failures non-retryable and persists them as real replies.
"""

from __future__ import annotations


from types import SimpleNamespace

import pytest

from tests.gateway.test_run_cleanup_progress import CleanupCaptureAdapter, _make_runner
from gateway.config import Platform
from gateway.session import SessionSource


async def _call_proxy(runner):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001")
    return await runner._run_agent_via_proxy(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:telegram:group:-1001",
    )


@pytest.mark.asyncio
async def test_missing_proxy_url_is_a_failed_result(monkeypatch):
    runner = _make_runner(CleanupCaptureAdapter())
    monkeypatch.setattr(runner, "_get_proxy_url", lambda: "", raising=False)

    result = await _call_proxy(runner)

    assert "Proxy URL not configured" in result["final_response"]
    assert result.get("failed") is True, (
        "error return lacks failed=True; consumers would treat it as a real answer"
    )
    assert result.get("completed") is False
    assert result.get("agent_persisted") is False


@pytest.mark.asyncio
async def test_missing_aiohttp_is_a_failed_result(monkeypatch):
    import builtins

    runner = _make_runner(CleanupCaptureAdapter())
    real_import = builtins.__import__

    def _no_aiohttp(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("no aiohttp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_aiohttp)

    result = await _call_proxy(runner)

    assert "aiohttp" in result["final_response"]
    assert result.get("failed") is True
    assert result.get("completed") is False
