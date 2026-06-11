"""SSRF regression tests for inbound Slack file downloads.

``_download_slack_file`` / ``_download_slack_file_bytes`` attach the bot
token and follow redirects, so they must validate the destination (CWE-918)
exactly like the already-guarded outbound ``send_image`` path: a pre-flight
``is_safe_url`` check plus a per-redirect guard.
"""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import _ssrf_redirect_guard


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" not in sys.modules:
        for name in (
            "slack_bolt",
            "slack_bolt.adapter",
            "slack_bolt.adapter.socket_mode",
            "slack_bolt.adapter.socket_mode.async_handler",
            "slack_bolt.async_app",
            "slack_sdk",
            "slack_sdk.web",
            "slack_sdk.web.async_client",
            "slack_sdk.errors",
        ):
            sys.modules.setdefault(name, MagicMock())
    if "aiohttp" not in sys.modules:
        sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _fake_adapter():
    self = SlackAdapter.__new__(SlackAdapter)
    self.config = SimpleNamespace(token="xoxb-test-token")
    self._team_clients = {}
    return self


class _NetworkTouched(RuntimeError):
    pass


class _RecordingClient:
    """Captures AsyncClient kwargs; refuses to perform real I/O."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        raise _NetworkTouched("network access attempted")


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
def test_unsafe_url_blocked_before_network(monkeypatch, method_name):
    import tools.url_safety as url_safety

    calls = {"checked": []}

    def fake_is_safe_url(url, *a, **k):
        calls["checked"].append(url)
        return False

    monkeypatch.setattr(url_safety, "is_safe_url", fake_is_safe_url)

    # If the guard is bypassed, the fake client raises _NetworkTouched; a
    # correct implementation raises ValueError *before* touching httpx.
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter()
    method = getattr(self, method_name)
    args = ("http://169.254.169.254/latest/meta-data/", ".jpg") \
        if method_name == "_download_slack_file" \
        else ("http://169.254.169.254/latest/meta-data/",)

    with pytest.raises(ValueError):
        asyncio.run(method(*args))

    assert calls["checked"], "download must call is_safe_url before fetching"


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
def test_redirect_guard_is_wired(monkeypatch, method_name):
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter()
    method = getattr(self, method_name)
    args = ("https://files.slack.com/x.jpg", ".jpg") \
        if method_name == "_download_slack_file" \
        else ("https://files.slack.com/x.jpg",)

    # The fake client raises when .get() is called; we only care that the
    # client was constructed with the redirect guard hook.
    with pytest.raises(_NetworkTouched):
        asyncio.run(method(*args))

    kwargs = _RecordingClient.last_kwargs
    assert kwargs is not None
    hooks = kwargs.get("event_hooks", {})
    assert _ssrf_redirect_guard in hooks.get("response", []), (
        "AsyncClient must register _ssrf_redirect_guard to block "
        "redirect-based SSRF"
    )
