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


# ---------------------------------------------------------------------------
# Slack-CDN allowlist (follow-up hardening on top of #44026)
#
# ``url_private`` / ``url_private_download`` legitimately only ever point at
# the Slack CDN. Because the download attaches the bot token as a Bearer
# header, a forged file object (malicious workspace app / compromised event
# stream) pointing at ANY public host would exfiltrate the token — a hole the
# generic private-IP SSRF check cannot close. The adapter therefore refuses
# every non-Slack-CDN https URL up front.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
@pytest.mark.parametrize(
    "url",
    [
        # Public non-Slack host: generic SSRF check passes, allowlist must not.
        "https://attacker.example.com/steal-token",
        # Lookalike host — suffix match must anchor on a dot boundary.
        "https://files.slack.com.evil.example/x.jpg",
        "https://notslack-files.example.com/x.jpg",
        # Plain http is never a Slack CDN URL (token would go out in clear).
        "http://files.slack.com/x.jpg",
    ],
)
def test_non_slack_cdn_url_blocked_before_network(monkeypatch, method_name, url):
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter()
    method = getattr(self, method_name)
    args = (url, ".jpg") if method_name == "_download_slack_file" else (url,)

    with pytest.raises(ValueError, match="non-Slack-CDN"):
        asyncio.run(method(*args))


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
@pytest.mark.parametrize(
    "url",
    [
        # The canonical file CDN host.
        "https://files.slack.com/files-pri/T123-F456/image.png",
        # Enterprise Grid workspaces serve from per-org subdomains.
        "https://mycorp.enterprise.slack.com/files-pri/T123-F456/doc.pdf",
        # Legacy public-share host.
        "https://slack-files.com/T123-F456-abc",
        "https://files.slack-files.com/T123-F456-abc",
    ],
)
def test_slack_cdn_urls_still_allowed(monkeypatch, method_name, url):
    """Legitimate Slack CDN URLs must reach the network layer (no regression)."""
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter()
    method = getattr(self, method_name)
    args = (url, ".png") if method_name == "_download_slack_file" else (url,)

    # _NetworkTouched means the guard chain passed and the request was issued.
    with pytest.raises(_NetworkTouched):
        asyncio.run(method(*args))


# ---------------------------------------------------------------------------
# Connect-time DNS pinning (composition with #57860): a Slack-CDN hostname
# whose DNS answer flips from public at preflight to a metadata IP at connect
# time must be blocked before any TCP connect is attempted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
def test_download_blocks_connect_time_dns_rebind(monkeypatch, method_name):
    import socket

    import httpcore
    from httpcore._backends.auto import AutoBackend

    from tools.url_safety import SSRFConnectionBlocked

    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)

    # First resolution (is_safe_url preflight) sees a public IP; the
    # connect-time resolution sees the metadata IP — classic rebinding TOCTOU.
    answers = iter(("93.184.216.34", "169.254.169.254"))

    def fake_getaddrinfo(_host, port, *_args, **_kwargs):
        ip = next(answers)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    connect_attempts = []

    async def fake_connect_tcp(
        _self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        connect_attempts.append((host, port))
        raise httpcore.ConnectError("stop before network")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(AutoBackend, "connect_tcp", fake_connect_tcp)

    self = _fake_adapter()
    method = getattr(self, method_name)
    url = "https://files.slack.com/files-pri/T123-F456/image.png"
    args = (url, ".png") if method_name == "_download_slack_file" else (url,)

    with pytest.raises(SSRFConnectionBlocked):
        asyncio.run(method(*args))

    assert connect_attempts == []
