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


def _fake_adapter(base_url=None):
    self = SlackAdapter.__new__(SlackAdapter)
    extra = {"base_url": base_url} if base_url else {}
    self.config = SimpleNamespace(token="xoxb-test-token", extra=extra)
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


# ---------------------------------------------------------------------------
# Connect-time DNS pinning (composition with #57860): a Slack-CDN hostname
# whose DNS answer flips from public at preflight to a metadata IP at connect
# time must be blocked before any TCP connect is attempted.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Custom ``slack.base_url`` endpoint
#
# Both guards above were written assuming Slack is always the real slack.com.
# With a custom ``slack.base_url`` (self-hosted relay / Enterprise proxy /
# mock Slack) the file links in the event payload point at THAT endpoint, so
# they must be downloadable even when they resolve to a private address. The
# trusted set mirrors the Slack-CDN allowlist: the configured host plus hosts
# below it (Slack itself splits ``slack.com`` from ``files.slack.com``).
# Anything outside keeps the upstream behaviour.
# ---------------------------------------------------------------------------

_RELAY_BASE_URL = "http://127.0.0.1:49917/api/"
_RELAY_FILE_URL = "http://127.0.0.1:49917/files/T1-F1/chart.png"


class _OkClient:
    """Minimal httpx.AsyncClient stand-in that serves one successful response."""

    last_kwargs = None
    requested = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        type(self).requested = (url, kwargs)
        return SimpleNamespace(
            content=b"png-bytes",
            headers={"content-type": "image/png"},
            raise_for_status=lambda: None,
        )


def _redirect_response(location):
    return SimpleNamespace(
        is_redirect=True,
        headers={"location": location},
        url=_RELAY_FILE_URL,
        next_request=None,
        status_code=302,
    )


def test_url_origin_contract():
    from plugins.platforms.slack import adapter as slack_adapter

    origin = slack_adapter._slack_url_origin
    # Scheme-default ports make the two spellings one origin.
    assert origin("https://host/api/") == origin("https://host:443/api/")
    assert origin("http://host/api/") == origin("http://HOST.:80/other")
    # Unusable inputs never produce an origin (so they can never match).
    for bad in (None, "", "   ", "ftp://host/x", "https://", "not a url"):
        assert origin(bad) is None


def test_base_url_trust_covers_the_endpoint_host_and_hosts_below_it():
    from plugins.platforms.slack import adapter as slack_adapter

    trusted = slack_adapter._is_slack_base_url_trusted
    api = "https://slack.internal.corp/api/"
    assert trusted(_RELAY_FILE_URL, _RELAY_BASE_URL)
    assert trusted("https://files.slack.internal.corp/files-pri/T1-F1/x", api)
    assert trusted("https://a.b.slack.internal.corp/x", api)
    # Below the configured host only: its parent, a sibling of that parent and
    # a host merely ending in the same letters are all outside it.
    assert not trusted("https://internal.corp/x", api)
    assert not trusted("https://files.internal.corp/x", api)
    assert not trusted("https://evilslack.internal.corp/x", api)
    # Scheme and port still have to match exactly.
    assert not trusted("http://127.0.0.1:8080/files/x", _RELAY_BASE_URL)
    assert not trusted("http://127.0.0.2:49917/files/x", _RELAY_BASE_URL)
    assert not trusted("https://127.0.0.1:49917/files/x", _RELAY_BASE_URL)
    # With no base_url configured nothing is trusted this way.
    assert not trusted(_RELAY_FILE_URL, None)
    assert not trusted("https://files.slack.com/x.png", None)


@pytest.mark.parametrize(
    "method_name",
    ["_download_slack_file", "_download_slack_file_bytes"],
)
def test_configured_base_url_origin_is_downloadable(monkeypatch, method_name, tmp_path):
    """A file URL on the configured base_url origin downloads even though it
    resolves to a private address (that is the point of a local relay)."""
    import tools.url_safety as url_safety

    checked = []

    def fake_is_safe_url(url, *a, **k):
        checked.append(url)
        return False  # a relay on 127.0.0.1 never passes the private-IP check

    monkeypatch.setattr(url_safety, "is_safe_url", fake_is_safe_url)
    monkeypatch.setattr("httpx.AsyncClient", _OkClient)
    monkeypatch.setattr(
        "gateway.platforms.base.cache_image_from_bytes",
        lambda content, ext: str(tmp_path / f"cached{ext}"),
    )

    self = _fake_adapter(base_url=_RELAY_BASE_URL)
    args = (_RELAY_FILE_URL, ".png") if method_name == "_download_slack_file" else (_RELAY_FILE_URL,)

    result = asyncio.run(getattr(self, method_name)(*args))

    if method_name == "_download_slack_file_bytes":
        assert result == b"png-bytes"
    else:
        assert result == str(tmp_path / "cached.png")
    assert _OkClient.requested[0] == _RELAY_FILE_URL
    assert checked == [], "origin-trusted URLs must not go through is_safe_url"

    # Skipping the private-IP checks is only safe because the client itself is
    # origin-pinned, so assert the guard is actually wired — and drive it, so
    # a hook that no longer refuses off-origin hops fails here too.
    hooks = (_OkClient.last_kwargs or {}).get("event_hooks", {}).get("response") or []
    assert hooks, "origin-trusted client must register a redirect guard"
    with pytest.raises(ValueError):
        asyncio.run(hooks[0](_redirect_response("https://evil.example.com/x.png")))


@pytest.mark.parametrize(
    "other_url",
    [
        "http://127.0.0.1:8080/files/x.png",  # same host, different port
        "http://169.254.169.254/latest/meta-data/",  # metadata endpoint
        "https://evil.example.com/x.png",  # public host off the CDN
    ],
)
def test_configured_base_url_does_not_widen_trust(monkeypatch, other_url):
    """Configuring base_url trusts that endpoint — nothing else."""
    import tools.url_safety as url_safety

    # Pass the private-IP pre-flight (no live DNS in tests) so the refusal has
    # to come from the origin/CDN trust decision itself.
    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: True)
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter(base_url=_RELAY_BASE_URL)
    with pytest.raises(ValueError) as exc:
        asyncio.run(self._download_slack_file_bytes(other_url))
    # The refusal must name the endpoint that IS trusted — otherwise a
    # misconfigured relay is indistinguishable from a genuine SSRF block.
    assert "http://127.0.0.1:49917" in str(exc.value)


def test_trusted_origin_label_omits_default_ports():
    from plugins.platforms.slack import adapter as slack_adapter

    label = slack_adapter._slack_trusted_origin_label
    assert label("https://slack.internal.corp/api/") == "https://slack.internal.corp"
    assert label("https://slack.internal.corp:443/api/") == "https://slack.internal.corp"
    assert label("http://127.0.0.1:49917/api/") == "http://127.0.0.1:49917"
    # No usable custom endpoint → nothing to advertise as trusted.
    for blank in (None, "", "ftp://host/x"):
        assert label(blank) is None
        assert slack_adapter._slack_trust_hint(blank) == ""


@pytest.mark.parametrize(
    "file_url",
    [
        "https://slack.internal.corp/files-pri/T1-F1/doc.pdf",
        "https://files.slack.internal.corp/files-pri/T1-F1/doc.pdf",
    ],
)
def test_file_host_below_the_endpoint_is_downloadable(monkeypatch, file_url):
    """A deployment that mirrors Slack's own API/file host split hands out
    file links on a host below base_url, and those must download."""
    import tools.url_safety as url_safety

    # False everywhere: reaching the download proves the endpoint trust path
    # was taken rather than the private-IP one.
    monkeypatch.setattr(url_safety, "is_safe_url", lambda *a, **k: False)
    monkeypatch.setattr("httpx.AsyncClient", _OkClient)

    self = _fake_adapter(base_url="https://slack.internal.corp/api/")

    assert asyncio.run(self._download_slack_file_bytes(file_url)) == b"png-bytes"
    assert _OkClient.requested[0] == file_url


@pytest.mark.parametrize(
    "slack_base_url",
    [
        "https://slack.com/api/",  # the documented default, spelled out
        "https://acme.enterprise.slack.com/api/",  # Enterprise Grid
    ],
)
def test_slack_owned_base_url_keeps_cdn_downloads_pinned(monkeypatch, slack_base_url):
    """A base_url on Slack itself must not make CDN links "configured".

    Host-and-below trust would otherwise cover the whole CDN, costing every
    ordinary download the SSRF pre-flight and the DNS-pinned client.
    """
    import tools.url_safety as url_safety

    checked = []

    def fake_is_safe_url(url, *a, **k):
        checked.append(url)
        return True

    pinned = []

    def fake_pinned_client(**kwargs):
        pinned.append(kwargs)
        return _OkClient(**kwargs)

    monkeypatch.setattr(url_safety, "is_safe_url", fake_is_safe_url)
    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", fake_pinned_client)
    # Taking the endpoint-trust path would build a plain client instead, and
    # this one refuses to perform I/O.
    monkeypatch.setattr("httpx.AsyncClient", _RecordingClient)

    self = _fake_adapter(base_url=slack_base_url)
    cdn_url = "https://files.slack.com/files-pri/T1-F1/x.png"

    assert asyncio.run(self._download_slack_file_bytes(cdn_url)) == b"png-bytes"
    assert checked == [cdn_url], "CDN downloads must keep the SSRF pre-flight"
    assert pinned, "CDN downloads must keep the DNS-pinned client"


def test_base_url_redirect_guard_allows_hops_within_the_endpoint():
    """A relay may 3xx within itself (auth handoff, path rewrite)."""
    from plugins.platforms.slack import adapter as slack_adapter

    guard = slack_adapter._slack_base_url_redirect_guard(_RELAY_BASE_URL)
    asyncio.run(guard(_redirect_response("http://127.0.0.1:49917/files/next.png")))
    # A non-redirect response carries no hop to vet.
    asyncio.run(guard(SimpleNamespace(is_redirect=False, headers={}, url=_RELAY_FILE_URL)))
    # Same trusted set as the up-front check: the API host may hand off to the
    # file host below it.
    named = slack_adapter._slack_base_url_redirect_guard(
        "https://slack.internal.corp/api/"
    )
    asyncio.run(
        named(_redirect_response("https://files.slack.internal.corp/files-pri/T1-F1/x"))
    )


@pytest.mark.parametrize(
    "location",
    [
        "https://files.slack.com/x.png",  # public, and even a real CDN host
        "http://127.0.0.1:8080/x.png",  # same host, different port
        "http://169.254.169.254/latest/meta-data/",
    ],
)
def test_base_url_redirect_guard_blocks_off_endpoint_hops(location):
    """The endpoint-trusted client has no connect-time DNS pinning, so a hop
    that leaves it is refused outright rather than hostname-checked."""
    from plugins.platforms.slack import adapter as slack_adapter

    guard = slack_adapter._slack_base_url_redirect_guard(_RELAY_BASE_URL)
    with pytest.raises(ValueError) as exc:
        asyncio.run(guard(_redirect_response(location)))
    # Same diagnosability requirement as the up-front refusals.
    assert "http://127.0.0.1:49917" in str(exc.value)


