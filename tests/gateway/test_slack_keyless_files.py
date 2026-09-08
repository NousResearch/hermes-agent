"""A keyless file edge is explicit, path-restricted and never followed on redirect."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

pytest.importorskip("slack_bolt")
from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def file_edge(monkeypatch):
    calls = []
    payload = b"%PDF-1.4\nprivate Slack file\n%%EOF"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization")))
            redirect = urlsplit(self.path).query.removeprefix("redirect=")
            if urlsplit(self.path).query.startswith("redirect="):
                self.send_response(302)
                self.send_header("Location", redirect)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    # The token-backed CDN path must still use its ordinary SSRF guard. Never
    # resolve or call Slack itself if the keyless routing is missing/broken.
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda *_: False)
    real_send = httpx.AsyncClient.send

    async def local_only(client, request, **kwargs):
        assert str(request.url).startswith(endpoint + "/"), "unexpected outbound destination"
        return await real_send(client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", local_only)
    try:
        yield endpoint, calls, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_private_file_flow_and_redirect_boundary(file_edge):
    endpoint, calls, payload = file_edge
    extra = {"keyless_api_base_url": endpoint + "/api/",
             "keyless_file_base_url": endpoint + "/private/"}
    adapter = SlackAdapter(PlatformConfig(token="", extra=extra))
    source = "https://files.slack.com/files-pri/T123-F456/report%20name.pdf?download=1"
    media, types, text = await adapter._collect_inbound_media(
        {"files": [{"name": "report.pdf", "size": len(payload), "mimetype": "application/pdf",
                    "url_private_download": source}]}, "C123", "T123", "document", [], [])
    assert len(media) == 1 and Path(media[0]).read_bytes() == payload
    assert types == ["application/pdf"] and text == "document"
    assert calls == [("/private/files-pri/T123-F456/report%20name.pdf?download=1", "Bearer xoxb-keyless")]

    # Every redirect is rejected: the edge returns bytes, not a redirect to a
    # CDN, another endpoint, or an API method that could receive its injected key.
    for location in (endpoint + "/private/files-pri/T123-F456/other.pdf",
                     endpoint + "/api/auth.test", "https://files.slack.com/files-pri/T123-F456/a.pdf",
                     "http://169.254.169.254/latest/meta-data/", "https://attacker.invalid/"):
        before = len(calls)
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._download_slack_file_bytes(source.split("?")[0] + "?redirect=" + location)
        assert len(calls) == before + 1, "redirect was followed"

    # Existing custom/token-backed download routing remains independent.
    standard = SlackAdapter(PlatformConfig(token="xoxb-standard", extra={"base_url": endpoint + "/api/"}))
    assert await standard._download_slack_file_bytes(endpoint + "/ordinary.pdf") == payload
    assert calls[-1] == ("/ordinary.pdf", "Bearer xoxb-standard")
    with pytest.raises(ValueError, match="SSRF"):
        await standard._download_slack_file_bytes(source)


@pytest.mark.asyncio
@pytest.mark.parametrize("source,file_base", [
    ("https://files.slack.com/files-pri/T123-F456/a.pdf", None),
    ("https://files.slack.com/files-pri/T123-F456/a.pdf", "https://user:pass@edge.invalid/private"),
    ("https://files.slack.com/files-pri/T123-F456/a.pdf", "https://edge.invalid/private/../api"),
    ("https://files.slack.com/files-pri/T123-F456/a.pdf", "https://edge.invalid/private/%2e%2e/api"),
    ("https://files.slack.com/files-pri/T123-F456/a.pdf", "https://edge.invalid/private?backend=other"),
    ("https://attacker.invalid/files-pri/T123-F456/a.pdf", "/private"),
    ("https://files.slack.com.attacker.invalid/files-pri/T123-F456/a.pdf", "/private"),
    ("https://other.slack.com/files-pri/T123-F456/a.pdf", "/private"),
    ("https://user@files.slack.com/files-pri/T123-F456/a.pdf", "/private"),
    ("https://files.slack.com:444/files-pri/T123-F456/a.pdf", "/private"),
    ("http://files.slack.com/files-pri/T123-F456/a.pdf", "/private"),
    ("https://files.slack.com/api/auth.test", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/../../api/auth.test", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/%2e%2e/auth.test", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/%252e%252e/auth.test", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/%2fapi/auth.test", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/%5capi", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/a\n.pdf", "/private"),
    ("https://files.slack.com/files-pri/T123-F456/a.pdf#fragment", "/private"),
])
async def test_keyless_unconfigured_or_untrusted_file_never_uses_network(file_edge, source, file_base):
    endpoint, calls, _ = file_edge
    extra = {"keyless_api_base_url": endpoint + "/api/"}
    if file_base is not None:
        extra["keyless_file_base_url"] = endpoint + file_base if file_base.startswith("/") else file_base
    adapter = SlackAdapter(PlatformConfig(token="xoxb-must-not-leak", extra=extra))
    with pytest.raises(ValueError, match="keyless.*file"):
        await adapter._download_slack_file_bytes(source)
    assert calls == []
    if file_base is None:
        media, types, text = await adapter._collect_inbound_media(
            {"files": [{"name": "report.pdf", "size": 30, "mimetype": "application/pdf",
                        "url_private_download": source}]}, "C123", "T123", "document", [], [])
        assert not media and not types and "keyless_file_base_url" in text
        assert "Slack attachment notice" in text and "https://" not in text
        assert calls == []
