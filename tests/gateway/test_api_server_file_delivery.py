"""Signed download links for ``MEDIA:<path>`` file delivery over the API server.

Exercises :mod:`gateway.platforms.api_server_file_delivery` end to end: the resolution pass that
turns a non-image tag into a link, the streaming rewriter that keeps a tag from being split across
deltas, and the signed one-shot download route. The load-bearing contracts are

* the trust model — the URL is the capability (HMAC over artifact id + expiry), and every failure is
  a 404 so the route is not an existence oracle for a frontend that holds no API key;
* isolation — this transport and browser control gate each other's routes in neither direction, and
  two profiles never share a store root;
* degradation — anything not deliverable leaves the tag exactly as it was, so the default-off path
  is byte-identical to the behavior that shipped before this transport existed.
"""

import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import gateway.platforms.api_server_file_delivery as delivery  # noqa: E402
from gateway.browser_control_artifacts import ArtifactScopeMismatch, ArtifactStore  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.api_server import APIServerAdapter  # noqa: E402
from gateway.platforms.api_server_file_delivery import (  # noqa: E402
    DEFAULT_FILE_DELIVERY_MIME_TYPES,
    FileDeliveryConfig,
    MediaTagStreamRewriter,
    delivery_root,
    file_delivery_config,
    resolve_media_to_download_urls,
    sign_download_token,
    store_and_link,
    verify_download_token,
)

PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
ZIP_BYTES = b"PK\x03\x04" + b"\x00" * 32
PNG_BYTES = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489")
API_KEY = "-".join(("fixture", "neutral", "api", "key", "123"))
BASE_URL = "https://chat.example"

_LINK_RE = re.compile(r"\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)")
_URL_RE = re.compile(r"/v1/artifacts/download/[0-9a-f]{32}\?expires=\d+&signature=[0-9a-f]{32}")

#: One honest sample per advertised document type (the OOXML formats are ZIP containers).
_SAMPLES = {
    ".txt": b"a plain note\n",
    ".md": b"# title\n\nbody\n",
    ".csv": b"a,b\n1,2\n",
    ".tsv": b"a\tb\n1\t2\n",
    ".json": b'{"a": 1}\n',
}
_SAMPLES[".pdf"] = PDF_BYTES
for _container in (".zip", ".xlsx", ".docx"):
    _SAMPLES[_container] = ZIP_BYTES


class _Scope:
    """Minimal attribute scope accepted by ``artifact_scope_key``."""

    def __init__(self, principal="delivery:default:fixture", family="file-delivery"):
        self.principal_id = principal
        self.session_id = ""
        self.transport_family = family


@pytest.fixture
def store(tmp_path, config):
    # Mirrors what the adapter builds: the delivery store carries the configured (extended) allowlist,
    # not the browser-control default, which deliberately cannot carry documents.
    return ArtifactStore(tmp_path / "store", allowed_mime_types=config.allowed_mime_types)


@pytest.fixture
def secret():
    return b"\x11" * 32


@pytest.fixture
def config():
    return FileDeliveryConfig(enabled=True, public_base_url=BASE_URL)


def _write(tmp_path, name, payload=PDF_BYTES):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def _link_from(text):
    match = _LINK_RE.search(text)
    assert match, f"no markdown link in {text!r}"
    return match


def _url_parts(url):
    path, _, query = url.partition("?")
    params = dict(pair.split("=", 1) for pair in query.split("&") if pair)
    return path.rsplit("/", 1)[-1], params


def _delta_contents(sse_body):
    """Assemble the ``delta.content`` a streaming client would render from an SSE response body."""
    contents = []
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        for choice in (json.loads(line[len("data: "):]).get("choices") or []):
            content = (choice.get("delta") or {}).get("content")
            if content:
                contents.append(content)
    return contents


class TestNonImageResolution:
    def test_non_image_tag_becomes_a_signed_link(self, tmp_path, store, secret, config):
        path = _write(tmp_path, "report.pdf")
        out = resolve_media_to_download_urls(
            f"Here it is: MEDIA:{path}", store=store, scope=_Scope(), config=config, secret=secret)

        match = _link_from(out)
        assert "MEDIA:" not in out
        # The host path never crosses the HTTP boundary, in any configuration.
        assert str(path) not in out and str(tmp_path) not in out
        assert match.group("name").startswith("report.pdf (")
        artifact_id, params = _url_parts(match.group("url"))
        assert match.group("url").startswith(f"{BASE_URL}/v1/artifacts/download/")
        assert verify_download_token(secret, artifact_id, params["expires"], params["signature"])
        assert store.count() == 1

    def test_flag_off_leaves_the_tag_literal(self, tmp_path, store, secret):
        path = _write(tmp_path, "report.pdf")
        text = f"Here it is: MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=FileDeliveryConfig(), secret=secret) == text
        assert store.count() == 0

    def test_missing_secret_leaves_the_tag_literal(self, tmp_path, store, config):
        path = _write(tmp_path, "report.pdf")
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=None) == text
        assert store.count() == 0

    def test_relative_link_when_no_base_url_is_configured(self, tmp_path, store, secret):
        """An unconfigured base URL is not a reason to drop the file."""
        path = _write(tmp_path, "report.pdf")
        out = resolve_media_to_download_urls(
            f"MEDIA:{path}", store=store, scope=_Scope(),
            config=FileDeliveryConfig(enabled=True), secret=secret)
        assert _link_from(out).group("url").startswith("/v1/artifacts/download/")

    def test_repeated_path_mints_one_artifact(self, tmp_path, store, secret, config):
        path = _write(tmp_path, "report.pdf")
        out = resolve_media_to_download_urls(
            f"first MEDIA:{path} and again MEDIA:{path}",
            store=store, scope=_Scope(), config=config, secret=secret)
        urls = {match.group("url") for match in _LINK_RE.finditer(out)}
        assert len(urls) == 1
        assert store.count() == 1

    def test_per_response_ceiling_bounds_minted_artifacts(self, tmp_path, store, secret, config):
        """A reply is model-authored text: how many files one response can mint must be bounded."""
        ceiling = delivery.MAX_DELIVERIES_PER_RESPONSE
        text = " ".join(f"MEDIA:{_write(tmp_path, f'file{index}.pdf')}"
                        for index in range(ceiling + 3))
        out = resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret)
        assert len(_LINK_RE.findall(out)) == ceiling
        assert store.count() == ceiling
        # Over the ceiling the tag stays literal — never a broken link.
        assert out.count("MEDIA:") == 3

    def test_denylisted_path_is_neither_stored_nor_linked(self, tmp_path, store, secret, config):
        """``validate_media_delivery_path`` stays the choke point: a credential dir must not deliver."""
        leaked = Path(os.path.expanduser("~")) / ".ssh" / "leak.pdf"
        leaked.parent.mkdir(parents=True, exist_ok=True)
        leaked.write_bytes(PDF_BYTES)
        assert store_and_link(str(leaked), store=store, scope=_Scope(), config=config,
                              secret=secret) is None
        text = f"MEDIA:{leaked}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret) == text
        assert store.count() == 0

    def test_missing_file_is_left_literal(self, tmp_path, store, secret, config):
        text = f"MEDIA:{tmp_path / 'not-there.pdf'}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret) == text
        assert store.count() == 0

    def test_oversize_file_is_left_literal(self, tmp_path, store, secret):
        path = _write(tmp_path, "report.pdf", PDF_BYTES + b"x" * 4096)
        capped = FileDeliveryConfig(enabled=True, public_base_url=BASE_URL, max_bytes=64)
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=capped, secret=secret) == text
        assert store.count() == 0

    def test_extension_outside_the_vocabulary_is_left_literal(self, tmp_path, store, secret, config):
        path = _write(tmp_path, "notes.log")
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret) == text
        assert store.count() == 0

    def test_content_sniff_rejects_a_mislabeled_file(self, tmp_path, store, secret, config):
        """Extension is not evidence: a .md that is really a binary must not be served as markdown."""
        path = _write(tmp_path, "notes.md", b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64)
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret) == text
        assert store.count() == 0

    def test_denied_mime_type_is_left_literal(self, tmp_path, store, secret):
        path = _write(tmp_path, "report.pdf")
        narrowed = FileDeliveryConfig(enabled=True, public_base_url=BASE_URL,
                                      allowed_mime_types=frozenset({"text/plain"}))
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=narrowed, secret=secret) == text
        assert store.count() == 0

    def test_image_tag_is_left_to_the_image_pass(self, tmp_path, store, secret, config):
        """Images keep their data-URL behavior; this pass must not claim them."""
        path = _write(tmp_path, "shot.png", PNG_BYTES)
        text = f"MEDIA:{path}"
        assert resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret) == text
        assert store.count() == 0

    @pytest.mark.parametrize("suffix", sorted(delivery.DELIVERY_DOCUMENT_EXT_MIME))
    def test_every_advertised_type_delivers_through_the_adapter(self, tmp_path, monkeypatch, suffix):
        """Every type the config advertises must be producible end to end.

        The store's allowlist and the config's are two separate declarations of one vocabulary; this
        pins them together by delivering a real sample of each (a mismatch would otherwise show up as
        a silently literal tag for exactly the files users ask for most).
        """
        config = FileDeliveryConfig(enabled=True, public_base_url=BASE_URL)
        monkeypatch.setattr(delivery, "file_delivery_config", lambda *args, **kwargs: config)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
        sample = _SAMPLES[suffix]
        text = f"Here you go: MEDIA:{_write(tmp_path, f'sample{suffix}', sample)}"
        out = adapter._resolve_media_tags(text)
        assert "MEDIA:" not in out, out
        artifact_id, params = _url_parts(_link_from(out).group("url"))
        secret = adapter._file_delivery_secret("default")
        assert verify_download_token(secret, artifact_id, params["expires"], params["signature"])


class TestStreamingRewriter:
    def _rewriter(self, store, secret, config):
        seen = {}

        def _resolve(raw_path):
            if raw_path not in seen:
                seen[raw_path] = store_and_link(
                    raw_path, store=store, scope=_Scope(), config=config, secret=secret)
            return seen[raw_path]

        return MediaTagStreamRewriter(_resolve)

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64])
    def test_emitted_text_matches_the_non_streaming_link(self, tmp_path, store, secret, config, chunk):
        """Same tag, same shape of link, whichever lane delivered the text — at EVERY delta size.

        Chunk sizes are parametrized because the keyword boundary is what breaks: a single-character
        delta can carry a lone "M", and if that is emitted the rest of "EDIA:/path" reads as ordinary
        text and the raw server path streams out. Sizes 1 and 7 both hit that split.
        """
        path = _write(tmp_path, "report.pdf")
        text = f"Here you go: MEDIA:{path}"
        non_streaming = resolve_media_to_download_urls(
            text, store=store, scope=_Scope(), config=config, secret=secret)

        rewriter = self._rewriter(store, secret, config)
        streamed = "".join(rewriter.feed(text[index:index + chunk])
                           for index in range(0, len(text), chunk)) + rewriter.flush()

        assert _URL_RE.sub("<URL>", streamed) == _URL_RE.sub("<URL>", non_streaming)
        assert "MEDIA:" not in streamed and str(path) not in streamed

    @pytest.mark.parametrize("chunk", [1, 7, 11])
    def test_a_tag_split_across_deltas_never_leaks_the_path(self, tmp_path, store, secret, config, chunk):
        path = _write(tmp_path, "quarterly report.pdf")
        text = f"Done: MEDIA:{path}"
        rewriter = self._rewriter(store, secret, config)
        emitted = [rewriter.feed(text[index:index + chunk])
                   for index in range(0, len(text), chunk)]
        emitted.append(rewriter.flush())
        joined = "".join(emitted)
        assert "MEDIA:" not in joined
        assert str(tmp_path) not in joined
        assert _LINK_RE.search(joined)

    def test_complete_text_does_not_stall_behind_the_hold_back(self):
        """Only a possible tag is held: ordinary prose streams through immediately."""
        rewriter = MediaTagStreamRewriter(lambda raw_path: "LINK")
        assert rewriter.feed("Hello ") == "Hello "
        assert rewriter.feed("world") == "world"
        assert rewriter.flush() == ""

    def test_eos_closed_tag_is_held_then_resolved(self, tmp_path, store, secret, config):
        """A match closed by end-of-buffer may still grow (``…/report.tar`` → ``…/report.tar.gz``),
        so resolution waits for a real boundary — and flush()'s the tag when the stream just ends."""
        path = _write(tmp_path, "archive.zip", ZIP_BYTES)
        seen = []
        rewriter = MediaTagStreamRewriter(
            lambda raw_path: seen.append(raw_path) or store_and_link(
                raw_path, store=store, scope=_Scope(), config=config, secret=secret))
        head = rewriter.feed(f"MEDIA:{path}")
        assert head == "" and seen == []
        resolved = rewriter.flush()
        assert seen == [str(path)]
        assert _LINK_RE.search(resolved)


class _DeliveryEnabled:
    """Turns the transport on for the classes that exercise it."""

    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        config = FileDeliveryConfig(enabled=True, public_base_url=BASE_URL)
        monkeypatch.setattr(delivery, "file_delivery_config", lambda *args, **kwargs: config)
        return config


def _adapter(key=API_KEY):
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))


class TestSignedDownloadRoute(_DeliveryEnabled):
    """End-to-end through the real route: the link the resolver emits must actually download."""

    def _app(self, adapter):
        app = web.Application()
        app.router.add_get("/v1/artifacts/download/{artifact_id}", adapter._handle_artifact_download)
        app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
        return app

    def _mint_url(self, adapter, tmp_path, name="report.pdf"):
        markdown = adapter._resolve_media_tags(f"Here you go: MEDIA:{_write(tmp_path, name)}")
        return _link_from(markdown).group("url")[len(BASE_URL):]

    @pytest.mark.asyncio
    async def test_link_downloads_once_then_404s(self, tmp_path):
        adapter = _adapter()
        path = self._mint_url(adapter, tmp_path)
        async with TestClient(TestServer(self._app(adapter))) as client:
            first = await client.get(path)
            assert first.status == 200
            assert await first.read() == PDF_BYTES
            assert first.headers["Content-Disposition"].startswith("attachment;")
            assert first.headers["Content-Length"] == str(len(PDF_BYTES))
            # Consumed: the same link is dead, which is the whole point of a one-shot delivery.
            second = await client.get(path)
            assert second.status == 404

    @pytest.mark.asyncio
    async def test_tampered_signature_is_404(self, tmp_path):
        adapter = _adapter()
        path = self._mint_url(adapter, tmp_path)
        head, _, query = path.partition("?")
        params = dict(pair.split("=", 1) for pair in query.split("&"))
        flipped = params["signature"][:-1] + ("0" if params["signature"][-1] != "0" else "1")
        async with TestClient(TestServer(self._app(adapter))) as client:
            tampered = await client.get(f"{head}?expires={params['expires']}&signature={flipped}")
            assert tampered.status == 404
            # A signature is bound to its artifact id, not reusable across artifacts.
            other = await client.get(f"/v1/artifacts/download/{'0' * 32}"
                                     f"?expires={params['expires']}&signature={params['signature']}")
            assert other.status == 404

    @pytest.mark.asyncio
    async def test_expired_link_is_404(self, tmp_path):
        adapter = _adapter()
        path = self._mint_url(adapter, tmp_path)
        head, _, _query = path.partition("?")
        artifact_id = head.rsplit("/", 1)[-1]
        secret = adapter._file_delivery_secret("default")
        past = int(time.time()) - 1
        expired = f"{head}?expires={past}&signature={sign_download_token(secret, artifact_id, past)}"
        async with TestClient(TestServer(self._app(adapter))) as client:
            response = await client.get(expired)
            assert response.status == 404

    @pytest.mark.asyncio
    async def test_disabled_transport_has_no_signed_route(self, tmp_path, monkeypatch):
        """Off is off: a syntactically perfect link must not open the route."""
        adapter = _adapter()
        path = self._mint_url(adapter, tmp_path)
        monkeypatch.setattr(delivery, "file_delivery_config",
                            lambda *args, **kwargs: FileDeliveryConfig())
        async with TestClient(TestServer(self._app(adapter))) as client:
            response = await client.get(path)
            assert response.status == 404

    @pytest.mark.asyncio
    async def test_browser_control_ladder_is_untouched(self, tmp_path):
        """A keyless, signature-less request still goes down the browser-control ladder.

        With browser control off that is a 404 — the delivery flag must not have widened it into a
        bearer-free artifact route.
        """
        adapter = _adapter()
        path = self._mint_url(adapter, tmp_path)
        unsigned = path.partition("?")[0]
        async with TestClient(TestServer(self._app(adapter))) as client:
            response = await client.get(unsigned)
            assert response.status == 404
            # …and the browser-control artifact ladder still answers its own prelude for a keyed
            # request rather than being hijacked by the delivery branch.
            keyed = await client.get(unsigned, headers={"Authorization": f"Bearer {API_KEY}"})
            assert keyed.status == 404

    @pytest.mark.asyncio
    async def test_capabilities_advertise_the_transport_separately(self):
        adapter = _adapter()
        async with TestClient(TestServer(self._app(adapter))) as client:
            response = await client.get("/v1/capabilities",
                                        headers={"Authorization": f"Bearer {API_KEY}"})
            assert response.status == 200
            features = (await response.json())["features"]
        assert features["file_delivery"]["enabled"] is True
        assert features["file_delivery"]["auth"] == "signed-url"
        assert features["file_delivery"]["allowed_mime_types"] == sorted(DEFAULT_FILE_DELIVERY_MIME_TYPES)
        # The browser-control block keeps advertising its own artifact transport unchanged.
        assert features["browser_extension_control"]["enabled"] is False
        assert features["browser_extension_control"]["artifact_transport"]["upload"]["path"] == \
            "/v1/artifacts/upload"


class TestProfileIsolation:
    def test_two_profiles_never_share_a_store_root(self):
        assert delivery_root("arman") != delivery_root("tina")
        assert delivery_root("arman").parent != delivery_root("tina").parent

    def test_stores_and_principals_are_per_profile(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
        config = FileDeliveryConfig(enabled=True, public_base_url=BASE_URL)
        first = adapter._file_delivery_store_for("arman", config)
        second = adapter._file_delivery_store_for("tina", config)
        assert first.root != second.root
        assert adapter._file_delivery_principal("arman") != adapter._file_delivery_principal("tina")

    def test_another_profiles_scope_cannot_read_the_artifact(self, tmp_path, secret, config):
        """The store's own scope binding still holds behind a signed link."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
        store = adapter._file_delivery_store_for("arman", config)
        path = _write(tmp_path, "report.pdf")
        markdown = store_and_link(str(path), store=store, scope=adapter._file_delivery_scope("arman"),
                                  config=config, secret=secret)
        assert markdown is not None
        artifact_id, _params = _url_parts(_link_from(markdown).group("url"))
        with pytest.raises(ArtifactScopeMismatch):
            store.load(artifact_id, scope=adapter._file_delivery_scope("tina"))
        # …and the same profile still can (the failed attempt did not consume it).
        data, _receipt = store.load(artifact_id, scope=adapter._file_delivery_scope("arman"))
        assert data == PDF_BYTES


class TestConfigLoader:
    def test_delivery_vocabulary_is_inside_the_tag_grammar(self):
        """A type this transport advertises must be one the anchored tag regex can actually match.

        ``MEDIA_DELIVERY_EXTS`` builds ``MEDIA_TAG_CLEANUP_RE``'s alternation, so an extension outside
        it would never be rewritten — the config key would promise a download the tag grammar cannot
        see. Both directions matter: documents stay a subset, and every document type has a sample.
        """
        from gateway.platforms.base import MEDIA_DELIVERY_EXTS

        assert set(delivery.DELIVERY_DOCUMENT_EXT_MIME) <= set(MEDIA_DELIVERY_EXTS)
        assert set(delivery.DELIVERY_DOCUMENT_EXT_MIME) == set(_SAMPLES)
        assert set(delivery.DELIVERY_DOCUMENT_EXT_MIME.values()) <= DEFAULT_FILE_DELIVERY_MIME_TYPES

    def test_block_is_read_from_the_profile_config(self):
        """The default-off contract, through the loader the runtime actually uses."""
        assert file_delivery_config() == FileDeliveryConfig()

    def test_enabled_block_reaches_the_runtime_reader(self):
        from hermes_constants import get_hermes_home
        import yaml

        home = Path(get_hermes_home())
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"api_server": {"file_delivery": {
                "enabled": True, "public_base_url": f"{BASE_URL}/p/arman", "max_bytes": 2048,
                "ttl_seconds": 60, "allowed_mime_types": ["application/pdf"]}}}}),
            encoding="utf-8")
        resolved = file_delivery_config()
        assert resolved.enabled is True
        assert resolved.public_base_url == f"{BASE_URL}/p/arman"
        assert resolved.max_bytes == 2048
        assert resolved.ttl_seconds == 60.0
        assert resolved.allowed_mime_types == frozenset({"application/pdf"})


class TestStreamingLane(_DeliveryEnabled):
    """The delta lane is the one a chat client actually renders.

    A streaming frontend renders ``choices[0].delta.content`` and finalizes its view with what
    streamed, so a link that existed only in the terminal payload would never reach the user — the
    same tag has to be resolved on the wire, split across deltas or not.
    """

    @pytest.mark.asyncio
    async def test_streamed_deltas_carry_the_link_instead_of_the_path(self, tmp_path):
        adapter = _adapter()
        pdf = _write(tmp_path, "report.pdf")
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        # The tag is deliberately split mid-path: a per-delta regex cannot see it whole.
        cut = len(str(pdf)) - 4

        async def _mock_run_agent(**kwargs):
            callback = kwargs.get("stream_delta_callback")
            if callback:
                callback(f"Here you go: MEDIA:{str(pdf)[:cut]}")
                callback(str(pdf)[cut:])
            return (
                {"final_response": f"Here you go: MEDIA:{pdf}", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            async with TestClient(TestServer(app)) as client:
                response = await client.post(
                    "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"model": "test",
                          "messages": [{"role": "user", "content": "send the report"}],
                          "stream": True})
                body = await response.text()

        assert response.status == 200 and "[DONE]" in body
        assert str(pdf) not in body          # no server path anywhere on the wire
        streamed = "".join(_delta_contents(body))
        assert "MEDIA:" not in streamed
        # Same link as the non-streaming lane, modulo the freshly minted artifact id.
        non_streaming = adapter._resolve_media_tags(f"Here you go: MEDIA:{pdf}")
        assert _URL_RE.sub("<URL>", streamed) == _URL_RE.sub("<URL>", non_streaming)
