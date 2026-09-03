"""Route-configurable webhook signature schemes.

The adapter can be told, per route, how a provider packages its HMAC into
headers instead of that packaging being hard-coded one vendor at a time.  The
cases below pin the three shapes that cover most of the ecosystem —

* ElevenLabs  ``ElevenLabs-Signature: t=<unix>,v0=<hex>``  over ``<t>.<body>``
* Stripe      ``Stripe-Signature: t=<unix>,v1=<hex>``      over ``<t>.<body>``
* Slack       ``X-Slack-Signature: v0=<hex>`` plus a separate timestamp header,
              over ``v0:<ts>:<body>``

— plus the failure modes that matter when the endpoint is reachable from the
public internet and the signature is the only thing in front of an agent: a
missing or empty header, a right-shaped wrong value, a tampered body, the wrong
secret, timestamps that are stale, in the future, or not numbers at all,
conflicting repeated header parts, and malformed configuration.  Every one of
those must fail closed; none may fall through to a different scheme.
"""

import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    WebhookAdapter,
    _parse_signature_spec,
    _render_signed_message,
    _split_signature_header,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ELEVENLABS_SIGNATURE = {
    "header": "ElevenLabs-Signature",
    "signature_part": "v0",
    "timestamp_part": "t",
    "template": "{timestamp}.{body}",
    "tolerance_seconds": 1800,
}

STRIPE_SIGNATURE = {
    "header": "Stripe-Signature",
    "signature_part": "v1",
    "timestamp_part": "t",
    "template": "{timestamp}.{body}",
}

SLACK_SIGNATURE = {
    "header": "X-Slack-Signature",
    "signature_prefix": "v0=",
    "timestamp_header": "X-Slack-Request-Timestamp",
    "template": "v0:{timestamp}:{body}",
}

BODY = json.dumps({"type": "post_call_transcription", "data": {"id": "c1"}}).encode()
SECRET = "wsec_0123456789abcdef0123456789abcdef"


def _make_adapter(routes, host="127.0.0.1", **extra_kw):
    extra = {"host": host, "port": 0, "routes": routes}
    extra.update(extra_kw)
    return WebhookAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _mock_request(headers=None, route_name="r"):
    req = MagicMock()
    req.headers = headers or {}
    req.match_info = {"route_name": route_name}
    req.method = "POST"
    return req


def _hex_hmac(secret: str, message: bytes, algorithm=hashlib.sha256) -> str:
    return hmac.new(secret.encode(), message, algorithm).hexdigest()


def _elevenlabs_header(body: bytes, secret: str, timestamp=None) -> str:
    """Reproduce elevenlabs-python's construct_event() header exactly."""
    ts = str(int(time.time())) if timestamp is None else str(timestamp)
    return f"t={ts},v0={_hex_hmac(secret, f'{ts}.'.encode() + body)}"


def _stripe_header(body: bytes, secrets, timestamp=None) -> str:
    ts = str(int(time.time())) if timestamp is None else str(timestamp)
    sigs = ",".join(
        f"v1={_hex_hmac(s, f'{ts}.'.encode() + body)}" for s in secrets
    )
    return f"t={ts},{sigs}"


def _slack_headers(body: bytes, secret: str, timestamp=None) -> dict:
    ts = str(int(time.time())) if timestamp is None else str(timestamp)
    signed = f"v0:{ts}:".encode() + body
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": "v0=" + _hex_hmac(secret, signed),
    }


def _route(signature, secret=SECRET, **extra):
    route = {"secret": secret, "prompt": "{__raw__}", "deliver": "log"}
    if signature is not None:
        route["signature"] = signature
    route.update(extra)
    return route


def _validate(signature, body=BODY, headers=None, secret=SECRET, route_secret=None):
    """Run one validation against a route configured with *signature*."""
    routes = {"r": _route(signature, secret=route_secret or secret)}
    adapter = _make_adapter(routes)
    return adapter._validate_signature(
        _mock_request(headers), body, route_secret or secret
    )


# ===================================================================
# ElevenLabs — the combined "t=,v0=" header
# ===================================================================


class TestElevenLabsShape:
    def test_valid_signature_accepted(self):
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is True

    def test_secret_used_raw_including_wsec_prefix(self):
        """ElevenLabs keys the HMAC with the literal secret, prefix included.

        Svix-style ``whsec_`` secrets are base64 payloads that must be decoded
        first; ``wsec_`` ones are not. Stripping or decoding here would break
        every real delivery, so pin the raw-key behaviour.
        """
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is True
        assert (
            _validate(
                ELEVENLABS_SIGNATURE,
                headers={
                    "ElevenLabs-Signature": _elevenlabs_header(
                        BODY, SECRET.removeprefix("wsec_")
                    )
                },
            )
            is False
        )

    def test_tampered_body_rejected(self):
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET)}
        assert _validate(ELEVENLABS_SIGNATURE, body=BODY + b" ", headers=headers) is False

    def test_wrong_secret_rejected(self):
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, "wsec_other")}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_expired_timestamp_rejected(self):
        stale = int(time.time()) - 1801
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET, stale)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_future_timestamp_rejected(self):
        """A clock far ahead is as suspect as a stale one — and accepting it
        would hand an attacker a replay ticket valid for as long as they like."""
        ahead = int(time.time()) + 1801
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET, ahead)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_timestamp_inside_window_accepted(self):
        recent = int(time.time()) - 1500
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET, recent)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is True

    def test_default_tolerance_is_tighter_than_elevenlabs(self):
        """Tolerance defaults to the adapter's existing 300s, not the
        provider's 30 minutes; a route opts into a looser window explicitly."""
        spec = dict(ELEVENLABS_SIGNATURE)
        spec.pop("tolerance_seconds")
        old = int(time.time()) - 400
        headers = {"ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET, old)}
        assert _validate(spec, headers=headers) is False

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "garbage",
            "t=,v0=",
            "v0=" + "a" * 64,                       # no timestamp part
            f"t={int(time.time())}",                # no signature part
            f"t={int(time.time())},v0=",            # empty signature
            f"t=not-a-number,v0={'a' * 64}",
            f"t={int(time.time())},v0={'a' * 64}",  # right shape, wrong value
            f"t={int(time.time())},v0=zz",
        ],
    )
    def test_malformed_or_forged_headers_rejected(self, value):
        assert _validate(ELEVENLABS_SIGNATURE, headers={"ElevenLabs-Signature": value}) is False

    def test_missing_header_rejected(self):
        assert _validate(ELEVENLABS_SIGNATURE, headers={}) is False

    def test_non_ascii_signature_rejected_not_raised(self):
        """Hostile non-ASCII must fail closed, not raise out of the handler."""
        value = f"t={int(time.time())},v0=ské-not-hex"
        assert _validate(ELEVENLABS_SIGNATURE, headers={"ElevenLabs-Signature": value}) is False

    def test_conflicting_timestamp_parts_rejected(self):
        """Repeating ``t=`` must not let a caller pair a fresh timestamp with
        a signature computed over a different one."""
        valid = _elevenlabs_header(BODY, SECRET)
        stale = int(time.time()) - 100000
        assert (
            _validate(
                ELEVENLABS_SIGNATURE,
                headers={"ElevenLabs-Signature": f"t={stale},{valid}"},
            )
            is False
        )

    def test_repeated_identical_timestamp_tolerated(self):
        valid = _elevenlabs_header(BODY, SECRET)
        ts = valid.split(",")[0]
        assert (
            _validate(
                ELEVENLABS_SIGNATURE,
                headers={"ElevenLabs-Signature": f"{valid},{ts}"},
            )
            is True
        )

    def test_padding_junk_in_header_ignored(self):
        valid = _elevenlabs_header(BODY, SECRET)
        assert (
            _validate(
                ELEVENLABS_SIGNATURE,
                headers={"ElevenLabs-Signature": f"junk,{valid},x=1,"},
            )
            is True
        )

    def test_uppercase_hex_accepted(self):
        valid = _elevenlabs_header(BODY, SECRET)
        ts, sig = valid.split(",v0=")
        assert (
            _validate(
                ELEVENLABS_SIGNATURE,
                headers={"ElevenLabs-Signature": f"{ts},v0={sig.upper()}"},
            )
            is True
        )


# ===================================================================
# Other provider shapes — the point of making this configurable
# ===================================================================


class TestOtherProviderShapes:
    def test_stripe_signature_accepted(self):
        headers = {"Stripe-Signature": _stripe_header(BODY, [SECRET])}
        assert _validate(STRIPE_SIGNATURE, headers=headers) is True

    def test_stripe_secret_rotation_accepts_either_signature(self):
        """Stripe sends one ``v1=`` per live secret while a roll is in flight."""
        headers = {"Stripe-Signature": _stripe_header(BODY, ["old-secret", SECRET])}
        assert _validate(STRIPE_SIGNATURE, headers=headers) is True

    def test_stripe_rotation_all_wrong_rejected(self):
        headers = {"Stripe-Signature": _stripe_header(BODY, ["old-secret", "other"])}
        assert _validate(STRIPE_SIGNATURE, headers=headers) is False

    def test_stripe_tampered_body_rejected(self):
        headers = {"Stripe-Signature": _stripe_header(BODY, [SECRET])}
        assert _validate(STRIPE_SIGNATURE, body=b"{}", headers=headers) is False

    def test_slack_signature_accepted(self):
        """Slack keeps the timestamp in its own header and signs a
        colon-delimited message — a different layout, same primitive."""
        assert _validate(SLACK_SIGNATURE, headers=_slack_headers(BODY, SECRET)) is True

    def test_slack_missing_timestamp_header_rejected(self):
        headers = _slack_headers(BODY, SECRET)
        headers.pop("X-Slack-Request-Timestamp")
        assert _validate(SLACK_SIGNATURE, headers=headers) is False

    def test_slack_stale_timestamp_rejected(self):
        headers = _slack_headers(BODY, SECRET, timestamp=int(time.time()) - 400)
        assert _validate(SLACK_SIGNATURE, headers=headers) is False

    def test_slack_missing_v0_prefix_rejected(self):
        headers = _slack_headers(BODY, SECRET)
        headers["X-Slack-Signature"] = headers["X-Slack-Signature"].removeprefix("v0=")
        assert _validate(SLACK_SIGNATURE, headers=headers) is False

    def test_slack_body_swapped_rejected(self):
        assert _validate(SLACK_SIGNATURE, body=b"{}", headers=_slack_headers(BODY, SECRET)) is False

    def test_base64_encoding_supported(self):
        import base64 as b64

        spec = {
            "header": "X-Provider-Signature",
            "template": "{body}",
            "encoding": "base64",
        }
        digest = hmac.new(SECRET.encode(), BODY, hashlib.sha256).digest()
        headers = {"X-Provider-Signature": b64.b64encode(digest).decode()}
        assert _validate(spec, headers=headers) is True
        assert _validate(spec, headers={"X-Provider-Signature": "AAAA"}) is False

    def test_sha1_and_sha512_supported(self):
        for name, algorithm in (("sha1", hashlib.sha1), ("sha512", hashlib.sha512)):
            spec = {
                "header": "X-Legacy-Signature",
                "template": "{body}",
                "algorithm": name,
            }
            good = {"X-Legacy-Signature": _hex_hmac(SECRET, BODY, algorithm)}
            assert _validate(spec, headers=good) is True, name
            bad = {"X-Legacy-Signature": _hex_hmac("nope", BODY, algorithm)}
            assert _validate(spec, headers=bad) is False, name

    def test_existing_generic_v2_scheme_is_expressible(self):
        """The built-in V2 scheme restated as config — which is how a route
        pins itself to V2 and drops the legacy V1 fallback."""
        spec = {
            "header": "X-Webhook-Signature-V2",
            "timestamp_header": "X-Webhook-Timestamp",
            "template": "{timestamp}.{body}",
        }
        ts = str(int(time.time()))
        headers = {
            "X-Webhook-Signature-V2": _hex_hmac(SECRET, f"{ts}.".encode() + BODY),
            "X-Webhook-Timestamp": ts,
        }
        assert _validate(spec, headers=headers) is True


# ===================================================================
# Exclusivity — no downgrade to another scheme
# ===================================================================


class TestExclusivity:
    def test_legacy_v1_signature_rejected_on_configured_route(self):
        """The V1 body-only header has no replay protection. A route that has
        declared its provider must not be authenticated through it."""
        headers = {"X-Webhook-Signature": _hex_hmac(SECRET, BODY)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_github_signature_rejected_on_configured_route(self):
        headers = {"X-Hub-Signature-256": "sha256=" + _hex_hmac(SECRET, BODY)}
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_gitlab_token_rejected_on_configured_route(self):
        assert _validate(ELEVENLABS_SIGNATURE, headers={"X-Gitlab-Token": SECRET}) is False

    def test_v2_headers_rejected_on_configured_route(self):
        ts = str(int(time.time()))
        headers = {
            "X-Webhook-Signature-V2": _hex_hmac(SECRET, f"{ts}.".encode() + BODY),
            "X-Webhook-Timestamp": ts,
        }
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is False

    def test_valid_provider_header_still_wins_alongside_junk(self):
        headers = {
            "ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET),
            "X-Hub-Signature-256": "sha256=deadbeef",
        }
        assert _validate(ELEVENLABS_SIGNATURE, headers=headers) is True

    def test_unconfigured_routes_keep_builtin_probing(self):
        """Backwards compatibility: without a `signature` block nothing moves."""
        headers = {"X-Hub-Signature-256": "sha256=" + _hex_hmac(SECRET, BODY)}
        assert _validate(None, headers=headers) is True


# ===================================================================
# Configuration validation
# ===================================================================


class TestSignatureConfigValidation:
    @pytest.mark.parametrize(
        "block",
        [
            "not-a-mapping",
            {},                                              # no header
            {"header": "   "},                               # blank header
            {"header": "H", "algorithm": "md5"},             # unsupported
            {"header": "H", "encoding": "base32"},           # unsupported
            {"header": "H", "template": "{timestamp}"},      # body not signed
            {"header": "H", "template": "{nope}.{body}"},    # unknown token
            {"header": "H", "template": "{body}", "timestamp_part": "t"},
            {"header": "H", "timestamp_part": "t", "timestamp_header": "X-T"},
            {"header": "H", "template": "{timestamp}.{body}"},  # no ts source
            {"header": "H", "template": "{body}", "tolerance_seconds": 0},
            {"header": "H", "template": "{body}", "tolerance_seconds": "300"},
            {"header": "H", "template": "{body}", "tolerance_seconds": True},
            {"header": 5, "template": "{body}"},
            {"header": "H", "template": "{body}", "signature_prefix": 1},
            {"header": "H", "template": "{body}", "signature_prefix": ["v0="]},
            {"header": "H", "template": "{body}", "signature_part": 1},
            {"header": "H", "template": "{body}", "algorithm": 256},
            {"header": "H", "template": "{body}", "encoding": 16},
            {"header": "H", "template": "{timestamp}.{body}", "timestamp_part": 1},
            {"header": "H", "template": "{timestamp}.{body}", "timestamp_header": 1},
            {"header": "H", "template": 5},
            {"header": "H", "template": "{body}:{body}"},
            {"header": "H", "template": "{timestamp}.{body}{body}", "timestamp_part": "t"},
        ],
    )
    def test_invalid_blocks_rejected(self, block):
        with pytest.raises(ValueError):
            _parse_signature_spec("r", block)

    def test_defaults_applied(self):
        spec = _parse_signature_spec("r", {"header": "H", "template": "{body}"})
        assert spec["algorithm_name"] == "sha256"
        assert spec["encoding"] == "hex"
        assert spec["tolerance_seconds"] == 300
        assert spec["uses_timestamp"] is False

    def test_elevenlabs_block_parses(self):
        spec = _parse_signature_spec("r", ELEVENLABS_SIGNATURE)
        assert spec["uses_timestamp"] is True
        assert spec["tolerance_seconds"] == 1800
        assert spec["signature_part"] == "v0"

    @pytest.mark.asyncio
    async def test_connect_refuses_malformed_block(self):
        adapter = _make_adapter({"r": _route({"header": "H", "algorithm": "rot13"})})
        with pytest.raises(ValueError, match="unknown algorithm"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_accepts_valid_block(self):
        adapter = _make_adapter({"r": _route(ELEVENLABS_SIGNATURE)})
        assert await adapter.connect() is True
        await adapter.disconnect()

    def test_malformed_block_on_a_live_route_fails_closed(self):
        """Dynamically-registered routes never see connect(); a bad block
        there must reject requests, not raise or accept them."""
        routes = {"r": _route({"header": "H", "encoding": "base32"})}
        adapter = _make_adapter(routes)
        req = _mock_request({"H": "anything"})
        assert adapter._validate_signature(req, BODY, SECRET) is False


# ===================================================================
# Message construction
# ===================================================================


class TestSignedMessage:
    def test_raw_body_bytes_are_spliced_not_decoded(self):
        body = b"\xff\xfe not utf-8"
        assert _render_signed_message("{timestamp}.{body}", "123", body) == b"123." + body

    def test_non_utf8_body_validates_end_to_end(self):
        body = b'{"blob":"\xc3\x28"}'
        ts = str(int(time.time()))
        headers = {
            "ElevenLabs-Signature": f"t={ts},v0={_hex_hmac(SECRET, f'{ts}.'.encode() + body)}"
        }
        assert _validate(ELEVENLABS_SIGNATURE, body=body, headers=headers) is True

    def test_timestamp_cannot_forge_a_body_marker(self):
        """The body slot is located before the timestamp is substituted, so a
        crafted timestamp cannot move where the payload is spliced."""
        assert (
            _render_signed_message("{timestamp}.{body}", "{body}", b"X")
            == b"{body}.X"
        )

    def test_split_collects_repeated_labels(self):
        assert _split_signature_header("t=1,v1=a,v1=b") == {"t": ["1"], "v1": ["a", "b"]}

    def test_split_ignores_valueless_chunks(self):
        assert _split_signature_header("junk,t=1") == {"t": ["1"]}


# ===================================================================
# End to end over HTTP
# ===================================================================


class TestOverHttp:
    @pytest.mark.asyncio
    async def test_valid_delivery_accepted_tampered_rejected(self):
        routes = {"elevenlabs-calls": _route(ELEVENLABS_SIGNATURE, deliver_only=False)}
        adapter = _make_adapter(routes)
        seen = []
        adapter.handle_message = lambda event: seen.append(event)

        async def _capture(event):
            seen.append(event)

        adapter.handle_message = _capture

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            good = await cli.post(
                "/webhooks/elevenlabs-calls",
                data=BODY,
                headers={
                    "Content-Type": "application/json",
                    "ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET),
                },
            )
            assert good.status == 202

            tampered = BODY.replace(b"c1", b"c2")
            bad = await cli.post(
                "/webhooks/elevenlabs-calls",
                data=tampered,
                headers={
                    "Content-Type": "application/json",
                    "ElevenLabs-Signature": _elevenlabs_header(BODY, SECRET),
                },
            )
            assert bad.status == 401

            missing = await cli.post(
                "/webhooks/elevenlabs-calls",
                data=BODY,
                headers={"Content-Type": "application/json"},
            )
            assert missing.status == 401

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_oversized_body_rejected_before_validation(self):
        routes = {"r": _route(ELEVENLABS_SIGNATURE)}
        adapter = _make_adapter(routes, max_body_bytes=512)
        big = b"x" * 4096

        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.post(
                "/webhooks/r",
                data=big,
                headers={
                    "Content-Type": "application/json",
                    "ElevenLabs-Signature": _elevenlabs_header(big, SECRET),
                },
            )
            assert resp.status == 413


# ===================================================================
# Review regressions — a malformed block must never raise
#
# Every string-valued key must be normalised at parse time. Before this was
# enforced, a truthy non-string `signature_prefix` passed startup validation
# and then reached `candidate.startswith(prefix)` in the verifier, raising
# TypeError — a 500 on the request path, on the one boundary that is supposed
# to fail closed. These pin the whole class, not just the reported key.
# ===================================================================


STRING_KEYS = [
    "header",
    "signature_part",
    "signature_prefix",
    "timestamp_part",
    "timestamp_header",
    "template",
]


def _block_with(key, value):
    """A block that is valid except for *key*."""
    block = {"header": "X-Sig", "template": "{timestamp}.{body}", "timestamp_part": "t"}
    block[key] = value
    return block


class TestMalformedBlockNeverRaises:
    @pytest.mark.parametrize("key", STRING_KEYS)
    @pytest.mark.parametrize("value", [1, 0.5, ["x"], {"x": 1}, object()])
    def test_non_string_values_rejected_at_parse(self, key, value):
        with pytest.raises(ValueError):
            _parse_signature_spec("r", _block_with(key, value))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", STRING_KEYS)
    async def test_connect_rejects_non_string(self, key):
        """Static routes must fail at startup, not at request time."""
        adapter = _make_adapter({"r": _route(_block_with(key, 1))})
        with pytest.raises(ValueError):
            await adapter.connect()

    @pytest.mark.parametrize("key", STRING_KEYS)
    def test_live_route_returns_false_without_raising(self, key):
        """A dynamically-registered route never reaches connect()'s check, so
        the request path must reject it rather than raise out of the handler."""
        adapter = _make_adapter({})
        adapter._routes["dyn"] = _route(_block_with(key, 1))
        req = _mock_request({"X-Sig": "v0=" + "a" * 64}, route_name="dyn")
        assert adapter._validate_signature(req, BODY, SECRET) is False

    def test_non_string_prefix_reaches_the_startswith_call_site(self):
        """The reported defect: a truthy non-string prefix used to survive
        parsing and blow up inside the verifier on `candidate.startswith()`."""
        block = _block_with("signature_prefix", 1)
        with pytest.raises(ValueError):
            _parse_signature_spec("r", block)

        adapter = _make_adapter({})
        adapter._routes["dyn"] = _route(block)
        ts = str(int(time.time()))
        header = f"t={ts},v0={_hex_hmac(SECRET, f'{ts}.'.encode() + BODY)}"
        req = _mock_request({"X-Sig": header}, route_name="dyn")
        assert adapter._validate_signature(req, BODY, SECRET) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", STRING_KEYS)
    async def test_over_http_is_401_never_500(self, key):
        """End to end: a malformed live route rejects with 401, and the
        request never reaches agent dispatch."""
        adapter = _make_adapter({})
        adapter._routes["dyn"] = _route(_block_with(key, 1))
        seen = []

        async def _capture(event):
            seen.append(event)

        adapter.handle_message = _capture

        ts = str(int(time.time()))
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.post(
                "/webhooks/dyn",
                data=BODY,
                headers={
                    "Content-Type": "application/json",
                    "X-Sig": f"t={ts},v0={_hex_hmac(SECRET, f'{ts}.'.encode() + BODY)}",
                },
            )
            assert resp.status == 401, f"{key}: got {resp.status}"
        assert seen == []


# ===================================================================
# Review regressions — accepted templates have deterministic semantics
#
# The renderer splices the raw body at a single point, so a repeated {body}
# used to leave a literal "{body}" in the signed message: "{body}:{body}"
# rendered `<raw-body>:{body}`. Rejecting at startup is the fail-closed
# reading and keeps the accepted set unambiguous.
# ===================================================================


class TestTemplateShape:
    @pytest.mark.parametrize(
        "template",
        ["{body}{body}", "{body}:{body}", "{body}.{body}.{body}"],
    )
    def test_repeated_body_marker_rejected(self, template):
        with pytest.raises(ValueError, match="exactly one"):
            _parse_signature_spec("r", {"header": "H", "template": template})

    def test_missing_body_marker_rejected(self):
        with pytest.raises(ValueError, match=r"must contain"):
            _parse_signature_spec("r", {"header": "H", "template": "{timestamp}",
                                        "timestamp_part": "t"})

    def test_single_body_marker_accepted(self):
        spec = _parse_signature_spec("r", {"header": "H", "template": "{body}"})
        assert spec["template"] == "{body}"

    @pytest.mark.asyncio
    async def test_connect_rejects_repeated_body_marker(self):
        adapter = _make_adapter({"r": _route({"header": "H", "template": "{body}{body}"})})
        with pytest.raises(ValueError, match="exactly one"):
            await adapter.connect()

    def test_repeated_timestamp_marker_is_substituted_everywhere(self):
        """{timestamp} may repeat — unlike {body} it is a plain string
        replacement, so every occurrence is substituted and the result is
        deterministic. Pinned so the two markers can't silently diverge."""
        assert (
            _render_signed_message("{timestamp}:{timestamp}.{body}", "42", b"X")
            == b"42:42.X"
        )

    def test_repeated_timestamp_validates_end_to_end(self):
        spec = {
            "header": "X-Sig",
            "signature_part": "v0",
            "timestamp_part": "t",
            "template": "{timestamp}:{timestamp}.{body}",
        }
        ts = str(int(time.time()))
        signed = f"{ts}:{ts}.".encode() + BODY
        headers = {"X-Sig": f"t={ts},v0={_hex_hmac(SECRET, signed)}"}
        assert _validate(spec, headers=headers) is True

    def test_no_literal_marker_survives_into_the_signed_message(self):
        """The bug this closes: any accepted template must render without a
        leftover placeholder."""
        for template in ("{body}", "{timestamp}.{body}", "v0:{timestamp}:{body}"):
            spec = _parse_signature_spec("r", dict(
                {"header": "H", "template": template},
                **({"timestamp_part": "t"} if "{timestamp}" in template else {}),
            ))
            rendered = _render_signed_message(spec["template"], "123", b"PAYLOAD")
            assert b"{body}" not in rendered
            assert b"{timestamp}" not in rendered
            assert b"PAYLOAD" in rendered
