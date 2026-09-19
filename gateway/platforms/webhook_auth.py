"""Webhook signature validation for the generic webhook adapter."""

import base64
import binascii
import hashlib
import hmac
import logging
import time

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


logger = logging.getLogger("gateway.platforms.webhook")


def _hmac_str_equal(provided: str, expected: str) -> bool:
    """Timing-safe equality for two ``str`` values, tolerant of non-ASCII input.

    ``hmac.compare_digest`` raises ``TypeError`` when given a ``str`` that
    contains non-ASCII characters. The ``provided`` value here is an
    attacker-controlled signature/token header on a public, unauthenticated
    webhook endpoint, so a single non-ASCII byte would otherwise raise out of
    the request handler and return a 500 instead of rejecting the request.
    Comparing as UTF-8 bytes keeps the constant-time guarantee while making a
    hostile header fail closed with a clean rejection.
    """
    return hmac.compare_digest(provided.encode(), expected.encode())


# Replay tolerance in seconds applied to timestamp-bearing schemes (generic V2
# and Svix). Kept as a module constant so tests can patch it deterministically.
DEFAULT_REPLAY_TOLERANCE_SECONDS = 300

# Supported provider schemes. Routes bind to exactly one of these via
# ``signature_mode``; validation never infers a scheme from request headers.
SIGNATURE_MODES = frozenset(
    {
        "github",
        "gitlab",
        "gitlab_standard",
        "hindsight",
        "svix",
        "generic_v2",
        "generic_v1",
    }
)


def _header(request: "web.Request", name: str) -> str:
    return (
        request.headers.get(name, "")
        or request.headers.get(name.lower(), "")
        or request.headers.get(name.upper(), "")
    )


class WebhookAuthMixin:
    """Validate webhook signatures while preserving the adapter seam."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._v1_signature_warned: set[str] = set()

    def _verify_github(self, request, body, secret) -> bool:
        gh_sig = _header(request, "X-Hub-Signature-256")
        if not gh_sig:
            return False
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return _hmac_str_equal(gh_sig, expected)

    def _verify_gitlab(self, request, secret) -> bool:
        gl_token = _header(request, "X-Gitlab-Token")
        if not gl_token:
            return False
        return _hmac_str_equal(gl_token, secret)

    def _verify_hindsight(self, request, body, secret) -> bool:
        # Hindsight (issue #80327, fix by sg-shag in #80329): same
        # sha256=<hex> raw-body contract as GitHub, different header name.
        hs_sig = _header(request, "X-Hindsight-Signature")
        if not hs_sig:
            return False
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        return _hmac_str_equal(hs_sig, expected)

    def _verify_generic_v2(self, request, body, secret) -> bool:
        v2_sig = _header(request, "X-Webhook-Signature-V2")
        if not v2_sig:
            return False
        v2_timestamp = _header(request, "X-Webhook-Timestamp")
        if not v2_timestamp:
            logger.warning(
                "[webhook] Route '%s' sent X-Webhook-Signature-V2 with "
                "no X-Webhook-Timestamp — rejecting rather than "
                "falling back to legacy V1",
                request.match_info.get("route_name", ""),
            )
            return False
        try:
            ts = int(v2_timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > DEFAULT_REPLAY_TOLERANCE_SECONDS:
            logger.warning(
                "[webhook] Route '%s' generic HMAC V2 timestamp outside replay window",
                request.match_info.get("route_name", ""),
            )
            return False
        signed_content = v2_timestamp.encode() + b"." + body
        expected_v2 = hmac.new(
            secret.encode(), signed_content, hashlib.sha256
        ).hexdigest()
        return _hmac_str_equal(v2_sig, expected_v2)

    def _verify_generic_v1(self, request, body, secret) -> bool:
        generic_sig = _header(request, "X-Webhook-Signature")
        if not generic_sig:
            return False
        expected = hmac.new(
            secret.encode(), body, hashlib.sha256
        ).hexdigest()
        route_name = request.match_info.get("route_name", "")
        if route_name not in self._v1_signature_warned:
            self._v1_signature_warned.add(route_name)
            logger.warning(
                "[webhook] Route '%s' uses legacy body-only HMAC (no "
                "timestamp), which is vulnerable to replay attacks. Add "
                "an 'X-Webhook-Timestamp' header and switch to "
                "'X-Webhook-Signature-V2' (HMAC-SHA256 of "
                "'<timestamp>.<body>').",
                route_name,
            )
        return _hmac_str_equal(generic_sig, expected)

    def _validate_signature(
        self,
        request: "web.Request",
        body: bytes,
        secret: str,
        signature_mode: str = "generic_v2",
    ) -> bool:
        """Validate webhook signature for an explicit provider scheme.

        The configured ``signature_mode`` decides the scheme; a route never
        infers a weaker scheme from attacker-controlled headers. An unknown or
        empty mode fails closed.
        """
        if signature_mode == "github":
            return self._verify_github(request, body, secret)
        if signature_mode == "gitlab":
            return self._verify_gitlab(request, secret)
        if signature_mode == "gitlab_standard":
            # Standard Webhooks (issue #47451, fix by HwangJohn in #47849):
            # webhook-id / webhook-timestamp / webhook-signature headers,
            # signed content "{id}.{timestamp}.{raw_body}" with a
            # v1,<base64-hmac-sha256> signature — same wire format as Svix.
            return self._validate_svix_signature(
                body=body,
                secret=secret,
                msg_id=_header(request, "webhook-id"),
                timestamp=_header(request, "webhook-timestamp"),
                signature_header=_header(request, "webhook-signature"),
            )
        if signature_mode == "hindsight":
            return self._verify_hindsight(request, body, secret)
        if signature_mode == "svix":
            svix_id = _header(request, "svix-id")
            svix_timestamp = _header(request, "svix-timestamp")
            svix_signature = _header(request, "svix-signature")
            return self._validate_svix_signature(
                body=body,
                secret=secret,
                msg_id=svix_id,
                timestamp=svix_timestamp,
                signature_header=svix_signature,
            )
        if signature_mode == "generic_v1":
            return self._verify_generic_v1(request, body, secret)
        if signature_mode == "generic_v2":
            return self._verify_generic_v2(request, body, secret)

        logger.warning(
            "[webhook] Route '%s' has unsupported signature_mode %r",
            request.match_info.get("route_name", ""),
            signature_mode,
        )
        return False

    def _validate_svix_signature(
        self,
        body: bytes,
        secret: str,
        msg_id: str,
        timestamp: str,
        signature_header: str,
        tolerance_seconds: int = DEFAULT_REPLAY_TOLERANCE_SECONDS,
    ) -> bool:
        """Validate Svix-compatible signatures used by AgentMail webhooks."""
        if not (msg_id and timestamp and signature_header and secret):
            return False

        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > tolerance_seconds:
            logger.warning("[webhook] Svix signature timestamp outside replay window")
            return False

        if secret.startswith("whsec_"):
            encoded_secret = secret.removeprefix("whsec_")
            try:
                key = base64.b64decode(encoded_secret, validate=True)
            except (binascii.Error, ValueError):
                logger.debug("[webhook] Invalid whsec_ Svix signing secret")
                return False
        else:
            # Be permissive for providers that document Svix-style headers but
            # hand out raw shared secrets rather than whsec_ base64 secrets.
            logger.debug("[webhook] Validating Svix-style signature with raw secret")
            key = secret.encode()

        signed_content = msg_id.encode() + b"." + timestamp.encode() + b"." + body
        expected = base64.b64encode(
            hmac.new(key, signed_content, hashlib.sha256).digest()
        ).decode()

        # Svix can send multiple signatures separated by spaces during secret
        # rotation. Each entry is formatted as "vN,<base64>".
        for part in signature_header.split():
            try:
                version, signature = part.split(",", 1)
            except ValueError:
                continue
            if version == "v1" and _hmac_str_equal(signature, expected):
                return True
        return False
