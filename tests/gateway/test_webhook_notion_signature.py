import hashlib
import hmac

from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


class DummyRequest:
    def __init__(self, headers):
        self.headers = CIMultiDict(headers)


def _adapter():
    return WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "secret": "global-secret",
                "routes": {},
            },
        )
    )


def _notion_signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_validate_notion_signature_header_accepts_valid_signature():
    adapter = _adapter()
    secret = "notion-verification-token"
    body = b'{"type":"comment.created","entity":{"id":"page-id"}}'
    request = DummyRequest({"X-Notion-Signature": _notion_signature(secret, body)})

    assert adapter._validate_signature(request, body, secret) is True


def test_validate_notion_signature_header_rejects_tampered_body():
    adapter = _adapter()
    secret = "notion-verification-token"
    original_body = b'{"type":"comment.created","entity":{"id":"page-id"}}'
    tampered_body = b'{"type":"comment.updated","entity":{"id":"page-id"}}'
    request = DummyRequest({"X-Notion-Signature": _notion_signature(secret, original_body)})

    assert adapter._validate_signature(request, tampered_body, secret) is False


def test_validate_notion_signature_header_is_case_insensitive():
    adapter = _adapter()
    secret = "notion-verification-token"
    body = b'{"type":"page.content_updated","entity":{"id":"page-id"}}'
    request = DummyRequest({"x-notion-signature": _notion_signature(secret, body)})

    assert adapter._validate_signature(request, body, secret) is True
