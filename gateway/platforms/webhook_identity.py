"""Per-route delivery identity: where a webhook route reads the id that makes a provider retry a duplicate.

Without a source, the id comes from a fixed header chain (``X-GitHub-Delivery``, ``svix-id``,
``webhook-id``, ``X-Request-ID``), else a fresh one per request. A provider whose stable id lives
somewhere else (a different header, or a field in the body) gets a fresh id per retry, so every retry
runs again. A route's ``identity`` key names the source:

* ``header:<Name>`` reads that request header. The sender chose the key, so a second delivery with the
  same key but a different body is a conflict, answered 409.
* ``body:<json-pointer>`` (RFC 6901, e.g. ``body:/data/event/event_id``) reads a string or integer
  field of the JSON body. A provider may re-serialize the body on retry, so the body is not compared.
* ``none`` gives every request its own id: nothing is deduped.

A source that finds no value in a request falls back to the header chain.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional

HEADER, BODY, NONE = "header", "body", "none"

# Headers that change on every attempt or that the transport owns: an identity read from them dedupes
# nothing (or everything), so ``identity`` refuses them.
_UNSTABLE_HEADERS = frozenset({
    "authorization", "content-length", "content-type", "host", "connection", "transfer-encoding",
    "x-webhook-timestamp", "x-webhook-signature", "x-webhook-signature-v2", "x-hub-signature",
    "x-hub-signature-256", "x-gitlab-token", "linear-signature", "webhook-timestamp", "webhook-signature",
    "svix-timestamp", "svix-signature", "x-forwarded-for", "x-real-ip"})
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")


@dataclass(frozen=True)
class IdentitySource:
    kind: str  # HEADER, BODY or NONE
    value: str = ""  # the header name, or the JSON pointer

    @property
    def compares_body(self) -> bool:
        """Whether a repeat of this identity must carry the same body (409 otherwise)."""
        return self.kind == HEADER


def parse_identity(spec: object) -> tuple[Optional[IdentitySource], Optional[str]]:
    """``(source, error)`` for a route's ``identity`` value; ``(None, None)`` when it is unset."""
    if spec is None:
        return None, None
    if not isinstance(spec, str) or not spec.strip():
        return None, f"identity must be 'header:<Name>', 'body:/json/pointer' or 'none', not {spec!r}"
    spec = spec.strip()
    if spec == NONE:
        return IdentitySource(NONE), None
    kind, sep, value = spec.partition(":")
    if kind == HEADER and sep:
        if not _HEADER_NAME_RE.match(value):
            return None, f"identity header {value!r} is not a valid header name"
        if value.lower() in _UNSTABLE_HEADERS:
            return None, (f"identity header {value!r} changes on every attempt or belongs to the transport; "
                          "name the header that carries the provider's delivery id")
        return IdentitySource(HEADER, value), None
    if kind == BODY and sep:
        if pointer_tokens(value) is None:
            return None, (f"identity body pointer {value!r} is not an RFC 6901 JSON pointer "
                          "such as /data/event/event_id")
        return IdentitySource(BODY, value), None
    return None, f"identity must be 'header:<Name>', 'body:/json/pointer' or 'none', not {spec!r}"


_BAD_ESCAPE_RE = re.compile(r"~(?![01])")
_ARRAY_INDEX_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


def pointer_tokens(pointer: str) -> Optional[list[str]]:
    """The reference tokens of an RFC 6901 pointer, or None when it is malformed. The whole-document
    pointer ``""`` is refused too: an identity must name one field."""
    if not pointer.startswith("/") or _BAD_ESCAPE_RE.search(pointer):
        return None
    return [raw.replace("~1", "/").replace("~0", "~") for raw in pointer[1:].split("/")]


def _pointer_value(payload: Any, pointer: str) -> Optional[str]:
    node = payload
    for token in pointer_tokens(pointer) or ():
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list) and _ARRAY_INDEX_RE.match(token) and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    if isinstance(node, bool) or not isinstance(node, (str, int)):
        return None
    text = str(node)
    return text or None


def resolve_identity(source: IdentitySource, headers, payload: Any) -> Optional[str]:
    """The delivery id *source* names in this request; None when the request does not carry one."""
    if source.kind == HEADER:
        return headers.get(source.value) or None
    if source.kind == BODY:
        return _pointer_value(payload, source.value)
    return None


def body_digest(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()
