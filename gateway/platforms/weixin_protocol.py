"""Wire metadata and network diagnostics shared by the iLink transport."""

import re
import socket
import ssl
from contextlib import contextmanager
from contextvars import ContextVar

_REQUEST_METADATA = ContextVar("weixin_request_metadata", default=None)


class WeixinCDNClientError(RuntimeError):
    """A rejected CDN upload must not be retried."""


@contextmanager
def bind_request_metadata(token: str, bot_agent: str, route_tag: str | None):
    scope = _REQUEST_METADATA.set((token, bot_agent, route_tag))
    try:
        yield
    finally:
        _REQUEST_METADATA.reset(scope)


def get_request_metadata(token: str | None):
    value = _REQUEST_METADATA.get()
    return value[1:] if value and value[0] == token else (None, None)


def normalize_route_tag(raw) -> str | None:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        return None
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    return str(raw).strip() or None


def sanitize_bot_agent(raw) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "Hermes"
    products, pending = [], None
    tokens = iter(raw.split())
    for token in tokens:
        if token.startswith("("):
            while not token.endswith(")"):
                following = next(tokens, None)
                if following is None:
                    break
                token += " " + following
            if pending:
                comment = token[1:-1] if token.endswith(")") else ""
                products.append(f"{pending} ({comment})" if re.fullmatch(r"[\x20-\x27\x2a-\x7e]{1,64}", comment) else pending)
                pending = None
            continue
        if pending:
            products.append(pending)
        pending = token if re.fullmatch(r"[A-Za-z0-9_.-]{1,32}/[A-Za-z0-9_.+-]{1,32}", token) else None
    if pending:
        products.append(pending)
    result = []
    for product in products:
        if len(" ".join([*result, product])) > 256:
            break
        result.append(product)
    return " ".join(result) or "Hermes"


def classify_network_error(error: BaseException) -> str:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ssl.SSLError):
            return "tls"
        if isinstance(error, socket.gaierror):
            return "dns"
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, ConnectionError):
            return "tcp"
        error = getattr(error, "os_error", None) or error.__cause__ or error.__context__
    return "unknown"
