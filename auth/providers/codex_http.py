"""codex http protocol/lifecycle responsibilities."""

from __future__ import annotations
import time
from contextlib import suppress
from typing import Any, Optional, Tuple
from auth.constants import _codex_err, httpx


def _ssl_interop_hint(exc: BaseException) -> str:
    """Actionable hint for device-login transport errors that look like TLS middlebox interference.

    OpenSSL 3.5+ advertises post-quantum hybrid groups (e.g. X25519MLKEM768) by default, and some
    intercepting middleboxes reject the resulting larger TLS 1.3 ClientHello — while curl, using a
    different TLS stack, still works, so the failure masquerades as a Codex outage (#106384).
    httpx wraps the ``ssl.SSLError`` in a ``ConnectError``/``ConnectTimeout`` whose text usually
    repeats the OpenSSL message; the cause chain is checked too in case it doesn't.
    """
    from auth.providers.codex import _SSL_TROUBLE_MARKERS
    import ssl

    chain = (exc, exc.__cause__, exc.__context__)
    if not any(
        isinstance(err, ssl.SSLError)
        or any(marker in str(err) for marker in _SSL_TROUBLE_MARKERS)
        for err in chain
        if err is not None
    ):
        return ""
    return (
        " This looks like a TLS handshake failure rather than a Codex outage: some networks reject"
        " the larger TLS 1.3 ClientHello that OpenSSL 3.5+ sends by default (post-quantum hybrid"
        " groups). Workaround: point OPENSSL_CONF at a config restricting Groups to classic curves"
        " (x25519:secp256r1:secp384r1:x448), or test with TLS 1.2 — see the Codex note in"
        " https://hermes-agent.nousresearch.com/docs/integrations/providers"
    )


def _is_transient_transport_error(exc: BaseException) -> bool:
    """True when *exc* is transport-level (connection/TLS/socket) and safe to retry.

    httpx raises ``httpx.TransportError`` subclasses wrapping the original ``ssl``/``socket``
    error as ``__cause__``, so both spellings count. Anything else (decode errors, bugs) is
    not a network blip and must surface immediately.
    """
    err: Optional[BaseException] = exc
    seen: set[int] = set()
    while err is not None and id(err) not in seen:
        if isinstance(err, (httpx.TransportError, OSError)):
            return True
        seen.add(id(err))
        err = err.__cause__
    return False


def _codex_login_post(
    url: str, *, failure: Tuple[str, str], **kwargs: Any
) -> "httpx.Response":
    """One 15s POST for the device-login flow; transport errors become ``_codex_err(*failure)``.

    A transient transport blip (a dropped connection mid-flow) is retried twice with a small
    linear backoff before failing: losing the token exchange to a single SSL EOF wastes a
    device-code approval the user already completed in the browser (#114610).
    """
    attempt, attempts = 1, 3
    while True:
        try:
            with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
                return client.post(url, **kwargs)
        except Exception as exc:
            if attempt == attempts or not _is_transient_transport_error(exc):
                raise _codex_err(
                    f"{failure[0]}: {exc}{_ssl_interop_hint(exc)}", failure[1]
                ) from exc
            time.sleep(attempt)
            attempt += 1


_CODEX_AUTH_BODY_MAX_BYTES = 1024 * 1024


def _cap_codex_response_body(response: "httpx.Response") -> None:
    """httpx response hook: refuse to buffer an auth body above ``_CODEX_AUTH_BODY_MAX_BYTES``.

    Runs before ``client.post()`` reads the body, so a hostile or broken endpoint/proxy answering
    200 with megabytes of "JSON" is cut off at the cap instead of being fully buffered and parsed
    (#55253). Same cap for every status: error bodies are small diagnostics too.
    """
    from auth.providers.codex import _capped_byte_stream_class

    response.stream = _capped_byte_stream_class()(response)


def _codex_http_client(**kwargs: Any) -> "httpx.Client":
    """Build an ``httpx.Client`` for Codex OAuth/probe endpoints with Happy-Eyeballs racing and a
    1 MiB response-body cap (``_cap_codex_response_body``).

    A host advertising AAAA records but blackholing IPv6 makes each serial connect eat the full
    timeout before IPv4 is tried (same failure mode as the chat transport). Best-effort: if the
    racing backend can't be installed (mocked client in tests), serial connect behavior remains.

    Same broken-IPv6 failure mode as the chat transport (#13834): a host that advertises AAAA records but
    blackholes IPv6 makes each serial connect attempt eat the full connect timeout before IPv4 is tried, so
    token refresh / device login / usage probes time out where the official Codex CLI (which races families
    per RFC 8305) works.
    """
    client = httpx.Client(
        event_hooks={"response": [_cap_codex_response_body]}, **kwargs
    )
    with suppress(Exception):
        from agent.process_bootstrap import enable_happy_eyeballs_on_client

        enable_happy_eyeballs_on_client(client)
    return client
