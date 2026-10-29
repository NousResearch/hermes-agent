"""Model probes share chat's TLS policy, with scoped redirect credentials."""
from __future__ import annotations

from contextlib import contextmanager

# Wall-clock cap on one probe (headers + whole body), on top of the per-op httpx timeouts.
TOTAL_DEADLINE_S = 15.0


def resolve_verify(base_url: str = ""):
    from agent.ssl_verify import resolve_httpx_verify

    settings = {}
    if base_url:
        try:
            from hermes_cli.config import get_custom_provider_tls_settings

            settings = get_custom_provider_tls_settings(base_url)
        except Exception:
            pass  # Metadata remains optional when config discovery fails.
    return resolve_httpx_verify(
        ca_bundle=settings.get("ssl_ca_cert"),
        ssl_verify=settings.get("ssl_verify"),
        base_url=base_url,
    )


@contextmanager
def stream(url: str, *, headers=None, params=None, timeout=10.0, verify=None):
    import httpx
    from agent.deadline import DeadlineExpired, run_bounded_sync
    from hermes_cli.urllib_security import url_origin

    origin = url_origin(url)
    private_headers = {name.lower() for name in headers or {} if name.lower() not in {"accept", "user-agent"}}
    private_headers.update({"authorization", "cookie", "proxy-authorization"})

    def scope_credentials(request):
        if url_origin(str(request.url)) != origin:
            for name in private_headers:
                request.headers.pop(name, None)

    if isinstance(timeout, tuple):
        connect, read = timeout
        timeout = httpx.Timeout(read, connect=connect)
    with httpx.Client(
        verify=resolve_verify() if verify is None else verify,
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [scope_credentials]},
    ) as client:
        def fetch():
            response = client.send(client.build_request("GET", url, headers=headers, params=params), stream=True)
            if response.is_success:  # a rejection closes on its headers, never waits for a body
                response.read()
            return response

        # Probes run inside requests (session.create resolves the context window through them), and
        # httpx timeouts bound each socket op, not the exchange: a body trickling in under the read
        # timeout, or a hung DNS lookup, would hold the caller for as long as it lasts. Leaving the
        # client block on expiry closes the pool under the abandoned worker.
        bounded = run_bounded_sync(fetch, TOTAL_DEADLINE_S, label=f"model metadata {origin}")
        if bounded.timed_out:
            raise DeadlineExpired(f"model metadata {origin}", TOTAL_DEADLINE_S)
        try:
            yield bounded.value
        finally:
            bounded.value.close()

def get(url: str, *, headers=None, params=None, timeout=10.0, verify=None):
    with stream(url, headers=headers, params=params, timeout=timeout, verify=verify) as response:
        return response
