"""Model probes share chat's TLS policy, with scoped redirect credentials."""
from __future__ import annotations

from contextlib import contextmanager


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
        with client.stream("GET", url, headers=headers, params=params) as response:
            yield response


def get(url: str, *, headers=None, params=None, timeout=10.0, verify=None):
    with stream(url, headers=headers, params=params, timeout=timeout, verify=verify) as response:
        response.read()
        return response
