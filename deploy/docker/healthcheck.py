#!/usr/bin/env python3
"""Container health check for the NOVA control plane.

Liveness, not readiness. ``/platform/v1/health`` is the one route that answers before a
tenant bundle is healthy, so a failure here means the process is down — not that someone's
configuration is wrong. Restarting a container because a tenant's policy file has a typo
would be a loop, not a recovery.

**401 and 403 count as healthy, and that is the point.** With ``--behind-tls-proxy`` set,
the control plane deliberately stops trusting loopback callers (a proxy terminating TLS on
the same host makes every forwarded request look local, so trusting local would mean
trusting the internet). This probe is such a caller. A 401 therefore proves two things at
once: the server is accepting connections, and its authentication layer is refusing an
unauthenticated request. Handing the container a bearer token to get a 200 instead would
put a live credential in the container's environment to learn strictly less.

Written in Python rather than shelling out to curl so the image needs no apt layer at all.
Standard library only, which is all this image is guaranteed to have.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 4.0

#: Statuses that prove the process is serving. See the module docstring on 401/403.
ALIVE_STATUSES = frozenset({200, 401, 403})


def main() -> int:
    host = os.environ.get("NOVA_BIND_HOST", "127.0.0.1")
    # A server bound to every interface is still reachable on loopback; probing loopback
    # keeps the check inside the container rather than going out and back.
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = os.environ.get("NOVA_BIND_PORT", "8787")
    scheme = "https" if os.environ.get("NOVA_TLS_CERT") else "http"
    url = f"{scheme}://{host}:{port}/platform/v1/health"

    context = None
    if scheme == "https":
        import ssl

        # The certificate is this process's own, presented to itself over loopback; its
        # name almost certainly does not match "127.0.0.1". Verifying it here would fail a
        # healthy server. Nothing is trusted as a result — the check only asks whether the
        # socket answers.
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=context) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:  # noqa: BLE001 — any failure to reach it is unhealthy
        print(f"health: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if status in ALIVE_STATUSES:
        return 0
    print(f"health: HTTP {status}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
