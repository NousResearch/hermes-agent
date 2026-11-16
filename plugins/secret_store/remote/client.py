"""HTTP client for the secrets plane's agent route, ``/v1/secrets/self``.

The secrets plane sits beside the config plane on the connector and authenticates the agent the
same way, so this client shares the config plane's whole transport: the destination policy (https,
or http to loopback only; no redirect followed with a credential), the bearer (the Nous access token
from the profile's ``auth.json``, or the on-prem IdP client-credentials token) and the instance id
header. Stdlib only, like the config client: it runs inside ``load_hermes_dotenv``.

No value ever appears in a message this module builds: a failure names the route, the HTTP status
and the plane's error code.
"""
from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Any, Optional

from plugins.config_backends.remote import client as plane_client
from plugins.config_backends.remote.credentials import plane_token
from plugins.config_backends.remote.destination import checked_url, display_url

URL_ENV = "HERMES_SECRETS_REMOTE_URL"
ROUTE = "/v1/secrets/self"

Response = plane_client.Response
TransportError = plane_client.TransportError


def base_url() -> str:
    """The secrets plane origin: ``HERMES_SECRETS_REMOTE_URL``, else the config plane's URL."""
    return (os.environ.get(URL_ENV, "").strip() or plane_client.base_url()).rstrip("/")


def shown_url() -> str:
    return display_url(base_url())


def instance_id() -> str:
    return plane_client.instance_id()


def bearer(home: Path) -> str:
    """``home``'s bearer, resolved only once the secrets plane destination passed the policy."""
    checked_url(base_url(), env_name=URL_ENV)
    return plane_token(home)


def request(method: str, home: Path, profile: str, *, etag: Optional[str] = None,
            body: Optional[dict[str, Any]] = None, token: Optional[str] = None) -> Response:
    """One call to ``/v1/secrets/self?profile=<profile>`` as ``home``'s agent (bearer + instance id;
    *token* is the bearer when the caller resolved it already)."""
    plane = checked_url(base_url(), env_name=URL_ENV)
    url = f"{plane}{ROUTE}?{urllib.parse.urlencode({'profile': profile})}"
    if token is None:
        token = plane_token(Path(home))
    return plane_client.send(method, url, token=token, etag=etag, body=body,
                             headers={"X-Hermes-Instance-Id": instance_id()})
