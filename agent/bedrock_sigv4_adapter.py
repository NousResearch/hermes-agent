"""AWS SigV4 request signing for the OpenAI-compatible Bedrock Mantle endpoint.

Amazon Bedrock exposes an OpenAI-compatible surface at
``https://bedrock-mantle.{region}.api.aws`` (the "Mantle" engine). Unlike the
native ``bedrock-runtime`` Converse path (which Hermes drives through the AWS
SDK in ``agent/bedrock_adapter.py``), Mantle speaks the OpenAI Chat Completions
and Responses wire formats — so Hermes talks to it with the ordinary OpenAI
Python SDK client.

What it does NOT accept is a static bearer ``api_key``: requests must be signed
with AWS SigV4 (service name ``bedrock``) using the standard botocore
credential chain (env vars, ``~/.aws/credentials``, SSO, instance roles, …).

The OpenAI SDK computes the ``Authorization`` header at construction time from a
static ``api_key`` string, which cannot carry a per-request SigV4 signature
(the signature is computed over the final method + canonical URI + body +
timestamp). To sign per request we install an ``httpx.Auth`` on the underlying
``httpx.Client`` and pass that client to the SDK via ``http_client=...``. The
auth's ``auth_flow`` sees the fully built request (method, URL, headers, body)
and overwrites the SDK's placeholder ``Authorization`` with the SigV4-signed
headers.

This mirrors the per-request bearer-hook pattern in
``agent/azure_identity_adapter.build_bearer_http_client`` (Entra ID on Foundry),
adapted from a bearer token to AWS SigV4.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# AWS service name used for the SigV4 signing scope on the Bedrock endpoints.
BEDROCK_SIGV4_SERVICE = "bedrock"

# Hostname fragment that identifies a Bedrock Mantle base URL. Mantle hosts look
# like ``bedrock-mantle.us-east-2.api.aws``.
MANTLE_HOST_FRAGMENT = "bedrock-mantle."


def is_mantle_base_url(base_url: str) -> bool:
    """Return True if ``base_url`` points at a Bedrock Mantle endpoint."""
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return MANTLE_HOST_FRAGMENT in host


def region_from_base_url(base_url: str) -> Optional[str]:
    """Extract the AWS region from a Mantle base URL.

    ``https://bedrock-mantle.us-east-2.api.aws/v1`` -> ``us-east-2``. Returns
    None when the host does not match the expected Mantle shape.
    """
    host = (urlparse(base_url).hostname or "").lower()
    # bedrock-mantle.<region>.api.aws
    parts = host.split(".")
    if len(parts) >= 4 and parts[0] == "bedrock-mantle" and parts[-2:] == ["api", "aws"]:
        return parts[1]
    return None


def build_sigv4_http_client(
    region: str,
    *,
    service: str = BEDROCK_SIGV4_SERVICE,
    base_client: Any = None,
    **httpx_kwargs: Any,
) -> Any:
    """Return an ``httpx.Client`` that SigV4-signs every outbound request.

    ``region`` is the AWS region for the signing scope (parsed from the Mantle
    base URL by the caller). ``service`` is the SigV4 service name (``bedrock``).

    ``base_client`` lets the caller pass an already-configured ``httpx.Client``
    (e.g. the keepalive client Hermes builds in
    ``run_agent._build_keepalive_http_client``) so its transport / proxy /
    socket options are preserved — we only attach the SigV4 auth to it. When
    ``base_client`` is given, ``httpx_kwargs`` are ignored and the existing
    client is returned with ``.auth`` set.

    Raises ``ImportError`` if httpx is unavailable (it ships transitively with
    the openai SDK, so in practice it is always present) and ``RuntimeError`` if
    no AWS credentials can be resolved from the botocore chain.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover — httpx ships with openai SDK
        raise ImportError(
            "httpx is required for AWS SigV4 signing on the Bedrock Mantle "
            "endpoint. It is normally a transitive dependency of the openai SDK."
        ) from exc

    auth = _build_sigv4_auth(httpx, region=region, service=service)

    if base_client is not None:
        # Preserve the caller's transport/proxy/socket-options; just sign.
        base_client.auth = auth
        return base_client

    return httpx.Client(auth=auth, **httpx_kwargs)


def _resolve_aws_credentials() -> Any:
    """Resolve AWS credentials from the standard botocore chain."""
    try:
        import botocore.session
    except ImportError as exc:  # pragma: no cover — boto3 is a hard dep for bedrock
        raise RuntimeError(
            "botocore is required for AWS SigV4 signing but is not installed."
        ) from exc
    creds = botocore.session.get_session().get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found for Bedrock Mantle SigV4 signing. "
            "Configure the AWS credential chain (env vars, ~/.aws/credentials, "
            "SSO, or an instance role)."
        )
    return creds


def _build_sigv4_auth(httpx_mod: Any, *, region: str, service: str) -> Any:
    """Construct an ``httpx.Auth`` subclass instance that SigV4-signs requests.

    The class is defined inside this factory (rather than at module scope) so
    the module imports cleanly when httpx is absent — ``httpx.Auth`` is only
    referenced once httpx has been successfully imported by the caller.
    """
    credentials = _resolve_aws_credentials()

    class _SigV4HttpxAuth(httpx_mod.Auth):
        """``httpx.Auth`` that signs requests with AWS SigV4 (botocore)."""

        requires_request_body = True
        requires_response_body = False

        def auth_flow(self, request: Any):
            """Sign ``request`` in place with SigV4, then yield it.

            Two Mantle-specific concerns are handled here:

            1. Route rewrite. Mantle serves the OpenAI Responses API (used by
               GPT-5.5 / GPT-5.4) under ``/openai/v1/responses`` while the Chat
               Completions API (used by the open models) lives under
               ``/v1/chat/completions``. Hermes binds a single ``base_url``
               (``.../v1``) per provider, so the OpenAI SDK emits
               ``/v1/responses`` for the Responses path. We rewrite that to
               ``/openai/v1/responses`` before signing so both model families
               work from one base URL. (``/v1/chat/completions`` is left
               untouched.)
            2. Signing. SigV4 puts the signature in the ``Authorization``
               header itself, so we overwrite whatever placeholder bearer value
               the OpenAI SDK set from the dummy ``api_key``.
            """
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            # 1) Route rewrite for the Responses API.
            path = request.url.path
            if path.endswith("/v1/responses") and "/openai/" not in path:
                request.url = request.url.copy_with(
                    path=path.replace("/v1/responses", "/openai/v1/responses")
                )

            # 2) SigV4 signing over the (possibly rewritten) final request.
            frozen = credentials.get_frozen_credentials()
            body = request.content or b""
            content_type = request.headers.get("Content-Type", "application/json")

            aws_request = AWSRequest(
                method=request.method,
                url=str(request.url),
                data=body,
                headers={"Content-Type": content_type},
            )
            SigV4Auth(frozen, service, region).add_auth(aws_request)

            # Copy the SigV4-signed headers onto the outbound httpx request,
            # overwriting the SDK's placeholder Authorization header.
            for key, value in aws_request.headers.items():
                request.headers[key] = value

            yield request

    return _SigV4HttpxAuth()


# Default region for Mantle when none is configured. us-east-1 hosts Claude
# models (Opus 4.8, Fable 5, Haiku 4.5) alongside the ~40 open models.
# GPT-5.5 / GPT-5.4 live only in us-east-2 and are routed there via
# mantle_region_for_model(). Region resolution order is handled by
# resolve_mantle_region().
DEFAULT_MANTLE_REGION = "us-east-1"

# Models hosted exclusively in us-east-2 (not available in the default region).
_MANTLE_USEAST2_MODELS: frozenset[str] = frozenset({
    "openai.gpt-5.5",
    "openai.gpt-5.4",
    "openai.gpt-5.5-2026-04-23",
    "openai.gpt-5.4-2026-03-05",
})


def mantle_region_for_model(model_id: str | None) -> str:
    """Return the Mantle AWS region that serves a given model ID.

    GPT-5.x models are only deployed in us-east-2. All other models
    (Claude, open models) use the default region.
    """
    if model_id and model_id.strip().lower() in _MANTLE_USEAST2_MODELS:
        return "us-east-2"
    return resolve_mantle_region()


def resolve_mantle_region() -> str:
    """Resolve the AWS region to use for the Mantle endpoint.

    Order: AWS_REGION / AWS_DEFAULT_REGION env vars → botocore configured
    region (``~/.aws/config`` / SSO profile) → DEFAULT_MANTLE_REGION.
    """
    import os

    for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    try:
        import botocore.session

        region = botocore.session.get_session().get_config_variable("region")
        if region:
            return str(region)
    except Exception:
        pass
    return DEFAULT_MANTLE_REGION


def mantle_base_url(region: str | None = None) -> str:
    """Return the Mantle Chat Completions base URL for ``region``.

    The OpenAI SDK appends ``/chat/completions`` and ``/responses`` to this
    base. The SigV4 auth rewrites the Responses path to ``/openai/v1`` at send
    time (see ``_build_sigv4_auth``), so a single ``/v1`` base serves both the
    open models (chat/completions) and GPT-5.5/5.4 (responses).
    """
    region = region or resolve_mantle_region()
    return f"https://bedrock-mantle.{region}.api.aws/v1"


def mantle_model_ids_or_none(
    region: str | None = None,
    *,
    timeout: float = 8.0,
) -> Optional[list[str]]:
    """Live-discover the Mantle model catalog via a SigV4-signed GET /v1/models.

    Returns a list of model-id strings, or None on any failure (so callers fall
    back to the curated list). Mirrors ``bedrock_model_ids_or_none`` but hits
    Mantle's OpenAI-compatible ``/v1/models`` endpoint instead of the AWS SDK
    control plane.
    """
    region = region or resolve_mantle_region()
    try:
        import json

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        creds = _resolve_aws_credentials().get_frozen_credentials()
        url = f"https://bedrock-mantle.{region}.api.aws/v1/models"
        aws_request = AWSRequest(method="GET", url=url, headers={})
        SigV4Auth(creds, BEDROCK_SIGV4_SERVICE, region).add_auth(aws_request)

        import urllib.request

        req = urllib.request.Request(url, method="GET")
        for key, value in aws_request.headers.items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        ids = [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
        return ids or None
    except Exception as exc:
        logger.debug("mantle_model_ids_or_none(%s) failed: %s", region, exc)
        return None

