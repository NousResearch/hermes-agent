"""Anthropic-on-Vertex adapter for Hermes Agent.

Constructs an ``anthropic.AnthropicVertex`` client using ADC-minted OAuth
tokens and Vertex-hosted Anthropic Claude model endpoints. Mirrors
:func:`agent.anthropic_adapter.build_anthropic_bedrock_client` in shape:
the Anthropic SDK ships a purpose-built ``AnthropicVertex`` class that
handles the URL construction (``publishers/anthropic/models/<model>:
rawPredict``) and the ``Authorization: Bearer <access-token>`` header
attachment natively — we just pass ``project_id`` + ``region`` + a
short-lived Google credentials object and the SDK does the rest.

Auth flows through :mod:`agent.vertex_adapter` — the same code path that
Gemini-on-Vertex uses. Everything the operator has to configure is
already there:

* ``GOOGLE_APPLICATION_CREDENTIALS`` / ``VERTEX_CREDENTIALS_PATH`` for a
  service-account JSON.
* Application Default Credentials via ``gcloud auth application-default
  login`` or the GCE metadata server (VM SA).
* ``VERTEX_PROJECT_ID`` env var or ``vertex.project_id`` in
  ``config.yaml`` to override the credentials' embedded project.
* ``VERTEX_REGION`` env var or ``vertex.region`` in ``config.yaml``
  (defaults to ``global``).

The two Vertex code paths — Gemini via OpenAI-compat endpoint, and
Anthropic Claude via native Anthropic Messages API — share credentials,
project/region config, and OAuth token cache. Adding this provider does
not introduce a second authentication surface.

Requires: ``pip install 'anthropic>=0.39.0'`` (for
``anthropic.AnthropicVertex``) plus ``google-auth``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from agent.vertex_adapter import (
    DEFAULT_REGION,
    _resolve_credentials_path,
    _resolve_project_override,
    _resolve_region,
    google,
)

logger = logging.getLogger(__name__)


def _get_anthropic_sdk():
    """Return the ``anthropic`` SDK module, importing lazily.

    Delegates to :mod:`agent.anthropic_adapter` so the SDK is imported at
    most once per process regardless of which adapter first triggers it.
    Returns ``None`` when the SDK is not installed (e.g. minimal install
    without the ``anthropic`` extra) — the callers surface a friendly
    ImportError with the install command.
    """
    from agent.anthropic_adapter import _get_anthropic_sdk as _get_from_adapter

    return _get_from_adapter()


def _resolve_google_credentials():
    """Return a ``google.auth.credentials.Credentials`` for Vertex Anthropic.

    Mirrors :func:`agent.vertex_adapter.get_vertex_credentials` up to the
    credentials-object step — but returns the credentials directly rather
    than a materialized access token. AnthropicVertex refreshes the token
    itself via the Google auth transport on each call, so handing it the
    Credentials object gives it the same short-lived-token guarantees as
    Gemini-on-Vertex while removing the token-plumbing complexity from
    this module.

    Returns ``(creds, project_id)`` or ``(None, None)`` on failure.
    """
    if google is None:
        logger.warning(
            "google-auth package not installed. Cannot use Anthropic on Vertex."
        )
        return None, None

    from google.oauth2 import service_account

    resolved_path = _resolve_credentials_path(None)

    try:
        if resolved_path:
            creds = service_account.Credentials.from_service_account_file(
                resolved_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            project_id = creds.project_id
        else:
            creds, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        override_project = _resolve_project_override()
        if override_project:
            project_id = override_project

        return creds, project_id
    except Exception as exc:
        logger.error("Failed to resolve Anthropic-Vertex credentials: %s", exc)
        return None, None


def get_anthropic_vertex_config(
    region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(project_id, region)`` for the Anthropic-on-Vertex call.

    Does NOT include an access token — the token minting happens inside
    the AnthropicVertex client via the Credentials object we hand it in
    :func:`build_anthropic_vertex_client`. This function exists so
    :mod:`hermes_cli.runtime_provider` can compute the display base URL
    and stage the routing dict without touching the SDK.

    Returns ``(None, None)`` when credentials cannot be resolved (missing
    google-auth, no ADC, no service-account JSON, etc.).
    """
    _creds, project_id = _resolve_google_credentials()
    if not project_id:
        return None, None
    return project_id, _resolve_region(region)


def build_anthropic_vertex_base_url(
    project_id: str, region: str = DEFAULT_REGION
) -> str:
    """Build a display-only base URL for the Anthropic-on-Vertex endpoint.

    Not consumed by the AnthropicVertex SDK (which builds its own URLs
    from project_id + region), but hermes-agent's runtime dict and
    provider auto-detection paths key off ``base_url`` — a URL shape
    that matches Vertex's actual publisher-model routes keeps the
    diagnostics honest and lets the ``aiplatform.googleapis.com`` host
    heuristic in :mod:`agent.usage_pricing` recognize the endpoint for
    billing attribution.

    The ``global`` location uses the bare ``aiplatform.googleapis.com``
    host; regional locations use ``{region}-aiplatform.googleapis.com``.
    Path is ``/v1/projects/{project}/locations/{region}/publishers/
    anthropic`` — one level above the ``:rawPredict`` endpoints the SDK
    actually hits, so log lines quoting the base URL point at "Anthropic
    on Vertex, this project, this region" without leaking the specific
    model name.
    """
    host = (
        "aiplatform.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{project_id}"
        f"/locations/{region}/publishers/anthropic"
    )


def build_anthropic_vertex_client(
    project_id: str,
    region: str = DEFAULT_REGION,
    timeout: Optional[float] = None,
) -> Any:
    """Create an ``AnthropicVertex`` client for Claude models on Vertex AI.

    Uses the Anthropic SDK's native ``AnthropicVertex`` adapter, which
    provides full Claude feature parity: prompt caching, thinking budgets,
    adaptive thinking, fast mode — the same set Bedrock-hosted Claude
    gets. The SDK constructs the correct
    ``.../publishers/anthropic/models/<model>:rawPredict`` URL and
    attaches ``Authorization: Bearer <access-token>`` per request; we
    only supply project_id, region, and a Google credentials object.

    Attaches the common Anthropic beta headers as client-level defaults
    so Vertex-hosted Claude models get the same enhanced features as
    native Anthropic (prompt caching, fine-grained tool streaming,
    interleaved thinking). Does NOT attach ``context-1m-2025-08-07``.
    Anthropic's March-2026 GA rollout made 1M context automatic on
    Vertex-hosted Opus 4.6+ / Sonnet 4.6+ — the beta header is accepted
    but ignored on that wire (see
    https://claude.com/blog/1m-context-ga). Sending a no-op header is
    misleading, so we omit it here; the 1M window is available with no
    per-call configuration on the Vertex path. Operators who want it
    attached for parity with a mixed native-Anthropic backend can pass
    ``default_headers={"anthropic-beta": "context-1m-2025-08-07,..."}``
    at construction — same effect either way on Vertex, harmless.

    Auth uses the Google credentials chain: service-account JSON via
    ``GOOGLE_APPLICATION_CREDENTIALS`` / ``VERTEX_CREDENTIALS_PATH``,
    Application Default Credentials via ``gcloud auth
    application-default login``, or the GCE metadata server on Google
    Cloud VMs. Refresh happens automatically inside the SDK.

    Raises ``ImportError`` when the ``anthropic`` package is not
    installed or is too old (``AnthropicVertex`` was added in 0.30.0).
    """
    _anthropic_sdk = _get_anthropic_sdk()
    if _anthropic_sdk is None:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic-on-Vertex "
            "provider. Install it with: pip install 'anthropic>=0.39.0'"
        )
    if not hasattr(_anthropic_sdk, "AnthropicVertex"):
        raise ImportError(
            "anthropic.AnthropicVertex not available. "
            "Upgrade with: pip install 'anthropic>=0.39.0'"
        )

    creds, resolved_project_id = _resolve_google_credentials()
    if creds is None or resolved_project_id is None:
        raise RuntimeError(
            "Anthropic-on-Vertex credentials could not be resolved. Vertex "
            "uses OAuth2 (not a static API key): provide a service-account "
            "JSON via GOOGLE_APPLICATION_CREDENTIALS (or "
            "VERTEX_CREDENTIALS_PATH) in ~/.hermes/.env, or run 'gcloud auth "
            "application-default login' for ADC. Set the GCP project under "
            "vertex.project_id in config.yaml if it isn't embedded in the "
            "credentials."
        )

    # Explicit ``project_id`` from the caller wins over the credentials'
    # embedded project. Matches the ``vertex_adapter._resolve_project_override``
    # precedence used by Gemini-on-Vertex.
    final_project_id = project_id or resolved_project_id

    from httpx import Timeout

    from agent.anthropic_adapter import _COMMON_BETAS

    read_timeout = (
        timeout
        if (isinstance(timeout, (int, float)) and timeout > 0)
        else 900.0
    )

    return _anthropic_sdk.AnthropicVertex(
        project_id=final_project_id,
        region=region,
        credentials=creds,
        timeout=Timeout(timeout=float(read_timeout), connect=10.0),
        # Delegate retry to hermes's outer loop (honors Retry-After); the SDK
        # default max_retries=2 ignores it and double-retries. Matches Bedrock.
        max_retries=0,
        default_headers={"anthropic-beta": ",".join(_COMMON_BETAS)},
    )


def has_anthropic_vertex_credentials() -> bool:
    """Fast check for whether Anthropic-on-Vertex credentials are configured.

    No network calls, no SDK import — safe for provider auto-detection
    and setup-status display. True when either a service-account JSON
    path is resolvable, or an explicit project ID is configured (env or
    config.yaml, implying ADC is intended).
    """
    if _resolve_credentials_path(None):
        return True
    if _resolve_project_override():
        return True
    return False


def is_anthropic_vertex_model(model_id: str) -> bool:
    """Return True if a Vertex model ID should route via the AnthropicVertex SDK.

    Used by :mod:`hermes_cli.runtime_provider` to dispatch the shared
    ``vertex`` provider onto the correct wire protocol at runtime:

    * ``anthropic/…`` → ``anthropic_messages`` mode via the
      ``AnthropicVertex`` SDK (this classifier).
    * everything else (``google/gemini-*``, and any future partner
      family Vertex Model Garden adds) → ``chat_completions`` mode via
      Vertex's OpenAI-compat aggregator.

    The vendor prefix is **required**. Contrast with
    :func:`agent.bedrock_adapter.is_anthropic_bedrock_model`, which
    matches ``anthropic.claude`` (and the regional variants) — Bedrock
    grew up as an Anthropic-only surface and its bare-``claude-*``
    acceptance is a legacy shortcut. Vertex Model Garden is
    multi-vendor from day one (Anthropic + Google + more coming), so
    the classifier accepts *only* the fully-qualified
    ``anthropic/<model>`` form. This surfaces a loud, actionable error
    when someone writes ``default: "claude-opus-4-8"`` under
    ``provider: vertex``: the request goes down the aggregator path
    and Vertex 404s with "publisher google — model claude-opus-4-8 not
    found", telling the user exactly what to fix.

    Case-insensitive. Whitespace is stripped.
    """
    if not isinstance(model_id, str):
        return False
    m = model_id.strip().lower()
    if not m:
        return False
    return m.startswith("anthropic/")
