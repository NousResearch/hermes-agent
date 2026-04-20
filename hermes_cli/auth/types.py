"""Typed credential containers for Hermes' auth resolvers (F-M7).

The four credential-resolver entry points
(``resolve_nous_runtime_credentials``, ``resolve_codex_runtime_credentials``,
``resolve_api_key_provider_credentials``, ``resolve_external_process_provider_credentials``)
historically returned ``Dict[str, Any]``. That lost the shape of the
payload, so every caller indexed the dict by string and type checkers
saw ``Any`` flowing through the model-routing, gateway, companion,
and setup-wizard paths.

Rather than switch the runtime representation to ``@dataclass`` — which
would require every caller to change from ``creds["api_key"]`` to
``creds.api_key`` and churn ~30 files — we keep the dict shape and use
``TypedDict`` to attach compile-time types. Type checkers enforce the
keys + value types; runtime access patterns stay identical.

Each resolver returns its own TypedDict so the intent of the payload
is clear from the signature alone. The per-provider status helpers
(``get_*_auth_status``) remain typed as ``Dict[str, Any]`` for now —
their shape legitimately varies by provider state, and formalizing
them gets no extra leverage.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class NousCredentials(TypedDict):
    """Runtime credentials for Nous Portal inference.

    Returned by ``resolve_nous_runtime_credentials``. ``source`` is
    ``"cache"`` when the cached agent_key satisfied the requested TTL,
    otherwise ``"portal"`` (a fresh mint round-tripped through the
    portal).
    """

    provider: str
    base_url: str
    api_key: str
    key_id: Optional[str]
    expires_at: Optional[str]
    expires_in: int
    source: str


class CodexCredentials(TypedDict):
    """Runtime credentials for OpenAI Codex (ChatGPT-plan).

    Returned by ``resolve_codex_runtime_credentials``. The access token
    is long-lived by OpenAI standards (minutes to hours); the resolver
    refreshes on expiry via the Hermes-owned auth store.
    """

    provider: str
    base_url: str
    api_key: str
    source: str
    last_refresh: Optional[str]
    auth_mode: str


class ApiKeyCredentials(TypedDict):
    """Runtime credentials for env-var-backed API-key providers.

    Returned by ``resolve_api_key_provider_credentials``. ``source``
    is the env var name the key came from (``OPENROUTER_API_KEY``,
    ``ZAI_API_KEY``, …), or ``"default"`` when no env var matched —
    callers use it to tell the user where to look if the key looks
    wrong.
    """

    provider: str
    api_key: str
    base_url: str
    source: str


class ExternalProcessCredentials(TypedDict):
    """Runtime credentials for subprocess-backed providers (Copilot ACP).

    Returned by ``resolve_external_process_provider_credentials``. The
    ``api_key`` slot is a sentinel (``"copilot-acp"``) because the
    subprocess handles its own auth — what the runtime actually needs
    is the resolved ``command`` path and the ``base_url`` to connect to.
    """

    provider: str
    api_key: str
    base_url: str
    command: str
    args: list[str]
    source: str
