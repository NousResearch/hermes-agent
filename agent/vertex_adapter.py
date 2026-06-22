"""Google Cloud Vertex AI adapter for Hermes Agent.

Provides native integration with Google Cloud Vertex AI, enabling access to
Gemini models through GCP's enterprise-grade API surface. Unlike the Gemini
AI Studio path (``gemini_native_adapter.py``), Vertex AI uses OAuth2/ADC
credentials instead of API keys and routes through regional or global
aiplatform.googleapis.com endpoints.

Why this exists
---------------
Vertex AI provides capabilities unavailable in AI Studio:
  - **Service account auth**: IAM-based credential management, no API keys
  - **VPC-SC**: Vertex endpoints are VPC Service Controls compatible
  - **Regional data residency**: Route to eu-west1, asia-northeast1, etc.
  - **Fine-tuned models**: Access customer-tuned model checkpoints via
    ``projects/{PROJECT}/locations/{LOC}/models/{MODEL_ID}``
  - **Enterprise quotas**: Higher RPM/TPM limits on paid GCP projects

Architecture follows ``bedrock_adapter.py``:
  - All Vertex-specific logic isolated in this module
  - Message format reuses ``gemini_native_adapter`` converters (same schema)
  - Responses normalized to OpenAI-compatible SimpleNamespace
  - Lazy imports — ``google-auth`` only loaded when Vertex provider is active

Credential resolution order
---------------------------
1. ``GOOGLE_APPLICATION_CREDENTIALS`` env var → service account JSON file
2. ``GOOGLE_CLOUD_ACCESS_TOKEN`` env var → raw Bearer token (CI/CD use)
3. Application Default Credentials (ADC) via ``google.auth.default()``
   - Looks in ~/.config/gcloud/application_default_credentials.json
   - Works with ``gcloud auth application-default login``
   - In GCE/GKE/Cloud Run: picks up attached service account automatically

Environment variables
---------------------
  GOOGLE_CLOUD_PROJECT      GCP project ID (required if not in ADC metadata)
  GOOGLE_CLOUD_REGION       Vertex region, default "us-central1"
  GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON
  GOOGLE_CLOUD_ACCESS_TOKEN Raw Bearer token (overrides all other auth)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import httpx

from agent.gemini_native_adapter import (
    GeminiAPIError,
    _iter_sse_events,
    bare_gemini_model_id,
    build_gemini_request,
    translate_gemini_response,
    translate_stream_event,
    _GeminiStreamChunk,
    _GeminiChatCompletions,
    _AsyncGeminiChatCompletions,
    _GeminiChatNamespace,
    _AsyncGeminiChatNamespace,
)

logger = logging.getLogger(__name__)

DEFAULT_VERTEX_REGION = "us-central1"
DEFAULT_VERTEX_PUBLISHER = "google"

# Token refresh margin: refresh the OAuth token this many seconds before it expires.
_TOKEN_REFRESH_MARGIN_SECS = 120


# ---------------------------------------------------------------------------
# GCP project / region resolution
# ---------------------------------------------------------------------------

def resolve_vertex_project(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return the GCP project ID to use for Vertex AI API calls.

    Priority:
      1. GOOGLE_CLOUD_PROJECT env var
      2. GCLOUD_PROJECT env var (legacy alias)
      3. Project embedded in ADC credentials (GOOGLE_APPLICATION_CREDENTIALS)
      4. Project from ``google.auth.default()`` quota_project_id
    """
    env = env if env is not None else os.environ
    for var in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"):
        project = env.get(var, "").strip()
        if project:
            return project

    # Try reading from service account file directly (no network)
    sa_file = env.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if sa_file and os.path.isfile(sa_file):
        try:
            with open(sa_file) as f:
                data = json.load(f)
            if "project_id" in data:
                return data["project_id"]
        except Exception:
            pass

    # Fall back to google-auth library which can query GCE metadata
    try:
        import google.auth
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass

    return None


def resolve_vertex_region(env: Optional[Dict[str, str]] = None) -> str:
    """Return the Vertex AI region to use.

    Priority:
      1. GOOGLE_CLOUD_REGION env var
      2. VERTEX_AI_REGION env var (alternate)
      3. CLOUDSDK_AI_REGION env var (gcloud config)
      4. DEFAULT_VERTEX_REGION ("us-central1")
    """
    env = env if env is not None else os.environ
    for var in ("GOOGLE_CLOUD_REGION", "VERTEX_AI_REGION", "CLOUDSDK_AI_REGION"):
        region = env.get(var, "").strip()
        if region:
            return region
    return DEFAULT_VERTEX_REGION


def has_vertex_credentials(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True if any GCP credential source is available.

    Checks env vars first (fast, no I/O), then falls back to google-auth's
    ADC resolver which covers GCE/GKE/Cloud Run instance metadata.
    """
    env = env if env is not None else os.environ

    # Raw access token (CI/CD pipelines)
    if env.get("GOOGLE_CLOUD_ACCESS_TOKEN", "").strip():
        return True

    # Explicit service account file
    sa_file = env.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if sa_file and os.path.isfile(sa_file):
        return True

    # ADC file created by gcloud auth application-default login
    adc_path = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if os.path.isfile(adc_path):
        return True

    # GCE / GKE / Cloud Run / Cloud Functions — instance metadata
    try:
        import google.auth
        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# OAuth2 token management
# ---------------------------------------------------------------------------

class _TokenCache:
    """Thread-safe OAuth2 Bearer token cache with automatic refresh."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expiry: float = 0.0  # unix timestamp
        self._creds: Any = None    # google.auth credentials object

    def _load_creds(self) -> None:
        """Load credentials from the environment (called once, lazily)."""
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError:
            raise ImportError(
                "The 'google-auth' package is required for Vertex AI. "
                "Install it: pip install google-auth\n"
                "Or: pip install -e '.[vertex]'"
            )
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._creds = creds

    def get_token(self) -> str:
        """Return a valid Bearer token, refreshing if needed."""
        # Fast path: raw token from env var (e.g. CI/CD `gcloud auth print-access-token`)
        raw = os.environ.get("GOOGLE_CLOUD_ACCESS_TOKEN", "").strip()
        if raw:
            return raw

        with self._lock:
            now = time.time()
            if self._token and now < self._expiry - _TOKEN_REFRESH_MARGIN_SECS:
                return self._token

            if self._creds is None:
                self._load_creds()

            try:
                import google.auth.transport.requests
                request = google.auth.transport.requests.Request()
                self._creds.refresh(request)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to refresh Google OAuth2 credentials for Vertex AI: {exc}\n"
                    "Run 'gcloud auth application-default login' or set "
                    "GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file."
                ) from exc

            self._token = self._creds.token
            expiry = getattr(self._creds, "expiry", None)
            if expiry is not None:
                self._expiry = expiry.timestamp()
            else:
                # Default: assume 1 hour validity (Google's standard)
                self._expiry = now + 3600

            return self._token


_token_cache = _TokenCache()


def get_vertex_bearer_token() -> str:
    """Return a valid GCP Bearer token for Vertex AI API calls."""
    return _token_cache.get_token()


def reset_token_cache() -> None:
    """Clear the token cache. Used in tests and credential rotation."""
    global _token_cache
    _token_cache = _TokenCache()


# ---------------------------------------------------------------------------
# Vertex AI endpoint construction
# ---------------------------------------------------------------------------

def build_vertex_url(
    *,
    project: str,
    region: str,
    model: str,
    publisher: str = DEFAULT_VERTEX_PUBLISHER,
    streaming: bool = False,
) -> str:
    """Construct a Vertex AI generateContent (or streamGenerateContent) URL.

    URL format:
        https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/
            locations/{REGION}/publishers/{PUBLISHER}/models/{MODEL}:generateContent

    For fine-tuned endpoints (``projects/{PROJECT}/locations/{LOC}/models/{ID}``),
    pass the full resource path as ``model`` — it is used verbatim.
    """
    host = f"{region}-aiplatform.googleapis.com"
    method = "streamGenerateContent" if streaming else "generateContent"

    if model.startswith("projects/"):
        # Full resource path for fine-tuned / endpoint models
        return f"https://{host}/v1/{model}:{method}"

    bare = bare_gemini_model_id(model)
    resource = (
        f"projects/{project}/locations/{region}"
        f"/publishers/{publisher}/models/{bare}"
    )
    return f"https://{host}/v1/{resource}:{method}"


# ---------------------------------------------------------------------------
# VertexNativeClient — OpenAI-SDK-compatible facade
# ---------------------------------------------------------------------------

class VertexNativeClient:
    """Minimal OpenAI-SDK-compatible facade over Google Cloud Vertex AI.

    Reuses the Gemini native adapter's message converters since the REST
    schema is identical — only auth and endpoint differ.
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        region: Optional[str] = None,
        publisher: str = DEFAULT_VERTEX_PUBLISHER,
        timeout: Any = None,
        http_client: Optional[httpx.Client] = None,
        **_: Any,
    ) -> None:
        self.project = project or resolve_vertex_project() or ""
        if not self.project:
            raise RuntimeError(
                "Vertex AI requires a GCP project ID. Set GOOGLE_CLOUD_PROJECT "
                "in your environment, or configure a service account that includes "
                "the project field."
            )
        self.region = region or resolve_vertex_region()
        self.publisher = publisher
        self.chat = _GeminiChatNamespace(self)
        self.is_closed = False
        self._http = http_client or httpx.Client(
            timeout=timeout
            or httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=30.0)
        )

    def close(self) -> None:
        self.is_closed = True
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _headers(self) -> Dict[str, str]:
        token = get_vertex_bearer_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-agent (vertex-ai)",
            "x-goog-user-project": self.project,
        }

    @staticmethod
    def _advance_stream_iterator(
        iterator: Iterator[_GeminiStreamChunk],
    ) -> tuple[bool, Optional[_GeminiStreamChunk]]:
        try:
            return False, next(iterator)
        except StopIteration:
            return True, None

    def _create_chat_completion(
        self,
        *,
        model: str = "gemini-2.5-flash",
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Any = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Any = None,
        **_: Any,
    ) -> Any:
        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = (
                extra_body.get("thinking_config")
                or extra_body.get("thinkingConfig")
            )

        request = build_gemini_request(
            messages=messages or [],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking_config=thinking_config,
        )

        if stream:
            return self._stream_completion(model=model, request=request, timeout=timeout)

        url = build_vertex_url(
            project=self.project,
            region=self.region,
            model=model,
            publisher=self.publisher,
            streaming=False,
        )
        response = self._http.post(
            url, json=request, headers=self._headers(), timeout=timeout
        )
        if response.status_code != 200:
            raise _vertex_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiAPIError(
                f"Invalid JSON from Vertex AI: {exc}",
                code="vertex_invalid_json",
                status_code=response.status_code,
                response=response,
            ) from exc
        return translate_gemini_response(payload, model=bare_gemini_model_id(model))

    def _stream_completion(
        self, *, model: str, request: Dict[str, Any], timeout: Any = None
    ) -> Iterator[_GeminiStreamChunk]:
        url = build_vertex_url(
            project=self.project,
            region=self.region,
            model=model,
            publisher=self.publisher,
            streaming=True,
        )
        # Vertex streaming needs alt=sse query param
        url = f"{url}?alt=sse"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"
        bare = bare_gemini_model_id(model)

        def _generator() -> Iterator[_GeminiStreamChunk]:
            try:
                with self._http.stream(
                    "POST", url, json=request, headers=stream_headers, timeout=timeout
                ) as response:
                    if response.status_code != 200:
                        response.read()
                        raise _vertex_http_error(response)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        for chunk in translate_stream_event(event, bare, tool_call_indices):
                            yield chunk
            except httpx.HTTPError as exc:
                raise GeminiAPIError(
                    f"Vertex AI streaming request failed: {exc}",
                    code="vertex_stream_error",
                ) from exc

        return _generator()


# ---------------------------------------------------------------------------
# Error construction
# ---------------------------------------------------------------------------

def _vertex_http_error(response: httpx.Response) -> GeminiAPIError:
    """Build a GeminiAPIError from a non-200 Vertex AI HTTP response."""
    status = response.status_code
    try:
        body = response.json()
    except Exception:
        body = {}

    error_obj = body.get("error", {}) if isinstance(body, dict) else {}
    message = error_obj.get("message") or response.text or f"HTTP {status}"
    code = error_obj.get("status", "UNKNOWN")

    retry_after: Optional[float] = None
    if status == 429:
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                retry_after = float(ra)
            except (TypeError, ValueError):
                pass

    return GeminiAPIError(
        f"Vertex AI error {status} ({code}): {message}",
        code=f"vertex_{status}",
        status_code=status,
        response=response,
        retry_after=retry_after,
        details=error_obj,
    )


# ---------------------------------------------------------------------------
# Convenience: build a VertexNativeClient from agent config
# ---------------------------------------------------------------------------

_vertex_client_cache: Dict[str, VertexNativeClient] = {}
_vertex_client_lock = threading.Lock()


def get_vertex_client(
    project: Optional[str] = None,
    region: Optional[str] = None,
) -> VertexNativeClient:
    """Return a cached VertexNativeClient for the given project/region pair."""
    resolved_project = project or resolve_vertex_project() or ""
    resolved_region = region or resolve_vertex_region()
    cache_key = f"{resolved_project}:{resolved_region}"

    with _vertex_client_lock:
        if cache_key not in _vertex_client_cache:
            _vertex_client_cache[cache_key] = VertexNativeClient(
                project=resolved_project,
                region=resolved_region,
            )
        return _vertex_client_cache[cache_key]


def reset_client_cache() -> None:
    """Clear cached Vertex clients. Used in tests and credential rotation."""
    with _vertex_client_lock:
        for client in _vertex_client_cache.values():
            try:
                client.close()
            except Exception:
                pass
        _vertex_client_cache.clear()
