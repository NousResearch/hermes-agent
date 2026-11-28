"""OpenAI Codex OAuth upstream adapter for the subscription proxy."""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from agent.auxiliary_client import resolve_provider_client
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL
from hermes_cli.codex_models import DEFAULT_CODEX_MODELS
from hermes_cli.proxy.adapters.base import (
    LocalAdapterResponse,
    UpstreamAdapter,
    UpstreamCredential,
)

_POOL_PROVIDER = "openai-codex"
_ALLOWED_PATHS = frozenset({"/chat/completions", "/models"})
logger = logging.getLogger(__name__)


class OpenAICodexAdapter(UpstreamAdapter):
    """Proxy upstream backed by Hermes-managed ChatGPT/Codex OAuth credentials."""

    auth_hint = "hermes auth add openai-codex"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex OAuth"

    @property
    def loopback_only(self) -> bool:
        return True

    @property
    def requires_downstream_auth(self) -> bool:
        return True

    @property
    def allowed_paths(self) -> frozenset[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            pool = self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No OpenAI Codex OAuth credentials found. Run `hermes auth add openai-codex` first."
                )
            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available OpenAI Codex OAuth credentials found. Re-authenticate with "
                    "`hermes auth add openai-codex`."
                )
            self._pool = pool
            return self._credential_from_entry(entry)

    def handle_local_request(
        self, rel_path: str, body: bytes
    ) -> Optional[LocalAdapterResponse]:
        if rel_path == "/models":
            models = [{"id": model, "object": "model", "owned_by": "openai-codex"}
                      for model in DEFAULT_CODEX_MODELS]
            return LocalAdapterResponse(
                status=200,
                body=json.dumps({"object": "list", "data": models}, separators=(",", ":")).encode("utf-8"),
            )
        if rel_path != "/chat/completions":
            return None
        try:
            payload = json.loads(body or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json_error(400, "Request body must be valid JSON.", "invalid_json")
        if not isinstance(payload, dict):
            return self._json_error(400, "Request body must be a JSON object.", "invalid_request")
        if payload.get("stream"):
            return self._json_error(
                400,
                "The OpenAI Codex subscription adapter currently supports non-streaming requests only.",
                "streaming_not_supported",
            )
        model = str(payload.get("model") or "").strip()
        messages = payload.get("messages")
        if not model or not isinstance(messages, list):
            return self._json_error(400, "model and messages are required.", "invalid_request")

        kwargs = dict(payload)
        kwargs.pop("stream", None)
        client = None
        try:
            from openai import APIError

            client, resolved_model = resolve_provider_client(_POOL_PROVIDER, model)
            if client is None:
                raise RuntimeError("OpenAI Codex OAuth credentials are unavailable.")
            kwargs["model"] = resolved_model or model
            response = client.chat.completions.create(**kwargs)
            return LocalAdapterResponse(status=200, body=self._response_bytes(response))
        except ImportError as exc:
            missing_module = str(exc.name or "")
            if missing_module != "openai" and not missing_module.startswith("openai."):
                raise
            return self._inference_error(exc)
        except APIError as exc:
            return self._inference_error(exc)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._inference_error(exc)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _inference_error(self, exc: BaseException) -> LocalAdapterResponse:
        logger.warning("Codex proxy inference failed: %s", type(exc).__name__)
        return self._json_error(502, "Codex inference failed.", "codex_inference_failed")

    @staticmethod
    def _response_bytes(response) -> bytes:
        if isinstance(response, bytes):
            return response
        if isinstance(response, dict):
            payload = response
        elif callable(getattr(response, "model_dump", None)):
            payload = response.model_dump(mode="json")
        elif hasattr(response, "__dict__"):
            payload = OpenAICodexAdapter._jsonable(vars(response))
        else:
            raise TypeError("Codex completion returned an unsupported response type")
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _jsonable(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): OpenAICodexAdapter._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [OpenAICodexAdapter._jsonable(item) for item in value]
        if hasattr(value, "__dict__"):
            return OpenAICodexAdapter._jsonable(vars(value))
        raise TypeError(f"Codex completion contains unsupported value type {type(value).__name__}")

    @staticmethod
    def _json_error(status: int, message: str, code: str) -> LocalAdapterResponse:
        body = {"error": {"message": message, "type": code, "code": code}}
        return LocalAdapterResponse(
            status=status,
            body=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        )

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Codex credential pool load failed: %s", type(exc).__name__)
            return None

    @staticmethod
    def _credential_from_entry(entry: PooledCredential) -> UpstreamCredential:
        bearer = str(getattr(entry, "runtime_api_key", None) or entry.access_token or "").strip()
        if not bearer:
            raise RuntimeError("OpenAI Codex credential did not contain a usable access token.")
        base_url = str(
            getattr(entry, "runtime_base_url", None) or entry.base_url or DEFAULT_CODEX_BASE_URL
        ).strip().rstrip("/")
        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url or DEFAULT_CODEX_BASE_URL,
            expires_at=entry.expires_at,
        )


__all__ = ["OpenAICodexAdapter"]
