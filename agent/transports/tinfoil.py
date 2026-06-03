"""Tinfoil EHBP transport — end-to-end encrypted enclave inference.

The TinfoilTransport extends ChatCompletionsTransport with the Tinfoil
``SecureClient``, which provides:

* Enclave attestation verification (fetched from atc.tinfoil.sh)
* TLS certificate pinning (public-key fingerprint from attestation)
* HPKE / EHBP request-body encryption
* Transparent re-verification on TLS certificate rotation

When the ``tinfoil`` SDK is not installed or attestation fails, the
transport falls back to plain TLS — the same behaviour as the current
``ChatCompletionsTransport`` path.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.types import NormalizedResponse

logger = logging.getLogger(__name__)

_TINFOIL_AVAILABLE: bool = False
try:
    from tinfoil import SecureClient

    _TINFOIL_AVAILABLE = True
except ImportError:
    SecureClient = None  # type: ignore[assignment]

_TINFOIL_REPO = "tinfoilsh/confidential-model-router"


class TinfoilSecureUnavailableError(RuntimeError):
    """Raised when the secure HPKE transport cannot be established and
    ``tinfoil.require_secure`` is enabled.

    Signals a deliberate fail-closed refusal to downgrade to plain TLS —
    the agent must not send inference payloads to the gateway unencrypted
    when the operator has demanded enclave-verified transport.
    """


class TinfoilTransport(ChatCompletionsTransport):
    """Transport for api_mode='tinfoil_ehbp'.

    Messages and tools are already in OpenAI format — message conversion,
    tool conversion, and response normalization are inherited from
    ``ChatCompletionsTransport``.

    The secure HTTP client is built by ``build_secure_openai_client`` and
    must be injected into the ``openai.OpenAI`` constructor at client
    construction time (see ``agent/agent_runtime_helpers.py``).  The
    transport caches the ``SecureClient`` instance per session so the
    attestation handshake only happens once.
    """

    @property
    def api_mode(self) -> str:
        return "tinfoil_ehbp"

    def __init__(self) -> None:
        super().__init__()
        self._secure_client: Any = None
        self._verification_document: Any = None

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        """Build OpenAI ``chat.completions.create()`` kwargs.

        Delegates to ``ChatCompletionsTransport.build_kwargs`` — the
        kwargs shape is identical; security lives in the ``http_client``.
        """
        return super().build_kwargs(model, messages, tools, **params)

    def build_secure_client(self, api_key: str) -> Any:
        """Build (or return cached) ``SecureClient`` for attestation.

        Returns ``None`` when the ``tinfoil`` SDK is not available.
        """
        if self._secure_client is not None:
            return self._secure_client
        if not _TINFOIL_AVAILABLE:
            return None
        secure = None
        try:
            secure = SecureClient(  # type: ignore[operator]
                enclave="",
                repo=_TINFOIL_REPO,
            )
        except Exception as exc:
            logger.warning(
                "Tinfoil SecureClient init failed, falling back to plain TLS: %s",
                exc,
            )
            return None
        try:
            self._verification_document = secure.get_verification_document()
        except Exception as exc:
            logger.debug("Tinfoil verification document fetch failed: %s", exc)
        if not secure.enclave:
            logger.warning("Tinfoil SecureClient: enclave not populated after init")
            return None
        self._secure_client = secure
        return secure

    def is_secure(self) -> bool:
        """``True`` when the secure HPKE transport is active."""
        return self._secure_client is not None

    def build_secure_openai_client(
        self,
        api_key: str,
        base_url: str,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        require_secure: bool = False,
    ) -> Any:
        """Build an ``openai.OpenAI`` client with a Tinfoil-verified transport.

        Falls back to a plain ``openai.OpenAI`` client when the SDK is
        unavailable or attestation fails — UNLESS ``require_secure`` is set,
        in which case the inability to establish the verified HPKE transport
        raises :class:`TinfoilSecureUnavailableError` instead of silently
        downgrading to plain TLS (fail-closed).
        """
        import openai

        secure = self.build_secure_client(api_key)
        if secure is not None:
            try:
                http_client = secure.make_secure_http_client()
                effective_base_url = urljoin(f"https://{secure.enclave}/", "v1/")
                logger.info(
                    "Tinfoil EHBP secure transport established for %s "
                    "(enclave %s)",
                    base_url,
                    secure.enclave,
                )
                return openai.OpenAI(
                    api_key=api_key,
                    base_url=effective_base_url,
                    http_client=http_client,
                    timeout=timeout,
                    default_headers=default_headers,
                )
            except Exception as exc:
                # Secure client built but the HTTP transport failed. Reset
                # the cached client so ``is_secure()`` reports the true
                # (insecure) state rather than a half-initialized handshake.
                self._secure_client = None
                if require_secure:
                    raise TinfoilSecureUnavailableError(
                        "Tinfoil secure HPKE transport could not be created "
                        f"and tinfoil.require_secure is enabled: {exc}"
                    ) from exc
                logger.warning(
                    "Tinfoil secure HTTP client creation failed, "
                    "falling back to plain TLS: %s",
                    exc,
                )

        if require_secure:
            raise TinfoilSecureUnavailableError(
                "Tinfoil secure HPKE transport is unavailable (SDK missing or "
                "attestation failed) and tinfoil.require_secure is enabled — "
                "refusing to downgrade to plain TLS. Install the 'tinfoil' "
                "extra (pip install hermes-agent[tinfoil]) or set "
                "tinfoil.require_secure: false to allow plain-TLS fallback."
            )

        logger.info(
            "Tinfoil EHBP transport unavailable, using plain TLS for %s",
            base_url,
        )
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
        )

    def get_verification_document(self) -> Any:
        """Return the attestation verification document, or None."""
        doc = self._verification_document
        if doc is not None:
            return doc
        if self._secure_client is not None:
            try:
                doc = self._secure_client.get_verification_document()
                self._verification_document = doc
            except Exception as exc:
                logger.debug("Tinfoil verification document fetch failed: %s", exc)
        return self._verification_document

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize — identical to ChatCompletionsTransport."""
        return super().normalize_response(response, **kwargs)


from agent.transports import register_transport  # noqa: E402

register_transport("tinfoil_ehbp", TinfoilTransport)