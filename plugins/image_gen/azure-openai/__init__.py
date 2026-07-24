"""Azure OpenAI image generation backend.

The provider is registered through the existing image-generation plugin edge.
This module owns Azure's setup metadata and profile-aware configuration
resolution; native SDK request handling is implemented separately.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
)


DEFAULT_AZURE_IMAGE_API_VERSION = "2024-10-21"
DEFAULT_API_VERSION = DEFAULT_AZURE_IMAGE_API_VERSION

_AZURE_IMAGE_KEY_SETUP_ACTION = (
    "Configure AZURE_OPENAI_IMAGE_KEY in Image Generation via hermes tools, "
    "or select another provider."
)
_AZURE_ENDPOINT_SETUP_ACTION = (
    "Configure image_gen.azure_openai.endpoint in Image Generation via "
    "hermes tools, or select another provider."
)
_AZURE_DEPLOYMENT_SETUP_ACTION = (
    "Configure image_gen.azure_openai.deployment_name in Image Generation "
    "via hermes tools, or select another provider."
)
_FOUNDRY_KEYLESS_SETUP_ACTION = (
    "Configure AZURE_OPENAI_IMAGE_KEY, or install the optional azure-identity "
    "package to use Foundry Entra ID authentication."
)
_AZURE_ENDPOINT_CORRECTION = (
    "Use https://<resource>.services.ai.azure.com/openai/v1 for Foundry, "
    "or https://<resource>.openai.azure.com for direct Azure OpenAI."
)

_ENDPOINT_CONFIG_KEY = "image_gen.azure_openai.endpoint"
_DEPLOYMENT_CONFIG_KEY = "image_gen.azure_openai.deployment_name"

EndpointFamily = Literal["azure-openai", "foundry-v1"]


@dataclass(frozen=True)
class AzureImageSettings:
    """Complete, normalized settings required by an Azure image client."""

    endpoint: str
    api_key: Optional[str]
    deployment_name: str
    api_version: Optional[str]
    endpoint_family: EndpointFamily


@dataclass(frozen=True)
class AzureImageConfigurationError:
    """A missing or invalid Azure setting without retaining credentials."""

    field: str
    error_type: str
    message: str
    canonical_key: Optional[str] = None
    setup_action: Optional[str] = None

    @property
    def config_key(self) -> Optional[str]:
        """Backward-friendly alias for callers building provider responses."""
        return self.canonical_key

    @property
    def missing_field(self) -> str:
        """Explicit alias used by readiness and error-mapping callers."""
        return self.field


def _trimmed(value: object) -> Optional[str]:
    """Return a stripped non-empty string, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _azure_config(config: Mapping[str, object]) -> Mapping[str, object]:
    image_gen = config.get("image_gen")
    if not isinstance(image_gen, Mapping):
        return {}
    azure = image_gen.get("azure_openai")
    return azure if isinstance(azure, Mapping) else {}


def normalize_azure_image_endpoint(
    endpoint: str,
) -> Optional[tuple[str, EndpointFamily]]:
    """Normalize and classify supported Azure image endpoint families.

    Foundry's OpenAI-compatible endpoint is already a complete ``base_url``;
    direct Azure OpenAI resources instead use the SDK's ``azure_endpoint``.
    Query strings, fragments, embedded credentials, non-default ports, and
    unexpected paths are rejected so routing cannot silently switch families.
    """
    try:
        parsed = urlsplit(endpoint.strip())
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        return None

    hostname = parsed.hostname.lower().rstrip(".")
    path = parsed.path.rstrip("/")

    if hostname.endswith(".services.ai.azure.com") and path.lower() == "/openai/v1":
        return f"https://{hostname}/openai/v1", "foundry-v1"
    if hostname.endswith(".openai.azure.com") and not path:
        return f"https://{hostname}", "azure-openai"
    return None


def normalize_azure_image_setup_endpoint(endpoint: str) -> str:
    """Validate and canonicalize an endpoint entered through provider setup."""
    normalized = normalize_azure_image_endpoint(endpoint)
    if normalized is None:
        raise ValueError(_AZURE_ENDPOINT_CORRECTION)
    return normalized[0]


def resolve_azure_image_settings(
    config: Mapping[str, object],
    environ: Mapping[str, str],
) -> Union[AzureImageSettings, AzureImageConfigurationError]:
    """Resolve endpoint-family-aware settings without mutating either input."""
    azure = _azure_config(config)
    api_key = _trimmed(environ.get("AZURE_OPENAI_IMAGE_KEY"))
    endpoint = _trimmed(azure.get("endpoint")) or _trimmed(
        environ.get("AZURE_OPENAI_ENDPOINT")
    )
    deployment_name = _trimmed(azure.get("deployment_name")) or _trimmed(
        environ.get("AZURE_IMAGE_DEPLOYMENT_NAME")
    )

    if endpoint is None:
        return AzureImageConfigurationError(
            field="endpoint",
            error_type="configuration_error",
            message="Azure OpenAI image endpoint is not configured.",
            canonical_key=_ENDPOINT_CONFIG_KEY,
            setup_action=_AZURE_ENDPOINT_SETUP_ACTION,
        )
    if deployment_name is None:
        return AzureImageConfigurationError(
            field="deployment_name",
            error_type="configuration_error",
            message="Azure OpenAI image deployment is not configured.",
            canonical_key=_DEPLOYMENT_CONFIG_KEY,
            setup_action=_AZURE_DEPLOYMENT_SETUP_ACTION,
        )

    endpoint_info = normalize_azure_image_endpoint(endpoint)
    if endpoint_info is None:
        return AzureImageConfigurationError(
            field="endpoint",
            error_type="configuration_error",
            message=(
                "Azure OpenAI image endpoint is invalid. Use a Foundry OpenAI "
                "v1 endpoint or a direct Azure OpenAI resource endpoint."
            ),
            canonical_key=_ENDPOINT_CONFIG_KEY,
            setup_action=_AZURE_ENDPOINT_SETUP_ACTION,
        )

    normalized_endpoint, endpoint_family = endpoint_info
    if api_key is None and endpoint_family == "azure-openai":
        return AzureImageConfigurationError(
            field="api_key",
            error_type="auth_required",
            message=(
                "Authentication is required for the configured image provider "
                "(Azure OpenAI)."
            ),
            setup_action=_AZURE_IMAGE_KEY_SETUP_ACTION,
        )

    api_version = None
    if endpoint_family == "azure-openai":
        api_version = (
            _trimmed(azure.get("api_version")) or DEFAULT_AZURE_IMAGE_API_VERSION
        )

    return AzureImageSettings(
        endpoint=normalized_endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        api_version=api_version,
        endpoint_family=endpoint_family,
    )


class AzureOpenAIImageGenProvider(ImageGenProvider):
    """Azure OpenAI text-to-image provider registered as ``azure-openai``."""

    @property
    def name(self) -> str:
        return "azure-openai"

    @property
    def display_name(self) -> str:
        return "Azure OpenAI"

    def is_available(self) -> bool:
        """Return whether the SDK and structurally required settings exist."""
        try:
            import openai
            from hermes_cli.config import load_config

            settings = resolve_azure_image_settings(load_config(), os.environ)
            if not isinstance(settings, AzureImageSettings):
                return False
            if settings.endpoint_family == "azure-openai":
                return hasattr(openai, "AzureOpenAI")
            if not hasattr(openai, "OpenAI"):
                return False
            if settings.api_key:
                return True

            from agent.azure_identity_adapter import has_azure_identity_installed

            return has_azure_identity_installed()
        except Exception:  # noqa: BLE001 - picker readiness must remain defensive
            return False

    def picker_readiness_status(
        self,
        config: Mapping[str, object],
        get_secret: Any,
    ) -> str:
        """Return endpoint-aware auth readiness without minting a token."""
        environ = dict(os.environ)
        api_key = _trimmed(get_secret("AZURE_OPENAI_IMAGE_KEY"))
        if api_key is None:
            environ.pop("AZURE_OPENAI_IMAGE_KEY", None)
        else:
            environ["AZURE_OPENAI_IMAGE_KEY"] = api_key

        settings = resolve_azure_image_settings(config, environ)
        if isinstance(settings, AzureImageConfigurationError):
            return "needs_keys" if settings.field == "api_key" else "needs_setup"
        if settings.endpoint_family != "foundry-v1" or settings.api_key:
            return "ready"

        from agent.azure_identity_adapter import has_azure_identity_installed

        return "ready" if has_azure_identity_installed() else "needs_auth"

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": "Azure OpenAI and Foundry image deployments — text-to-image",
            "env_vars": [
                {
                    "key": "AZURE_OPENAI_IMAGE_KEY",
                    "prompt": "Azure OpenAI image API key (optional with Foundry Entra ID)",
                    "password": True,
                    "required": False,
                },
            ],
            "config_fields": [
                {
                    "key": _ENDPOINT_CONFIG_KEY,
                    "prompt": "Azure endpoint (resource root or Foundry /openai/v1)",
                    "required": True,
                    "normalize": normalize_azure_image_setup_endpoint,
                },
                {
                    "key": _DEPLOYMENT_CONFIG_KEY,
                    "prompt": "Azure image deployment",
                    "required": True,
                },
            ],
            "readiness_check": self.picker_readiness_status,
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate and materialize an Azure image response."""
        import base64

        from agent.image_gen_provider import (
            normalize_reference_images,
            resolve_aspect_ratio,
            save_b64_image,
            save_url_image,
            success_response,
        )

        normalized_prompt = prompt.strip() if isinstance(prompt, str) else ""
        aspect = resolve_aspect_ratio(aspect_ratio)

        has_source_image = bool(
            (isinstance(image_url, str) and image_url.strip())
            or normalize_reference_images(reference_image_urls)
        )
        if has_source_image:
            return error_response(
                error=(
                    "Azure OpenAI image generation is text-to-image only; "
                    "image_url and reference_image_urls are unsupported."
                ),
                error_type="modality_unsupported",
                provider=self.name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )

        if not normalized_prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        from hermes_cli.config import load_config

        settings = resolve_azure_image_settings(load_config(), os.environ)
        if isinstance(settings, AzureImageConfigurationError):
            error = settings.message
            if settings.setup_action:
                error = f"{error} {settings.setup_action}"
            payload = error_response(
                error=error,
                error_type=settings.error_type,
                provider=self.name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )
            if settings.setup_action:
                payload["setup_action"] = settings.setup_action
            if settings.config_key:
                payload["config_key"] = settings.config_key
            return payload

        def sanitized_exception_text(exc: BaseException) -> str:
            """Remove the exact Azure key before an exception reaches any sink."""
            try:
                text = str(exc)
            except Exception:  # noqa: BLE001 - hostile exceptions stay safe
                text = type(exc).__name__
            if settings.api_key:
                text = text.replace(settings.api_key, "[REDACTED]")

            from agent.redact import redact_sensitive_text

            return redact_sensitive_text(text, force=True)

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider=self.name,
                model=settings.deployment_name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )

        sizes = {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        }
        size = sizes.get(aspect, sizes[DEFAULT_ASPECT_RATIO])

        # The deployment name is authoritative; generic model/quality kwargs
        # are deliberately ignored. Foundry's endpoint is already a complete
        # OpenAI-compatible base URL and must never receive Azure REST options.
        if settings.endpoint_family == "foundry-v1" and settings.api_key is None:
            from agent.azure_identity_adapter import has_azure_identity_installed

            if not has_azure_identity_installed():
                payload = error_response(
                    error=(
                        "Authentication is required for the configured image "
                        "provider (Azure OpenAI). "
                        f"{_FOUNDRY_KEYLESS_SETUP_ACTION}"
                    ),
                    error_type="auth_required",
                    provider=self.name,
                    model=settings.deployment_name,
                    prompt=normalized_prompt,
                    aspect_ratio=aspect,
                )
                payload["setup_action"] = _FOUNDRY_KEYLESS_SETUP_ACTION
                return payload

        try:
            if settings.endpoint_family == "foundry-v1":
                api_credential: Any = settings.api_key
                if api_credential is None:
                    from agent.azure_identity_adapter import (
                        SCOPE_AI_AZURE_DEFAULT,
                        build_token_provider,
                    )

                    api_credential = build_token_provider(
                        scope=SCOPE_AI_AZURE_DEFAULT
                    )
                client = openai.OpenAI(
                    base_url=settings.endpoint,
                    api_key=api_credential,
                )
            else:
                client = openai.AzureOpenAI(
                    api_key=settings.api_key,
                    azure_endpoint=settings.endpoint,
                    api_version=settings.api_version,
                )
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            import logging

            safe_error = sanitized_exception_text(exc)
            logging.getLogger(__name__).error(
                "Azure image client construction failed: %s", safe_error
            )
            return error_response(
                error=f"Could not initialize Azure image client: {safe_error}",
                error_type="api_error",
                provider=self.name,
                model=settings.deployment_name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )

        try:
            response = client.images.generate(
                model=settings.deployment_name,
                prompt=normalized_prompt,
                size=size,
                n=1,
            )
        except Exception as exc:  # noqa: BLE001 - SDK boundary
            import logging

            safe_error = sanitized_exception_text(exc)
            logging.getLogger(__name__).error(
                "Azure OpenAI image generation failed: %s", safe_error
            )
            return error_response(
                error=f"Azure OpenAI image generation failed: {safe_error}",
                error_type="api_error",
                provider=self.name,
                model=settings.deployment_name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )

        try:
            data = getattr(response, "data", None)
            first = data[0] if data else None
            b64 = getattr(first, "b64_json", None) if first is not None else None
            url = getattr(first, "url", None) if first is not None else None
            revised_prompt = (
                getattr(first, "revised_prompt", None) if first is not None else None
            )
        except (IndexError, KeyError, TypeError):
            b64 = url = revised_prompt = None

        if isinstance(b64, str) and b64.strip():
            b64 = b64.strip()
            try:
                # Validate separately so malformed service output is classified
                # as an empty response, not as a local cache I/O failure.
                base64.b64decode(b64, validate=True)
            except (ValueError, TypeError):
                return error_response(
                    error="Azure OpenAI returned invalid base64 image data",
                    error_type="empty_response",
                    provider=self.name,
                    model=settings.deployment_name,
                    prompt=normalized_prompt,
                    aspect_ratio=aspect,
                )

            try:
                saved_path = save_b64_image(b64, prefix="azure_openai")
            except Exception as exc:  # noqa: BLE001 - persistence boundary
                safe_error = sanitized_exception_text(exc)
                return error_response(
                    error=f"Could not save Azure OpenAI image to cache: {safe_error}",
                    error_type="io_error",
                    provider=self.name,
                    model=settings.deployment_name,
                    prompt=normalized_prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif isinstance(url, str) and url.strip():
            url = url.strip()
            try:
                saved_path = save_url_image(url, prefix="azure_openai")
            except Exception:  # noqa: BLE001 - original URL is the required fallback
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="Azure OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider=self.name,
                model=settings.deployment_name,
                prompt=normalized_prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if isinstance(revised_prompt, str) and revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=settings.deployment_name,
            prompt=normalized_prompt,
            aspect_ratio=aspect,
            provider=self.name,
            modality="text",
            extra=extra,
        )


def register(ctx) -> None:
    """Register the Azure OpenAI image provider with the shared registry."""
    ctx.register_image_gen_provider(AzureOpenAIImageGenProvider())
