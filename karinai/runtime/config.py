"""Configuration model for KarinAI managed runtime containers.

This module intentionally has no dependency on upstream gateway internals. It
validates the runtime-manager handoff before the productized agent starts and
provides stable prompt variables for KarinAI-owned templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

from .tool_policy import (
    BETA_DISABLED_TOOLSETS,
    BETA_ENABLED_TOOLSETS,
    validate_beta_tool_policy,
)


class ManagedRuntimeConfigError(ValueError):
    """Raised when managed runtime configuration is missing or unsafe."""


def _clean(value: object) -> str:
    return str(value or "").strip()


def parse_bool(value: object, *, default: bool = False) -> bool:
    text = _clean(value).lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def parse_csv(value: object) -> tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _require_non_empty(name: str, value: str, errors: list[str]) -> None:
    if not value:
        errors.append(f"{name} is required in KarinAI managed runtime mode")


def _require_absolute_path(name: str, value: str, errors: list[str]) -> None:
    if value and not Path(value).is_absolute():
        errors.append(f"{name} must be an absolute path in KarinAI managed runtime mode")


@dataclass(frozen=True)
class ManagedRuntimeConfig:
    """Trusted runtime-manager handoff for one KarinAI agent container."""

    managed_runtime: bool
    user_id: str
    workspace_id: str
    workspace_dir: str
    runtime_state_dir: str
    api_server_key: str = field(repr=False)
    api_server_host: str = "127.0.0.1"
    api_server_port: int = 8000
    assistant_name: str = "KarinAI"
    product_name: str = "KarinAI"
    runtime_name: str = "KarinAI agent"
    brand_name: str = "KarinAI"
    policy_mode: str = "beta"
    model_gateway_url: str = ""
    model_gateway_model: str = "karinai/default"
    model_gateway_api_mode: str = "chat_completions"
    model_gateway_backend_provider: str = ""
    image_gateway_url: str = ""
    image_gateway_provider: str = ""
    image_gateway_model: str = ""
    tool_gateway_url: str = ""
    runtime_token: str = field(default="", repr=False)
    enabled_toolsets: tuple[str, ...] = BETA_ENABLED_TOOLSETS
    disabled_toolsets: tuple[str, ...] = BETA_DISABLED_TOOLSETS
    local_cron_enabled: bool = False
    plugin_install_enabled: bool = False
    dashboard_enabled: bool = False

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        require_managed: bool = True,
    ) -> "ManagedRuntimeConfig":
        source = env or {}
        managed = parse_bool(source.get("KARINAI_MANAGED_RUNTIME"), default=False)
        if require_managed and not managed:
            raise ManagedRuntimeConfigError(
                "KARINAI_MANAGED_RUNTIME must be true to load managed runtime config"
            )

        raw_port = _clean(source.get("API_SERVER_PORT")) or "8000"
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ManagedRuntimeConfigError("API_SERVER_PORT must be an integer") from exc

        cfg = cls(
            managed_runtime=managed,
            user_id=_clean(source.get("KARINAI_USER_ID")),
            workspace_id=_clean(source.get("KARINAI_WORKSPACE_ID")),
            workspace_dir=_clean(source.get("KARINAI_WORKSPACE_DIR")),
            runtime_state_dir=_clean(source.get("KARINAI_RUNTIME_STATE_DIR")),
            api_server_key=_clean(source.get("API_SERVER_KEY")),
            api_server_host=_clean(source.get("API_SERVER_HOST")) or "127.0.0.1",
            api_server_port=port,
            assistant_name=_clean(source.get("KARINAI_ASSISTANT_NAME")) or "KarinAI",
            product_name=_clean(source.get("KARINAI_PRODUCT_NAME")) or "KarinAI",
            runtime_name=_clean(source.get("KARINAI_RUNTIME_NAME")) or "KarinAI agent",
            brand_name=_clean(source.get("KARINAI_BRAND_NAME")) or "KarinAI",
            policy_mode=_clean(source.get("KARINAI_POLICY_MODE")) or "beta",
            model_gateway_url=_clean(source.get("KARINAI_MODEL_GATEWAY_URL")),
            model_gateway_model=_clean(source.get("KARINAI_MODEL_GATEWAY_MODEL"))
            or "karinai/default",
            model_gateway_api_mode=_clean(source.get("KARINAI_MODEL_GATEWAY_API_MODE"))
            or "chat_completions",
            model_gateway_backend_provider=_clean(
                source.get("KARINAI_MODEL_GATEWAY_BACKEND_PROVIDER")
            ),
            image_gateway_url=_clean(source.get("KARINAI_IMAGE_GATEWAY_URL")),
            image_gateway_provider=_clean(source.get("KARINAI_IMAGE_GATEWAY_PROVIDER")),
            image_gateway_model=_clean(source.get("KARINAI_IMAGE_GATEWAY_MODEL")),
            tool_gateway_url=_clean(source.get("KARINAI_TOOL_GATEWAY_URL")),
            runtime_token=_clean(source.get("KARINAI_RUNTIME_TOKEN")),
            enabled_toolsets=parse_csv(source.get("KARINAI_ENABLED_TOOLSETS"))
            or BETA_ENABLED_TOOLSETS,
            disabled_toolsets=parse_csv(source.get("KARINAI_DISABLED_TOOLSETS"))
            or BETA_DISABLED_TOOLSETS,
            local_cron_enabled=parse_bool(
                source.get("KARINAI_LOCAL_CRON_ENABLED"), default=False
            ),
            plugin_install_enabled=parse_bool(
                source.get("KARINAI_PLUGIN_INSTALL_ENABLED"), default=False
            ),
            dashboard_enabled=parse_bool(
                source.get("KARINAI_DASHBOARD_ENABLED"), default=False
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        errors: list[str] = []
        if not self.managed_runtime:
            errors.append("managed_runtime must be true")
        _require_non_empty("KARINAI_USER_ID", self.user_id, errors)
        _require_non_empty("KARINAI_WORKSPACE_ID", self.workspace_id, errors)
        _require_non_empty("KARINAI_WORKSPACE_DIR", self.workspace_dir, errors)
        _require_non_empty("KARINAI_RUNTIME_STATE_DIR", self.runtime_state_dir, errors)
        _require_absolute_path("KARINAI_WORKSPACE_DIR", self.workspace_dir, errors)
        _require_absolute_path("KARINAI_RUNTIME_STATE_DIR", self.runtime_state_dir, errors)
        _require_non_empty("API_SERVER_KEY", self.api_server_key, errors)
        if self.api_server_port <= 0 or self.api_server_port > 65535:
            errors.append("API_SERVER_PORT must be between 1 and 65535")
        if self.policy_mode != "beta":
            errors.append("only KARINAI_POLICY_MODE=beta is supported in this runtime layer")
        if self.local_cron_enabled:
            errors.append("KARINAI_LOCAL_CRON_ENABLED must be false; backend owns schedules")
        if self.plugin_install_enabled:
            errors.append("KARINAI_PLUGIN_INSTALL_ENABLED must be false by default")
        if self.dashboard_enabled:
            errors.append("KARINAI_DASHBOARD_ENABLED must be false in beta managed runtime")
        if self.model_gateway_url:
            parsed = urlparse(self.model_gateway_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append("KARINAI_MODEL_GATEWAY_URL must be an absolute HTTP(S) URL")
            _require_non_empty("KARINAI_MODEL_GATEWAY_MODEL", self.model_gateway_model, errors)
            _require_non_empty("KARINAI_RUNTIME_TOKEN", self.runtime_token, errors)
            if self.model_gateway_api_mode not in {"chat_completions", "codex_responses"}:
                errors.append(
                    "KARINAI_MODEL_GATEWAY_API_MODE must be chat_completions or codex_responses"
                )
        image_gateway_hints = [
            self.image_gateway_url,
            self.image_gateway_provider,
            self.image_gateway_model,
        ]
        if any(image_gateway_hints) and not self.image_gateway_url:
            errors.append(
                "KARINAI_IMAGE_GATEWAY_URL is required when image gateway provider/model hints are set"
            )
        if self.image_gateway_url:
            parsed = urlparse(self.image_gateway_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                errors.append("KARINAI_IMAGE_GATEWAY_URL must be an absolute HTTP(S) URL")
            _require_non_empty("KARINAI_RUNTIME_TOKEN", self.runtime_token, errors)
        try:
            validate_beta_tool_policy(self.enabled_toolsets, self.disabled_toolsets)
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            raise ManagedRuntimeConfigError("; ".join(errors))

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir)

    @property
    def runtime_state_path(self) -> Path:
        return Path(self.runtime_state_dir)

    def prompt_variables(self) -> dict[str, str]:
        """Variables exposed to KarinAI prompt templates. Secrets are omitted."""
        return {
            "assistant_name": self.assistant_name,
            "product_name": self.product_name,
            "runtime_name": self.runtime_name,
            "brand_name": self.brand_name,
            "policy_mode": self.policy_mode,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "workspace_dir": self.workspace_dir,
            "runtime_state_dir": self.runtime_state_dir,
            "enabled_toolsets": ", ".join(self.enabled_toolsets),
            "disabled_toolsets": ", ".join(self.disabled_toolsets),
            "model_gateway_configured": "true" if self.model_gateway_url else "false",
            "model_gateway_model": self.model_gateway_model,
            "model_gateway_api_mode": self.model_gateway_api_mode,
            "image_gateway_configured": "true" if self.image_gateway_url else "false",
            "image_gateway_model": self.image_gateway_model,
            "tool_gateway_configured": "true" if self.tool_gateway_url else "false",
        }

    def gateway_env(self) -> dict[str, str]:
        """Environment values the managed entrypoint should enforce before startup."""
        return {
            "API_SERVER_ENABLED": "true",
            "API_SERVER_HOST": self.api_server_host,
            "API_SERVER_PORT": str(self.api_server_port),
            "API_SERVER_MODEL_NAME": self.runtime_name,
            "HERMES_HOME": self.runtime_state_dir,
            "HOME": str(self.runtime_state_path / "home"),
            "HERMES_WRITE_SAFE_ROOT": self.workspace_dir,
            "HERMES_DASHBOARD": "false",
            "TERMINAL_CWD": self.workspace_dir,
            "KARINAI_MODEL_GATEWAY_API_MODE": self.model_gateway_api_mode,
            "KARINAI_MODEL_GATEWAY_BACKEND_PROVIDER": self.model_gateway_backend_provider,
        }


def names_csv(names: Sequence[str]) -> str:
    return ",".join(str(name).strip() for name in names if str(name).strip())
