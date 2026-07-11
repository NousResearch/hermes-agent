"""OpenAI-compatible shim that forwards Hermes requests to ``grok agent stdio``.

Mirrors :class:`agent.copilot_acp_client.CopilotACPClient` for xAI's Grok
Build CLI ACP mode (JSON-RPC over stdio). Process reuse and per-prompt
``session/new`` follow the shared ACP client lifecycle (see parent module).

Model selection (verified against Grok CLI 0.2.x on Windows):
  - ``grok agent stdio`` ignores process-level ``--model``.
  - The working switch is ACP ``session/set_model`` with a ``modelId`` from
    ``session/new`` → ``models.availableModels`` (e.g. ``grok-4.5``,
    ``grok-composer-2.5-fast``).
  - The CLI also advertises ``authMethods`` after ``initialize``; we call
    ``authenticate`` with the cached token or ``xai.api_key`` when present.

Docs: https://docs.x.ai/build/cli/headless-scripting ·
      https://agentclientprotocol.com/protocol/v1/session
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

from agent.copilot_acp_client import CopilotACPClient, _coalesce_acp_args

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://grok"

# Hermes-side placeholder ids that mean "use Grok's own default model".
_HERMES_PLACEHOLDER_MODELS = frozenset({
    "grok-acp",
    "grok-cli",
    "grok-build",
    "xai-grok-cli",
})


def _resolve_command() -> str:
    env = (
        os.getenv("HERMES_GROK_ACP_COMMAND", "").strip()
        or os.getenv("GROK_CLI_PATH", "").strip()
    )
    if env:
        return env
    local = os.path.expanduser("~/.grok/bin/grok")
    if os.path.exists(local):
        return local
    return "grok"


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_GROK_ACP_ARGS", "").strip()
    if not raw:
        # --no-auto-update is a global CLI flag and must precede the agent
        # subcommand; it is safe for automation (docs recommend it for headless
        # and ACP uses).  User can override with HERMES_GROK_ACP_ARGS.
        return ["--no-auto-update", "agent", "stdio"]
    return shlex.split(raw)


def _backend_model_id(model: str | None) -> str | None:
    """Return Hermes model id for binding, or None to leave Grok's default."""
    m = (model or "").strip()
    if not m or m.lower() in _HERMES_PLACEHOLDER_MODELS:
        return None
    return m


def _norm_model_token(value: str) -> str:
    """Normalize model tokens for fuzzy matching (dots/underscores → hyphens)."""
    s = (value or "").strip().lower()
    s = s.replace("_", "-").replace(".", "-")
    s = re.sub(r"-+", "-", s)
    return s


def resolve_grok_acp_model_value(
    hermes_model: str | None,
    available_models: list[dict[str, Any]] | None,
) -> str | None:
    """Map a Hermes/CLI model id onto a Grok ACP ``modelId``.

    ACP ``modelId`` values are dotted ids like ``grok-4.5`` or
    ``grok-composer-2.5-fast``.  Hermes may pass them as-is or as ``grok-4-5``.
    Returns None when no mapping is possible (caller should leave the session
    default).
    """
    wanted = _backend_model_id(hermes_model)
    if not wanted:
        return None

    models = []
    for entry in available_models or []:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("modelId") or "").strip()
        name = str(entry.get("name") or "").strip()
        if model_id:
            models.append({"modelId": model_id, "name": name})

    if not models:
        # No catalog — best-effort: keep the requested id as-is.
        return _norm_model_token(wanted)

    wanted_n = _norm_model_token(wanted)

    # 1) Exact modelId match (normalized).
    for entry in models:
        model_id = entry["modelId"]
        if model_id == wanted or _norm_model_token(model_id) == wanted_n:
            return model_id

    # 2) Display name match (exact or normalized).
    for entry in models:
        name = entry["name"]
        if name == wanted or _norm_model_token(name) == wanted_n:
            return entry["modelId"]

    # 3) Fuzzy word containment in display name.
    wanted_words = re.sub(r"[-_.]+", " ", wanted_n).strip()
    for entry in models:
        name = entry["name"]
        name_n = _norm_model_token(name).replace("-", " ")
        if wanted_words and wanted_words in name_n:
            return entry["modelId"]
        name_compact = name_n.replace(" ", "-")
        if name_compact and name_compact in wanted_n:
            return entry["modelId"]

    # 4) Family alias (grok/composer/build) → first matching model.
    family = wanted_n.split("-")[0]
    family_map = {
        "grok": "grok-",
        "composer": "composer",
        "build": "grok-build",
        "4.5": "grok-4.5",
        "4.3": "grok-4.3",
        "4": "grok-4",
        "3": "grok-3",
        "2": "grok-2",
    }
    needle = family_map.get(family, wanted_n)
    for entry in models:
        model_id_n = _norm_model_token(entry["modelId"])
        name_n = _norm_model_token(entry["name"])
        if needle in model_id_n or needle in name_n:
            return entry["modelId"]

    return None


class GrokACPClient(CopilotACPClient):
    """Minimal OpenAI-client-compatible facade for Grok CLI ACP."""

    _acp_display_name = "Grok ACP"
    _default_model_name = "grok-acp"
    _install_hint = (
        "Install Grok CLI (https://docs.x.ai/build/cli) and run "
        "`grok login`, or set HERMES_GROK_ACP_COMMAND/GROK_CLI_PATH."
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **kwargs: Any,
    ):
        # Resolve against Grok defaults *before* super(), so an empty
        # ``args=[]`` from incomplete call-site wiring cannot fall through to
        # CopilotACPClient's module-level defaults (``--acp --stdio``).
        resolved_command = acp_command or command or _resolve_command()
        resolved_args = _coalesce_acp_args(acp_args, args, _resolve_args)
        super().__init__(
            api_key=api_key or "grok-acp",
            base_url=base_url or ACP_MARKER_BASE_URL,
            default_headers=default_headers,
            acp_command=resolved_command,
            acp_args=resolved_args,
            acp_cwd=acp_cwd,
            **kwargs,
        )
        # Re-assert Grok's resolved argv in case kwargs still carried a stale
        # empty list.
        self._acp_command = resolved_command
        self._acp_args = resolved_args
        # Session-level model binding (None = leave CLI default alone).
        self._desired_session_model: str | None = None
        self._session_model_value: str | None = None

    def _prepare_for_model(self, model: str | None) -> None:
        desired = _backend_model_id(model)
        self._desired_session_model = desired
        # Model change means the warm session's current model is stale; force a
        # fresh session/new so session/set_model can bind the new model.
        if (
            desired is not None
            and self._session_id is not None
            and self._process_alive()
            and self._session_model_value is not None
            and self._session_model_value != desired
        ):
            self._reset_session_state()

    def _ensure_initialized(self, *, timeout_seconds: float) -> None:
        """Spawn (if needed), run ACP ``initialize`` and ``authenticate``."""
        if self._process_alive() and self._initialized:
            return
        if not self._process_alive():
            self._reset_transport(mark_closed=False)
            self._spawn_process()

        init = self._rpc(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {
                        "readTextFile": True,
                        "writeTextFile": True,
                    }
                },
                "clientInfo": {
                    "name": "hermes-agent",
                    "title": "Hermes Agent",
                    "version": "0.0.0",
                },
            },
            timeout_seconds=timeout_seconds,
        ) or {}
        self._initialized = True
        self._authenticate(init, timeout_seconds=timeout_seconds)

    def _authenticate(
        self,
        init: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> None:
        """Select and call the advertised ACP auth method if present."""
        auth_methods = init.get("authMethods") or []
        if not isinstance(auth_methods, list) or not auth_methods:
            return

        available_ids = {
            str(m.get("id") or "") for m in auth_methods if isinstance(m, dict)
        }
        if not available_ids:
            return

        method_id: str | None = None
        if os.environ.get("XAI_API_KEY") and "xai.api_key" in available_ids:
            method_id = "xai.api_key"
        elif "cached_token" in available_ids:
            method_id = "cached_token"
        elif "grok.com" in available_ids:
            method_id = "grok.com"

        if not method_id:
            return

        try:
            self._rpc(
                "authenticate",
                {
                    "methodId": method_id,
                    "_meta": {"headless": True},
                },
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Grok ACP: authenticate(%r) failed: %s",
                method_id,
                exc,
                exc_info=True,
            )
            # The CLI may still proceed if a cached token is ambient, but the
            # error is preserved if the subsequent session/new also fails.
            raise

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        desired = _backend_model_id(model) or self._desired_session_model

        models = session.get("models")
        if not isinstance(models, dict):
            models = {}
        current_model_id = str(models.get("currentModelId") or "").strip() or None
        available_models = models.get("availableModels")
        if not isinstance(available_models, list):
            available_models = []

        # Track the session's current model even when we don't change it.
        if current_model_id:
            self._session_model_value = current_model_id

        if not desired:
            return

        acp_value = resolve_grok_acp_model_value(desired, available_models)
        if not acp_value:
            logger.warning(
                "Grok ACP: could not map Hermes model %r onto available models",
                desired,
            )
            return

        if current_model_id == acp_value:
            logger.info("Grok ACP: session already on model %r", acp_value)
            return

        try:
            result = self._rpc(
                "session/set_model",
                {
                    "sessionId": session_id,
                    "modelId": acp_value,
                },
                timeout_seconds=timeout_seconds,
            ) or {}
        except Exception as exc:
            logger.warning(
                "Grok ACP: session/set_model(%r) failed: %s",
                acp_value,
                exc,
            )
            return

        # Confirm from response when present.
        applied = acp_value
        ok = result.get("_meta", {}).get("model", {}).get("Ok")
        if isinstance(ok, str) and ok:
            applied = ok
        self._session_model_value = applied
        logger.info(
            "Grok ACP: session model set hermes=%r -> acp=%r",
            desired,
            applied,
        )

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super()._create_chat_completion(
            model=model or "grok-acp",
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        try:
            return super()._run_prompt(prompt_text, timeout_seconds=timeout_seconds)
        except RuntimeError as exc:
            msg = str(exc)
            if "Could not start Copilot ACP command" in msg:
                raise RuntimeError(
                    f"Could not start Grok ACP command '{self._acp_command}'. "
                    "Install Grok CLI (https://docs.x.ai/build/cli) and run "
                    "`grok login`, or set HERMES_GROK_ACP_COMMAND/GROK_CLI_PATH."
                ) from exc
            if "Copilot ACP process did not expose" in msg:
                raise RuntimeError(
                    "Grok ACP process did not expose stdin/stdout pipes."
                ) from exc
            raise
