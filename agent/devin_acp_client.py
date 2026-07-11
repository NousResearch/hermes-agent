"""OpenAI-compatible shim that forwards Hermes requests to ``devin acp``.

Mirrors :class:`agent.copilot_acp_client.CopilotACPClient` for Cognition's
Devin CLI ACP mode (JSON-RPC over stdio). Process reuse and per-prompt
``session/new`` follow the shared ACP client lifecycle (see parent module).

Model selection (verified against Devin CLI 3000.x on Windows):
  - CLI root ``devin --model X -p …`` honors ``X`` (non-ACP).
  - ``devin acp`` **ignores** process-level ``--model`` / ``DEVIN_MODEL`` and
    starts at config default (often ``swe-1-7``).
  - The working switch is ACP ``session/set_config_option`` with
    ``configId="model"`` and an ACP model *value* from
    ``session/new`` → ``configOptions`` (hyphen ids such as ``swe-1-6-fast``,
    not the dotted CLI ids from ``devin --model`` help).

Hermes therefore:
  1. Still passes ``--model`` / ``DEVIN_MODEL`` as a belt-and-suspenders hint.
  2. After every ``session/new``, maps the Hermes model id onto an ACP option
     value and calls ``session/set_config_option``.
  3. Respawns the warm process when the Hermes model id changes.

Docs: https://docs.devin.ai/cli/acp/jetbrains ·
      https://agentclientprotocol.com/protocol/v1/session-config-options
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

from agent.copilot_acp_client import CopilotACPClient, _coalesce_acp_args

logger = logging.getLogger(__name__)

ACP_MARKER_BASE_URL = "acp://devin"

# Hermes-side placeholder ids that mean "use Devin's own default model".
_HERMES_PLACEHOLDER_MODELS = frozenset({
    "devin-acp",
    "devin",
    "devin-cli",
    "cognition-devin",
})


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_DEVIN_ACP_COMMAND", "").strip()
        or os.getenv("DEVIN_CLI_PATH", "").strip()
        or "devin"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_DEVIN_ACP_ARGS", "").strip()
    if not raw:
        # Official JetBrains / Zed ACP config uses a single ``acp`` argument.
        return ["acp"]
    return shlex.split(raw)


def _backend_model_id(model: str | None) -> str | None:
    """Return Hermes model id for binding, or None to leave Devin's default."""
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


def resolve_devin_acp_model_value(
    hermes_model: str | None,
    config_options: list[dict[str, Any]] | None,
) -> str | None:
    """Map a Hermes/CLI model id onto a Devin ACP ``configOptions`` value.

    ACP values look like ``swe-1-6-fast`` / ``claude-sonnet-5-medium`` /
    ``MODEL_PRIVATE_11`` (Haiku). CLI ``--model`` help uses dotted short names
    like ``swe-1.6-fast`` / ``claude-haiku-4.5``. Returns None when no mapping
    is possible (caller should leave the session default).
    """
    wanted = _backend_model_id(hermes_model)
    if not wanted:
        return None

    options: list[dict[str, str]] = []
    for opt in config_options or []:
        if not isinstance(opt, dict):
            continue
        if str(opt.get("id") or "") != "model":
            continue
        for entry in opt.get("options") or []:
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("value") or "").strip()
            name = str(entry.get("name") or "").strip()
            if value:
                options.append({"value": value, "name": name})
        break

    if not options:
        # No catalog — best-effort: convert dots to hyphens (CLI→ACP common case).
        return _norm_model_token(wanted)

    wanted_n = _norm_model_token(wanted)
    # 1) Exact value match (already ACP id).
    for entry in options:
        if entry["value"] == wanted or _norm_model_token(entry["value"]) == wanted_n:
            return entry["value"]

    # 2) Value ends with / contains normalized token (e.g. swe-1-6-fast).
    for entry in options:
        vn = _norm_model_token(entry["value"])
        if wanted_n == vn or vn.endswith(wanted_n) or wanted_n.endswith(vn):
            return entry["value"]

    # 3) Display name match (e.g. "Claude Haiku 4.5" ← claude-haiku-4.5).
    wanted_words = re.sub(r"[-_.]+", " ", wanted_n).strip()
    for entry in options:
        name_n = _norm_model_token(entry["name"]).replace("-", " ")
        if wanted_words and wanted_words in name_n:
            return entry["value"]
        # reverse: name tokens subset of wanted
        name_compact = name_n.replace(" ", "-")
        if name_compact and name_compact in wanted_n:
            return entry["value"]

    # 4) Family alias: opus/sonnet/swe/codex/gemini → first matching option.
    family = wanted_n.split("-")[0]
    family_map = {
        "opus": "claude-opus",
        "sonnet": "claude-sonnet",
        "haiku": "haiku",
        "swe": "swe-",
        "codex": "codex",
        "gemini": "gemini",
        "gpt": "gpt-",
        "adaptive": "adaptive",
    }
    needle = family_map.get(family, wanted_n)
    for entry in options:
        vn = _norm_model_token(entry["value"])
        nn = _norm_model_token(entry["name"])
        if needle in vn or needle in nn:
            return entry["value"]

    return None


class DevinACPClient(CopilotACPClient):
    """Minimal OpenAI-client-compatible facade for Devin CLI ACP."""

    _acp_display_name = "Devin ACP"
    _default_model_name = "devin-acp"
    _install_hint = (
        "Install Devin CLI (https://docs.devin.ai/cli) and run "
        "`devin auth login`, or set HERMES_DEVIN_ACP_COMMAND/DEVIN_CLI_PATH."
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
        # Resolve against Devin defaults *before* super(), so an empty
        # ``args=[]`` from incomplete call-site wiring cannot fall through to
        # CopilotACPClient's module-level ``_resolve_args()`` (``--acp --stdio``).
        resolved_command = acp_command or command or _resolve_command()
        resolved_args = _coalesce_acp_args(acp_args, args, _resolve_args)
        super().__init__(
            api_key=api_key or "devin-acp",
            base_url=base_url or ACP_MARKER_BASE_URL,
            default_headers=default_headers,
            acp_command=resolved_command,
            acp_args=resolved_args,
            acp_cwd=acp_cwd,
            **kwargs,
        )
        # Parent stores args via its own default_args_fn; re-assert Devin's
        # resolved argv in case kwargs still carried a stale empty list.
        self._acp_command = resolved_command
        self._acp_args = resolved_args
        # Process-level model binding (None = leave CLI default alone).
        self._desired_process_model: str | None = None
        self._process_bound_model: str | None = None
        # Last ACP config option value successfully applied (for tests/metrics).
        self._session_model_value: str | None = None

    def _prepare_for_model(self, model: str | None) -> None:
        desired = _backend_model_id(model)
        # Model change requires a new session (and preferably a clean process)
        # so set_config_option cannot fight a stale warm session.
        if desired != self._process_bound_model and self._process_alive():
            self._reset_transport(mark_closed=False)
        self._desired_process_model = desired

    def _subprocess_env(self) -> dict[str, str]:
        env = super()._subprocess_env()
        desired = self._desired_process_model
        if desired:
            env["DEVIN_MODEL"] = desired
        else:
            # Avoid inheriting a stale host DEVIN_MODEL when Hermes wants default.
            env.pop("DEVIN_MODEL", None)
        return env

    def _spawn_argv(self) -> list[str]:
        # Keep CLI --model as a hint (honored for non-ACP; ignored by acp mode
        # on current Devin builds — real switch is set_config_option below).
        desired = self._desired_process_model
        if desired:
            return [self._acp_command, "--model", desired, *list(self._acp_args)]
        return [self._acp_command, *list(self._acp_args)]

    def _spawn_process(self):
        proc = super()._spawn_process()
        self._process_bound_model = self._desired_process_model
        self._session_model_value = None
        return proc

    def _apply_session_model(
        self,
        session_id: str,
        session: dict[str, Any],
        model: str | None,
        *,
        timeout_seconds: float,
    ) -> None:
        desired = _backend_model_id(model) or self._desired_process_model
        if not desired:
            return

        config_options = session.get("configOptions")
        if not isinstance(config_options, list):
            config_options = []

        acp_value = resolve_devin_acp_model_value(desired, config_options)
        if not acp_value:
            logger.warning(
                "Devin ACP: could not map Hermes model %r onto session configOptions",
                desired,
            )
            return

        # Skip RPC when session already on the requested value.
        for opt in config_options:
            if isinstance(opt, dict) and str(opt.get("id") or "") == "model":
                if str(opt.get("currentValue") or "") == acp_value:
                    self._session_model_value = acp_value
                    return
                break

        try:
            result = self._rpc(
                "session/set_config_option",
                {
                    "sessionId": session_id,
                    "configId": "model",
                    "value": acp_value,
                },
                timeout_seconds=timeout_seconds,
            ) or {}
        except Exception as exc:
            logger.warning(
                "Devin ACP: session/set_config_option(model=%r) failed: %s",
                acp_value,
                exc,
            )
            return

        # Confirm from response when present.
        applied = acp_value
        for opt in result.get("configOptions") or []:
            if isinstance(opt, dict) and str(opt.get("id") or "") == "model":
                cur = str(opt.get("currentValue") or "").strip()
                if cur:
                    applied = cur
                break
        self._session_model_value = applied
        logger.info(
            "Devin ACP: session model set hermes=%r → acp=%r",
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
            model=model or "devin-acp",
            messages=messages,
            timeout=timeout,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            **kwargs,
        )
