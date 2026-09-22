"""Compression config parsing and threshold resolution, split out of ``agent/agent_init.py``.

``agent_init`` (2K-law shard, #79953) keeps ``init_agent`` as a thin orchestrator; the
``compression`` config section end to end lives here: strict int/flag coercion
(``_parse_config_int`` / ``_positive_int`` / ``_cfg_flag`` / ``_cfg_dict``), the
``CompressionSettings`` record, per-model threshold merging (Codex autoraise vs. Arcee
Trinity) and the init-time checkpoint gate. ``init_agent`` imports every name below, so
``agent.agent_init.<name>`` still resolves and no caller or patch target moved.
"""

from __future__ import annotations
import sys
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Dict, Optional

from agent.agent_runtime_helpers import _ra
from utils import is_truthy_value


def _resolve_compression_threshold(
    global_threshold: float, model_cthresh: Optional[float], *, model: Optional[str] = None,
    is_codex_autoraise: bool,
) -> tuple[float, Optional[Dict[str, Any]]]:
    """Global compaction threshold merged with a per-model override.

    Returns ``(threshold, autoraise_notice)``; the notice is set only when a Codex autoraise
    actually RAISES the threshold — it never lowers a higher user value (the user deliberately
    keeps more raw context). Other overrides (Arcee Trinity) stay unconditional.
    """
    if model_cthresh is None:
        return global_threshold, None
    if is_codex_autoraise:
        if model_cthresh <= global_threshold + 1e-9:
            return global_threshold, None
        return model_cthresh, {"model": model, "from": global_threshold, "to": model_cthresh}
    return model_cthresh, None


def _refuse_checkpoint_required_on_codex_app_server(
    checkpoint_required: bool, api_mode: Optional[str]
) -> None:
    """Fail closed at init: the codex app-server compacts its own thread without a truthful
    pre-compaction boundary (default "native" mode), so a required checkpoint can't be
    guaranteed — the compress_context() guard alone cannot cover native turns."""
    if checkpoint_required and api_mode == "codex_app_server":
        raise RuntimeError(
            "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required "
            "is incompatible with the codex_app_server API mode: the codex "
            "agent compacts its own thread without a truthful pre-compaction "
            "transcript boundary, so a required pre-compress checkpoint "
            "cannot be guaranteed. Disable compression.checkpoint_required "
            "or use a non-app-server API mode."
        )


def _parse_config_int(raw: Any, default: int) -> int:
    """Strict int coercion: rejects bool (YAML ``true`` → 1) and fractional floats."""
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _cfg_flag(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    """Legacy string-set truthiness used by the ``compression`` section."""
    return str(cfg.get(key, default)).lower() in {"true", "1", "yes"}


def _cfg_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """``cfg[key]`` if it is a mapping, else ``{}`` (malformed sections are ignored)."""
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


class CompressionSettings(SimpleNamespace):
    """Parsed ``compression`` config section (see ``_parse_compression_config``)."""


def _positive_int(raw: Any, *, reject: tuple = ()) -> Optional[int]:
    """``int(raw)`` when positive, else None. ``reject`` lists types refused outright (bool, float)."""
    if reject and isinstance(raw, reject):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _compression_threshold(agent, cfg: Dict[str, Any]) -> tuple[float, bool]:
    """Global threshold merged with the per-model override; stashes the autoraise notice.
    Codex gpt-5.4/5.5 raise to 85% (272K cap → 50% would compact at ~136K); the opt-out flag
    restores the global value, and the notice has its own display gate."""
    threshold = float(cfg.get("threshold", 0.50))
    autoraise = _cfg_flag(cfg, "codex_gpt55_autoraise", True)
    notice_enabled = _cfg_flag(cfg, "codex_gpt55_autoraise_notice", True)
    agent._compression_threshold_autoraised = None
    with suppress(Exception):
        from agent.auxiliary_client import (
            _compression_threshold_for_model as _cthresh_fn,
            _is_codex_gpt54_or_gpt55 as _is_codex_gpt54_or_gpt55_fn,
            _is_codex_spark as _is_codex_spark_fn,
        )
        _model_cthresh = _cthresh_fn(
            agent.model, agent.provider, allow_codex_gpt55_autoraise=autoraise,
        )
        # Codex autoraises apply only when they RAISE; Arcee Trinity keeps its
        # unconditional override.
        threshold, agent._compression_threshold_autoraised = _resolve_compression_threshold(
            threshold,
            _model_cthresh,
            model=agent.model,
            is_codex_autoraise=(
                _is_codex_gpt54_or_gpt55_fn(agent.model, agent.provider)
                or _is_codex_spark_fn(agent.model, agent.provider)
            ),
        )
    return threshold, notice_enabled


def _compression_codex_settings(cfg: Dict[str, Any]) -> tuple[str, bool, Optional[int]]:
    """``codex_app_server_auto`` / ``codex_responses_native`` / ``codex_responses_compact_threshold``."""
    app_server_auto = str(cfg.get("codex_app_server_auto", "native") or "native").lower()
    if app_server_auto not in {"native", "hermes", "off"}:
        _ra().logger.warning(
            "Invalid compression.codex_app_server_auto=%r; using 'native'. "
            "Valid values are: native, hermes, off.",
            app_server_auto,
        )
        app_server_auto = "native"
    # Native Responses server-side compaction (opt-in; gate in agent/native_compaction.py).
    # Truthy coercion so "false"/"off" strings stay disabled.
    responses_native = is_truthy_value(cfg.get("codex_responses_native", False))
    _raw = cfg.get("codex_responses_compact_threshold")
    compact_threshold = None
    if _raw is not None:
        compact_threshold = _positive_int(_raw, reject=(bool, float))
        if compact_threshold is None:
            _ra().logger.warning(
                "Invalid compression.codex_responses_compact_threshold=%r; "
                "using the automatic threshold derived from local compression.",
                _raw,
            )
    return app_server_auto, responses_native, compact_threshold


def _parse_compression_config(agent, _agent_cfg) -> CompressionSettings:
    """Parse the ``compression`` section. Defaults here MUST match DEFAULT_CONFIG."""
    cfg = _cfg_dict(_agent_cfg, "compression")
    threshold, autoraise_notice_enabled = _compression_threshold(agent, cfg)
    # Plain int()/float() coercions raise on garbage; evaluated up front, in config order.
    target_ratio = float(cfg.get("target_ratio", 0.20))
    protect_last = int(cfg.get("protect_last_n", 20))
    # max_attempts: retry rounds before "max compression attempts reached"; some sessions
    # need >3 (incompressible tool schemas). Default 3, floor 1, cap 10.
    max_attempts = _parse_config_int(cfg.get("max_attempts", 3), 3)
    if max_attempts < 1:
        max_attempts = 3
    # threshold_tokens: absolute cap (lower of ratio threshold and this); clamped to the
    # window at apply-time.
    threshold_tokens = cfg.get("threshold_tokens")
    if threshold_tokens is not None:
        threshold_tokens = _positive_int(threshold_tokens)
    # Non-system head messages to protect (system prompt is always protected); 0 is a
    # legitimate "system prompt + summary + tail".
    protect_first = max(0, int(cfg.get("protect_first_n", 3)))
    checkpoint_required = is_truthy_value(cfg.get("checkpoint_required"), default=False)
    _refuse_checkpoint_required_on_codex_app_server(
        checkpoint_required, getattr(agent, "api_mode", None)
    )
    app_server_auto, responses_native, compact_threshold = _compression_codex_settings(cfg)
    # Opt-in idle compaction: compact up front when a session resumes after this many
    # seconds idle (0 = disabled). Consumed by build_turn_context().
    idle_compact_after_seconds = max(0, int(cfg.get("idle_compact_after_seconds", 0)))
    return CompressionSettings(
        threshold=threshold,
        autoraise_notice_enabled=autoraise_notice_enabled,
        enabled=_cfg_flag(cfg, "enabled", True),
        target_ratio=target_ratio,
        protect_last=protect_last,
        # "lean" keeps a clamped 2.5%/10K-25K verbatim tail (continuity rides the summary);
        # "legacy" restores the 0.20*threshold tail. Unknown → lean inside the compressor.
        tail_mode=str(cfg.get("tail_mode", "lean")).strip().lower(),
        # Actionable user messages guaranteed to survive in the tail (default 1, floor 1).
        min_tail_users=max(1, _parse_config_int(cfg.get("min_tail_user_messages", 1), 1)),
        max_attempts=min(max_attempts, 10),
        # Opt-in proactive tool-result prune trigger (0 = disabled; negatives = disabled).
        proactive_prune_tokens=max(0, _parse_config_int(cfg.get("proactive_prune_tokens", 0), 0)),
        proactive_prune_min_chars=_parse_config_int(
            cfg.get("proactive_prune_min_result_chars", 8000), 8000
        ),
        proactive_prune_min_reclaim=max(
            0, _parse_config_int(cfg.get("proactive_prune_min_reclaim_tokens", 4096), 4096)
        ),
        protect_first=protect_first,
        abort_on_summary_failure=_cfg_flag(cfg, "abort_on_summary_failure", False),
        # Per-model threshold overrides: keys substring-matched against the model name
        # (longest match wins); {} = global threshold for all models.
        model_thresholds={
            str(k): float(v) for k, v in _cfg_dict(cfg, "model_thresholds").items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        threshold_tokens=threshold_tokens,
        checkpoint_required=checkpoint_required,
        # In-place compaction: no session-id rotation. default=True MUST match DEFAULT_CONFIG
        # (a False default flipped agents into rotation mode when the key was omitted).
        in_place=is_truthy_value(cfg.get("in_place"), default=True),
        # Opt-in: micro-compaction rewrites sent history per turn (breaks the cache prefix).
        micro_compact=is_truthy_value(cfg.get("micro_compact"), default=False),
        # Pass cadence in completed turns; each pass costs one prompt-cache break (>= 1).
        micro_compact_every_n_turns=max(
            1, _parse_config_int(cfg.get("micro_compact_every_n_turns", 1), 1)
        ),
        # Rolling-summary defrag threshold, in tokens.
        micro_compact_defrag_tokens=max(
            1, _parse_config_int(cfg.get("micro_compact_defrag_threshold_tokens", 2000), 2000)
        ),
        codex_app_server_auto=app_server_auto,
        codex_responses_native=responses_native,
        codex_responses_compact_threshold=compact_threshold,
        idle_compact_after_seconds=idle_compact_after_seconds,
    )


def _warn_invalid_config_int(
    what: str, value: Any, requirement: str, fallback: str, print_fallback: str = "",
    agent: Any = None,
) -> None:
    """Log + stderr-print an invalid integer config value (``print_fallback``: user-facing
    wording where it differs from the log line). The print is an automatic diagnostic and
    honors the warning-notification policy; the log line never does."""
    _ra().logger.warning(
        "Invalid %s: %r — %s. Falling back to %s.", what, value, requirement, fallback,
    )
    from gateway.warning_notifications import warning_notifications_enabled
    try:
        if not warning_notifications_enabled(
            getattr(agent, "_notification_platform", getattr(agent, "platform", "cli")),
            getattr(agent, "_notification_config", None),
        ):
            return
    except Exception:
        pass
    print(
        f"\n⚠ Invalid {what}: {value!r}\n"
        f"  {requirement[0].upper() + requirement[1:]}.\n"
        f"  Falling back to {print_fallback or fallback}.\n",
        file=sys.stderr,
    )
