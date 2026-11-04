"""Parse compression settings without expanding agent initialization."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.config import DEFAULT_CONFIG, cfg_get
from utils import is_truthy_value


class CompressionSettings(SimpleNamespace):
    """Parsed ``compression`` config section (see ``_parse_compression_config``)."""


def _parse_compression_config(agent, _agent_cfg) -> CompressionSettings:
    """Parse the ``compression`` section. Defaults here MUST match DEFAULT_CONFIG."""
    from agent.agent_init import (
        _cfg_dict, _cfg_flag, _compression_codex_settings, _compression_threshold,
        _parse_config_int, _positive_int, _refuse_checkpoint_required_on_codex_app_server,
        logger,
    )

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
    # threshold_tokens: absolute cap (lower of ratio threshold and this); clamped to the window at
    # apply-time. Explicit null is the ratio-only opt-out and stays None.
    threshold_tokens = cfg.get("threshold_tokens", cfg_get(DEFAULT_CONFIG, "compression", "threshold_tokens"))
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
    summary_source = str(cfg.get(
        "summary_source", cfg_get(DEFAULT_CONFIG, "compression", "summary_source")
    ) or "previous").strip().lower()
    if summary_source not in {"previous", "original"}:
        logger.warning(
            "Invalid compression.summary_source=%r; expected 'previous' or 'original'. Using previous.",
            cfg.get("summary_source"),
        )
        summary_source = "previous"
    summary_source_windows = max(
        1, min(8, _parse_config_int(
            cfg.get("summary_source_windows", cfg_get(DEFAULT_CONFIG, "compression", "summary_source_windows")), 2
        ))
    )
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
        prepare_ahead=is_truthy_value(cfg.get("prepare_ahead"), default=False),
        summary_source=summary_source,
        summary_source_windows=summary_source_windows,
    )
