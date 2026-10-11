"""Discord config.yaml to adapter environment/extra bridge."""

from __future__ import annotations

from gateway.platforms._shared import yaml_env_setter as _yaml_env_setter


_YAML_BOOL_ENV_KEYS = (
    ("require_mention", "DISCORD_REQUIRE_MENTION"),
    ("thread_require_mention", "DISCORD_THREAD_REQUIRE_MENTION"),
    ("bots_require_inline_mention", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"),
)
# (public websocket_* key, legacy liveness_* alias, env bridge var)
_YAML_WEBSOCKET_LIVENESS_KEYS = (
    ("websocket_liveness_interval_seconds", "liveness_interval_seconds", "HERMES_DISCORD_LIVENESS_INTERVAL_SECONDS"),
    ("websocket_liveness_failure_threshold", "liveness_failure_threshold", "HERMES_DISCORD_LIVENESS_FAILURE_THRESHOLD"),
    ("websocket_heartbeat_ack_max_age_seconds", None, None),
    ("websocket_max_latency_seconds", None, None),
    ("websocket_event_max_silence_seconds", None, None),
)


def _apply_yaml_config(yaml_cfg: dict, discord_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``discord:`` keys into env vars (``apply_yaml_config_fn``).
    The adapter reads ``DISCORD_*`` via ``os.getenv()`` at ~50 sites, so this hook owns YAML→env;
    ``extra`` stays the per-adapter truth for liveness (multiplex isolation). Returns liveness settings.

    Implements the ``apply_yaml_config_fn`` contract (#24836). Mirrors the legacy ``discord_cfg`` block that
    used to live in ``gateway/config.py::load_gateway_config()`` before this migration.
    """
    # Every env write is first-writer-wins (an explicit env var beats YAML) and is skipped for a
    # profile-scoped multiplex load: a secondary profile's settings must never land in process-global
    # env where they'd become another profile's policy (#72348). Everything is seeded into extra too.
    _env_default = _yaml_env_setter()

    def _csv(value) -> str:
        return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)

    seeded_extra = {}
    for key, env_key in _YAML_BOOL_ENV_KEYS:
        if key in discord_cfg:
            seeded_extra[key] = discord_cfg[key]  # original type: the shared-key loop seeds bools as bools
            _env_default(env_key, str(discord_cfg[key]).lower())
    platforms_cfg = yaml_cfg.get("platforms")
    platform_extra_cfg = {}
    if isinstance(platforms_cfg, dict):
        discord_platform_cfg = platforms_cfg.get("discord")
        if isinstance(discord_platform_cfg, dict):
            candidate_extra = discord_platform_cfg.get("extra")
            if isinstance(candidate_extra, dict):
                platform_extra_cfg = candidate_extra

    def _gate(key: str, env_key: str, *, from_platform_extra: bool, lower: bool = False) -> None:
        value = discord_cfg[key] if key in discord_cfg else (platform_extra_cfg.get(key) if from_platform_extra else None)
        if value is None:
            return
        text = str(value).lower() if lower else _csv(value)
        seeded_extra[key] = text
        _env_default(env_key, text)

    _gate("allow_from", "DISCORD_ALLOWED_USERS", from_platform_extra=True)
    _gate("allowed_roles", "DISCORD_ALLOWED_ROLES", from_platform_extra=True)
    _gate("allow_all_users", "DISCORD_ALLOW_ALL_USERS", from_platform_extra=True, lower=True)
    _gate("allow_bots", "DISCORD_ALLOW_BOTS", from_platform_extra=True, lower=True)
    approval_mentions_cfg = (
        discord_cfg["approval_mentions"] if "approval_mentions" in discord_cfg
        else platform_extra_cfg.get("approval_mentions")
    )
    if approval_mentions_cfg is not None:
        seeded_extra["approval_mentions"] = approval_mentions_cfg
        _env_default("DISCORD_APPROVAL_MENTIONS", str(approval_mentions_cfg).lower())
    _gate("free_response_channels", "DISCORD_FREE_RESPONSE_CHANNELS", from_platform_extra=False)
    for key, env_key in (
        ("auto_thread", "DISCORD_AUTO_THREAD"),
        ("free_response_auto_thread", "DISCORD_FREE_RESPONSE_AUTO_THREAD"),
        ("reactions", "DISCORD_REACTIONS"),
    ):
        if key in discord_cfg:
            seeded_extra[key] = discord_cfg[key]
            _env_default(env_key, str(discord_cfg[key]).lower())
    backfill_cfg = discord_cfg.get("missed_message_backfill")
    if isinstance(backfill_cfg, dict):
        seeded_extra["missed_message_backfill"] = dict(backfill_cfg)
    _gate("ignored_channels", "DISCORD_IGNORED_CHANNELS", from_platform_extra=False)
    _gate("allowed_channels", "DISCORD_ALLOWED_CHANNELS", from_platform_extra=False)
    _gate("no_thread_channels", "DISCORD_NO_THREAD_CHANNELS", from_platform_extra=False)
    # history_backfill: recover mention-gated channel messages between bot turns.
    if "history_backfill" in discord_cfg:
        seeded_extra["history_backfill"] = discord_cfg["history_backfill"]
        _env_default("DISCORD_HISTORY_BACKFILL", str(discord_cfg["history_backfill"]).lower())
    hbl = discord_cfg.get("history_backfill_limit")
    if hbl is not None:
        seeded_extra["history_backfill_limit"] = hbl
        _env_default("DISCORD_HISTORY_BACKFILL_LIMIT", str(hbl))
    # allow_mentions: safe defaults live in the adapter; these keys only override when set.
    allow_mentions_cfg = discord_cfg.get("allow_mentions")
    if isinstance(allow_mentions_cfg, dict):
        seeded_extra["allow_mentions"] = dict(allow_mentions_cfg)
        for yaml_key in ("everyone", "roles", "users", "replied_user"):
            if yaml_key in allow_mentions_cfg:
                _env_default(f"DISCORD_ALLOW_MENTION_{yaml_key.upper()}", str(allow_mentions_cfg[yaml_key]).lower())
    # reply_to_mode: top-level preferred, falls back to extra; YAML 1.1 parses bare 'off' as False.
    _discord_extra = discord_cfg.get("extra") if isinstance(discord_cfg.get("extra"), dict) else {}
    _discord_rtm = discord_cfg["reply_to_mode"] if "reply_to_mode" in discord_cfg else _discord_extra.get("reply_to_mode")
    if _discord_rtm is not None:
        _env_default("DISCORD_REPLY_TO_MODE", "off" if _discord_rtm is False else str(_discord_rtm).lower())
    # Public config keys win over the generic ``extra`` form.
    _websocket_liveness_cfg = {**_discord_extra, **discord_cfg}
    # WebSocket health knobs (REST 200 is not Gateway health); legacy liveness_* aliases accepted.
    for primary_key, legacy_key, env_key in _YAML_WEBSOCKET_LIVENESS_KEYS:
        value = _websocket_liveness_cfg.get(primary_key)
        if value is None and legacy_key:
            value = _websocket_liveness_cfg.get(legacy_key)
        if value is not None:
            seeded_extra[primary_key] = value
            if env_key:
                _env_default(env_key, str(value))
    return seeded_extra or None
