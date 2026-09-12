"""Discord plugin setup, configuration bridging, and registration."""

from __future__ import annotations

import os

from gateway.platforms._shared import yaml_env_setter as _yaml_env_setter


def _adapter_module():
    from . import adapter
    return adapter


def _profile_scoped_config_load():
    return _adapter_module()._profile_scoped_config_load()


def _discord_deps_present():
    return _adapter_module().discord_deps_present()


def _check_discord_requirements():
    return _adapter_module().check_discord_requirements()


def _standalone_send(*args, **kwargs):
    return _adapter_module()._standalone_send(*args, **kwargs)

def _clean_discord_user_ids(raw: str) -> list:
    """Strip common Discord mention prefixes from a comma-separated ID string."""
    cleaned = []
    for uid in raw.replace(" ", "").split(","):
        uid = uid.strip()
        if uid.startswith("<@") and uid.endswith(">"):
            uid = uid.lstrip("<@!").rstrip(">")
        if uid.lower().startswith("user:"):
            uid = uid[5:]
        if uid:
            cleaned.append(uid)
    return cleaned


def interactive_setup() -> None:
    """Guide the user through Discord bot setup: token, allowlist, home channel (lazy CLI imports)."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt, prompt_yes_no, print_header, print_info, print_success,
    )
    def _info_lines(*lines: str) -> None:
        for line in lines:
            print_info(line)

    def _save_allowlist(allowed_users: str) -> None:
        save_env_value("DISCORD_ALLOWED_USERS", ",".join(_clean_discord_user_ids(allowed_users)))
        print_success("Discord allowlist configured")

    print_header("Discord")
    existing = get_env_value("DISCORD_BOT_TOKEN")
    if existing:
        print_info("Discord: already configured")
        if not prompt_yes_no("Reconfigure Discord?", False):
            if not get_env_value("DISCORD_ALLOWED_USERS"):
                print_info(
                    "⚠️  Discord has no user allowlist. With the fail-closed default, "
                    "messages are denied unless you configure allowed users, roles, "
                    "or channels, or set DISCORD_ALLOW_ALL_USERS=true."
                )
                if prompt_yes_no("Add allowed users now?", True):
                    print_info("   To find Discord ID: Enable Developer Mode, right-click name → Copy ID")
                    allowed_users = prompt("Allowed user IDs (comma-separated)")
                    if allowed_users:
                        _save_allowlist(allowed_users)
            return
    _info_lines(
        "Create a bot at https://discord.com/developers/applications",
        "On Bot → Privileged Gateway Intents, enable:",
        "  - Message Content Intent (required — without it Discord rejects the connection)",
        "  - Server Members Intent (required if you use usernames or role allowlists)",
        "Save Changes in the Developer Portal before starting the gateway.",
        "Docs: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/discord",
    )
    token = prompt("Discord bot token", password=True)
    if not token:
        return
    save_env_value("DISCORD_BOT_TOKEN", token)
    print_success("Discord token saved")
    print()
    _info_lines(
        "🔒 Security: Restrict who can use your bot", "   To find your Discord user ID:",
        "   1. Enable Developer Mode in Discord settings", "   2. Right-click your name → Copy ID",
    )
    print()
    print_info("   You can also use Discord usernames (resolved on gateway start).")
    print()
    allowed_users = prompt("Allowed user IDs or usernames (comma-separated, leave empty for open access)")
    if allowed_users:
        _save_allowlist(allowed_users)
    else:
        print_info(
            "⚠️  No allowlist set. Discord will deny messages until you set "
            "DISCORD_ALLOWED_USERS, DISCORD_ALLOWED_ROLES, DISCORD_ALLOWED_CHANNELS, "
            "or DISCORD_ALLOW_ALL_USERS=true for open access."
        )
    print()
    _info_lines(
        "📬 Home Channel: where Hermes delivers cron job results,",
        "   cross-platform messages, and notifications.",
        "   To get a channel ID: right-click a channel → Copy Channel ID",
        "   (requires Developer Mode in Discord settings)",
        "   You can also set this later by typing /set-home in a Discord channel.",
    )
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("DISCORD_HOME_CHANNEL", home_channel)
    elif remove_env_value("DISCORD_HOME_CHANNEL"):
        print_info("Home channel cleared.")


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
)


def _apply_yaml_config(yaml_cfg: dict, discord_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``discord:`` keys into env vars (``apply_yaml_config_fn``).
    The adapter reads ``DISCORD_*`` via ``os.getenv()`` at ~50 sites, so this hook owns YAML→env;
    ``extra`` stays the per-adapter truth for liveness (multiplex isolation). Returns liveness settings.

    Implements the ``apply_yaml_config_fn`` contract (#24836). Mirrors the legacy ``discord_cfg`` block that
    used to live in ``gateway/config.py::load_gateway_config()`` before this migration.
    """
    # Keep legacy env consumers working while avoiding process-global leakage between
    # multiplexed profiles; the setter also seeds the active profile's environment.
    _env_default = _yaml_env_setter()

    def _csv(value) -> str:
        return ",".join(str(v) for v in value) if isinstance(value, list) else str(value)

    seeded_extra = {}
    for key, env_key in _YAML_BOOL_ENV_KEYS:
        if key in discord_cfg:
            seeded_extra[key] = discord_cfg[key]
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
    for key, env_key in (("auto_thread", "DISCORD_AUTO_THREAD"), ("reactions", "DISCORD_REACTIONS")):
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


def _is_connected(config) -> bool:
    """Connected when DISCORD_BOT_TOKEN is set.
    Looks up ``hermes_cli.gateway.get_env_value`` at call time so tests can patch it (ambient env)."""
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("DISCORD_BOT_TOKEN") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs DiscordAdapter from a PlatformConfig."""
    return _adapter_module().DiscordAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="discord",
        label="Discord",
        adapter_factory=_build_adapter,
        check_fn=_discord_deps_present,
        ensure_deps_fn=_check_discord_requirements,
        is_connected=_is_connected,
        required_env=["DISCORD_BOT_TOKEN"],
        install_hint="Run `hermes setup` to install Discord support.",
        setup_fn=interactive_setup,
        # YAML→env bridge: ``discord:`` config keys → ``DISCORD_*`` env vars read via os.getenv().
        # YAML→env config bridge — owns the translation of ``config.yaml`` ``discord:`` keys
        # (require_mention, free_response_channels, auto_thread, reactions, ignored_channels,
        # allowed_channels, no_thread_channels, allow_mentions.*, reply_to_mode, thread_require_mention)
        # into ``DISCORD_*`` env vars that the adapter reads via ``os.getenv()``. Replaces the hardcoded
        # block that used to live in ``gateway/config.py``. Hook contract: #24836.
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="DISCORD_ALLOWED_USERS",
        allow_all_env="DISCORD_ALLOW_ALL_USERS",
        cron_deliver_env_var="DISCORD_HOME_CHANNEL",
        # Out-of-process cron delivery via REST, else ``deliver=discord`` jobs fail with "No live adapter".
        standalone_sender_fn=_standalone_send,
        max_message_length=2000,
        emoji="🎮",
        allow_update_command=True,
    )
