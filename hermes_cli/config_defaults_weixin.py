"""Weixin channel defaults shared by the CLI configuration registry."""

DEFAULT_WEIXIN_CONFIG = {"extra": {
    "allow_all_users": False, "use_platform_transcription": True, "reply_progress_messages": True, "bot_agent": "Hermes",
    "route_tag": None,
    "accounts": {}, "default_account": None,
    "block_streaming": {"min_chars": 200, "idle_ms": 3000},
    "quote_cache": {"enabled": True, "retention_days": 30, "max_messages_per_account": 10_000,
                    "media_retention_days": 7, "max_media_bytes_per_account": 256 * 1024 * 1024,
                    "max_single_media_bytes": 25 * 1024 * 1024},
}}
