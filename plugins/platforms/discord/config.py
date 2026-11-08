"""Discord activity settings resolved in the owning profile's scope."""

from __future__ import annotations

from typing import Any, Callable


def seed_presence_config(discord_cfg: dict, seeded_extra: dict) -> None:
    """Public keys take precedence over the legacy ``discord.extra`` form."""
    extra = discord_cfg.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    for key in ("activity", "activity_type"):
        if key in discord_cfg:
            seeded_extra[key] = discord_cfg[key]
        elif key in extra:
            seeded_extra[key] = extra[key]


def build_presence(config: Any, discord_module: Any, extra_or_secret: Callable) -> Any:
    """Resolve env precedence and construct an activity before connect leaves its profile scope."""
    extra = getattr(config, "extra", None)
    extra = extra if isinstance(extra, dict) else {}
    text = extra_or_secret(extra, "activity", "DISCORD_ACTIVITY", None)
    raw_type = extra_or_secret(extra, "activity_type", "DISCORD_ACTIVITY_TYPE", "playing")
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    activity_type = str(raw_type or "playing").strip().lower()
    if activity_type == "watching":
        return discord_module.Activity(type=discord_module.ActivityType.watching, name=text)
    if activity_type == "listening":
        return discord_module.Activity(type=discord_module.ActivityType.listening, name=text)
    if activity_type == "custom":
        return discord_module.CustomActivity(name=text)
    return discord_module.Game(name=text)


async def apply_presence(client: Any, presence: Any, adapter_name: str, discord_module: Any, logger: Any) -> None:
    """Discord presence is best effort; transport failures must not block readiness."""
    import aiohttp

    if presence is None:
        return
    try:
        await client.change_presence(activity=presence)
    except (discord_module.HTTPException, discord_module.ClientException, aiohttp.ClientError,
            OSError, TimeoutError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("[%s] Failed to set Discord activity: %s", adapter_name, exc)
