"""Profile-local credential gates shared by startup and reconnect."""

from __future__ import annotations

from pathlib import Path
from gateway.config import Platform, PlatformConfig

def _platform_has_bot_credential(platform: Platform, platform_config: PlatformConfig) -> bool:
    """Return True when a token-authenticated platform has a usable bot credential; platforms not using
    ``PlatformConfig.token`` (Signal session paths, port-binding HTTP adapters) always return True."""
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES, Platform
    if platform is Platform.WEIXIN:
        from gateway.platforms.weixin_group import has_weixin_credentials
        return has_weixin_credentials(platform_config)
    if platform is Platform.WHATSAPP:
        from hermes_constants import get_hermes_dir
        session = Path(platform_config.extra.get(
            "session_path", get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")))
        return (session / "creds.json").exists()
    if platform not in PLATFORM_TOKEN_ENV_NAMES:
        return True
    for attr in ("token", "api_key"):  # some adapters accept api_key as the primary credential
        value = getattr(platform_config, attr, None) or ""
        if isinstance(value, str) and value.strip():
            return True
    # Matrix also authenticates by password; a token-only check would evict a reconnectable config from
    # the retry queue. Read ONLY extra (build_config() copies env there): env fallback = every config OK.
    # Those credentials land in ``extra`` rather than ``.token``, so a token-only check reads a perfectly
    # reconnectable password-auth config as credential-less and evicts it from the retry queue on the first
    # transient failure — after which it stays down until the gateway is restarted by hand. Mirror the
    # adapter's own gate: homeserver + user_id + password. Read ONLY from extra, never os.getenv:
    # build_config() already copies all three env vars onto extra, and importing this module loads
    # ~/.hermes/.env, so an env fallback would report "has credential" for every Matrix config on the box —
    # including the empty-primary multiplex case (#64674) this check exists to evict.
    if platform is not Platform.MATRIX:
        return False
    extra = getattr(platform_config, "extra", None) or {}
    return all(str(extra.get(key) or "").strip() for key in ("homeserver", "user_id", "password"))
