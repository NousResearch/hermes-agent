"""Email env -> gateway config bridge (``PlatformEntry.env_overrides``)."""

from gateway.config import Platform, is_email_send_only, _getenv_str
from gateway.platform_env import EnvCred, home_step

ENV_ENABLE_CREDENTIALS = ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST")


def _email_credentials(config):
    """Require IMAP unless the resolved Email config explicitly selects send-only."""
    existing = config.platforms.get(Platform.EMAIL)
    credentials = ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST")
    if not is_email_send_only(existing.extra if existing else None):
        credentials += ("EMAIL_IMAP_HOST",)
    fixed = (("address", "EMAIL_ADDRESS"), ("smtp_host", "EMAIL_SMTP_HOST"))
    if _getenv_str("EMAIL_IMAP_HOST"):
        fixed += (("imap_host", "EMAIL_IMAP_HOST"),)
    EnvCred(Platform.EMAIL, credentials, fixed=fixed)(config)


ENV_OVERRIDES = (
    _email_credentials,
    home_step(Platform.EMAIL, "EMAIL_HOME_ADDRESS"),
)

# Authorization env names (``PlatformEntry`` fields). Core reads them from this file when the
# live entry lacks them: a user plugin overriding this platform, safe mode, the plugin disabled.
AUTHZ = {
    "allowed_users_env": "EMAIL_ALLOWED_USERS",
    "allow_all_env": "EMAIL_ALLOW_ALL_USERS",
    "allowlist_suppresses_pairing": True,
}
