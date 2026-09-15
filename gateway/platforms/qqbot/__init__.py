"""QQBot platform package. Re-exports adapter symbols so existing import paths
(``from gateway.platforms.qqbot import QQAdapter, check_qq_requirements``) keep working.
Sub-modules: constants, utils, crypto (AES-256-GCM), onboard (QR), chunked_upload, keyboards,
outbound (shared REST client), standalone (out-of-process sender)."""

from .adapter import QQAdapter, QQCloseError, check_qq_requirements, _coerce_list, _ssrf_redirect_guard  # noqa: F401
from .onboard import BindStatus, build_connect_url, qr_register  # noqa: F401
from .crypto import decrypt_secret, generate_bind_key  # noqa: F401
from .utils import build_user_agent, get_api_headers, coerce_list  # noqa: F401
from .chunked_upload import ChunkedUploader, UploadDailyLimitExceededError, UploadFileTooLargeError  # noqa: F401
from .outbound import QQApiClient, classify_media_type, resolve_target, split_for_qq  # noqa: F401
from .standalone import _standalone_send  # noqa: F401
from .keyboards import (  # noqa: F401
    ApprovalRequest, InlineKeyboard, InteractionEvent, build_approval_keyboard, build_approval_text,
    build_update_prompt_keyboard, parse_approval_button_data, parse_interaction_event,
    parse_update_prompt_button_data,
)

# Register QQBot in the platform registry so send_message can reach the standalone
# sender when no live adapter exists in this process (CLI/cron): built-in adapters are
# instantiated directly by gateway.run, so without this entry `_send_via_adapter` has no
# fallback outside the gateway. Idempotent — an existing (plugin) registration wins.


def _register_qqbot_in_registry() -> None:
    from gateway.platform_registry import PlatformEntry, platform_registry

    if platform_registry.is_registered("qqbot"):
        return
    platform_registry.register(PlatformEntry(
        name="qqbot",
        label="QQBot",
        adapter_factory=lambda cfg: QQAdapter(cfg),
        check_fn=check_qq_requirements,
        standalone_sender_fn=_standalone_send,
        cron_deliver_env_var="QQBOT_HOME_CHANNEL"))


_register_qqbot_in_registry()

__all__ = [
    "QQAdapter", "QQCloseError", "check_qq_requirements", "_coerce_list", "_ssrf_redirect_guard",
    "BindStatus", "build_connect_url", "qr_register",
    "decrypt_secret", "generate_bind_key",
    "build_user_agent", "get_api_headers", "coerce_list",
    "ChunkedUploader", "UploadDailyLimitExceededError", "UploadFileTooLargeError",
    "QQApiClient", "classify_media_type", "resolve_target", "split_for_qq", "_standalone_send",
    "ApprovalRequest", "InlineKeyboard", "InteractionEvent",
    "build_approval_keyboard", "build_approval_text", "build_update_prompt_keyboard",
    "parse_approval_button_data", "parse_interaction_event", "parse_update_prompt_button_data",
]
