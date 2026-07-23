"""
QQBot platform package.

Re-exports the main adapter symbols from ``adapter.py`` (the original
``qqbot.py``) so that **all existing import paths remain unchanged**::

    from gateway.platforms.qqbot import QQAdapter          # works
    from gateway.platforms.qqbot import check_qq_requirements  # works

New modules:
    - ``constants`` — shared constants (API URLs, timeouts, message types)
    - ``utils`` — User-Agent builder, config helpers
    - ``crypto`` — AES-256-GCM key generation and decryption
    - ``onboard`` — QR-code scan-to-configure flow
    - ``outbound`` — shared REST API client (QQApiClient)
    - ``standalone`` — out-of-process sender
"""

# -- Adapter (original qqbot.py) ------------------------------------------
from .adapter import (  # noqa: F401
    QQAdapter,
    QQCloseError,
    check_qq_requirements,
    _coerce_list,
    _ssrf_redirect_guard,
)

# -- Onboard (QR-code scan-to-configure) -----------------------------------
from .onboard import (  # noqa: F401
    BindStatus,
    build_connect_url,
    qr_register,
)
from .crypto import decrypt_secret, generate_bind_key  # noqa: F401

# -- Utils -----------------------------------------------------------------
from .utils import build_user_agent, get_api_headers, coerce_list  # noqa: F401

# -- Chunked upload --------------------------------------------------------
from .chunked_upload import (  # noqa: F401
    ChunkedUploader,
    UploadDailyLimitExceededError,
    UploadFileTooLargeError,
)

# -- Shared outbound (QQApiClient) -----------------------------------------
from .outbound import QQApiClient, classify_media_type, resolve_target, split_for_qq  # noqa: F401

# -- Standalone (out-of-process) sender ------------------------------------
from .standalone import _standalone_send  # noqa: F401

# -- Platform registry registration -----------------------------------------
# Register QQBot in the platform_registry so tools/send_message_tool.py can
# find the standalone_sender_fn when no live adapter is present.  This is a
# transitional step toward the bundled-plugin migration in PR #41065.
# Uses the real QQAdapter factory and check from adapter.py so the
# entry is valid and won't be shadowed by a future plugin registration.


def _register_qqbot_in_registry() -> None:
    """Register QQBot in the platform_registry (idempotent).

    Raises on failure so missing registration is noticed at startup,
    not silently swallowed until a standalone send fails 10 minutes later.
    """
    from gateway.platform_registry import PlatformEntry, platform_registry

    if platform_registry.is_registered("qqbot"):
        return

    platform_registry.register(
        PlatformEntry(
            name="qqbot",
            label="QQBot",
            adapter_factory=lambda cfg: QQAdapter(cfg),
            check_fn=check_qq_requirements,
            standalone_sender_fn=_standalone_send,
            cron_deliver_env_var="QQBOT_HOME_CHANNEL",
        )
    )


_register_qqbot_in_registry()

# -- Inline keyboards ------------------------------------------------------
from .keyboards import (  # noqa: F401
    ApprovalRequest,
    ApprovalSender,
    InlineKeyboard,
    InteractionEvent,
    build_approval_keyboard,
    build_approval_text,
    build_update_prompt_keyboard,
    parse_approval_button_data,
    parse_interaction_event,
    parse_update_prompt_button_data,
)

__all__ = [
    # adapter
    "QQAdapter",
    "QQCloseError",
    "check_qq_requirements",
    "_coerce_list",
    "_ssrf_redirect_guard",
    # onboard
    "BindStatus",
    "build_connect_url",
    "qr_register",
    # crypto
    "decrypt_secret",
    "generate_bind_key",
    # utils
    "build_user_agent",
    "get_api_headers",
    "coerce_list",
    # chunked upload
    "ChunkedUploader",
    "UploadDailyLimitExceededError",
    "UploadFileTooLargeError",
    # outbound
    "QQApiClient",
    "classify_media_type",
    "resolve_target",
    "split_for_qq",
    # standalone
    "_standalone_send",
    # keyboards
    "ApprovalRequest",
    "ApprovalSender",
    "InlineKeyboard",
    "InteractionEvent",
    "build_approval_keyboard",
    "build_approval_text",
    "build_update_prompt_keyboard",
    "parse_approval_button_data",
    "parse_interaction_event",
    "parse_update_prompt_button_data",
]
