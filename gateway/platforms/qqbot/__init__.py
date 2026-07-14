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

# -- Standalone (out-of-process) sender ------------------------------------
from .standalone import _standalone_send  # noqa: F401

# -- Platform registry registration -----------------------------------------
# Register QQBot in the platform_registry so tools/send_message_tool.py can
# find the standalone_sender_fn when no live adapter is present.  This is a
# transitional step toward the bundled-plugin migration in PR #41065.
# Registry writes are idempotent — calling this more than once is harmless.


def _register_qqbot_in_registry() -> None:
    """Register QQBot in the platform_registry (idempotent)."""
    try:
        from gateway.platform_registry import PlatformEntry, platform_registry

        if platform_registry.is_registered("qqbot"):
            return

        platform_registry.register(
            PlatformEntry(
                name="qqbot",
                label="QQBot",
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=_standalone_send,
                cron_deliver_env_var="QQBOT_HOME_CHANNEL",
            )
        )
    except Exception:
        pass  # Non-fatal — standalone path will surface a clear error.


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
