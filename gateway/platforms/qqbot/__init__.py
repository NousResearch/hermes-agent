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
