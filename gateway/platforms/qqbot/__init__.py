"""QQBot platform package. Re-exports adapter symbols so existing import paths
(``from gateway.platforms.qqbot import QQAdapter, check_qq_requirements``) keep working.
Sub-modules: constants, utils, crypto (AES-256-GCM), onboard (QR), chunked_upload, keyboards."""

from .adapter import QQAdapter, QQCloseError, check_qq_requirements, _coerce_list, _ssrf_redirect_guard  # noqa: F401
from .onboard import BindStatus, build_connect_url, qr_register  # noqa: F401
from .crypto import decrypt_secret, generate_bind_key  # noqa: F401
from .utils import build_user_agent, get_api_headers, coerce_list  # noqa: F401
from .chunked_upload import ChunkedUploader, UploadDailyLimitExceededError, UploadFileTooLargeError  # noqa: F401
from .keyboards import (  # noqa: F401
    ApprovalRequest, InlineKeyboard, InteractionEvent, build_approval_keyboard, build_approval_text,
    build_update_prompt_keyboard, parse_approval_button_data, parse_interaction_event,
    parse_update_prompt_button_data,
)

__all__ = [
    "QQAdapter", "QQCloseError", "check_qq_requirements", "_coerce_list", "_ssrf_redirect_guard",
    "BindStatus", "build_connect_url", "qr_register",
    "decrypt_secret", "generate_bind_key",
    "build_user_agent", "get_api_headers", "coerce_list",
    "ChunkedUploader", "UploadDailyLimitExceededError", "UploadFileTooLargeError",
    "ApprovalRequest", "InlineKeyboard", "InteractionEvent",
    "build_approval_keyboard", "build_approval_text", "build_update_prompt_keyboard",
    "parse_approval_button_data", "parse_interaction_event", "parse_update_prompt_button_data",
]
