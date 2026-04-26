"""Error card renderer for Feishu streaming cards.

Builds red-framed error cards with user-facing messages, recommended
actions, and optional retry buttons that route through the existing
Feishu card callback mechanism.
"""

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Error type constants
# ---------------------------------------------------------------------------

ERROR_NEED_AUTHORIZATION = "NeedAuthorizationError"
ERROR_APP_SCOPE_MISSING = "AppScopeMissingError"
ERROR_USER_AUTH_REQUIRED = "UserAuthRequiredError"
ERROR_USER_SCOPE_INSUFFICIENT = "UserScopeInsufficientError"
ERROR_GENERIC = "GenericError"

# ---------------------------------------------------------------------------
# Per-error-type configuration
# ---------------------------------------------------------------------------

# Each entry:
#   title:      Card header title (zh)
#   message:    User-facing explanation (zh)
#   advice:     Recommended action (zh)
#   action_id:  Feishu card callback action_id for the retry button (or None)
#   retry_label: Button label (zh), shown only when action_id is set
_ERROR_CONFIG: dict[str, dict[str, Any]] = {
    ERROR_NEED_AUTHORIZATION: {
        "title": "需要授权",
        "message": "当前操作需要您完成账号授权，请先运行 /setup 进行授权，然后重试。",
        "advice": "运行 /setup 完成授权后重试",
        "action_id": "retry_after_setup",
        "retry_label": "重新授权",
    },
    ERROR_APP_SCOPE_MISSING: {
        "title": "应用权限不足",
        "message": "应用缺少所需的 API 权限范围，请联系管理员在飞书开放平台添加对应权限。",
        "advice": "联系管理员开通应用权限",
        "action_id": None,
        "retry_label": None,
    },
    ERROR_USER_AUTH_REQUIRED: {
        "title": "需要用户授权",
        "message": "此操作需要使用您的身份（user_access_token），请先完成用户授权。",
        "advice": "运行 /setup 完成用户授权后重试",
        "action_id": "retry_after_user_auth",
        "retry_label": "去授权",
    },
    ERROR_USER_SCOPE_INSUFFICIENT: {
        "title": "用户权限不足",
        "message": "您的账号没有执行此操作所需的权限，请联系空间管理员授权或切换账号。",
        "advice": "联系管理员赋予相关权限，或切换有权限的账号",
        "action_id": "retry_action",
        "retry_label": "重试",
    },
    ERROR_GENERIC: {
        "title": "出错了",
        "message": "操作执行时发生错误，请稍后重试。",
        "advice": "稍后重试，若问题持续请联系管理员",
        "action_id": "retry_action",
        "retry_label": "重试",
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate a string to max_len chars, appending ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _build_header(title: str) -> dict[str, Any]:
    """Build a red card header block."""
    return {
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"❌ **{title}**",
        },
        "extra": {
            "tag": "img",
            "img_key": "",  # placeholder; real deployments supply an icon
            "alt": {"tag": "plain_text", "content": "error"},
        },
    }


def _build_divider() -> dict[str, Any]:
    return {"tag": "hr"}


def _build_text_block(content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": content,
        },
    }


def _build_retry_button(label: str, action_id: str, extra: Optional[dict] = None) -> dict[str, Any]:
    """Build a Feishu card button that fires a callback on click."""
    value: dict[str, Any] = {"action": action_id}
    if extra:
        value.update(extra)
    return {
        "tag": "button",
        "text": {
            "tag": "plain_text",
            "content": label,
        },
        "type": "danger",
        "value": value,
    }


def _build_action_row(button: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag": "action",
        "actions": [button],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_error_card(
    error_type: str,
    message: Optional[str] = None,
    retry_action: Optional[str] = None,
    extra_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a Feishu card JSON payload for an error state.

    Args:
        error_type: One of the ``ERROR_*`` constants defined in this module.
                    Unknown values fall back to ``ERROR_GENERIC``.
        message:    Optional override for the user-facing error detail.
                    When provided, appended after the default explanation.
        retry_action: Optional override for the callback ``action_id`` on
                      the retry button. When ``None``, the per-type default
                      is used (if any).
        extra_context: Optional dict merged into the button ``value`` payload,
                       e.g. ``{"thread_id": "xxx"}`` for routing callbacks.

    Returns:
        A Feishu interactive card dict ready to be serialised as JSON and
        sent via the card message API.
    """
    config = _ERROR_CONFIG.get(error_type, _ERROR_CONFIG[ERROR_GENERIC])

    title: str = config["title"]
    base_message: str = config["message"]
    advice: str = config["advice"]
    action_id: Optional[str] = retry_action or config["action_id"]
    retry_label: Optional[str] = config["retry_label"]

    # Build message body
    body_lines = [f"> {base_message}"]
    if message:
        body_lines.append(f"\n**详情：** {_truncate(message)}")
    body_lines.append(f"\n💡 **建议：** {advice}")
    body_content = "\n".join(body_lines)

    elements: list[dict[str, Any]] = [
        _build_header(title),
        _build_divider(),
        _build_text_block(body_content),
    ]

    # Append retry button row if applicable
    if action_id and retry_label:
        btn = _build_retry_button(retry_label, action_id, extra=extra_context)
        elements.append(_build_action_row(btn))

    card: dict[str, Any] = {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
        },
        "header": {
            "template": "red",
            "title": {
                "tag": "plain_text",
                "content": f"❌ {title}",
            },
        },
        "elements": elements,
    }

    return card


def build_error_card_for_exception(
    exc: Exception,
    retry_action: Optional[str] = None,
    extra_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Convenience wrapper: infer error_type from exception class name.

    Args:
        exc: The caught exception instance.
        retry_action: Forwarded to ``build_error_card``.
        extra_context: Forwarded to ``build_error_card``.

    Returns:
        A Feishu error card dict.
    """
    type_name = type(exc).__name__
    # Map well-known exception names to our constants
    known_types = {
        ERROR_NEED_AUTHORIZATION,
        ERROR_APP_SCOPE_MISSING,
        ERROR_USER_AUTH_REQUIRED,
        ERROR_USER_SCOPE_INSUFFICIENT,
    }
    error_type = type_name if type_name in known_types else ERROR_GENERIC
    return build_error_card(
        error_type=error_type,
        message=str(exc) or None,
        retry_action=retry_action,
        extra_context=extra_context,
    )
