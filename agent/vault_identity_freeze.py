"""Origin-bound vision / CDP freeze after a successful identity vault fill.

While frozen on origin O, ``browser_vision`` and screenshot / input-value CDP
reads refuse so a filled SSN cannot be recovered from a screenshot or
``input.value``. The freeze clears when the current page origin is no longer O.
If the origin cannot be determined, fail closed and keep the freeze.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_frozen_origin: Optional[str] = None

_RE_INPUT_VALUE = re.compile(
    r"input\.value|\.value\b|querySelector(?:All)?\s*\([^)]*\)\s*\.value",
    re.I,
)


def activate_identity_freeze(origin: str) -> None:
    global _frozen_origin
    with _LOCK:
        _frozen_origin = origin


def clear_identity_freeze() -> None:
    global _frozen_origin
    with _LOCK:
        _frozen_origin = None


def _probe_origin(task_id: Optional[str]) -> Optional[str]:
    try:
        from tools.browser_vault_tool import _current_page_origin

        return _current_page_origin(task_id or "default")
    except Exception:
        return None


def _still_frozen(task_id: Optional[str]) -> bool:
    with _LOCK:
        frozen = _frozen_origin
    if not frozen:
        return False
    current = _probe_origin(task_id)
    if current is None:
        return True
    if current != frozen:
        clear_identity_freeze()
        return False
    return True


def _refusal(message: str) -> str:
    return json.dumps(
        {
            "success": False,
            "error_type": "identity_vision_frozen",
            "error": message,
        }
    )


def refuse_if_identity_frozen(task_id: Optional[str] = None) -> Optional[str]:
    if not _still_frozen(task_id):
        return None
    with _LOCK:
        origin = _frozen_origin
    return _refusal(
        f"Identity fields were filled on {origin}; screenshots and vision are frozen "
        "until the page origin changes."
    )


def expression_reads_input_value(expression: str) -> bool:
    return bool(_RE_INPUT_VALUE.search(expression or ""))


def refuse_identity_cdp(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Optional[str]:
    if not _still_frozen(task_id):
        return None
    method = (method or "").strip()
    if method == "Page.captureScreenshot":
        return _refusal(
            "Identity fill is frozen: Page.captureScreenshot is refused until the page origin changes."
        )
    if method == "Runtime.evaluate" and expression_reads_input_value(
        str((params or {}).get("expression") or "")
    ):
        return _refusal(
            "Identity fill is frozen: reading input values via Runtime.evaluate is refused "
            "until the page origin changes."
        )
    return None
