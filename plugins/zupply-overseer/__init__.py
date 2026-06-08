"""Zupply Overseer plugin.

Deterministic preflight rules for known Zupply business-process failures.
This is intentionally hard-rule first: if a known-bad workflow is detected,
block before the tool executes and tell the agent which approved workflow to use.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional


_APPROVED_COURIER_PATHS = ("gosweetspot", "bookpickup", "book-driver", "book driver", "/api/bookpickup")
_EMAIL_SEND_MARKERS = (
    "himalaya",
    "send email",
    "send_email",
    "smtp",
    "mail send",
    "send mail",
    "email",
)
_COURIER_PICKUP_MARKERS = (
    "ping nzc",
    "nzc",
    "nz couriers",
    "new zealand couriers",
    "courier pickup",
    "courier collect",
    "collection request",
    "request to collect",
    "pickup request",
    "book pickup",
    "book driver",
)
_PAYMENT_EXECUTION_MARKERS = (
    "submit transfer",
    "execute payment",
    "approve payment",
    "make payment",
    "send payment",
    "pay now",
    "mark paid",
    "mark as paid",
    "approve bill",
    "authorise bill",
    "authorize bill",
)
_PAYMENT_SYSTEM_MARKERS = ("wise", "xero", "rewardpay", "payment", "bill")
_BRANDED_PDF_MARKERS = ("ghostmark", "pdf-generator", "zupply-pdf-generator", "branded", "locked template")
_ZUPPLY_DOC_MARKERS = ("zupply", "quote-7", "po-7", "quote ", "purchase order", "customer quote")
_DOC_WRITE_PATH_MARKERS = (
    "/zupply-os-vault/",
    "/operations/",
    "/templates/",
    "/suppliers/",
    "/customers/",
)
_SIDE_EFFECT_TOOLS = {
    "terminal",
    "execute_code",
    "browser_click",
    "browser_type",
    "browser_press",
    "computer_use",
    "send_message",
    "write_file",
    "patch",
    "skill_manage",
}


def _disabled() -> bool:
    return os.environ.get("ZUPPLY_OVERSEER_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}


def _collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _collect_strings(v)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _collect_strings(item)
    elif value is not None:
        yield str(value)


def _haystack(tool_name: str, args: Dict[str, Any]) -> str:
    return (tool_name + "\n" + "\n".join(_collect_strings(args))).lower()


def _has_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def _external_send_context(text: str) -> bool:
    return _has_any(text, _EMAIL_SEND_MARKERS) or "send_message" in text


def _is_documentation_write(tool_name: str, args: Dict[str, Any]) -> bool:
    """Allow notes/docs that merely reference PDFs; they are not final PDF creation/delivery."""
    if tool_name not in {"write_file", "patch", "skill_manage"}:
        return False
    paths = [str(args.get(k, "")).lower() for k in ("path", "file_path") if args.get(k)]
    return any(path.endswith(".md") and _has_any(path, _DOC_WRITE_PATH_MARKERS) for path in paths)


def _rule_courier_pickup_email(text: str) -> Optional[str]:
    if not _has_any(text, _COURIER_PICKUP_MARKERS):
        return None
    if _has_any(text, _APPROVED_COURIER_PATHS):
        return None
    if not _external_send_context(text):
        return None
    return (
        "Zupply Overseer blocked this action: courier pickup/collection pings must not be emailed. "
        "Use GoSweetSpot book-driver/bookpickup instead. For NZC/NZ Couriers, normalize carrier to "
        "`NZCouriers`, find outstanding unpicked consignments, then POST `/api/bookpickup`."
    )


def _rule_payment_execution(text: str) -> Optional[str]:
    if not _has_any(text, _PAYMENT_SYSTEM_MARKERS):
        return None
    if not _has_any(text, _PAYMENT_EXECUTION_MARKERS):
        return None
    if "draft" in text or "review" in text or "prep" in text or "prepare" in text:
        # Zupply allows review-only Wise packs and Xero draft bills.
        return None
    return (
        "Zupply Overseer blocked this action: payment/bill execution is not allowed. "
        "Prepare review packs or Xero DRAFT bills only; never submit transfers, approve bills, "
        "make payments, or mark paid unless Logan explicitly confirms outside the tool action."
    )


def _extract_margin_values(text: str) -> List[float]:
    values: List[float] = []
    patterns = [
        r"\bmargin\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*%",
        r"\bgross\s+margin\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return values


def _rule_quote_margin(text: str) -> Optional[str]:
    if "quote" not in text and "sell price" not in text and "customer-facing" not in text:
        return None
    margins = _extract_margin_values(text)
    low = [m for m in margins if m < 20.0]
    if not low:
        return None
    low_txt = ", ".join(f"{m:g}%" for m in low)
    return (
        f"Zupply Overseer blocked this action: customer quote margin below 20% detected ({low_txt}). "
        "Minimum margin is 20%, target margin is 28%. Reprice or get explicit Logan approval."
    )


def _rule_plain_pdf(text: str) -> Optional[str]:
    looks_like_pdf_create = ".pdf" in text and _has_any(text, _ZUPPLY_DOC_MARKERS)
    if not looks_like_pdf_create:
        return None
    if _has_any(text, _BRANDED_PDF_MARKERS):
        return None
    if "rough" in text or "internal" in text or "preview" in text:
        return None
    return (
        "Zupply Overseer blocked this action: final Zupply PDFs must use the branded Ghostmark/PDF-generator workflow. "
        "Do not create or deliver plain PDFs for customer quotes or POs. Use the Zupply PDF generator and QA totals/spacing first."
    )


def evaluate_tool_call(tool_name: str, args: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return block message for a policy violation, else None."""
    if _disabled():
        return None
    if tool_name not in _SIDE_EFFECT_TOOLS:
        return None
    safe_args = args if isinstance(args, dict) else {}
    text = _haystack(tool_name, safe_args)
    for rule in (
        _rule_courier_pickup_email,
        _rule_payment_execution,
        _rule_quote_margin,
    ):
        message = rule(text)
        if message:
            return message
    if not _is_documentation_write(tool_name, safe_args):
        message = _rule_plain_pdf(text)
        if message:
            return message
    return None


def _on_pre_tool_call(tool_name: str, args: Optional[Dict[str, Any]], **_: Any) -> Optional[Dict[str, str]]:
    message = evaluate_tool_call(tool_name, args)
    if not message:
        return None
    return {"action": "block", "message": message}


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
