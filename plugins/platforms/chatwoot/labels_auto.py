"""Automatic Chatwoot conversation labeling on every turn.

The ``chatwoot-conversation-labels`` skill instructs the agent to call
``chatwoot_labels``, but models often skip optional triage steps. This module
applies labels deterministically via ``post_llm_call`` so inbox tags appear even
when the agent never invokes the tool.

Classification is keyword/heuristic-based (fast, no extra LLM call). The skill
remains the source of truth for nuanced multi-label rules; this hook covers the
common CRWD Coach intents.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from plugins.platforms.chatwoot.labels import PREDEFINED_LABEL_TITLES
from plugins.platforms.chatwoot.labels_tool import (
    _assign_labels,
    _create_labels_if_not_exists,
    _resolve_conversation,
    check_chatwoot_labels_requirements,
)

logger = logging.getLogger(__name__)

_MAX_LABELS = 3

# (label, patterns) — first match wins per label; multiple labels can apply.
_LABEL_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "handoff-escalation",
        (
            r"\bhand\s*off\b",
            r"\bhuman\s+agent\b",
            r"\bloop\s+in\b",
            r"\bfrustrat",
            r"\bangry\b",
            r"\brejected\b",
            r"\bdispute\b",
            r"\bchargeback\b",
            r"\brefund\b",
            r"\bban(?:ned|s)?\b",
            r"\bsuspend",
        ),
    ),
    (
        "payment-payout",
        (
            r"\bpaid\b",
            r"\bpayment\b",
            r"\bpayout\b",
            r"\bwhere(?:'s| is) my money\b",
            r"\bwhen will i (?:get|be) paid\b",
            r"\bpayment history\b",
            r"\bdot\b",
        ),
    ),
    (
        "troubleshooting",
        (
            r"\bwon'?t load\b",
            r"\bbroken\b",
            r"\bnot working\b",
            r"\bdoesn'?t work\b",
            r"\bcan'?t (?:open|load|click)\b",
            r"\berror\b",
            r"\bbug\b",
            r"\bstuck\b",
        ),
    ),
    (
        "app-navigation",
        (
            r"\bwhere (?:is|do i find)\b",
            r"\bhow do i (?:find|open|get to)\b",
            r"\bhome tab\b",
            r"\bexplore tab\b",
            r"\bin the app\b",
            r"\bnavigate\b",
        ),
    ),
    (
        "account-membership",
        (
            r"\bmy account\b",
            r"\bmembership\b",
            r"\baccount status\b",
            r"\bdeactivat",
        ),
    ),
    (
        "gig-discovery",
        (
            r"\bfind (?:a )?gig",
            r"\bbrowse\b",
            r"\bavailable gig",
            r"\bnear me\b",
            r"\bnew gig",
            r"\bdiscover\b",
            r"\bwhat gig",
            r"\bany gig",
        ),
    ),
    (
        "gig-execution",
        (
            r"\bdetails? about\b",
            r"\bgig details?\b",
            r"\bsubmit\b",
            r"\bproof\b",
            r"\bsubmission\b",
            r"\bdeadline\b",
            r"\brequirements?\b",
            r"\bcomplete (?:the )?gig\b",
            r"\bamazon gig\b",
            r"\btell me about (?:the )?\w+ gig\b",
            r"\bgive me details\b",
            r"\bhow (?:do|to) (?:i )?(?:complete|do)\b",
        ),
    ),
)

_COMPILED_RULES: Tuple[Tuple[str, Tuple[re.Pattern[str], ...]], ...] = tuple(
    (label, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for label, patterns in _LABEL_RULES
)


def _is_chatwoot(platform: Any) -> bool:
    return str(platform or "").strip().lower() == "chatwoot"


def _text_for_classification(user_message: str, conversation_history: Sequence[Any]) -> str:
    """Build lowercase text from the latest user message plus recent user turns."""
    parts: List[str] = []
    if user_message and user_message.strip():
        parts.append(user_message.strip())
    if conversation_history:
        user_turns = 0
        for msg in reversed(conversation_history):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
            user_turns += 1
            if user_turns >= 3:
                break
    return " ".join(parts).lower()


def classify_conversation_labels(
    user_message: str = "",
    conversation_history: Optional[Sequence[Any]] = None,
) -> List[str]:
    """Return 1–3 predefined label titles for the conversation text."""
    text = _text_for_classification(user_message, conversation_history or ())
    if not text.strip():
        return ["general-inquiry"]

    matched: List[str] = []
    for label, patterns in _COMPILED_RULES:
        if label not in PREDEFINED_LABEL_TITLES:
            continue
        if any(p.search(text) for p in patterns):
            matched.append(label)
        if len(matched) >= _MAX_LABELS:
            break

    # Broad gig mention without a finer match → gig-execution (specific gig ask)
    # or gig-discovery (browsing).
    if not matched and re.search(r"\bgig", text, re.IGNORECASE):
        if re.search(r"\b(details?|about|submit|proof|complete|amazon)\b", text, re.IGNORECASE):
            matched.append("gig-execution")
        else:
            matched.append("gig-discovery")

    if not matched:
        matched.append("general-inquiry")

    return matched[:_MAX_LABELS]


def auto_label_conversation(
    user_message: str = "",
    conversation_history: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Bootstrap labels and assign classified tags to the current conversation."""
    if not check_chatwoot_labels_requirements():
        return {"success": False, "skipped": True, "reason": "chatwoot not configured"}

    account_id, conversation_id = _resolve_conversation()
    if not account_id or not conversation_id:
        return {"success": False, "skipped": True, "reason": "no chatwoot conversation"}

    labels = classify_conversation_labels(user_message, conversation_history)
    bootstrap = _create_labels_if_not_exists(account_id)
    if not bootstrap.get("success") and not bootstrap.get("existing"):
        return {
            "success": False,
            "skipped": False,
            "labels": labels,
            "error": bootstrap.get("error"),
        }

    result = _assign_labels(account_id, conversation_id, labels, replace=True)
    result["classified"] = labels
    result["skipped"] = False
    if not result.get("success"):
        logger.warning(
            "[chatwoot-labels-auto] assign failed for %s:%s — %s",
            account_id,
            conversation_id,
            result.get("error"),
        )
    else:
        logger.info(
            "[chatwoot-labels-auto] applied %s to conversation %s:%s",
            labels,
            account_id,
            conversation_id,
        )
    return result


def labeling_reminder_hook(**kwargs: Any) -> Optional[Dict[str, str]]:
    """``pre_llm_call`` — remind the agent that labels are auto-applied post-turn."""
    if not _is_chatwoot(kwargs.get("platform")):
        return None
    if not check_chatwoot_labels_requirements():
        return None
    return {
        "context": (
            "[Chatwoot triage] Conversation labels are applied automatically after "
            "each turn. You may also call `chatwoot_labels` with `assign_labels` if "
            "you want to override the auto-classification. Do not mention labels to "
            "the member."
        ),
    }


def auto_label_hook(**kwargs: Any) -> None:
    """``post_llm_call`` — classify and assign labels every Chatwoot turn."""
    if not _is_chatwoot(kwargs.get("platform")):
        return
    try:
        auto_label_conversation(
            user_message=str(kwargs.get("user_message") or ""),
            conversation_history=kwargs.get("conversation_history"),
        )
    except Exception as exc:
        logger.warning("[chatwoot-labels-auto] hook failed: %s", exc)
