from __future__ import annotations

from .models import Classification, Observation

_URGENT_WORDS = ("interview", "screening", "phone call", "schedule", "next step")
_PROMO_WORDS = ("discount", "sale", "offer", "unsubscribe")


def urgent_matches(item: Observation) -> bool:
    sender = item.sender.casefold()
    subject = item.subject.casefold()
    return not sender.startswith(("noreply@", "donotreply@")) and any(word in subject for word in _URGENT_WORDS)


def classify(item: Observation) -> Classification:
    subject = item.subject.casefold()
    category, confidence = ("promo", 0.95) if any(word in subject for word in _PROMO_WORDS) else ("uncertain", 0.5)
    return Classification(item, category, confidence, urgent_matches(item))
