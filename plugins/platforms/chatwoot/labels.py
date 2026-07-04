"""Predefined Chatwoot label taxonomy for CRWD Coach triage.

Swapping labels for another inbox: edit this list and add a matching skill.
Titles are lowercase — Chatwoot normalizes label titles to lowercase.
"""

from __future__ import annotations

from typing import Any, Dict, List

PREDEFINED_LABELS: List[Dict[str, Any]] = [
    {
        "title": "gig-discovery",
        "description": "Finding or browsing gigs",
        "color": "#1f93ff",
    },
    {
        "title": "gig-execution",
        "description": "Doing a gig, submissions, proof",
        "color": "#47c479",
    },
    {
        "title": "payment-payout",
        "description": "Payment timing, payout status, Dot",
        "color": "#ffc53d",
    },
    {
        "title": "app-navigation",
        "description": "How to use the CRWD app",
        "color": "#7b68ee",
    },
    {
        "title": "troubleshooting",
        "description": "Broken links, pages, buttons",
        "color": "#ff6b6b",
    },
    {
        "title": "handoff-escalation",
        "description": "Human takeover needed",
        "color": "#c0392b",
    },
    {
        "title": "account-membership",
        "description": "Account status, bans, membership",
        "color": "#95a5a6",
    },
    {
        "title": "general-inquiry",
        "description": "Other questions",
        "color": "#bdc3c7",
    },
]

PREDEFINED_LABEL_TITLES = frozenset(
    str(entry["title"]).strip().lower() for entry in PREDEFINED_LABELS
)
