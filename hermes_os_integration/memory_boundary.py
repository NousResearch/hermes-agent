"""Guardrails for keeping Hermes OS authoritative over project state."""

from .errors import STATE_CONFLICT, adapter_error

AUTHORITATIVE_DOMAINS = {
    "projects",
    "tasks",
    "research",
    "experiments",
    "reports",
    "reviews",
}


def classify_memory_write(domain, intent):
    domain = str(domain)
    intent = str(intent)
    if domain in AUTHORITATIVE_DOMAINS and intent not in {"cache", "note"}:
        return None, adapter_error(
            STATE_CONFLICT,
            "Runtime cannot directly mutate Hermes OS authoritative domain: " + domain,
        )
    return {
        "domain": domain,
        "intent": intent,
        "source": "official-hermes-agent",
        "authoritative": False,
    }, None
