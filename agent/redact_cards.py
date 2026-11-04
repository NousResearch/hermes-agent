"""Payment-card redaction for vault fills: grouped card numbers and the CVC of a live checkout page.

The exact-value registry in ``agent.redact`` matches the bytes a fill wrote, but card inputs reformat
what they receive (``4242424242424242`` becomes ``4242 4242 4242 4242`` or ``4242-4242-…``), and a
3-digit CVC registered as a substring masks every ``123`` in later output for the life of the process.

- A filled card number registers a digit pattern that tolerates one space, dash or dot between any two
  digits, so every grouping the page prints is masked; the mask keeps the last four digits, which are
  not secret and tell the model which card went in. Only a Luhn-valid 12–19 digit number registers:
  a short or malformed value would compile into a pattern that eats ordinary numbers.
- The CVC is masked only as a whole number and only while the page it was filled into is still the
  same document; ``tools.browser_vault_card_guard`` drops it on navigation.

Memory only, profile-scoped and bounded like the exact-value registry.
"""

from __future__ import annotations

import re
import threading

_MAX_CARDS_PER_PROFILE = 8
_SEPARATOR = r"[ .\-]?"
_CARD_PATTERNS: dict = {}  # profile home → ordered {digits: compiled pattern}
_PAGE_CVCS: dict = {}  # profile home → {page key: compiled whole-number pattern}
_LOCK = threading.Lock()


def _scope() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def register_card_number(number: str) -> bool:
    """Mask every grouping of ``number`` in later output. False (nothing registered) unless it is a
    Luhn-valid 12–19 digit card number."""
    digits = "".join(ch for ch in str(number or "") if ch.isdigit())
    if not 12 <= len(digits) <= 19 or not luhn_valid(digits):
        return False
    pattern = re.compile(r"(?<!\d)" + _SEPARATOR.join(digits) + r"(?!\d)")
    with _LOCK:
        bucket = _CARD_PATTERNS.setdefault(_scope(), {})
        bucket.pop(digits, None)
        bucket[digits] = pattern
        while len(bucket) > _MAX_CARDS_PER_PROFILE:
            del bucket[next(iter(bucket))]
    return True


def set_page_cvc(page_key: str, cvc: str) -> None:
    """Mask ``cvc`` as a whole number until ``drop_page_cvc(page_key)`` (the filled page navigated)."""
    digits = str(cvc or "").strip()
    if not digits.isdigit():
        return
    with _LOCK:
        _PAGE_CVCS.setdefault(_scope(), {})[page_key] = re.compile(r"(?<!\d)" + digits + r"(?!\d)")


def drop_page_cvc(page_key: str) -> None:
    with _LOCK:
        _PAGE_CVCS.get(_scope(), {}).pop(page_key, None)


def clear_card_redactions() -> None:
    with _LOCK:
        _CARD_PATTERNS.pop(_scope(), None)
        _PAGE_CVCS.pop(_scope(), None)


def redact_card_values(text: str) -> str:
    """Mask registered card numbers (any grouping, last 4 kept) and live-page CVCs in ``text``."""
    if not isinstance(text, str) or not text:
        return text
    scope = _scope()
    with _LOCK:
        cards = list(_CARD_PATTERNS.get(scope, {}).items())
        cvcs = list(_PAGE_CVCS.get(scope, {}).values())
    for digits, pattern in cards:
        text = pattern.sub(f"«redacted-card ••••{digits[-4:]}»", text)
    for pattern in cvcs:
        text = pattern.sub("«redacted-cvc»", text)
    return text
