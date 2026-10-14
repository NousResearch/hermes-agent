"""Conservative registration selection, separate from login autofill.

Selection is not consent. Callers must obtain human approval, pin the owned
original document and enforce exact origin before resolving or writing a secret.
"""
from __future__ import annotations

import re
from typing import Iterable

from agent.vault_login_classifier import LoginControl

_CONFIRM = re.compile(r"\b(?:confirm|repeat|retype)\s*password\b", re.I)


def registration_pair(controls: Iterable[LoginControl]) -> tuple[LoginControl, LoginControl] | None:
    """Recognize only one new-password/confirmation pair in the same form.

    Refuse password-change, OTP, ambiguous and unassociated controls. Existing
    login classification and its single-current-password behavior are untouched.
    """
    passwords = [c for c in controls if c.type == "password"]
    if len(passwords) != 2 or len({c.index for c in passwords}) != 2:
        return None
    if any(set(c.autocomplete.lower().split()) & {"current-password", "one-time-code"}
           for c in passwords):
        return None
    if any(c.max_length is not None and 0 < c.max_length < 8 for c in passwords):
        return None
    confirmations = [c for c in passwords if _CONFIRM.search(c.name + " " + c.label)]
    if len(confirmations) != 1:
        return None
    confirm = confirmations[0]
    new = next(c for c in passwords if c.index != confirm.index)
    if "new-password" not in new.autocomplete.lower().split():
        return None
    if new.form_index is None or new.form_index != confirm.form_index:
        return None
    return new, confirm


def confirm_registration(label: str, origin: str) -> bool:
    """Require an explicit human decision; headless/expired prompts fail closed."""
    from tools.approval_prompt import request_elicitation_consent

    return request_elicitation_consent(
        f"Create account using saved login '{label}' on {origin}",
        "Enter the saved password into the new-password and confirmation fields "
        "of this registration form. This does not submit registration. Approve "
        "only if you intend to create this account. No origin binding is changed.",
        surface="vault-registration", title="Confirm registration password fill?",
    ) == "accept"
