"""Registered vault values must not redact innocent dates or account numbers.

Repro of the bug this pins (fixed 2026-10-14):

    register_vault_redaction_value("12")   -> redact_registered_vault_values("12/10/2024")
                                            == "«redacted-vault-secret»/10/2024"   # month eaten
    register_vault_redaction_value("100")  -> "Business Platinum Card (-21008)"
                                            == "...(-2«redacted-vault-secret»8)"    # digits eaten

`redact_registered_vault_values` is a catch-all for a secret echoed back into tool output, so short
registered values (a CVC-length string, an expiry fragment like "12/27") are common — and plain
`str.replace` has neither word boundaries nor a date guard. The fix keeps the protection: a value is
still masked at a delimiter boundary, just never inside a date/time token or a longer digit run.
"""
import pytest

from agent.redact import (
    clear_vault_redaction_values,
    redact_registered_vault_values as redact,
    register_vault_redaction_value as register,
)

MASK = "«redacted-vault-secret»"


@pytest.fixture(autouse=True)
def _clean_values():
    clear_vault_redaction_values()
    yield
    clear_vault_redaction_values()


def test_short_value_does_not_eat_a_date():
    register("12")
    assert redact("12/10/2024 - 12/12/2025") == "12/10/2024 - 12/12/2025"


def test_short_value_does_not_eat_an_account_number():
    register("100")
    assert redact("Business Platinum Card (-21008)") == "Business Platinum Card (-21008)"


def test_short_value_does_not_eat_an_amount():
    register("100")
    assert redact("total 1,100.50") == "total 1,100.50"


def test_expiry_fragment_is_not_redacted_inside_a_full_date():
    register("12/27")
    assert redact("exp 12/27/2026") == "exp 12/27/2026"
    assert redact("expiry 12/27") == f"expiry {MASK}"      # standalone: still the secret


def test_short_value_still_masks_when_standalone():
    register("100")
    assert redact("CVC 100 echo") == f"CVC {MASK} echo"


def test_long_value_still_masks_verbatim():
    register("4111111111111111")
    assert redact("pan=4111111111111111 done") == f"pan={MASK} done"
    register("hunter2secret")
    assert redact("hunter2secret used") == f"{MASK} used"
