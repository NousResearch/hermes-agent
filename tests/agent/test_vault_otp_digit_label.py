"""OTP digit-count labels observed on Alaska's Auth0 challenge; no input values."""

import pytest

from agent.vault_login_classifier import LoginControl, classify_otp_controls


@pytest.mark.parametrize("label", [
    "Enter the 6-digit code",
    "Enter the 6-digit code *  Enter the 6-digit code *  ",
    "Enter a 4-digit code",
    "Enter 5 digit code",
    "ENTER THE 7-DIGIT CODE",
    "Enter 8 digit code",
])
def test_explicit_enter_digit_count_code_label(label):
    control = LoginControl.from_dict({
        "autocomplete": "off", "formIndex": 0, "index": 1, "maxLength": None,
        "label": label, "name": "code", "type": "text",
    })
    classified = classify_otp_controls([control])
    assert len(classified) == 1
    assert classified[0].control == control
    assert classified[0].token == "one-time-code"
    assert classified[0].score == 70


@pytest.mark.parametrize("label", [
    "Code", "Postal code", "Country code", "Promo code", "Security question",
    "Enter the 6-digit postal code", "Enter the 4-digit country code",
    "Enter the 8-digit promo code", "Enter the 3-digit code", "Enter the 9-digit code",
    "Enter the 16-digit code", "6-digit code", "Security preferences",
])
def test_unrelated_or_unspecific_code_labels_are_not_otp(label):
    control = LoginControl.from_dict({"label": label, "name": "code", "type": "text"})
    assert classify_otp_controls([control]) == []
