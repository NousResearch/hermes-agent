"""Masked card security codes are checkout targets unless authentication metadata vetoes them."""

import pytest

from agent.vault_login_classifier import (
    LoginControl,
    classify_checkout_control,
    classify_login_control,
    classify_otp_controls,
    select_checkout_fills,
)


@pytest.mark.parametrize("autocomplete,name,label", [
    ("cc-csc", "", ""),
    ("section-payment billing cc-csc", "", ""),
    ("", "", "Security code"),
    ("off", "cvv", ""),
    ("", "cvc", ""),
    ("", "csc", ""),
    ("", "", "Card code"),
])
def test_masked_cvv_classifies_and_fills_as_payment(autocomplete, name, label):
    control = LoginControl(autocomplete, 0, 2, label, name, "password")
    target = classify_checkout_control(control)
    assert target is not None
    assert target.token == "cc-csc"
    assert select_checkout_fills([target], {"cvc": "123"}, {"cvc": "cc-csc"}) == [
        {"index": 2, "token": "cc-csc", "value": "123"}
    ]


@pytest.mark.parametrize("autocomplete,name,label", [
    (token, "cvv", "Security code")
    for token in ("username", "email", "tel", "current-password", "new-password", "one-time-code")
] + [
    ("", "cvv", label) for label in (
        "Password", "Login", "Log in", "Sign in", "Username", "User name",
        "OTP", "TOTP", "2FA", "MFA", "Passcode", "One-time code",
        "Authentication", "Authenticator", "Two-factor code", "SMS",
        "Verification code", "Verification PIN", "Verification token",
        "Security PIN", "Security token",
    )
] + [
    ("", "verification", "Security code"),
    ("", "code", "CVV Verification"),
    ("", "cvv verification", "Security code"),
    ("", "cvv", "Card verification code OTP"),
    ("", "verification code", "Card verification code CVV"),
    ("", "code", "Card verification value CVV verification"),
    ("", "card", "Verification code CVV"),
    ("", "verification code", "Card CVV"),
    ("section-auth one-time-code", "cvv", "Security code"),
])
def test_authentication_metadata_vetoes_masked_cvv(autocomplete, name, label):
    control = LoginControl(autocomplete, 0, 2, label, name, "password")
    assert classify_checkout_control(control) is None


@pytest.mark.parametrize("name,label", [
    ("cvv", "Card verification code"),
    ("", "Card verification value (CVV)"),
    ("card-verification-code", "Security code"),
    ("Security code", "Card verification"),
])
def test_card_verification_words_do_not_veto_cvv(name, label):
    target = classify_checkout_control(LoginControl("", 0, 2, label, name, "password"))
    assert target is not None
    assert target.token == "cc-csc"


@pytest.mark.parametrize("input_type,name,label", [
    ("password", "", ""),
    ("password", "", "Card number"),
    ("password", "", "Expiry date"),
    ("password", "", "Address"),
    ("password", "", "Card verification code"),
    ("password", "cvvalue", ""),
    ("email", "cvv", "Security code"),
])
def test_masking_does_not_enable_other_checkout_heuristics(input_type, name, label):
    assert classify_checkout_control(LoginControl("", 0, 2, label, name, input_type)) is None


def test_mixed_checkout_and_auth_controls_only_fill_the_card_code():
    controls = [
        LoginControl("", 0, 0, "Security code", "cvv", "password"),
        LoginControl("current-password", 1, 1, "Password", "", "password"),
        LoginControl("one-time-code", 1, 2, "Security code", "", "password"),
        LoginControl("", 1, 3, "Verification", "code", "password"),
    ]
    targets = [target for control in controls if (target := classify_checkout_control(control))]
    assert select_checkout_fills(targets, {"cvc": "123"}, {"cvc": "cc-csc"}) == [
        {"index": 0, "token": "cc-csc", "value": "123"}
    ]
    assert classify_login_control(controls[1]).token == "current-password"
    assert classify_otp_controls([controls[2]])[0].token == "one-time-code"


def test_exact_checkout_autocomplete_retains_precedence():
    control = LoginControl("section-payment cc-csc", 0, 2, "Verification code", "", "password")
    target = classify_checkout_control(control)
    assert target is not None
    assert (target.token, target.score) == ("cc-csc", 100)
