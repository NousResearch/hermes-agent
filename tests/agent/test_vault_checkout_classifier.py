"""Checkout classification of masked card security codes."""

import pytest

from agent.vault_login_classifier import LoginControl, classify_checkout_control


@pytest.mark.parametrize(("label", "name"), [
    ("CVV", ""), ("CVC", ""), ("CSC", ""), ("Security code", ""),
    ("Card code", ""), ("", "payment-cvv"), ("", "card_cvc"),
])
def test_masked_cvv_without_autocomplete_is_checkout_target(label, name):
    control = LoginControl("", 0, 0, label, name, "password")
    classified = classify_checkout_control(control)
    assert classified is not None
    assert classified.token == "cc-csc"
    assert classified.score == 70


@pytest.mark.parametrize(("autocomplete", "label"), [
    ("current-password", "CVV"), ("new-password", "CVV"),
    ("one-time-code", "Security code"), ("username", "CVV"),
    ("", "Password"), ("", "New password CVV"),
    ("", "Verification code"), ("", "OTP"),
    ("", "Card number"), ("", "Name on card"),
    ("", "Expiration date"), ("", "Billing address"),
])
def test_password_controls_without_checkout_cvv_intent_are_excluded(autocomplete, label):
    control = LoginControl(autocomplete, 0, 0, label, "", "password")
    assert classify_checkout_control(control) is None


@pytest.mark.parametrize(("autocomplete", "expected"), [
    ("section-payment cc-csc", "cc-csc"),
    ("billing cc-number", "cc-number"),
    ("country", "country-name"),
    ("current-password cc-csc", "cc-csc"),
])
def test_explicit_checkout_autocomplete_precedes_password_heuristics(autocomplete, expected):
    control = LoginControl(autocomplete, 0, 0, "CVV", "", "password")
    classified = classify_checkout_control(control)
    assert classified is not None
    assert (classified.token, classified.score) == (expected, 100)


def test_email_cvv_label_is_not_a_checkout_fallback():
    control = LoginControl("", 0, 0, "CVV", "", "email")
    assert classify_checkout_control(control) is None


@pytest.mark.parametrize(("label", "name"), [
    ("Security code", "otp"),
    ("OTP security code", ""),
    ("CVV", "verification_code"),
    ("One-time code CVV", ""),
    ("Authentication code CVV", ""),
    ("Account password", "cvv"),
    ("Login password CVV", ""),
    ("Confirm your password CVV", ""),
    ("CVV", "login"),
    ("CVV", "password"),
])
def test_masked_cvv_conflicting_authentication_metadata_is_excluded(label, name):
    control = LoginControl("", 0, 0, label, name, "password")
    assert classify_checkout_control(control) is None


def test_checkout_selection_skips_conflicting_authentication_before_real_cvv():
    from agent.vault_login_classifier import select_checkout_fills

    controls = [
        LoginControl("", 0, 0, "Security code", "otp", "password"),
        LoginControl("", 0, 1, "Account password", "cvv", "password"),
        LoginControl("", 1, 2, "Security code", "", "password"),
    ]
    classified = [c for control in controls if (c := classify_checkout_control(control))]
    assert select_checkout_fills(classified, {"cvc": "123"}, {"cvc": "cc-csc"}) == [
        {"index": 2, "token": "cc-csc", "value": "123"},
    ]


def test_explicit_checkout_autocomplete_precedes_conflicting_authentication_metadata():
    control = LoginControl("one-time-code current-password cc-csc", 0, 0,
                           "OTP login password CVV", "verification_code", "password")
    classified = classify_checkout_control(control)
    assert classified is not None
    assert (classified.token, classified.score) == ("cc-csc", 100)


@pytest.mark.parametrize(("label", "name"), [
    ("Card verification value (CVV)", ""),
    ("CVV", "card_verification"),
])
def test_card_verification_metadata_preserves_masked_cvv(label, name):
    classified = classify_checkout_control(LoginControl("", 0, 0, label, name, "password"))
    assert classified is not None
    assert (classified.token, classified.score) == ("cc-csc", 70)
