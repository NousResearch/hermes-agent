"""Tests for agent.vault_login_classifier — OTP control classification."""

from agent.vault_login_classifier import LoginControl, classify_otp_controls, _normalize_text


def _control(name: str = "", label: str = "", type_: str = "text", autocomplete: str = "") -> LoginControl:
    return LoginControl(autocomplete=autocomplete, form_index=0, index=0, label=label, name=name, type=type_)


# ── _normalize_text ────────────────────────────────────────────────────

def test_normalize_splits_camelcase():
    assert _normalize_text("totpPin") == "totp pin"
    assert _normalize_text("otpCode") == "otp code"
    assert _normalize_text("verificationCode") == "verification code"


def test_normalize_handles_accents_and_punctuation():
    assert _normalize_text("Código-de-Verificação") == "codigo de verificacao"
    assert _normalize_text("  OTP  code! ") == "otp code"


# ── classify_otp_controls ──────────────────────────────────────────────

def test_classifies_google_totp_pin_field():
    """Google's PT-BR TOTP input: name=totpPin, label in Portuguese, type=tel."""
    controls = [_control(name="totpPin totpPin", label="Inserir código", type_="tel")]
    classified = classify_otp_controls(controls)
    assert len(classified) == 1
    assert classified[0].token == "one-time-code"


def test_autocomplete_one_time_code_is_authoritative():
    controls = [_control(autocomplete="one-time-code", name="whatever", type_="text")]
    classified = classify_otp_controls(controls)
    assert len(classified) == 1
    assert classified[0].score == 100


def test_non_matching_control_is_ignored():
    assert classify_otp_controls([_control(name="username", label="Username")]) == []
    # English-only keyword regex: a PT-only label with no OTP-ish name still misses.
    assert classify_otp_controls([_control(name="", label="codigo de verificacao")]) == []
