"""Context-only policy for synthetic CVC/OTP; no real credentials."""
import json

import pytest

from agent import redact


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    redact.clear_vault_redaction_values()
    yield
    redact.clear_vault_redaction_values()


@pytest.mark.parametrize("ordinary", ["1683", "$683.245", "683 requests", "683", '{"total":"683.245"}', "CVC field is empty; total=$683.245"])
def test_cvc_does_not_poison_ordinary_numbers(ordinary):
    redact.register_vault_redaction_value("683", kind="cvc")
    assert redact.redact_registered_vault_values(ordinary) == ordinary


@pytest.mark.parametrize("text", ['CVC: 683', 'cvv=683', 'Код безопасности: 683', '{"checkout":{"cvc":683,"total":"683.245"}}', "{'securityCode': '683'}", '<input value="683" autocomplete="cc-csc">', '{"name":"securityCode","value":"683","total":"683.245"}', '<input name="cvv" value="683">'])
def test_cvc_explicit_context_is_masked(text):
    redact.register_vault_redaction_value("683", kind="cvc")
    result = redact.redact_registered_vault_values(text)
    assert "«redacted-vault-secret»" in result
    if "683.245" in text:
        assert "683.245" in result


def test_nested_serialized_result_and_escaped_key():
    redact.register_vault_redaction_value("683", kind="cvc")
    raw = json.dumps({"output": json.dumps({"cvc": "683", "total": "683.245"})})
    assert json.loads(json.loads(redact.redact_registered_vault_values(raw))["output"])["cvc"] != "683"
    assert "683" not in redact.redact_registered_vault_values('{"c\\u0076c":"683"}')


@pytest.mark.parametrize("text", ['OTP: 743821', 'verification code=743821', 'Код подтверждения: 743821', '{"otp":"743821"}', '{"autocomplete":"one-time-code","value":"743821"}', '<input autocomplete="one-time-code" value="743821">'])
def test_otp_explicit_context_is_masked(text):
    redact.register_vault_redaction_value("743821", kind="otp")
    assert "743821" not in redact.redact_registered_vault_values(text)


def test_otp_unlabelled_numbers_preserved_as_accepted_tradeoff():
    redact.register_vault_redaction_value("743821", kind="otp")
    ordinary = '743821 requests; price=743821.25; {"statusCode":743821}'
    assert redact.redact_registered_vault_values(ordinary) == ordinary


def test_global_password_pan_and_tokens_unchanged():
    for value in ("XY", "4111111111111111", "opaque-token-for-test"):
        redact.register_vault_redaction_value(value)
        assert value not in redact.redact_registered_vault_values("prefix" + value + "suffix")


def test_same_bytes_password_protection_wins_over_code_policy():
    redact.register_vault_redaction_value("683")
    redact.register_vault_redaction_value("683", kind="cvc")
    assert "683" not in redact.redact_registered_vault_values("1683")


def test_code_registration_can_be_promoted_to_password():
    redact.register_vault_redaction_value("683", kind="cvc")
    redact.register_vault_redaction_value("683")
    assert "683" not in redact.redact_registered_vault_values("1683")


def test_egress_boundary_applies_context_policy():
    redact.register_vault_redaction_value("683", kind="cvc")
    result = redact.redact_for_egress('cvc=683; total=$683.245')
    assert "cvc=683" not in result
    assert "$683.245" in result


def test_profile_isolation(tmp_path, monkeypatch):
    redact.register_vault_redaction_value("683", kind="cvc")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    assert redact.redact_registered_vault_values("cvc=683") == "cvc=683"
    redact.clear_vault_redaction_values()


def test_clear_removes_context_values():
    redact.register_vault_redaction_value("683", kind="cvc")
    redact.clear_vault_redaction_values()
    assert redact.redact_registered_vault_values("cvc=683") == "cvc=683"


@pytest.mark.parametrize("api", ["registered", "sensitive", "egress"])
@pytest.mark.parametrize("secret", ["XY", "4111111111111111", "opaque-token-for-test"])
@pytest.mark.parametrize("nested", [False, True])
def test_json_code_masking_does_not_reexpose_global_secret(api, secret, nested):
    redact.register_vault_redaction_value(secret)
    redact.register_vault_redaction_value("683", kind="cvc")
    escaped = "".join(r"\u" + format(ord(char), "04x") for char in secret)
    raw = '{"cvc":"683","note":"' + escaped + '"}'
    if nested:
        raw = json.dumps({"output": raw})
    fn = {"registered": redact.redact_registered_vault_values,
          "sensitive": lambda text: redact.redact_sensitive_text(text, force=True),
          "egress": redact.redact_for_egress}[api]
    result = fn(raw)
    assert secret not in result
    assert "«redacted-vault-secret»" in result


@pytest.mark.parametrize("ordinary", [
    "identifier cvc683", "identifier otp743821",
    '{"cvc683":"public","amount":683}',
    '{"otp743821":"public","amount":743821}',
    '{"cvc 683":"public","amount":683}',
    '<input name="cvc683" value="public">',
    '<input id="otp743821" value="public">',
])
def test_identifiers_and_json_keys_are_not_code_values(ordinary):
    redact.register_vault_redaction_value("683", kind="cvc")
    redact.register_vault_redaction_value("743821", kind="otp")
    assert redact.redact_registered_vault_values(ordinary) == ordinary


@pytest.mark.parametrize("text", ["CVC 683", "CVC:683", "cvv=683", "OTP 743821", "OTP:743821"])
def test_required_relationship_separator_keeps_valid_labels(text):
    redact.register_vault_redaction_value("683", kind="cvc")
    redact.register_vault_redaction_value("743821", kind="otp")
    assert "«redacted-vault-secret»" in redact.redact_registered_vault_values(text)


@pytest.mark.parametrize("number", ["9007199254740993.0", "1e400", "0.12345678901234567890123456789", "1.00000000000000000000", "-0.0"])
@pytest.mark.parametrize("nested", [False, True])
def test_json_masking_preserves_numeric_lexemes(number, nested):
    redact.register_vault_redaction_value("683", kind="cvc")
    raw = '{"cvc":"683","total":' + number + '}'
    if nested:
        raw = json.dumps({"output": raw})
    result = redact.redact_registered_vault_values(raw)
    if nested:
        result = json.loads(result)["output"]
    assert number in result
    assert "Infinity" not in result
    assert "«redacted-vault-secret»" in result


@pytest.mark.parametrize("raw, fragment", [
    ('{"cvc":"683","total":1,"total":2}', '"total":1,"total":2'),
    ('{"cvc":"683","nested":{"total":1,"total":2}}', '"total":1,"total":2'),
    ('{"cvc":"683","cvc":"683"}', '"cvc":"«redacted-vault-secret»","cvc":"«redacted-vault-secret»"'),
    ('{"cvc":"683","nested":[{"total":1,"total":2}]}', '"total":1,"total":2'),
])
def test_json_masking_preserves_duplicate_pairs(raw, fragment):
    redact.register_vault_redaction_value("683", kind="cvc")
    result = redact.redact_registered_vault_values(raw)
    assert fragment in result
    assert "683" not in result
    json.loads(result)  # The emitted object still has valid JSON syntax.
