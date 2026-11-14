"""``security.redact_phone_numbers: false`` keeps only E.164 numbers in chat replies (#68911).

The opt-in is read from a real config.yaml in a temp ``HERMES_HOME`` and judged through the gateway's
production chat boundaries: the non-streamed final reply (``_sanitize_gateway_final_response``) and the
streamed reply (``GatewayStreamConsumer._clean_for_display``). Every value is synthetic.
"""

import pytest

from gateway.run import _sanitize_gateway_final_response
from gateway.stream_consumer import GatewayStreamConsumer

PHONE = "+" + "".join(str((i % 9) + 1) for i in range(11))
# One per secret class the forced egress scrub masks; none may survive the phone opt-in.
SECRETS = {
    "openai_key": "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0",
    "github_pat": "ghp_" + "Ab3Cd4Ef5Gh6Ij7Kl8Mn9Op0Qr1St2Uv3Wx",
    "telegram_bot_token": "bot1234567890:" + "AAH" * 13 + "x",
    "bearer_opaque": "Bearer " + "Zq8Xw7Vu6Ts5Rq4Po3Nm2Lk1Ji0Hg9Fe8Dc7",
    "jwt": "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0." + "Sfl" * 12,
    "db_connstr": "postgres://admin:" + "Hunter2Pw" * 3 + "@db.example.invalid:5432/app",
    "env_assignment": "\nDB_PASSWORD=" + "Tr0ub4dor" * 3 + "\n",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEow" + "IBAAKCAQEA" * 6 + "\n-----END RSA PRIVATE KEY-----",
}
_BODY = {
    "openai_key": "a1B2c3D4e5F6g7H8i9J0",
    "bearer_opaque": "Zq8Xw7Vu6Ts5Rq4Po3Nm",
    "db_connstr": "Hunter2PwHunter2Pw",
    "env_assignment": "Tr0ub4dorTr0ub4dor",
    "private_key": "IBAAKCAQEAIBAAKCAQEA",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)

    def write(security):
        body = "security:\n" + "".join(f"  {k}: {v}\n" for k, v in security.items()) if security else "{}\n"
        (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    return write


def _replies(text):
    return {"final": _sanitize_gateway_final_response("telegram", text),
            "stream": GatewayStreamConsumer._clean_for_display(text)}


@pytest.mark.parametrize("redact_secrets", [True, False])
def test_opt_in_keeps_the_phone_number_and_masks_every_other_secret(home, redact_secrets):
    home({"redact_phone_numbers": "false", "redact_secrets": str(redact_secrets).lower()})
    assert all(body in SECRETS[name] for name, body in _BODY.items())
    for name, secret in SECRETS.items():
        for surface, out in _replies(f"Call {PHONE} now. {secret} done.").items():
            assert PHONE in out, (surface, out)
            assert _BODY.get(name, secret) not in out, (name, surface, out)


@pytest.mark.parametrize("security", [{}, {"redact_phone_numbers": "true"}, {"redact_phone_numbers": "'no'"},
                                      {"redact_phone_numbers": "0"}])
def test_phone_numbers_stay_masked_unless_explicitly_false(home, security):
    home(security)
    for surface, out in _replies(f"Call {PHONE} now.").items():
        assert PHONE not in out, (surface, out)
        assert PHONE[:4] + "****" + PHONE[-4:] in out, (surface, out)

