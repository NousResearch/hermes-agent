"""RFC 6238 TOTP helper used as the dashboard's second factor."""

from __future__ import annotations

from hermes_cli.dashboard_auth import totp

# RFC 6238 Appendix B, SHA-1 vectors (8-digit values; the last 6 digits are the 6-digit code).
_RFC_SECRET = b"12345678901234567890"
_RFC_VECTORS = [(59, "94287082"), (1111111109, "07081804"), (1111111111, "14050471"),
                (1234567890, "89005924"), (2000000000, "69279037"), (20000000000, "65353130")]


def test_rfc6238_vectors():
    for t, eight in _RFC_VECTORS:
        assert totp.totp_code(_RFC_SECRET, counter=totp.totp_counter(t)) == eight[-6:]


def test_verify_accepts_adjacent_steps_and_reports_counter():
    now = 1111111111
    center = totp.totp_counter(now)
    for delta in (-1, 0, 1):
        code = totp.totp_code(_RFC_SECRET, counter=center + delta)
        assert totp.verify_totp(_RFC_SECRET, code, now=now) == center + delta
    two_away = totp.totp_code(_RFC_SECRET, counter=center + 2)
    assert totp.verify_totp(_RFC_SECRET, two_away, now=now) is None


def test_verify_tolerates_spaces_and_rejects_junk():
    now = 59
    assert totp.verify_totp(_RFC_SECRET, "287 082", now=now) is not None
    assert totp.verify_totp(_RFC_SECRET, "", now=now) is None
    assert totp.verify_totp(_RFC_SECRET, "28708", now=now) is None
    assert totp.verify_totp(_RFC_SECRET, "2870821", now=now) is None
    assert totp.verify_totp(_RFC_SECRET, "abcdef", now=now) is None


def test_secret_round_trip_and_forgiving_decode():
    s = totp.generate_totp_secret()
    assert len(s) == 32 and "=" not in s
    raw = totp.decode_totp_secret(s)
    assert len(raw) == 20
    # As an operator would paste it: lower case, grouped, padded.
    grouped = " ".join(s[i:i + 4] for i in range(0, 32, 4)).lower()
    assert totp.decode_totp_secret(grouped) == raw
    assert totp.decode_totp_secret(s + "=") == raw


def test_decode_rejects_invalid():
    import pytest
    for bad in ("", "   ", "not base32 !!", "1189"):
        with pytest.raises(ValueError):
            totp.decode_totp_secret(bad)


def test_provisioning_uri_shape():
    uri = totp.totp_provisioning_uri("abcd efgh", account="admin", issuer="Hermes Dashboard")
    assert uri.startswith("otpauth://totp/Hermes%20Dashboard%3Aadmin?")
    assert "secret=ABCDEFGH" in uri
    assert "issuer=Hermes+Dashboard" in uri
    assert "algorithm=SHA1" in uri and "digits=6" in uri and "period=30" in uri
