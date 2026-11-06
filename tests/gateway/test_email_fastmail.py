"""Fastmail split results: PR #68606, using synthetic wire headers only."""
from email.parser import BytesHeaderParser
import pytest
from plugins.platforms.email.adapter import _verify_sender_authentication

HOST = "phl-mx-05.messagingengine.com"
AUX = ["x-csa=none; x-me-sender=none; x-ptr=pass", "bimi=none", "arc=none"]
PASS = "dkim=pass header.d=example.com; dmarc=pass header.from=example.com; spf=pass smtp.mailfrom=owner@example.com"

def message(results, separator=None):
    fields = ["Received: by our receiver", "From: owner@example.com"]
    for i, result in enumerate(results):
        if i == 1 and separator:
            fields.append(separator)
        fields.append("Authentication-Results: " + result)
    return BytesHeaderParser().parsebytes(("\r\n".join(fields) + "\r\n\r\n").encode())

def verify(results, pin=HOST, separator=None):
    return _verify_sender_authentication(message(results, separator), "owner@example.com", authserv_id=pin)[0]

def test_four_field_fastmail_and_single_verdict_control():
    assert verify([HOST + "; " + PASS])
    assert verify([HOST + "; " + x for x in AUX + [PASS]])

@pytest.mark.parametrize("separator", ["Received: from attacker.example", "X-Other: intervening", "Subject: boundary"])
def test_lower_pass_cannot_cross_wire_header_boundary(separator):
    assert not verify([HOST + "; bimi=none", HOST + "; " + PASS], separator=separator)

@pytest.mark.parametrize("host", ["mx.example.com", "messagingengine.com.evil.test", "evilmessagingengine.com"])
def test_split_exception_is_not_generalized(host):
    assert not verify([host + "; bimi=none", host + "; " + PASS], pin=host)

@pytest.mark.parametrize("pin", ["", "messagingengine.com", "other.messagingengine.com"])
def test_existing_exact_receiver_pin_is_preserved(pin):
    assert not verify([HOST + "; bimi=none", HOST + "; " + PASS], pin=pin)

@pytest.mark.parametrize("first", ["dmarc=fail header.from=example.com", "spf=fail smtp.mailfrom=owner@example.com", "dkim=fail header.d=example.com", "bimi=none (unbalanced"])
def test_lower_pass_cannot_override_a_verdict_or_malformed_field(first):
    assert not verify([HOST + "; " + first, HOST + "; " + PASS])

def test_changed_receiver_ends_block_even_if_later_pin_matches():
    assert not verify([HOST + "; bimi=none", "other.messagingengine.com; arc=none", HOST + "; " + PASS])

@pytest.mark.parametrize("verdict", [
    'dmarc=fail reason="; dmarc=pass header.from=example.com"',
    'dmarc=fail (dmarc=pass header.from=example.com)',
    'dkim=pass header.d=evil.test; dkim=fail header.d=example.com',
    'spf=pass smtp.mailfrom=owner@evil.test; dmarc=fail header.from=example.com',
    'dmarc=pass header.from=evil.test',
    'dmarc=pass header.from=example.com; dmarc=fail header.from=example.com',
])
def test_split_verdict_keeps_clause_aware_smuggling_protections(verdict):
    assert not verify([HOST + "; " + x for x in AUX + [verdict]])

def test_auxiliary_fields_without_authentication_fail_closed():
    assert not verify([HOST + "; " + x for x in AUX])


def test_imap_preflight_preserves_boundary_and_fetches_only_authenticated_body(monkeypatch):
    from contextlib import contextmanager
    from unittest.mock import MagicMock
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter, _MAX_PREAUTH_HEADER_BYTES
    adapter = EmailAdapter(PlatformConfig(enabled=True, extra={"authserv_id": HOST}))
    adapter._authserv_id = HOST
    good = message([HOST + "; " + x for x in AUX + [PASS]])
    bad = message([HOST + "; bimi=none", HOST + "; " + PASS], "Received: from attacker.example")
    wires = {b"1": good.as_bytes(), b"2": bad.as_bytes()}
    fetched_bodies = []
    imap = MagicMock()
    def uid(command, *args):
        if command == "search":
            return "OK", [b"1 2"]
        if command == "fetch":
            key, spec = args
            if spec == "(RFC822)":
                fetched_bodies.append(key)
                return "OK", [(key, wires[key] + b"body")]
            # Model HEADER.FIELDS filtering: it must not erase the intervening Received.
            raw = wires[key]
            if "HEADER.FIELDS" in spec:
                raw = b"\n".join(line for line in raw.splitlines() if not line.startswith(b"Received:"))
            assert f"<0.{_MAX_PREAUTH_HEADER_BYTES + 1}>" in spec
            return "OK", [(key, raw)]
        return "OK", [b""]
    imap.uid.side_effect = uid
    @contextmanager
    def inbox():
        yield imap
    monkeypatch.setattr(adapter, "_inbox", inbox)
    result = adapter._fetch_new_messages(lambda candidate: candidate["sender_authenticated"])
    assert fetched_bodies == [b"1"]
    assert len(result) == 1
    assert result[0]["sender_authenticated"]
