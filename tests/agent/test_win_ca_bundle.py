"""Tests for agent.win_ca_bundle (written FIRST per TDD)."""
import base64
import ssl
from pathlib import Path
from typing import cast

import certifi
import pytest


@pytest.mark.windows_only
def test_real_store_loads():
    from agent.win_ca_bundle import windows_merged_ca_bundle
    result = windows_merged_ca_bundle()
    # On native Windows: returns a path that exists and loads >= 1 cert.
    # On non-Windows this test is skipped by the marker.
    assert result is not None
    p = Path(result)
    assert p.exists()
    ctx = ssl.create_default_context(cafile=str(p))
    try:
        certs = ctx.get_ca_certs()
        assert len(certs) >= 1
    except NotImplementedError:
        pass  # truststore-backed; creation success sufficient


def test_gate_non_win32(monkeypatch):
    import agent.win_ca_bundle as m
    monkeypatch.setattr(m.sys, "platform", "darwin")
    assert m.windows_merged_ca_bundle() is None


def _patch_win32(monkeypatch, entries):
    """Force the win32 gate open, reset the memo, and stub store enumeration."""
    import agent.win_ca_bundle as m
    monkeypatch.setattr(m.sys, "platform", "win32")
    monkeypatch.setattr(m, "_bundle_path", None)
    monkeypatch.setattr(m.ssl, "enum_certificates", lambda store: entries, raising=False)
    return m


def test_cache_write_is_atomic_and_no_residue_on_validation_failure(monkeypatch, tmp_path):
    """Regression (cross-vendor review, cross-process cache race): the final cache
    path must only ever hold validated content, and a validation failure must leave
    no temp residue and NO final-path file that another process already memoized."""
    import agent.win_ca_bundle as m

    der = base64.b64decode(
        Path(certifi.where()).read_text().split("-----BEGIN CERTIFICATE-----")[1]
        .split("-----END CERTIFICATE-----")[0]
        .strip()
    )
    m = _patch_win32(monkeypatch, [(der, "x509_asn", True)])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Broken validation: create_default_context rejects the temp file -> return None,
    # and NOTHING is left behind in the cache dir (no .pem, no .tmp residue).
    def _boom(cafile):
        raise ssl.SSLError("simulated validation failure")

    monkeypatch.setattr(m.ssl, "create_default_context", _boom)
    assert m.windows_merged_ca_bundle() is None
    assert list(tmp_path.joinpath("cache").iterdir()) == []


def test_cache_publish_leaves_no_residue_and_republishes_atomically(monkeypatch, tmp_path):
    """The published path must exist and validate after a successful build; a second
    build after memo reset republishes without leaving temp residue next to the final
    file. (Sequential stand-in for the cross-process race — the atomic os.replace
    plus temp-only-on-failure unlink is what makes concurrent processes safe.)"""
    import agent.win_ca_bundle as m

    der = base64.b64decode(
        Path(certifi.where()).read_text().split("-----BEGIN CERTIFICATE-----")[1]
        .split("-----END CERTIFICATE-----")[0]
        .strip()
    )
    m = _patch_win32(monkeypatch, [(der, "x509_asn", True)])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = m.windows_merged_ca_bundle()
    assert first is not None and Path(first).exists()
    m._bundle_path = None  # simulate a fresh process
    second = m.windows_merged_ca_bundle()
    assert second == first
    leftovers = [p for p in tmp_path.joinpath("cache").iterdir() if p.name != "windows-ca-bundle.pem"]
    assert leftovers == []


def test_pure_pem_helper_includes_serverauth_excludes_others():
    from agent.win_ca_bundle import _pem_from_store_entries
    # Read one real PEM block from certifi, decode to DER
    cert_path = certifi.where()
    pem_content = Path(cert_path).read_text()
    # Extract first PEM block
    lines = pem_content.splitlines()
    start_idx = lines.index("-----BEGIN CERTIFICATE-----")
    end_idx = lines.index("-----END CERTIFICATE-----", start_idx)
    pem_block = "\n".join(lines[start_idx:end_idx + 1]) + "\n"
    der_bytes = base64.b64decode("".join(lines[start_idx + 1:end_idx]))

    # Included: x509_asn + serverAuth
    included = _pem_from_store_entries([(der_bytes, "x509_asn", True)])
    assert len(included) == 1
    assert "BEGIN CERTIFICATE" in included[0]

    # Excluded: pkcs_7_asn
    excluded_pkcs = _pem_from_store_entries([(der_bytes, "pkcs_7_asn", True)])
    assert excluded_pkcs == []

    # Excluded: non-serverAuth OID tuple
    excluded_oid = _pem_from_store_entries([(der_bytes, "x509_asn", ("1.3.6.1.5.2.3.4",))])
    assert excluded_oid == []

    # Included: serverAuth OID tuple
    included_oid = _pem_from_store_entries([(der_bytes, "x509_asn", ("1.3.6.1.5.5.7.3.1",))])
    assert len(included_oid) == 1

    # Included: serverAuth OID frozenset — the ACTUAL shape ssl.enum_certificates
    # returns on Windows (CPython Modules/_ssl.c parseKeyUsage -> PyFrozenSet_New).
    # Regression: the filter originally checked isinstance(trust, tuple), which
    # silently dropped every corporate CA carrying explicit serverAuth EKUs.
    included_frozenset = _pem_from_store_entries([(der_bytes, "x509_asn", frozenset({"1.3.6.1.5.5.7.3.1"}))])
    assert len(included_frozenset) == 1

    # Excluded: non-serverAuth OID frozenset
    excluded_frozenset = _pem_from_store_entries([(der_bytes, "x509_asn", frozenset({"1.3.6.1.5.2.3.4"}))])
    assert excluded_frozenset == []

    # Excluded: trust=False (explicitly distrusted) must never be accepted
    excluded_false = _pem_from_store_entries([(der_bytes, "x509_asn", False)])
    assert excluded_false == []

    # Excluded: empty frozenset (cert present but no EKUs listed) and None
    # (not a shape enum_certificates yields, but must not blow up either).
    assert _pem_from_store_entries([(der_bytes, "x509_asn", frozenset())]) == []
    assert _pem_from_store_entries([(der_bytes, "x509_asn", cast("bool | frozenset | tuple", None))]) == []

    # Dedup: same DER twice -> one PEM
    deduped = _pem_from_store_entries([
        (der_bytes, "x509_asn", True),
        (der_bytes, "x509_asn", True),
    ])
    assert len(deduped) == 1


def test_pem_loads_via_ssl_create_default_context():
    from agent.win_ca_bundle import _pem_from_store_entries
    cert_path = certifi.where()
    pem_content = Path(cert_path).read_text()
    lines = pem_content.splitlines()
    start = lines.index("-----BEGIN CERTIFICATE-----")
    end = lines.index("-----END CERTIFICATE-----", start)
    der_bytes = base64.b64decode("".join(lines[start + 1:end]))

    pem_blocks = _pem_from_store_entries([(der_bytes, "x509_asn", True)])
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
        for block in pem_blocks:
            f.write(block)
        tmp_path = f.name

    ctx = ssl.create_default_context(cafile=tmp_path)
    try:
        certs = ctx.get_ca_certs()
        assert len(certs) >= 1
    except NotImplementedError:
        pass
    Path(tmp_path).unlink(missing_ok=True)
