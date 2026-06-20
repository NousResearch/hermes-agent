"""Unit tests for the IOC extractor tool (no network required)."""
from __future__ import annotations

import json
import sys
import types

# Minimal stubs so the tool module imports cleanly without the full hermes stack
for mod in ("tools.registry",):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.registry = types.SimpleNamespace(register=lambda **_kw: None)
        stub.tool_error = lambda msg: msg
        sys.modules[mod] = stub

from tools.cyber_ioc_extractor import _extract_iocs, _handle  # noqa: E402


class TestExtractIocs:
    def test_extracts_public_ipv4(self) -> None:
        result = _extract_iocs("Attacker IP: 203.0.113.42 connected to 198.51.100.7")
        assert "203.0.113.42" in result["iocs"]["ipv4"]
        assert "198.51.100.7" in result["iocs"]["ipv4"]

    def test_excludes_private_ipv4_by_default(self) -> None:
        result = _extract_iocs("Internal: 192.168.1.1, 10.0.0.1, 172.16.5.5")
        assert result["iocs"]["ipv4"] == []

    def test_includes_private_when_flag_set(self) -> None:
        result = _extract_iocs("192.168.1.1 and 10.0.0.5", include_private_ips=True)
        assert "192.168.1.1" in result["iocs"]["ipv4"]
        assert "10.0.0.5" in result["iocs"]["ipv4"]

    def test_extracts_sha256(self) -> None:
        h = "a" * 64
        result = _extract_iocs(f"Malware hash: {h}")
        assert h in result["iocs"]["sha256"]

    def test_extracts_sha1_without_sha256_collision(self) -> None:
        sha1 = "b" * 40
        sha256 = "c" * 64
        result = _extract_iocs(f"{sha256} and {sha1}")
        assert sha256 in result["iocs"]["sha256"]
        assert sha1 in result["iocs"]["sha1"]
        # sha1 must not also appear in sha256
        assert sha1 not in result["iocs"]["sha256"]

    def test_extracts_md5(self) -> None:
        md5 = "d" * 32
        result = _extract_iocs(f"MD5: {md5}")
        assert md5 in result["iocs"]["md5"]

    def test_extracts_cve(self) -> None:
        result = _extract_iocs("Vulnerability CVE-2024-12345 is critical.")
        assert "CVE-2024-12345" in result["iocs"]["cve"]

    def test_cve_case_insensitive(self) -> None:
        result = _extract_iocs("cve-2023-99999 was patched")
        assert "CVE-2023-99999" in result["iocs"]["cve"]

    def test_extracts_url(self) -> None:
        result = _extract_iocs("C2 at https://evil.example.com/payload/drop")
        assert any("evil.example.com" in u for u in result["iocs"]["url"])

    def test_defangs_hxxp(self) -> None:
        result = _extract_iocs("hxxps://malicious.example.com/stage2")
        assert any("malicious.example.com" in u for u in result["iocs"]["url"])

    def test_defangs_dot_notation(self) -> None:
        result = _extract_iocs("evil[.]example[.]com payload")
        # After defanging, should appear as a domain
        assert any("evil.example.com" in d for d in result["iocs"]["domain"])

    def test_extracts_email(self) -> None:
        result = _extract_iocs("Phishing from attacker@evil.com to victim@corp.org")
        emails = result["iocs"]["email"]
        assert "attacker@evil.com" in emails
        assert "victim@corp.org" in emails

    def test_deduplicates(self) -> None:
        result = _extract_iocs("203.0.113.1 and again 203.0.113.1")
        assert result["iocs"]["ipv4"].count("203.0.113.1") == 1

    def test_total_count_correct(self) -> None:
        result = _extract_iocs("IP: 203.0.113.1, CVE-2024-1111, CVE-2024-2222")
        assert result["total"] == 3

    def test_empty_text_returns_zeros(self) -> None:
        result = _extract_iocs("")
        assert result["total"] == 0

    def test_handler_requires_text(self) -> None:
        out = json.loads(_handle({}))
        assert "error" in out

    def test_handler_returns_json(self) -> None:
        out = json.loads(_handle({"text": "no IOCs here"}))
        assert "iocs" in out
        assert "total" in out
