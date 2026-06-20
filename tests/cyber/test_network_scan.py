"""Unit tests for cyber_network_scan tool (no network, no nmap required)."""
from __future__ import annotations

import json
import sys
import types
import textwrap

for mod in ("tools.registry",):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        stub.registry = types.SimpleNamespace(register=lambda **_kw: None)
        sys.modules[mod] = stub

from tools.cyber_network_scan import (  # noqa: E402
    _handle,
    _parse_nmap_xml,
    _risk_label,
    _make_summary,
    _ScanStore,
)


# ---------------------------------------------------------------------------
# Sample nmap XML fixture
# ---------------------------------------------------------------------------

_NMAP_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <nmaprun scanner="nmap">
      <host>
        <status state="up"/>
        <address addr="10.0.0.1" addrtype="ipv4"/>
        <hostnames><hostname name="router.local" type="user"/></hostnames>
        <ports>
          <port protocol="tcp" portid="22">
            <state state="open"/>
            <service name="ssh" product="OpenSSH" version="8.9p1"/>
          </port>
          <port protocol="tcp" portid="80">
            <state state="open"/>
            <service name="http" product="Apache httpd" version="2.4.54"/>
          </port>
          <port protocol="tcp" portid="443">
            <state state="open"/>
            <service name="https" product="Apache httpd" version="2.4.54"/>
          </port>
          <port protocol="tcp" portid="9090">
            <state state="closed"/>
            <service name="zeus-admin"/>
          </port>
        </ports>
        <os><osmatch name="Linux 5.15" accuracy="90"/></os>
      </host>
      <runstats><hosts up="1" down="0" total="1"/></runstats>
    </nmaprun>
""")

_NMAP_XML_EMPTY = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <nmaprun scanner="nmap">
      <runstats><hosts up="0" down="1" total="1"/></runstats>
    </nmaprun>
""")


# ---------------------------------------------------------------------------
# XML parser tests
# ---------------------------------------------------------------------------

class TestParseNmapXml:
    def test_parses_host_and_ports(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML, "10.0.0.1", "standard")
        assert result["hosts_up"] == 1
        assert len(result["hosts"]) == 1
        host = result["hosts"][0]
        assert host["ip"] == "10.0.0.1"
        assert host["hostname"] == "router.local"

    def test_only_open_ports_included(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML, "10.0.0.1", "standard")
        host = result["hosts"][0]
        ports = [p["port"] for p in host["open_ports"]]
        assert 22 in ports
        assert 80 in ports
        assert 443 in ports
        assert 9090 not in ports   # closed port excluded

    def test_service_version_captured(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML, "10.0.0.1", "standard")
        ssh = next(p for p in result["hosts"][0]["open_ports"] if p["port"] == 22)
        assert ssh["product"] == "OpenSSH"
        assert ssh["version"] == "8.9p1"

    def test_os_detection(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML, "10.0.0.1", "standard")
        host = result["hosts"][0]
        assert "Linux" in host["os"]
        assert host["os_accuracy"] == 90

    def test_empty_scan_no_hosts(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML_EMPTY, "10.0.0.2", "fast")
        assert result["hosts_up"] == 0
        assert result["hosts"] == []

    def test_bad_xml_returns_error(self) -> None:
        result = _parse_nmap_xml("not xml at all <<<", "10.0.0.1", "fast")
        assert "error" in result

    def test_metadata_fields(self) -> None:
        result = _parse_nmap_xml(_NMAP_XML, "10.0.0.0/24", "standard")
        assert result["target"] == "10.0.0.0/24"
        assert result["profile"] == "standard"
        assert "scanned_at" in result


# ---------------------------------------------------------------------------
# Risk label + summary
# ---------------------------------------------------------------------------

class TestRiskLabel:
    def test_critical(self) -> None: assert _risk_label(9.8) == "CRITICAL"
    def test_high(self) -> None:     assert _risk_label(7.5) == "HIGH"
    def test_medium(self) -> None:   assert _risk_label(5.0) == "MEDIUM"
    def test_low(self) -> None:      assert _risk_label(2.0) == "LOW"
    def test_info(self) -> None:     assert _risk_label(0.0) == "INFO"
    def test_boundary_9(self) -> None: assert _risk_label(9.0) == "CRITICAL"
    def test_boundary_7(self) -> None: assert _risk_label(7.0) == "HIGH"
    def test_boundary_4(self) -> None: assert _risk_label(4.0) == "MEDIUM"


class TestMakeSummary:
    def test_counts_by_risk(self) -> None:
        findings = [
            {"risk": "CRITICAL", "host": "1.2.3.4", "port": 80,
             "product": "Apache 2.4", "cves": [{"cve_id": "CVE-2021-1234"}]},
            {"risk": "HIGH", "host": "1.2.3.4", "port": 443,
             "product": "Apache 2.4", "cves": [{"cve_id": "CVE-2021-5678"}]},
            {"risk": "MEDIUM", "host": "1.2.3.4", "port": 22,
             "product": "OpenSSH 8.9", "cves": [{"cve_id": "CVE-2020-9999"}]},
        ]
        summary = _make_summary(findings)
        assert summary["by_risk"]["CRITICAL"] == 1
        assert summary["by_risk"]["HIGH"] == 1
        assert summary["by_risk"]["MEDIUM"] == 1
        assert summary["by_risk"]["LOW"] == 0

    def test_critical_items_list(self) -> None:
        findings = [
            {"risk": "CRITICAL", "host": "10.0.0.1", "port": 80,
             "product": "Apache 2.4.54", "cves": [{"cve_id": "CVE-2021-41773"}]},
        ]
        summary = _make_summary(findings)
        assert len(summary["critical_items"]) == 1
        assert "CVE-2021-41773" in summary["critical_items"][0]


# ---------------------------------------------------------------------------
# ScanStore
# ---------------------------------------------------------------------------

class TestScanStore:
    def test_save_and_retrieve(self) -> None:
        store = _ScanStore()
        scan = {"target": "10.0.0.0/24", "hosts_up": 3}
        scan_id = store.save(scan)
        assert scan_id == "scan-001"
        retrieved = store.get(scan_id)
        assert retrieved is not None
        assert retrieved["target"] == "10.0.0.0/24"

    def test_last_scan(self) -> None:
        store = _ScanStore()
        store.save({"target": "10.0.0.1"})
        store.save({"target": "10.0.0.2"})
        assert store.last()["target"] == "10.0.0.2"

    def test_last_scan_empty(self) -> None:
        assert _ScanStore().last() is None

    def test_list_summaries(self) -> None:
        store = _ScanStore()
        store.save({"target": "192.168.1.0/24", "hosts_up": 5, "scanned_at": "2024-01-01"})
        summaries = store.list_summaries()
        assert len(summaries) == 1
        assert summaries[0]["id"] == "scan-001"
        assert summaries[0]["hosts_up"] == 5

    def test_get_unknown_id_returns_none(self) -> None:
        store = _ScanStore()
        assert store.get("scan-999") is None

    def test_sequential_ids(self) -> None:
        store = _ScanStore()
        id1 = store.save({"target": "a"})
        id2 = store.save({"target": "b"})
        id3 = store.save({"target": "c"})
        assert id1 == "scan-001"
        assert id2 == "scan-002"
        assert id3 == "scan-003"


# ---------------------------------------------------------------------------
# Handler dispatch (no nmap / network)
# ---------------------------------------------------------------------------

class TestHandlerDispatch:
    def test_unknown_action(self) -> None:
        out = json.loads(_handle({"action": "nuke"}))
        assert "error" in out
        assert "valid_actions" in out

    def test_no_action(self) -> None:
        out = json.loads(_handle({}))
        assert "error" in out

    def test_scan_missing_target(self) -> None:
        out = json.loads(_handle({"action": "scan"}))
        assert "error" in out
        assert "target" in out["error"].lower()

    def test_scan_invalid_profile(self) -> None:
        out = json.loads(_handle({"action": "scan", "target": "10.0.0.1", "profile": "badprofile"}))
        assert "error" in out
        assert "valid" in out

    def test_correlate_no_scans_returns_error(self) -> None:
        out = json.loads(_handle({"action": "correlate"}))
        assert "error" in out

    def test_correlate_bad_scan_id(self) -> None:
        out = json.loads(_handle({"action": "correlate", "scan_id": "scan-999"}))
        assert "error" in out

    def test_status_returns_nmap_info(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert "nmap_available" in out
        assert "valid_profiles" in out
        assert "session_scans" in out
        assert isinstance(out["session_scans"], list)

    def test_status_profiles_complete(self) -> None:
        out = json.loads(_handle({"action": "status"}))
        assert set(out["valid_profiles"]) == {"fast", "standard", "full", "vuln"}
