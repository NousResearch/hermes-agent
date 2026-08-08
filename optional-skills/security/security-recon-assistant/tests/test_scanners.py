"""Tests unitaires pour les scanners."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import json

from security_recon_assistant.scanners.base import BaseScanner, ScanResult, ScanFinding
from security_recon_assistant.scanners.subfinder_scanner import SubfinderScanner
from security_recon_assistant.scanners.nmap_scanner import NmapScanner
from security_recon_assistant.core.executor import ExecutionResult
from security_recon_assistant.core.guardian import ViolationError


pytestmark = pytest.mark.unit


class TestBaseScanner:
    """Tests pour la classe abstraite BaseScanner."""

    def test_abstract_class_cannot_instantiate(self):
        """BaseScanner ne devrait pas pouvoir être instanciée directement."""
        with pytest.raises(TypeError):
            BaseScanner()

    def test_scan_result_creation(self):
        """Teste la création d'un ScanResult."""
        result = ScanResult(
            scanner_name="test",
            success=True,
            execution_time=1.5,
            findings=[
                ScanFinding(
                    target="example.com",
                    severity="high",
                    title="Test finding",
                    description="A test vulnerability",
                    evidence="{\"port\": 80}",
                    remediation="Close port 80"
                )
            ]
        )
        assert result.scanner_name == "test"
        assert result.success is True
        assert result.execution_time == 1.5
        assert len(result.findings) == 1
        assert result.findings[0].severity == "high"

    def test_scan_finding_severity_values(self):
        """Teste les valeurs de sévérité acceptables."""
        severities = ["low", "medium", "high", "critical", "info"]
        for severity in severities:
            finding = ScanFinding(
                target="test.com",
                severity=severity,
                title="Test",
                description="Test"
            )
            assert finding.severity == severity

    def test_scan_finding_invalid_severity(self):
        """Une sévérité invalide devrait lever une erreur."""
        with pytest.raises(ValueError):
            ScanFinding(
                target="test.com",
                severity="invalid",
                title="Test",
                description="Test"
            )

    def test_scan_result_to_dict(self):
        """Teste la sérialisation en dict."""
        result = ScanResult(
            scanner_name="nmap",
            success=True,
            command="nmap -sV example.com",
            execution_time=2.0,
            stdout="Some output",
            stderr="",
            findings=[]
        )
        data = result.to_dict()
        assert data["scanner_name"] == "nmap"
        assert data["success"] is True
        assert data["execution_time"] == 2.0
        assert "timestamp" in data

    def test_scan_finding_to_dict(self):
        """Teste la sérialisation d'un finding."""
        finding = ScanFinding(
            target="example.com",
            severity="high",
            title="Open Port",
            description="Port 80 is open",
            evidence="{\"port\": 80, \"service\": \"http\"}",
            remediation="Implement firewall rules"
        )
        data = finding.to_dict()
        assert data["target"] == "example.com"
        assert data["severity"] == "high"
        assert data["title"] == "Open Port"
        assert "remediation" in data


class TestSubfinderScanner:
    """Tests pour le scanner Subfinder."""

    @pytest.fixture
    def scanner(self):
        """Crée une instance du scanner Subfinder."""
        return SubfinderScanner()

    def test_scanner_metadata(self, scanner):
        """Teste les métadonnées du scanner."""
        assert scanner.name == "subfinder"
        assert "subfinder" in scanner.description.lower()
        assert "subdomain" in scanner.description.lower()

    def test_build_command_simple(self, scanner):
        """Teste la construction d'une commande simple."""
        cmd = scanner.build_command("example.com", rate_limit=50)
        assert "subfinder" in cmd
        assert "-d example.com" in cmd
        assert "-rate-limit 50" in cmd

    def test_build_command_with_exclusions(self, scanner):
        """Teste la commande avec exclusions."""
        cmd = scanner.build_command(
            "example.com",
            rate_limit=100,
            excluded_sources=["crt.sh", "dnsdumpster"]
        )
        assert "-exclude-sources crt.sh,dnsdumpster" in cmd

    def test_build_command_with_recursive(self, scanner):
        """Teste l'option recursive."""
        cmd = scanner.build_command("example.com", recursive=True)
        assert "-recursive" in cmd

    def test_parse_output_valid_json(self, scanner):
        """Teste le parsing d'une sortie JSON valide."""
        json_output = json.dumps([
            {"host": "www.example.com", "source": "crt.sh"},
            {"host": "api.example.com", "source": "dnsdumpster"},
            {"host": "test.example.com", "source": "virustotal"}
        ])
        result = scanner.parse_output(json_output, "example.com")
        assert result.success is True
        assert len(result.findings) == 3
        assert result.findings[0].target == "www.example.com"
        assert result.findings[0].evidence is not None

    def test_parse_output_empty(self, scanner):
        """Teste le parsing d'une sortie vide."""
        result = scanner.parse_output("", "example.com")
        assert result.success is True
        assert len(result.findings) == 0

    def test_parse_output_invalid_json(self, scanner):
        """Teste la gestion d'une sortie invalide."""
        result = scanner.parse_output("not json at all", "example.com")
        assert result.success is False
        assert len(result.findings) == 0
        assert result.stderr is not None

    def test_parse_output_with_duplicates(self, scanner):
        """Teste que les doublons sont éliminés."""
        json_output = json.dumps([
            {"host": "www.example.com", "source": "crt.sh"},
            {"host": "www.example.com", "source": "dnsdumpster"}
        ])
        result = scanner.parse_output(json_output, "example.com")
        assert len(result.findings) == 2  # Différentes sources = différents findings

    def test_full_scan_execution(self, scanner):
        """Teste l'exécution complète avec mock de l'exécuteur."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout=json.dumps([{"host": "www.example.com", "source": "crt.sh"}]),
            stderr="",
            duration=1.2
        )

        result = scanner.scan("example.com", mock_executor)

        assert result.success is True
        assert result.execution_time == 1.2
        assert len(result.findings) == 1
        assert result.findings[0].target == "www.example.com"

    def test_scan_failure_handling(self, scanner):
        """Teste la gestion d'échec d'exécution."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="subfinder: command not found",
            duration=0.5
        )

        result = scanner.scan("example.com", mock_executor)

        assert result.success is False
        assert result.exit_code == 1
        assert "command not found" in result.stderr


class TestNmapScanner:
    """Tests pour le scanner Nmap."""

    @pytest.fixture
    def scanner(self):
        """Crée une instance du scanner Nmap."""
        return NmapScanner()

    def test_scanner_metadata(self, scanner):
        """Teste les métadonnées."""
        assert scanner.name == "nmap"
        assert "nmap" in scanner.description.lower()
        assert "port" in scanner.description.lower()

    def test_build_command_basic(self, scanner):
        """Teste une commande nmap de base."""
        cmd = scanner.build_command("example.com", ports=[80, 443])
        assert "nmap" in cmd
        assert "-p 80,443" in cmd
        assert "-sV" in cmd  # Version detection par défaut
        assert "example.com" in cmd

    def test_build_command_all_ports(self, scanner):
        """Teste l'option -p- pour tous les ports."""
        cmd = scanner.build_command("example.com", all_ports=True)
        assert "-p-" in cmd

    def test_build_command_with_timing(self, scanner):
        """Teste les options de timing."""
        cmd = scanner.build_command("example.com", timing="T4")
        assert "-T4" in cmd

    def test_build_command_with_script(self, scanner):
        """Teste l'exécution de scripts NSE."""
        cmd = scanner.build_command("example.com", scripts=["ssl-cert", "http-title"])
        assert "--script ssl-cert,http-title" in cmd

    def test_parse_output_valid_xml(self, scanner):
        """Teste le parsing XML de nmap."""
        # Simulation d'un output nmap en XML (simplifié)
        xml_output = """<?xml version="1.0"?>
<nmaprun scanner="nmap">
  <host>
    <address addr="93.184.216.34" addrtype="ipv4"/>
    <hostnames>
      <hostname name="example.com" type="user"/>
    </hostnames>
    <ports>
      <port protocol="tcp" portid="80">
        <state state="open"/>
        <service name="http" version="Apache 2.4.41"/>
      </port>
      <port protocol="tcp" portid="443">
        <state state="open"/>
        <service name="https" version="nginx"/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml_output, "example.com")
        assert result.success is True
        assert len(result.findings) >= 2  # Au moins 2 ports ouverts

    def test_parse_output_with_service_versions(self, scanner):
        """Teste l'extraction des versions de service."""
        xml_output = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <ports>
      <port portid="22" protocol="tcp">
        <state state="open"/>
        <service name="ssh" version="OpenSSH 7.6p1 Ubuntu 4ubuntu0.3"/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml_output, "example.com")
        ssh_finding = next((f for f in result.findings if f.port == 22), None)
        assert ssh_finding is not None
        assert "OpenSSH" in ssh_finding.service_version

    def test_parse_output_no_hosts(self, scanner):
        """Teste le parsing quand aucun hôte n'est trouvé."""
        xml_output = """<?xml version="1.0"?>
<nmaprun>
</nmaprun>"""
        result = scanner.parse_output(xml_output, "example.com")
        assert result.success is True
        assert len(result.findings) == 0

    def test_parse_output_invalid_xml(self, scanner):
        """Teste la gestion d'XML invalide."""
        result = scanner.parse_output("not xml < unclosed", "example.com")
        assert result.success is False

    def test_scan_with_ports(self, scanner):
        """Teste un scan avec ports spécifiques."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout='<?xml version="1.0"?><nmaprun></nmaprun>',
            stderr="",
            duration=5.0
        )

        result = scanner.scan("example.com", mock_executor, ports=[80, 443])

        assert result.success is True
        assert "-p 80,443" in result.command

    def test_scan_with_udp_ports(self, scanner):
        """Teste un scan UDP."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout='<?xml version="1.0"?><nmaprun></nmaprun>',
            stderr="",
            duration=10.0
        )

        result = scanner.scan("example.com", mock_executor, ports=[53, 161], udp=True)

        assert result.success is True
        assert "-sU" in result.command  # Scan UDP

    def test_scan_os_detection(self, scanner):
        """Teste l'option de détection d'OS."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout='<?xml version="1.0"?><nmaprun></nmaprun>',
            stderr="",
            duration=15.0
        )

        result = scanner.scan("example.com", mock_executor, os_detect=True)

        assert result.success is True
        assert "-O" in result.command

    def test_scan_aggressive(self, scanner):
        """Teste le mode agressif (-A)."""
        mock_executor = MagicMock()
        mock_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout='<?xml version="1.0"?><nmaprun></nmaprun>',
            stderr="",
            duration=20.0
        )

        result = scanner.scan("example.com", mock_executor, aggressive=True)

        assert result.success is True
        assert "-A" in result.command

    def test_parse_realistic_nmap_output(self, scanner):
        """Teste avec un output nmap plus réaliste."""
        xml_output = """<?xml version="1.0" encoding="UTF-8"?>
<nmaprun scanner="nmap" args="nmap -sV -p 80,443 example.com" start="1616161616">
  <scaninfo type="connect" protocol="tcp" numservices="2" services="80,443"/>
  <host>
    <address addr="93.184.216.34" addrtype="ipv4"/>
    <hostnames>
      <hostname name="example.com" type="PTR"/>
    </hostnames>
    <ports>
      <port protocol="tcp" portid="80">
        <state state="open" reason="syn-ack" reason_ttl="51"/>
        <service name="http" product="Apache httpd" version="2.4.41" extrainfo="(Ubuntu)"/>
      </port>
      <port protocol="tcp" portid="443">
        <state state="open" reason="syn-ack" reason_ttl="51"/>
        <service name="https" product="nginx" version="1.18.0 (Ubuntu)"/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml_output, "example.com")

        assert result.success is True
        assert len(result.findings) == 2
        ports = [f.port for f in result.findings]
        assert 80 in ports
        assert 443 in ports

        http_finding = next(f for f in result.findings if f.port == 80)
        assert "Apache" in http_finding.service_name
        assert "2.4.41" in http_finding.service_version


@pytest.mark.integration
class TestScannerIntegration:
    """Tests d'intégration entre Guardian et scanners."""

    def test_guardian_blocks_unauthorized_scanner(self, sample_scope):
        """Le Guardian devrait empêcher un scanner sur une cible non autorisée."""
        from security_recon_assistant.core.guardian import Guardian
        guardian = Guardian(sample_scope)

        scanner = SubfinderScanner()

        # Ceci devrait lever ViolationError
        with pytest.raises(ViolationError):
            guardian.check_command("subfinder -d evil.com", {"target": "evil.com"})

    def test_guardian_allows_authorized_target(self, sample_scope):
        """Le Guardian devrait autoriser une cible valide."""
        from security_recon_assistant.core.guardian import Guardian
        guardian = Guardian(sample_scope)

        scanner = NmapScanner()
        cmd = scanner.build_command("scanme.nmap.org", ports=[80])
        assert guardian.check_command(cmd, {"target": "scanme.nmap.org"}) is True
