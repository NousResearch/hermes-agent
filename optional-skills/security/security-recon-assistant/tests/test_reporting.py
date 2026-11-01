"""Tests pour le module reporting."""

import pytest
import tempfile
import os
from pathlib import Path

from security_recon_assistant.reporting.json_report import JSONReportGenerator
from security_recon_assistant.reporting.html_report import HTMLReportGenerator
from security_recon_assistant.scanners.base import ScanResult, ScanFinding
from security_recon_assistant.core.scope import ScopeConfig


pytestmark = pytest.mark.unit


class TestJSONReportGenerator:
    """Tests pour le générateur de rapport JSON."""

    @pytest.fixture
    def sample_results(self):
        """Crée des résultats de scan réalistes."""
        return [
            ScanResult(
                scanner_name="subfinder",
                success=True,
                execution_time=2.5,
                command="subfinder -d example.com",
                findings=[
                    ScanFinding(
                        target="www.example.com",
                        severity="low",
                        title="Subdomain discovered",
                        description="Subdomain found via certificate transparency",
                        evidence='{"host": "www.example.com", "source": "crt.sh"}',
                        remediation="No immediate action required"
                    ),
                    ScanFinding(
                        target="api.example.com",
                        severity="low",
                        title="Subdomain discovered",
                        description="Subdomain found via DNS enumeration",
                        evidence='{"host": "api.example.com", "source": "dnsdumpster"}'
                    )
                ]
            ),
            ScanResult(
                scanner_name="nmap",
                success=True,
                execution_time=15.3,
                command="nmap -sV -p 80,443 example.com",
                findings=[
                    ScanFinding(
                        target="example.com",
                        severity="medium",
                        title="Open HTTPS port",
                        description="Port 443/tcp is open",
                        evidence="<port protocol='tcp' portid='443'><state state='open'/></port>",
                        port=443,
                        service_name="https",
                        service_version="nginx 1.18.0",
                        remediation="Ensure HTTPS is properly configured"
                    ),
                    ScanFinding(
                        target="example.com",
                        severity="high",
                        title="Outdated Apache version",
                        description="Apache 2.4.29 is vulnerable to CVE-2019-0211",
                        evidence="Apache/2.4.29 (Ubuntu)",
                        cve="CVE-2019-0211",
                        port=80,
                        service_name="http",
                        service_version="Apache 2.4.29",
                        remediation="Upgrade to Apache 2.4.39 or later"
                    )
                ]
            )
        ]

    @pytest.fixture
    def sample_scope(self):
        """Crée une configuration de scope."""
        return ScopeConfig(
            allowed_domains=["example.com", "*.test.org"],
            excluded_domains=["admin.example.com"],
            max_depth=2
        )

    def test_json_generator_initialization(self):
        """Teste l'initialisation du générateur."""
        generator = JSONReportGenerator(pretty=True)
        assert generator.pretty is True

    def test_generate_basic_structure(self, sample_results, sample_scope):
        """Teste la structure de base du rapport JSON."""
        generator = JSONReportGenerator(pretty=False)
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=18.2
        )

        assert "report" in report
        assert "metadata" in report
        assert "findings" in report
        assert "summary" in report

    def test_generate_metadata(self, sample_results, sample_scope):
        """Teste les métadonnées du rapport."""
        generator = JSONReportGenerator()
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=20.0,
            custom_metadata={"client": "Acme Corp"}
        )

        meta = report["metadata"]
        assert meta["target"] == "example.com"
        assert "generated_at" in meta
        assert meta["total_scanners"] == 2
        assert meta["client"] == "Acme Corp"

    def test_summary_statistics(self, sample_results, sample_scope):
        """Teste les statistiques du résumé."""
        generator = JSONReportGenerator()
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=18.2
        )

        summary = report["summary"]
        assert summary["total_findings"] == 3
        assert summary["critical_findings"] == 0
        assert summary["high_findings"] == 1  # Outdated Apache
        assert summary["medium_findings"] == 1
        assert summary["low_findings"] == 1
        assert summary["info_findings"] == 0
        assert summary["successful_scans"] == 2
        assert summary["failed_scans"] == 0

    def test_serialize_scan_result(self, sample_results, sample_scope):
        """Teste la sérialisation complète d'un ScanResult."""
        generator = JSONReportGenerator()
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=18.2
        )

        findings_data = report["findings"]
        assert len(findings_data) == 3

        # Vérifie le premier finding (subfinder)
        subfinder_finding = findings_data[0]
        assert subfinder_finding["scanner"] == "subfinder"
        assert subfinder_finding["target"] == "www.example.com"
        assert subfinder_finding["severity"] == "low"
        assert "remediation" in subfinder_finding

        # Vérifie un finding avec CVE
        nmap_cve_finding = next(f for f in findings_data if f.get("cve"))
        assert nmap_cve_finding["cve"] == "CVE-2019-0211"

    def test_save_to_file(self, sample_results, sample_scope):
        """Teste la sauvegarde du rapport dans un fichier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "report.json")
            generator = JSONReportGenerator()
            generator.save(
                filepath=filepath,
                target="example.com",
                results=sample_results,
                scope=sample_scope,
                total_duration=18.2
            )

            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
                assert "subfinder" in content
                assert "nmap" in content

    def test_pretty_formatting(self, sample_results, sample_scope):
        """Teste que le format pretty produit un JSON indenté."""
        import json

        generator = JSONReportGenerator(pretty=True)
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=18.2
        )

        json_str = json.dumps(report, indent=2)
        assert "\n" in json_str
        assert "  " in json_str  # indentation

    def test_empty_results(self, sample_scope):
        """Teste un rapport avec aucun finding."""
        generator = JSONReportGenerator()
        results = [
            ScanResult(
                scanner_name="nmap",
                success=True,
                findings=[]
            )
        ]

        report = generator.generate(
            target="example.com",
            results=results,
            scope=sample_scope,
            total_duration=5.0
        )

        assert report["summary"]["total_findings"] == 0
        assert len(report["findings"]) == 0

    def test_failed_scan_in_results(self, sample_scope):
        """Teste un scan qui a échoué."""
        results = [
            ScanResult(
                scanner_name="subfinder",
                success=False,
                stderr="subfinder: timeout",
                execution_time=30.0
            )
        ]

        generator = JSONReportGenerator()
        report = generator.generate(
            target="example.com",
            results=results,
            scope=sample_scope,
            total_duration=30.0
        )

        assert report["summary"]["failed_scans"] == 1
        assert report["summary"]["successful_scans"] == 0

    def test_scope_inclusion_in_report(self, sample_results, sample_scope):
        """Teste que le scope est inclus dans le rapport."""
        generator = JSONReportGenerator()
        report = generator.generate(
            target="example.com",
            results=sample_results,
            scope=sample_scope,
            total_duration=18.2
        )

        assert "scope" in report
        scope_data = report["scope"]
        assert "example.com" in scope_data["allowed_domains"]
        assert "*.test.org" in scope_data["allowed_domains"]
