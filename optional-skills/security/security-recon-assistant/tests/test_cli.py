"""Tests pour l'interface CLI."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from security_recon_assistant.cli import cli
pytestmark = pytest.mark.integration
from security_recon_assistant.scanners.base import ScanResult


class TestCLIBasic:
    """Tests de base pour l'interface CLI."""

    @pytest.fixture
    def runner(self):
        """Crée un CliRunner."""
        return CliRunner()

    def test_cli_version(self, runner):
        """Teste l'option --version."""
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_help(self, runner):
        """Teste l'option --help."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "security-recon-assistant" in result.output.lower()
        assert "target" in result.output.lower()
        assert "scope" in result.output.lower()

    def test_cli_missing_required_args(self, runner):
        """Teste l'absence des arguments requis."""
        result = runner.invoke(cli, [])
        assert result.exit_code != 0
        assert (
            "Missing argument" in result.output
            or "Missing option" in result.output
            or "requires" in result.output.lower()
        )

    def test_cli_invalid_scope(self, runner):
        """Teste avec un fichier de scope invalide."""
        result = runner.invoke(cli, [
            '--target', 'example.com',
            '--scope', '/nonexistent/scope.yaml'
        ])
        assert result.exit_code != 0
        assert "scope" in result.output.lower() or "not found" in result.output.lower()

    def test_cli_invalid_target(self, runner):
        """Teste avec une cible invalide."""
        # Crée un scope temporaire valide
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("allowed_domains:\n  - example.com\n")
            f.flush()
            scope_path = f.name

        try:
            result = runner.invoke(cli, [
                '--target', 'evil.com',
                '--scope', scope_path
            ])
            # Devrait échouer car evil.com n'est pas dans le scope
            assert result.exit_code != 0
            assert "scope" in result.output.lower() or "not allowed" in result.output.lower()
        finally:
            os.unlink(scope_path)


class TestCLIOptions:
    """Tests des options de la CLI."""

    @pytest.fixture
    def valid_scope_file(self):
        """Crée un fichier de scope valide."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
allowed_domains:
  - scanme.nmap.org
  - example.com
max_depth: 2
rate_limit: 100
""")
            f.flush()
            return f.name

    def test_output_json(self, runner, valid_scope_file):
        """Teste l'option --output-format json."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(
                    scanner_name="test",
                    success=True,
                    findings=[]
                )
            ]

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--output-format', 'json',
                '--output', '/tmp/report.json'
            ])

            assert result.exit_code == 0

    def test_output_html(self, runner, valid_scope_file):
        """Teste l'option --output-format html."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(
                    scanner_name="test",
                    success=True,
                    findings=[]
                )
            ]

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--output-format', 'html',
                '--output', '/tmp/report.html'
            ])

            assert result.exit_code == 0

    def test_quiet_mode(self, runner, valid_scope_file):
        """Teste l'option --quiet."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(
                    scanner_name="test",
                    success=True,
                    findings=[]
                )
            ]

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--quiet'
            ])

            assert result.exit_code == 0
            # En mode quiet, moins de sortie
            assert len(result.output) < 1000

    def test_verbose_mode(self, runner, valid_scope_file):
        """Teste l'option --verbose."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(
                    scanner_name="test",
                    success=True,
                    command="test command",
                    findings=[]
                )
            ]

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--verbose'
            ])

            assert result.exit_code == 0
            # En mode verbose, plus de détails
            assert "command" in result.output.lower() or "test command" in result.output

    def test_workers_option(self, runner, valid_scope_file):
        """Teste l'option --workers."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(scanner_name="test", success=True, findings=[])
            ]

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--workers', '4'
            ])

            assert result.exit_code == 0
            # Vérifier que l'executor a été configuré avec 4 workers
            # (on ne peut pas tester directement, mais on vérifie que la CLI accepte l'option)


class TestCLIErrorHandling:
    """Tests de gestion d'erreurs de la CLI."""

    @pytest.fixture
    def valid_scope_file(self):
        """Crée un fichier de scope valide."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("allowed_domains:\n  - scanme.nmap.org\n")
            f.flush()
            return f.name

    def test_keyboard_interrupt(self, runner, valid_scope_file):
        """Teste l'interruption clavier (Ctrl+C)."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            # Simule un long running qui lève KeyboardInterrupt
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file
            ])

            assert result.exit_code == 130  # Code pour SIGINT
            assert "interrupted" in result.output.lower()

    def test_scanner_exception(self, runner, valid_scope_file):
        """Teste une exception dans un scanner."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.side_effect = Exception("Unexpected error in scanner")

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file
            ])

            assert result.exit_code != 0
            assert "error" in result.output.lower()

    def test_guardian_violation(self, runner, valid_scope_file):
        """Teste une violation du guardian."""
        from security_recon_assistant.core.guardian import ViolationError

        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.side_effect = ViolationError("Target out of scope")

            result = runner.invoke(cli, [
                '--target', 'evil.com',
                '--scope', valid_scope_file
            ])

            assert result.exit_code != 0
            assert "scope" in result.output.lower()

    def test_executor_failure(self, runner, valid_scope_file):
        """Teste l'échec de l'exécuteur (commande système)."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.side_effect = RuntimeError("Command execution failed")

            result = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file
            ])

            assert result.exit_code != 0


class TestCLIRealExecution:
    """Tests avec exécution réelle (mocks légers)."""

    def test_full_workflow_mocked_scanners(self, runner):
        """Teste un workflow complet avec scanners mockés."""
        yaml_content = """
allowed_domains:
  - scanme.nmap.org
max_depth: 2
rate_limit: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            scope_path = f.name

        output_path = tempfile.mktemp(suffix='.json')

        try:
            # Mock les scanners pour éviter de lancer de vrais outils
            with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
                from security_recon_assistant.scanners.base import ScanResult, ScanFinding

                mock_run.return_value = [
                    ScanResult(
                        scanner_name="subfinder",
                        success=True,
                        execution_time=2.5,
                        findings=[
                            ScanFinding(
                                target="www.scanme.nmap.org",
                                severity="low",
                                title="Subdomain",
                                description="Found via DNS",
                                evidence="{}"
                            )
                        ]
                    ),
                    ScanResult(
                        scanner_name="nmap",
                        success=True,
                        execution_time=10.0,
                        findings=[
                            ScanFinding(
                                target="scanme.nmap.org",
                                severity="medium",
                                title="Open port",
                                description="Port 80 open",
                                port=80,
                                evidence=""
                            )
                        ]
                    )
                ]

                result = runner.invoke(cli, [
                    '--target', 'scanme.nmap.org',
                    '--scope', scope_path,
                    '--output-format', 'json',
                    '--output', output_path,
                    '--log-level', 'INFO'
                ])

                assert result.exit_code == 0
                assert os.path.exists(output_path)

                # Vérifie le contenu du fichier de sortie
                import json
                with open(output_path, 'r') as f:
                    report = json.load(f)
                    assert report["summary"]["total_findings"] == 2
                    assert len(report["findings"]) == 2

        finally:
            if os.path.exists(scope_path):
                os.unlink(scope_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_with_verbosity_levels(self, runner, valid_scope_file):
        """Teste les différents niveaux de verbosité."""
        with patch('security_recon_assistant.orchestrator.pipeline.Pipeline.run') as mock_run:
            mock_run.return_value = [
                ScanResult(scanner_name="test", success=True, findings=[])
            ]

            # Sans verbose
            result_quiet = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file
            ])

            # Avec verbose
            result_verbose = runner.invoke(cli, [
                '--target', 'scanme.nmap.org',
                '--scope', valid_scope_file,
                '--verbose'
            ])

            assert result_quiet.exit_code == 0
            assert result_verbose.exit_code == 0
            # Normalement le verbose donne plus d'infos
            assert len(result_verbose.output) >= len(result_quiet.output)
