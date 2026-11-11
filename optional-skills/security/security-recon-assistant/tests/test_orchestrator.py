"""Tests pour l'orchestrateur et le pipeline."""

import pytest
from unittest.mock import MagicMock, patch

from security_recon_assistant.orchestrator.pipeline import Pipeline, PipelineConfig
from security_recon_assistant.scanners.base import ScanResult
from security_recon_assistant.core.executor import ExecutionResult


pytestmark = pytest.mark.unit


class TestPipelineConfig:
    """Tests pour la configuration du pipeline."""

    def test_default_config(self):
        """Teste les valeurs par défaut."""
        config = PipelineConfig()
        assert config.sequential is True
        assert config.retry_failed is False
        assert config.max_retries == 1
        assert config.stop_on_critical is True

    def test_custom_config(self):
        """Teste une configuration personnalisée."""
        config = PipelineConfig(
            sequential=False,
            retry_failed=True,
            max_retries=3,
            stop_on_critical=False
        )
        assert config.sequential is False
        assert config.retry_failed is True
        assert config.max_retries == 3
        assert config.stop_on_critical is False


class TestPipeline:
    """Tests pour l'orchestrateur Pipeline."""

    @pytest.fixture
    def mock_scanners(self):
        """Crée des scanners mockés."""
        scanner1 = MagicMock()
        scanner1.name = "subfinder"
        scanner1.scan.return_value = ScanResult(
            scanner_name="subfinder",
            success=True,
            findings=[]
        )

        scanner2 = MagicMock()
        scanner2.name = "nmap"
        scanner2.scan.return_value = ScanResult(
            scanner_name="nmap",
            success=True,
            findings=[]
        )

        return [scanner1, scanner2]

    @pytest.fixture
    def mock_guardian(self):
        """Crée un Guardian mocké."""
        guardian = MagicMock()
        guardian.check_command.return_value = True
        return guardian

    @pytest.fixture
    def mock_executor(self):
        """Crée un exécuteur mocké."""
        executor = MagicMock()
        executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration=1.0
        )
        return executor

    def test_pipeline_initialization(self, mock_scanners, mock_guardian, mock_executor):
        """Teste l'initialisation du pipeline."""
        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )
        assert len(pipeline.scanners) == 2
        assert pipeline.config.sequential is True

    def test_pipeline_sequential_execution(self, mock_scanners, mock_guardian, mock_executor):
        """Teste l'exécution séquentielle."""
        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        assert len(results) == 2
        mock_scanners[0].scan.assert_called_once()
        mock_scanners[1].scan.assert_called_once()

    def test_pipeline_parallel_execution(self, mock_scanners, mock_guardian, mock_executor):
        """Teste l'exécution parallèle."""
        config = PipelineConfig(sequential=False)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        assert len(results) == 2
        # En parallèle, les deux sont appelés sans ordre forcé
        mock_scanners[0].scan.assert_called_once()
        mock_scanners[1].scan.assert_called_once()

    def test_pipeline_handles_failure(self, mock_scanners, mock_guardian, mock_executor):
        """Teste la gestion d'échec d'un scanner."""
        # Premier scanner échoue
        mock_scanners[0].scan.return_value = ScanResult(
            scanner_name="subfinder",
            success=False,
            stderr="error occurred"
        )
        mock_scanners[1].scan.return_value = ScanResult(
            scanner_name="nmap",
            success=True,
            findings=[]
        )

        config = PipelineConfig(sequential=True, retry_failed=False)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is True

    def test_pipeline_stop_on_critical(self, mock_scanners, mock_guardian, mock_executor):
        """Teste l'arrêt sur criticité."""
        # Premier scanner retourne un finding CRITICAL
        critical_finding = MagicMock()
        critical_finding.severity = "critical"

        mock_scanners[0].scan.return_value = ScanResult(
            scanner_name="subfinder",
            success=True,
            findings=[critical_finding]
        )

        config = PipelineConfig(sequential=True, stop_on_critical=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        # Le deuxième scanner ne devrait pas être exécuté
        assert len(results) == 1
        mock_scanners[1].scan.assert_not_called()

    def test_pipeline_continue_on_non_critical(self, mock_scanners, mock_guardian, mock_executor):
        """Teste la continuation malgré un finding non critique."""
        # Premier scanner retourne un finding MEDIUM
        medium_finding = MagicMock()
        medium_finding.severity = "medium"

        mock_scanners[0].scan.return_value = ScanResult(
            scanner_name="subfinder",
            success=True,
            findings=[medium_finding]
        )

        config = PipelineConfig(sequential=True, stop_on_critical=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        assert len(results) == 2
        mock_scanners[1].scan.assert_called_once()

    def test_pipeline_guardian_integration(self, mock_scanners, mock_executor):
        """Teste que le Guardian est appelé pour chaque commande."""
        guardian = MagicMock()
        guardian.check_command.return_value = True

        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=guardian,
            executor=mock_executor,
            config=config
        )

        pipeline.run("example.com")

        # Le Guardian doit être appelé au moins une fois par scanner
        assert guardian.check_command.call_count >= 2

    def test_pipeline_guardian_violation_stops_pipeline(self, mock_scanners, mock_executor):
        """Teste qu'une violation du Guardian arrête le pipeline."""
        from security_recon_assistant.core.guardian import ViolationError

        guardian = MagicMock()
        # Premier appel succeed, deuxième lève ViolationError
        guardian.check_command.side_effect = [True, ViolationError("Out of scope")]

        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        # Le deuxième scanner ne devrait pas avoir été appelé car Guardian a échoué
        assert len(results) == 1
        mock_scanners[1].scan.assert_not_called()

    def test_pipeline_empty_scanners(self, mock_guardian, mock_executor):
        """Teste un pipeline sans scanner."""
        config = PipelineConfig()
        pipeline = Pipeline(
            scanners=[],
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")
        assert results == []

    def test_pipeline_aggregates_statistics(self, mock_scanners, mock_guardian, mock_executor):
        """Teste que le pipeline agrège les statistiques."""
        mock_scanners[0].scan.return_value = ScanResult(
            scanner_name="subfinder",
            success=True,
            execution_time=1.0,
            findings=[]
        )
        mock_scanners[1].scan.return_value = ScanResult(
            scanner_name="nmap",
            success=True,
            execution_time=5.0,
            findings=[]
        )

        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        # Le pipeline devrait retourner tous les résultats
        assert len(results) == 2
        total_time = sum(r.execution_time for r in results if r.execution_time)
        assert total_time > 0

    def test_pipeline_retry_logic(self, mock_scanners, mock_guardian, mock_executor):
        """Teste la logique de retry."""
        # Premier appel échoue, deuxième réussit
        mock_scanners[0].scan.side_effect = [
            ScanResult(scanner_name="subfinder", success=False, stderr="error"),
            ScanResult(scanner_name="subfinder", success=True, findings=[])
        ]

        config = PipelineConfig(sequential=True, retry_failed=True, max_retries=1)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=mock_executor,
            config=config
        )

        results = pipeline.run("example.com")

        assert len(results) == 2  # Deux appels au premier scanner
        assert results[0].success is False
        assert results[1].success is True

    def test_executor_propagation(self, mock_scanners, mock_guardian):
        """Teste que l'exécuteur est passé correctement aux scanners."""
        custom_executor = MagicMock()
        custom_executor.run.return_value = ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration=1.0
        )

        config = PipelineConfig(sequential=True)
        pipeline = Pipeline(
            scanners=mock_scanners,
            guardian=mock_guardian,
            executor=custom_executor,
            config=config
        )

        pipeline.run("example.com")

        # Vérifie que c'est bien custom_executor qui a été utilisé
        mock_scanners[0].scan.assert_called_with("example.com", custom_executor)
        mock_scanners[1].scan.assert_called_with("example.com", custom_executor)
