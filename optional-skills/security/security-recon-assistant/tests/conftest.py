"""Fixtures pytest partagés pour les tests."""

import pytest
import tempfile
import os
from pathlib import Path
from click.testing import CliRunner

from security_recon_assistant.core.scope import ScopeConfig, load_scope_from_yaml
from security_recon_assistant.core.guardian import Guardian
from security_recon_assistant.scanners.base import ScanResult, ScanFinding


@pytest.fixture
def temp_scope_file():
    """Crée un fichier de scope temporaire."""
    yaml_content = """
allowed_domains:
  - scanme.nmap.org
  - example.com
  - "*.test.org"
excluded_domains:
  - admin.scanme.nmap.org
  - "*.dev.test.org"
max_depth: 2
rate_limit: 100
check_ssl: true
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


@pytest.fixture
def sample_scope_config(temp_scope_file):
    """Charge une configuration de scope depuis le fichier temporaire."""
    return load_scope_from_yaml(temp_scope_file)


@pytest.fixture
def sample_scope(sample_scope_config):
    """Alias de compatibilité pour les tests qui attendent sample_scope."""
    return sample_scope_config


@pytest.fixture
def guardian(sample_scope_config):
    """Crée une instance du Guardian avec le scope de test."""
    return Guardian(sample_scope_config)


@pytest.fixture
def sample_scan_finding():
    """Crée un ScanFinding de test."""
    return ScanFinding(
        target="example.com",
        severity="medium",
        title="Open Port 80",
        description="Port 80 is open and serving HTTP",
        evidence='{"port": 80, "state": "open"}',
        remediation="Implement firewall rules",
        port=80,
        service_name="http",
        service_version="Apache 2.4.41"
    )


@pytest.fixture
def sample_scan_result(sample_scan_finding):
    """Crée un ScanResult de test avec un finding."""
    return ScanResult(
        scanner_name="nmap",
        success=True,
        command="nmap -sV -p 80 example.com",
        execution_time=5.2,
        stdout="output",
        stderr="",
        findings=[sample_scan_finding]
    )


@pytest.fixture
def mock_executor():
    """Crée un mock d'exécuteur qui retourne un résultat réussi."""
    from security_recon_assistant.core.executor import ExecutionResult

    executor = MagicMock()
    executor.run.return_value = ExecutionResult(
        exit_code=0,
        stdout="Mock output",
        stderr="",
        duration=1.0
    )
    return executor


@pytest.fixture
def failing_mock_executor():
    """Crée un mock d'exécuteur qui échoue."""
    from security_recon_assistant.core.executor import ExecutionResult

    executor = MagicMock()
    executor.run.return_value = ExecutionResult(
        exit_code=1,
        stdout="",
        stderr="Command failed",
        duration=0.5
    )
    return executor


@pytest.fixture
def tmp_report_dir():
    """Crée un répertoire temporaire pour les rapports."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmpdir)
    except OSError:
        pass


@pytest.fixture
def runner():
    """CliRunner partagé pour tous les tests CLI."""
    return CliRunner()


@pytest.fixture
def valid_scope_file():
    """Fichier scope valide partagé pour tests CLI."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
allowed_domains:
  - scanme.nmap.org
  - example.com
max_depth: 2
rate_limit: 100
""")
        f.flush()
        path = f.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# Import nécessaire pour les fixtures
from unittest.mock import MagicMock
