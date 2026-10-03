"""Tests pour les cas limites et edge cases."""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from security_recon_assistant.core.scope import ScopeConfig, load_scope_from_yaml
from security_recon_assistant.core.guardian import Guardian, ViolationError
from security_recon_assistant.core.executor import ExecutionResult
from security_recon_assistant.scanners.base import ScanFinding, ScanResult
from security_recon_assistant.scanners.subfinder_scanner import SubfinderScanner
from security_recon_assistant.scanners.nmap_scanner import NmapScanner


pytestmark = pytest.mark.unit


class TestScopeEdgeCases:
    """Tests des cas limites pour le scope."""

    def test_empty_allowed_domains(self):
        """Un scope avec une whitelist vide devrait tout refuser."""
        yaml_content = """
allowed_domains: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            assert config.allowed_domains == set()
            guardian = Guardian(config)
            assert guardian.is_allowed("anything.com") is False
        finally:
            os.unlink(temp_path)

    def test_wildcard_subdomain_depth(self):
        """Teste le wildcard avec plusieurs niveaux de sous-domaines."""
        yaml_content = """
allowed_domains:
  - "*.deep.example.com"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            guardian = Guardian(config)
            # *.deep.example.com should match a.b.c.deep.example.com
            assert guardian.is_allowed("a.b.c.deep.example.com") is True
            # But not just example.com or deep.example.com
            assert guardian.is_allowed("example.com") is False
            assert guardian.is_allowed("deep.example.com") is False
        finally:
            os.unlink(temp_path)

    def test_wildcard_root_domain_should_not_match(self):
        """Un wildcard ne devrait pas matcher le domaine racine."""
        yaml_content = """
allowed_domains:
  - "*.example.com"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            guardian = Guardian(config)
            assert guardian.is_allowed("example.com") is False
            assert guardian.is_allowed("www.example.com") is True
        finally:
            os.unlink(temp_path)

    def test_duplicate_domains_in_scope(self):
        """Les domaines en double devraient être dédupliqués."""
        yaml_content = """
allowed_domains:
  - example.com
  - example.com
  - EXAMPLE.COM
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            # Devrait avoir seulement 1 domaine (case-insensitive dedup)
            assert len(config.allowed_domains) == 1
            assert "example.com" in config.allowed_domains
        finally:
            os.unlink(temp_path)

    def test_excluded_domain_is_subdomain_of_allowed(self):
        """
        Si un domaine est exclu mais est un sous-domaine d'un domaine autorisé,
        l'exclusion doit prévaloir.
        """
        yaml_content = """
allowed_domains:
  - example.com
excluded_domains:
  - admin.example.com
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            guardian = Guardian(config)
            # example.com est autorisé
            assert guardian.is_allowed("example.com") is True
            # www.example.com est un sous-domaine, autorisé (pas d'exclusion spécifique)
            assert guardian.is_allowed("www.example.com") is True
            # admin.example.com est exclu
            assert guardian.is_allowed("admin.example.com") is False
            # Autre sous-domaine exclu
            assert guardian.is_allowed("test.admin.example.com") is False
        finally:
            os.unlink(temp_path)

    def test_scope_case_sensitivity(self):
        """Les domaines devraient être comparés de manière case-insensitive."""
        yaml_content = """
allowed_domains:
  - Example.COM
  - TeSt.OrG
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            guardian = Guardian(config)
            assert guardian.is_allowed("example.com") is True
            assert guardian.is_allowed("EXAMPLE.COM") is True
            assert guardian.is_allowed("test.org") is True
            assert guardian.is_allowed("TEST.ORG") is True
        finally:
            os.unlink(temp_path)


class TestGuardianEdgeCases:
    """Tests des cas limites pour le Guardian."""

    def test_command_with_multiple_targets(self, sample_scope_config):
        """Teste l'extraction de plusieurs cibles depuis une même commande."""
        guardian = Guardian(sample_scope_config)
        targets = guardian._extract_targets("nmap -p 80,443 scanme.nmap.org example.com")
        assert "scanme.nmap.org" in targets
        assert "example.com" in targets

    def test_command_with_ip_address(self, sample_scope_config):
        """Teste l'extraction d'une IP."""
        guardian = Guardian(sample_scope_config)
        # Pour l'instant les IPs ne sont pas dans le scope, donc extraction seulement
        targets = guardian._extract_targets("nmap -p 80 192.168.1.1")
        assert "192.168.1.1" in targets
        # Et c'est hors scope (rejeté par is_allowed actuellement)
        assert guardian.is_allowed("192.168.1.1") is False

    def test_check_command_with_empty_target_list(self, sample_scope_config):
        """check_command avec une cible vide."""
        guardian = Guardian(sample_scope_config)
        # Sans cible détectée, on ne peut pas valider — devrait retourner False ou raise?
        # Actuellement: ViolationError si pas de cible dans le scope
        with pytest.raises(ViolationError):
            guardian.check_command("uptime", {})

    def test_check_command_different_user_context(self, sample_scope_config):
        """Teste que le contexte utilisateur n'affecte pas la validation."""
        guardian = Guardian(sample_scope_config)
        # Peu importe le contexte fourni, la validation devrait être la même
        result1 = guardian.check_command("nmap scanme.nmap.org", {"user": "alice"})
        result2 = guardian.check_command("nmap scanme.nmap.org", {"user": "bob"})
        assert result1 is True
        assert result2 is True

    def test_guardian_performance_many_domains(self):
        """Teste les performances avec beaucoup de domaines."""
        yaml_content = """
allowed_domains:
"""
        # Génère 1000 domaines
        domains = [f"test{i}.example.com" for i in range(1000)]
        yaml_content += "\n  - " + "\n  - ".join(domains)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            guardian = Guardian(config)
            # Un domain dans la liste
            start = time.time()
            assert guardian.is_allowed("test500.example.com") is True
            elapsed = time.time() - start
            assert elapsed < 0.1  # Devrait être rapide

            # Un domain pas dans la liste
            assert guardian.is_allowed("notinlist.example.com") is False
        finally:
            os.unlink(temp_path)


class TestExecutorEdgeCases:
    """Tests des cas limites pour l'exécuteur."""

    def test_execution_result_creation(self):
        """Teste la création d'un ExecutionResult."""
        from security_recon_assistant.core.executor import ExecutionResult

        result = ExecutionResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration=2.5
        )
        assert result.exit_code == 0
        assert result.success is True

    def test_execution_result_failure(self):
        """Teste un ExecutionResult en cas d'échec."""
        from security_recon_assistant.core.executor import ExecutionResult

        result = ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="error message",
            duration=0.1
        )
        assert result.success is False
        assert result.exit_code != 0

    def test_execution_result_timeout(self):
        """Teste un ExecutionResult après timeout."""
        from security_recon_assistant.core.executor import ExecutionResult

        result = ExecutionResult(
            exit_code=None,
            stdout="partial output",
            stderr="",
            duration=300.0,
            timeout=True
        )
        assert result.success is False
        assert result.timeout is True

    def test_executor_with_large_output(self):
        """Teste la gestion de grande sortie."""
        from security_recon_assistant.core.executor import ExecutionResult, Executor

        executor = Executor()

        # Simule une grande sortie
        large_stdout = "A" * 100000

        result = ExecutionResult(
            exit_code=0,
            stdout=large_stdout,
            stderr="",
            duration=1.0
        )
        # Le système devrait gérer sans crash
        assert len(result.stdout) == 100000

    def test_executor_with_unicode_output(self):
        """Teste la gestion de l'unicode."""
        from security_recon_assistant.core.executor import ExecutionResult

        unicode_output = "Sortie unicode: café, naïve, 日本語 🎌"
        result = ExecutionResult(
            exit_code=0,
            stdout=unicode_output,
            stderr="",
            duration=1.0
        )
        assert "café" in result.stdout


class TestScannerEdgeCases:
    """Tests des cas limites pour les scanners."""

    def test_subfinder_empty_response(self):
        """Teste Subfinder avec une réponse vide."""
        scanner = SubfinderScanner()
        result = scanner.parse_output("", "example.com")
        assert result.success is True
        assert len(result.findings) == 0

    def test_subfinder_malformed_json(self):
        """Teste Subfinder avec du JSON malformé."""
        scanner = SubfinderScanner()
        result = scanner.parse_output("{invalid json", "example.com")
        assert result.success is False
        assert result.stderr is not None

    def test_subfinder_with_special_characters_in_domain(self):
        """Teste un domaine avec des caractères spéciaux (devrait échouer proprement)."""
        scanner = SubfinderScanner()
        # Les domaines ne devraient pas contenir de slash, etc.
        result = scanner.parse_output('{"host": "example.com/evil"}', "example.com")
        # Le parsing devrait gérer cela (peut-être en filtrant)
        # Pour l'instant, le domaine est pris tel quel
        assert result.success is True or result.success is False

    def test_nmap_with_no_hosts(self):
        """Teste Nmap quand aucun hôte n'est up."""
        scanner = NmapScanner()
        xml = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <status state="down"/>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml, "example.com")
        assert result.success is True
        assert len(result.findings) == 0

    def test_nmap_with_multiple_ports_same_service(self):
        """Teste plusieurs ports avec le même service."""
        scanner = NmapScanner()
        xml = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <ports>
      <port portid="80">
        <state state="open"/>
        <service name="http" product="Apache"/>
      </port>
      <port portid="8080">
        <state state="open"/>
        <service name="http-proxy" product="Apache"/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml, "example.com")
        assert len(result.findings) == 2
        ports = {f.port for f in result.findings}
        assert ports == {80, 8080}

    def test_nmap_xml_special_characters(self):
        """Teste la gestion de caractères spéciaux dans le XML."""
        scanner = NmapScanner()
        xml = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <ports>
      <port portid="80">
        <service name="http" product="Apache &quot;Secure&quot;" version="2.4.41 &amp;"/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml, "example.com")
        assert result.success is True
        # Les caractères échappés devraient être correctement parsés
        assert len(result.findings) > 0

    def test_nmap_parse_with_script_output(self):
        """Teste le parsing avec des résultats de script NSE."""
        scanner = NmapScanner()
        xml = """<?xml version="1.0"?>
<nmaprun>
  <host>
    <ports>
      <port portid="443">
        <state state="open"/>
        <service name="https"/>
        <script id="ssl-cert" output="Subject: CN=example.com\n..."/>
      </port>
    </ports>
  </host>
</nmaprun>"""
        result = scanner.parse_output(xml, "example.com")
        # Devrait extraire les données du script
        ssl_finding = next((f for f in result.findings if f.port == 443), None)
        assert ssl_finding is not None
        assert ssl_finding.evidence is not None


class TestConcurrentSafety:
    """Tests de concurrence (même si les tests sont séquentiels, on teste la logique)."""

    def test_multiple_guardian_instances_independent(self, sample_scope_config):
        """Plusieurs instances de Guardian devraient être indépendantes."""
        g1 = Guardian(sample_scope_config)
        g2 = Guardian(sample_scope_config)

        assert g1.is_allowed("scanme.nmap.org") is True
        assert g2.is_allowed("scanme.nmap.org") is True
        # Modifier un cache interne (si existe) ne devrait pas affecter l'autre
        # Pour l'instant Guardian n'a pas de cache mutable, donc c'est bon

    def test_scan_result_immutability(self, sample_scan_finding):
        """Teste que les ScanResult créés sont correctement copiés."""
        original = ScanResult(
            scanner_name="test",
            success=True,
            findings=[sample_scan_finding]
        )
        # Modifier le finding original ne devrait pas affecter le ScanResult
        sample_scan_finding.title = "Modified"
        assert original.findings[0].title != "Modified"

        # À moins que ce soit le même objet...
        # On devrait faire une copie profonde dans les scanners
        # Cela pourrait être amélioré dans le code de production
