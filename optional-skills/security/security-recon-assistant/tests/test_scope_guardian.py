"""Tests unitaires pour le module scope et guardian."""

import pytest
import tempfile
import os
from pathlib import Path

from security_recon_assistant.core.scope import ScopeConfig, load_scope_from_yaml
from security_recon_assistant.core.guardian import Guardian, ViolationError


pytestmark = pytest.mark.unit


class TestScopeConfig:
    """Tests pour la configuration de scope."""

    def test_empty_scope(self):
        """Un scope vide devrait être valide."""
        config = ScopeConfig()
        assert config.allowed_domains == set()
        assert config.excluded_domains == set()
        assert config.max_depth == 3
        assert config.rate_limit == 50

    def test_parse_domains(self):
        """Teste le parsing des listes de domaines."""
        config = ScopeConfig(
            allowed_domains=["example.com", "*.test.org"],
            excluded_domains=["*.excluded.com", "bad.net"]
        )
        assert "example.com" in config.allowed_domains
        assert "*.test.org" in config.allowed_domains
        assert len(config.allowed_domains) == 2
        assert len(config.excluded_domains) == 2

    def test_load_from_yaml_minimal(self):
        """Teste le chargement d'un YAML minimal."""
        yaml_content = """
allowed_domains:
  - example.com
excluded_domains: []
max_depth: 2
rate_limit: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            assert config.allowed_domains == {"example.com"}
            assert config.max_depth == 2
            assert config.rate_limit == 100
        finally:
            os.unlink(temp_path)

    def test_load_from_yaml_full(self):
        """Teste le chargement d'un YAML complet avec toutes les options."""
        yaml_content = """
# Scope de sécurité
allowed_domains:
  - scanme.nmap.org
  - test.example.com
  - "*.demo.local"
excluded_domains:
  - "*.staging.example.com"
  - admin.scanme.nmap.org
max_depth: 3
rate_limit: 200
check_ssl: true
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            config = load_scope_from_yaml(temp_path)
            assert "scanme.nmap.org" in config.allowed_domains
            assert "*.demo.local" in config.allowed_domains
            assert "*.staging.example.com" in config.excluded_domains
            assert config.max_depth == 3
            assert config.rate_limit == 200
            assert config.check_ssl is True
        finally:
            os.unlink(temp_path)

    def test_load_from_nonexistent_file(self):
        """Un fichier inexistant devrait lever une erreur."""
        with pytest.raises(FileNotFoundError):
            load_scope_from_yaml("/nonexistent/path.yaml")

    def test_invalid_yaml(self):
        """Un YAML invalide devrait lever une erreur."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # PY YAML Error
                load_scope_from_yaml(temp_path)
        finally:
            os.unlink(temp_path)


class TestGuardian:
    """Tests pour le Guardian de scope."""

    @pytest.fixture
    def sample_scope(self):
        """Crée un scope de test avec des règles claires."""
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
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        config = load_scope_from_yaml(temp_path)
        os.unlink(temp_path)
        return config

    def test_guardian_initialization(self, sample_scope):
        """Teste l'initialisation du Guardian."""
        guardian = Guardian(sample_scope)
        assert guardian.scope_config == sample_scope
        assert len(guardian.allowed_patterns) == 3
        assert len(guardian.excluded_patterns) == 2

    def test_is_allowed_exact_match(self, sample_scope):
        """Teste qu'un domaine exact dans la whitelist est autorisé."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("scanme.nmap.org") is True
        assert guardian.is_allowed("example.com") is True

    def test_is_allowed_wildcard_match(self, sample_scope):
        """Teste le matching wildcard."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("api.test.org") is True
        assert guardian.is_allowed("foo.bar.test.org") is True

    def test_is_allowed_subdomain_exact(self, sample_scope):
        """Teste qu'un sous-domaine précis est autorisé si le parent est whitelisté."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("www.example.com") is True
        assert guardian.is_allowed("mail.example.com") is True

    def test_is_excluded_exact(self, sample_scope):
        """Teste qu'un domaine exclu est refusé."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("admin.scanme.nmap.org") is False

    def test_is_excluded_wildcard(self, sample_scope):
        """Teste le matching wildcard pour l'exclusion."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("dev.api.test.org") is False
        assert guardian.is_allowed("staging.api.test.org") is False

    def test_exclusion_overrides_allowed(self, sample_scope):
        """L'exclusion doit override l'autorisation."""
        guardian = Guardian(sample_scope)
        # scanme.nmap.org est whitelisté, mais admin.scanme.nmap.org est exclu
        assert guardian.is_allowed("admin.scanme.nmap.org") is False

    def test_is_allowed_unknown_domain(self, sample_scope):
        """Un domaine non whitelisté devrait être refusé."""
        guardian = Guardian(sample_scope)
        assert guardian.is_allowed("evil.com") is False
        assert guardian.is_allowed("example.org") is False

    def test_check_command_allowed(self, sample_scope):
        """Teste check_command pour une commande valide."""
        guardian = Guardian(sample_scope)
        # Devrait passer car scanme.nmap.org est whitelisté
        result = guardian.check_command(
            "nmap -sV scanme.nmap.org",
            {"target": "scanme.nmap.org"}
        )
        assert result is True

    def test_check_command_blocked_target(self, sample_scope):
        """Teste check_command pour une cible non autorisée."""
        guardian = Guardian(sample_scope)
        with pytest.raises(ViolationError, match="hors scope"):
            guardian.check_command(
                "nmap -sV evil.com",
                {"target": "evil.com"}
            )

    def test_check_command_blocked_excluded(self, sample_scope):
        """Teste check_command pour une cible exclue."""
        guardian = Guardian(sample_scope)
        with pytest.raises(ViolationError, match="exclu"):
            guardian.check_command(
                "nmap -sV admin.scanme.nmap.org",
                {"target": "admin.scanme.nmap.org"}
            )

    def test_extract_targets_simple(self, sample_scope):
        """Teste l'extraction de cibles depuis une commande."""
        guardian = Guardian(sample_scope)
        targets = guardian._extract_targets("nmap -sV scanme.nmap.org")
        assert "scanme.nmap.org" in targets

    def test_extract_targets_multiple(self, sample_scope):
        """Teste l'extraction de multiples cibles."""
        guardian = Guardian(sample_scope)
        targets = guardian._extract_targets("nmap -sV scanme.nmap.org example.com")
        assert "scanme.nmap.org" in targets
        assert "example.com" in targets

    def test_extract_targets_subfinder(self, sample_scope):
        """Teste l'extraction depuis subfinder."""
        guardian = Guardian(sample_scope)
        targets = guardian._extract_targets("subfinder -d scanme.nmap.org")
        assert "scanme.nmap.org" in targets

    def test_wildcard_matching_logic(self):
        """Test détaillé du matching wildcard."""
        yaml_content = """
allowed_domains:
  - "*.test.org"
  - "foo.*.example.com"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        config = load_scope_from_yaml(temp_path)
        os.unlink(temp_path)
        guardian = Guardian(config)

        # *.test.org
        assert guardian.is_allowed("api.test.org") is True
        assert guardian.is_allowed("www.test.org") is True
        assert guardian.is_allowed("deep.sub.api.test.org") is True
        assert guardian.is_allowed("test.org") is False  # Pas de wildcard pour le domaine nu
        assert guardian.is_allowed("other.org") is False

    def test_case_insensitive_domain_match(self):
        """Les domaines devraient être comparés cas-insensitifs."""
        yaml_content = """
allowed_domains:
  - Example.COM
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        config = load_scope_from_yaml(temp_path)
        os.unlink(temp_path)
        guardian = Guardian(config)

        assert guardian.is_allowed("example.com") is True
        assert guardian.is_allowed("EXAMPLE.COM") is True
        assert guardian.is_allowed("ExAmPlE.CoM") is True

    def test_scope_with_ip_ranges(self):
        """Teste le scope avec des IPs (future feature)."""
        yaml_content = """
allowed_domains:
  - 192.168.1.0/24
  - 10.0.0.1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        config = load_scope_from_yaml(temp_path)
        os.unlink(temp_path)
        guardian = Guardian(config)

        # Pour l'instant, les IPs ne sont pas supportées — devraient être refusées
        # (on peut décider de les supporter plus tard)
        # On s'assure qu'au moins le domain parsing fonctionne
        assert "192.168.1.0/24" in guardian.allowed_patterns
