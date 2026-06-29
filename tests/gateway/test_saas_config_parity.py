"""
tests/gateway/test_saas_config_parity.py

Parity tests for cli-config.saas.yaml - the config baked into the Hermes
SaaS/cloud image.  These tests act as a regression guard: any accidental
removal of a cloud-required key, or drift of a cloud-critical value, will
be caught here before it reaches a deployed image.

Phase: age737h-245-A
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SAAS_CONFIG_PATH = REPO_ROOT / "cli-config.saas.yaml"


def _load() -> Dict[str, Any]:
    """Load and return the parsed SaaS config."""
    with SAAS_CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _get(cfg: Dict[str, Any], *keys: str) -> Any:
    """Drill into nested dict; raise KeyError with a readable path on miss."""
    node: Any = cfg
    path = ""
    for key in keys:
        path = f"{path}.{key}" if path else key
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing key '{path}' in cli-config.saas.yaml")
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def saas_cfg() -> Dict[str, Any]:
    return _load()


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------

class TestModelSection:
    def test_provider_is_bedrock(self, saas_cfg):
        assert _get(saas_cfg, "model", "provider") == "bedrock", (
            "SaaS config must use provider=bedrock for data sovereignty"
        )

    def test_default_model_present(self, saas_cfg):
        default = _get(saas_cfg, "model", "default")
        assert isinstance(default, str) and default.strip(), (
            "model.default must be a non-empty string"
        )

    def test_tier_present(self, saas_cfg):
        tier = _get(saas_cfg, "model", "tier")
        assert isinstance(tier, str) and tier.strip(), (
            "model.tier must be a non-empty string"
        )


# ---------------------------------------------------------------------------
# Kanban section
# ---------------------------------------------------------------------------

class TestKanbanSection:
    def test_dispatch_in_gateway_disabled(self, saas_cfg):
        assert _get(saas_cfg, "kanban", "dispatch_in_gateway") is False, (
            "kanban.dispatch_in_gateway must be false in cloud (OOM-risk)"
        )


# ---------------------------------------------------------------------------
# Bedrock section
# ---------------------------------------------------------------------------

class TestBedrockSection:
    def test_bedrock_section_present(self, saas_cfg):
        _get(saas_cfg, "bedrock")

    def test_discovery_disabled(self, saas_cfg):
        assert _get(saas_cfg, "bedrock", "discovery", "enabled") is False, (
            "bedrock.discovery.enabled must be false (no ListFoundationModels IAM perm)"
        )


# ---------------------------------------------------------------------------
# Security section
# ---------------------------------------------------------------------------

class TestSecuritySection:
    def test_redact_secrets_enabled(self, saas_cfg):
        assert _get(saas_cfg, "security", "redact_secrets") is True, (
            "security.redact_secrets must be true in cloud (Plan 007-D)"
        )


# ---------------------------------------------------------------------------
# Sessions section
# ---------------------------------------------------------------------------

class TestSessionsSection:
    def test_auto_prune_disabled(self, saas_cfg):
        assert _get(saas_cfg, "sessions", "auto_prune") is False, (
            "sessions.auto_prune must be false in cloud (full history in Neon)"
        )


# ---------------------------------------------------------------------------
# Memory section
# ---------------------------------------------------------------------------

class TestMemorySection:
    def test_provider_is_atlas(self, saas_cfg):
        assert _get(saas_cfg, "memory", "provider") == "atlas", (
            "memory.provider must be 'atlas' in cloud (Plan 011-C.2)"
        )

    def test_retention_days_present(self, saas_cfg):
        days = _get(saas_cfg, "memory", "retention_days")
        assert isinstance(days, int) and days > 0, (
            "memory.retention_days must be a positive integer"
        )


# ---------------------------------------------------------------------------
# Storage section (phase age737h-245-A)
# ---------------------------------------------------------------------------

class TestStorageSection:
    def test_storage_section_present(self, saas_cfg):
        _get(saas_cfg, "storage")

    def test_backend_is_s3(self, saas_cfg):
        assert _get(saas_cfg, "storage", "backend") == "s3", (
            "storage.backend must be 's3' in the SaaS/cloud image"
        )

    def test_s3_subsection_present(self, saas_cfg):
        _get(saas_cfg, "storage", "s3")

    def test_s3_prefix_present(self, saas_cfg):
        prefix = _get(saas_cfg, "storage", "s3", "prefix")
        assert isinstance(prefix, str), (
            "storage.s3.prefix must be a string (may be empty — populated at runtime)"
        )

    def test_s3_sse_present(self, saas_cfg):
        sse = _get(saas_cfg, "storage", "s3", "sse")
        assert isinstance(sse, str) and sse.strip(), (
            "storage.s3.sse must be set (compliance requirement)"
        )

    def test_s3_sse_is_kms(self, saas_cfg):
        assert _get(saas_cfg, "storage", "s3", "sse") == "aws:kms", (
            "storage.s3.sse must be 'aws:kms' for KMS-backed encryption"
        )

    def test_s3_bucket_key_present(self, saas_cfg):
        # Bucket value is intentionally empty (populated via env var at runtime).
        # We only assert the key exists so a typo in the key name is caught.
        s3 = _get(saas_cfg, "storage", "s3")
        assert "bucket" in s3, (
            "storage.s3.bucket key must be present (value populated via env var)"
        )

    def test_s3_region_key_present(self, saas_cfg):
        s3 = _get(saas_cfg, "storage", "s3")
        assert "region" in s3, (
            "storage.s3.region key must be present (value populated via AWS_REGION env var)"
        )


# ---------------------------------------------------------------------------
# No local-only platforms
# ---------------------------------------------------------------------------

class TestNoPlatformDrift:
    """Guard against accidentally enabling local-only platforms in cloud."""

    LOCAL_ONLY_PLATFORMS = {"telegram", "discord", "whatsapp", "signal"}

    def test_no_local_only_platforms_configured(self, saas_cfg):
        platforms = saas_cfg.get("platforms", {})
        enabled = set(platforms.keys()) & self.LOCAL_ONLY_PLATFORMS
        assert not enabled, (
            f"Local-only platforms must not be configured in cloud: {enabled}"
        )
