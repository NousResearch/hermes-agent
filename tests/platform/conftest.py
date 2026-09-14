"""Shared fixtures for NOVA platform tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.audit import AuditLog
from nova.runtime.hermes import HermesRuntime
from nova.spec import load_bundle

EXAMPLE_BUNDLE = Path(__file__).resolve().parents[2] / "nova" / "examples" / "acme"

#: Agents the example bundle materializes. The third is a channel-scoped variant: the
#: example's Telegram connection requires approval for ``send_external_email``, which
#: ``operations`` can perform, so that agent gets its own profile with its own compiled
#: policy (see ``nova/channels/derive.py``). Named here rather than repeated as a literal
#: in every count assertion, so the reason a third agent exists is written down once.
EXAMPLE_AGENTS = {"customer-support", "operations", "operations__acme-support-telegram"}


@pytest.fixture
def bundle():
    return load_bundle(EXAMPLE_BUNDLE)


@pytest.fixture
def home(tmp_path) -> Path:
    """An empty runtime home."""
    root = tmp_path / "home"
    root.mkdir()
    return root


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.jsonl", tenant_id="acme", actor="test")


@pytest.fixture
def runtime(home) -> HermesRuntime:
    return HermesRuntime(home=home, tenant_id="acme")
