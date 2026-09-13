"""
tests/test_skill_ledger_signing.py

Rigorous tests for optional cryptographic execution signing for Hermes skill ledger mutations.
Verifies persistent key management, multi-mutation chain verification, and tamper detection.
"""

import os
import tempfile
from pathlib import Path
import pytest
from tools.skill_ledger import append_entry, verify_skill_ledger_integrity


def test_unsigned_ledger_reports_none_chain_valid(monkeypatch):
    """When signing is not active, verification reports chain_valid as None (not False)."""
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HERMES_AUDIT_SIGNING", raising=False)

        # Append standard unsigned entry
        entry_id = append_entry(action="create", skill="normal-skill", actor="user")
        assert entry_id is not None

        # Verify reports unsigned_ledger_active without falsely failing
        summary = verify_skill_ledger_integrity()
        assert summary.get("chain_valid") is None
        assert summary.get("status") == "unsigned_ledger_active"


def test_signed_ledger_persistent_key_and_tamper_detection(monkeypatch):
    """When signing is active, verifies persistent key reuse across mutations and catches tampering."""
    import importlib.util
    if importlib.util.find_spec("guardclaw") is None:
        pytest.skip("guardclaw not installed")

    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_AUDIT_SIGNING", "1")

        # 1. Record first mutation
        e1 = append_entry(
            action="create",
            skill="skill-alpha",
            actor="curator",
            evidence={"reason": "Autonomous evolution #1"},
        )
        assert e1 is not None

        # Check key persistence
        vault_dir = home / "skills" / ".guardclaw_curator_vault"
        key_file = vault_dir / "curator_signing_key.json"
        assert key_file.exists()
        key_data_1 = key_file.read_text(encoding="utf-8")

        # 2. Record second mutation and verify key is persisted and reused
        e2 = append_entry(
            action="update",
            skill="skill-alpha",
            actor="curator",
            evidence={"reason": "Autonomous evolution #2"},
        )
        assert e2 is not None
        key_data_2 = key_file.read_text(encoding="utf-8")
        assert key_data_1 == key_data_2, "Signing key must be persistent across mutations"

        # 3. Positive verification proof
        summary = verify_skill_ledger_integrity()
        assert summary["chain_valid"] is True
        assert summary["verified_count"] == 3  # Genesis + Mutation 1 + Mutation 2

        # 4. Negative tamper verification proof
        ledger_file = vault_dir / "ledger.jsonl"
        assert ledger_file.exists()
        original_content = ledger_file.read_text(encoding="utf-8")
        tampered_content = original_content.replace("Autonomous evolution #1", "Malicious forged reason")
        ledger_file.write_text(tampered_content, encoding="utf-8")

        tamper_summary = verify_skill_ledger_integrity()
        assert tamper_summary["chain_valid"] is False, "Tampered ledger must fail cryptographic verification"
