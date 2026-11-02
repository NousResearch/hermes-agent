"""Regression for #118388: a platform singleton secret (``TELEGRAM_BOT_TOKEN``) duplicated across
local profile homes was invisible to ``hermes doctor`` / ``hermes gateway status``; the only signal
was the losing standalone gateway's log. Doctor, status and the migrate preflight now share one
duplicate-credential helper, so all three name the same profiles + key names (never the value)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import doctor_state, gateway, gateway_migrate as gm

SECRET = "123456:shared-secret-value"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles" / "worker").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: None)
    monkeypatch.setattr(gm, "_installed_services", lambda home: [])
    return root, root / "profiles" / "worker"


def _surfaces() -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        doctor_state._check_profiles(False)
        gateway._print_duplicate_credential_warnings()
    return out.getvalue()


def test_duplicate_singleton_secret_named_identically_by_doctor_status_and_preflight(homes):
    default, worker = homes
    (default / ".env").write_text(f"TELEGRAM_BOT_TOKEN={SECRET}\nOPENAI_API_KEY=sk-shared\n", encoding="utf-8")
    (worker / ".env").write_text(f"TELEGRAM_BOT_TOKEN={SECRET}\nOPENAI_API_KEY=sk-shared\n", encoding="utf-8")

    findings = gm.duplicate_credential_findings()
    assert len(findings) == 1
    line = findings[0]
    assert "'default'" in line and "'worker'" in line and "TELEGRAM_BOT_TOKEN" in line
    assert gm.MIGRATE_COMMAND in line
    # Same helper, same words: the preflight blocker IS the doctor/status finding.
    assert gm.build_migration_plan().blockers == findings
    out = _surfaces()
    assert out.count(line) == 2  # once from doctor, once from gateway status
    # Absence: the value (or a hash a user could mistake for it) never reaches any surface.
    assert SECRET not in out and "shared-secret" not in out and "sk-shared" not in out


def test_distinct_tokens_and_shared_non_singleton_keys_are_not_findings(homes):
    default, worker = homes
    (default / ".env").write_text(f"TELEGRAM_BOT_TOKEN={SECRET}\nOPENAI_API_KEY=sk-shared\n", encoding="utf-8")
    (worker / ".env").write_text("TELEGRAM_BOT_TOKEN=999999:other-token\nOPENAI_API_KEY=sk-shared\n", encoding="utf-8")

    assert gm.duplicate_credential_findings() == []
    assert not gm.build_migration_plan().blocked
    assert "both hold" not in _surfaces()
