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


def test_duplicate_signal_account_is_a_finding(homes):
    """Signal has no token: the registered number is the credential. A secondary profile whose
    ``.env`` carries a copy of the default profile's ``SIGNAL_*`` block must surface exactly like a
    duplicated ``TELEGRAM_BOT_TOKEN`` — before the multiplexer starts two adapters on one account."""
    default, worker = homes
    block = "SIGNAL_HTTP_URL=http://127.0.0.1:8080\nSIGNAL_ACCOUNT=+15550001111\n"
    (default / ".env").write_text(block, encoding="utf-8")
    (worker / ".env").write_text(block, encoding="utf-8")

    findings = gm.duplicate_credential_findings()
    assert len(findings) == 1
    line = findings[0]
    assert "'default'" in line and "'worker'" in line and "SIGNAL_ACCOUNT" in line
    assert "signal token" not in line  # there is no Signal token; the remedy must name the key
    assert gm.build_migration_plan().blockers == findings
    out = _surfaces()
    assert out.count(line) == 2
    assert "5550001111" not in out  # the number never reaches any surface


def test_distinct_signal_accounts_are_not_findings(homes):
    default, worker = homes
    (default / ".env").write_text(
        "SIGNAL_HTTP_URL=http://127.0.0.1:8080\nSIGNAL_ACCOUNT=+15550001111\n", encoding="utf-8")
    (worker / ".env").write_text(
        "SIGNAL_HTTP_URL=http://127.0.0.1:8080\nSIGNAL_ACCOUNT=+15550002222\n", encoding="utf-8")

    assert gm.duplicate_credential_findings() == []
    assert not gm.build_migration_plan().blocked


def test_signal_probe_matches_runtime_when_a_stray_token_is_present():
    """Probe/runtime parity for mixed config: a generic ``token`` left in a Signal block is never
    consumed by the adapter, so neither side may key the claim on it."""
    from gateway.config import PlatformConfig
    from gateway.run import GatewayRunner

    fp = GatewayRunner._adapter_credential_fingerprint

    class _SignalAdapter:
        def __init__(self, config):
            self.config = config
            self.http_url = "http://127.0.0.1:8080"
            self.account = config.extra["account"]

    same_account_a = PlatformConfig(enabled=True, token="stray-1", extra={"account": "+15550001111"})
    same_account_b = PlatformConfig(enabled=True, token="stray-2", extra={"account": "+15550001111"})
    other_account = PlatformConfig(enabled=True, token="stray-1", extra={"account": "+15550002222"})

    # runtime: same account collides, different account does not — regardless of the stray token
    assert fp(_SignalAdapter(same_account_a)) == fp(_SignalAdapter(same_account_b))
    assert fp(_SignalAdapter(same_account_a)) != fp(_SignalAdapter(other_account))
    # preflight probe: identical verdicts, and identical fingerprints to the runtime adapter
    assert fp(gm._credential_probe(same_account_a)) == fp(_SignalAdapter(same_account_a))
    assert fp(gm._credential_probe(same_account_a)) == fp(gm._credential_probe(same_account_b))
    assert fp(gm._credential_probe(same_account_a)) != fp(gm._credential_probe(other_account))
    # token-identified platforms are untouched: no account → the token still decides
    assert fp(gm._credential_probe(PlatformConfig(enabled=True, token="bot-token"))) == \
        fp(gm._credential_probe(PlatformConfig(enabled=True, token="bot-token", extra={})))
