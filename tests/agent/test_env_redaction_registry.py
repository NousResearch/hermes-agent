"""Profile ``.env`` values in the exact-value redaction registry.

A bare opaque credential (no vendor prefix, no ``KEY=`` shape) quoted in a project document passes
the regex redactor and can reach stored tool results through three paths: a subdirectory context
hint spliced onto a terminal result, ``search_files`` with context lines, and ``read_file``. The
profile already holds the exact bytes in its ``.env``; registering them makes the exact-substring
scrub — until now fed only by browser vault fills — catch every echo, on every tool result.

Every test runs under the conftest-isolated HERMES_HOME (the registry is keyed by profile home) and
resets the registry around itself so module-global state never leaks.
"""

from __future__ import annotations

import time

import pytest

import agent.redact as redact_mod
from agent.redact import (
    _is_env_secret_candidate,
    clear_env_redaction_values,
    clear_vault_redaction_values,
    redact_registered_vault_values,
    redact_sensitive_text,
    register_env_redaction_values,
    register_vault_redaction_value,
)
from agent.tool_dispatch_helpers import _scrub_registered_secrets

APP_TOKEN = "q7Vm2kPz9RtL4xWc8NbY"  # 20 chars, mixed case + digits, no prefix: the shape no regex can see
SERVICE_PASS = "Tr0ub4dor&3-example-pass"
PROFILE_ENV = {
    "APP_TOKEN": APP_TOKEN,
    "APP_URL": "http://192.0.2.10:8080",
    "SERVICE_PASS": SERVICE_PASS,
    "SERVICE_HOST": "192.0.2.53",
    "SERVICE_USER": "alice",
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "alice",
    "TERMINAL_MODAL_IMAGE": "python:3.11-slim-bookworm-hermes",
    "OBSIDIAN_VAULT_PATH": "/home/alice/notes",
    "SSH_AUTH_SOCK": "$HOME/.ssh/agent.sock",
    "BACKUP_AWS_ACCESS_KEY_ID": "READER01KEYIDENTIFIER",
    "BACKUP_AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "SOME_TOKEN": "changeme",
    "OTHER_TOKEN": "<fill me in>",
    "GATE_PIN_PASS": "123456789",
    "SHORT_PW": "abc1234",
}


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_env_redaction_values()
    clear_vault_redaction_values()
    yield
    clear_env_redaction_values()
    clear_vault_redaction_values()


# ── Baseline: documents the gap ─────────────────────────────────────────────


def test_bare_opaque_token_in_backticks_passes_pattern_redaction():
    text = f"Service auth token for the API: `{APP_TOKEN}`"
    assert APP_TOKEN in redact_sensitive_text(text, force=True, file_read=True)


# ── Selection policy ────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", [
    "APP_TOKEN", "SERVICE_PASS", "BACKUP_AWS_SECRET_ACCESS_KEY",
])
def test_credential_keyed_values_are_candidates(key):
    assert _is_env_secret_candidate(key, PROFILE_ENV[key])


@pytest.mark.parametrize("key, why", [
    ("APP_URL", "locator, not a credential"),
    ("SERVICE_HOST", "no credential keyword in the key"),
    ("SERVICE_USER", "identifier suffix"),
    ("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "AUTH keyword but an identifier suffix"),
    ("TERMINAL_MODAL_IMAGE", "a shape-only test would register this image tag"),
    ("OBSIDIAN_VAULT_PATH", "path"),
    ("SSH_AUTH_SOCK", "variable reference under an AUTH key"),
    ("BACKUP_AWS_ACCESS_KEY_ID", "key id, public half"),
    ("SOME_TOKEN", "placeholder"),
    ("OTHER_TOKEN", "angle-bracket placeholder"),
    ("GATE_PIN_PASS", "short numeric value"),
    ("SHORT_PW", "below the length floor"),
])
def test_non_secret_values_are_not_candidates(key, why):
    assert not _is_env_secret_candidate(key, PROFILE_ENV[key]), why


def test_known_vendor_prefix_registers_under_any_key_name():
    assert _is_env_secret_candidate("MY_THING", "ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    # The prefix rule precedes the identifier-suffix exclusion: an AKIA… key id is a vendor credential shape.
    assert _is_env_secret_candidate("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")


def test_registration_selects_exactly_the_secret_values():
    assert register_env_redaction_values(PROFILE_ENV) == 3
    bucket = redact_mod._ENV_REDACTION_VALUES[redact_mod._vault_scope()]
    assert set(bucket.values()) == {"APP_TOKEN", "SERVICE_PASS", "BACKUP_AWS_SECRET_ACCESS_KEY"}


# ── Scrubbing ───────────────────────────────────────────────────────────────


def test_registered_env_value_is_masked_with_its_key_name():
    register_env_redaction_values(PROFILE_ENV)
    out = redact_sensitive_text(f"Service auth token for the API: `{APP_TOKEN}`", force=True, file_read=True)
    assert APP_TOKEN not in out
    assert "«redacted-env-secret:APP_TOKEN»" in out


def test_env_scrub_is_a_hard_boundary_even_with_redaction_disabled(monkeypatch):
    monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
    register_env_redaction_values(PROFILE_ENV)
    out = redact_sensitive_text(f"MAX_TOKENS=100 {SERVICE_PASS}", code_file=True)
    assert SERVICE_PASS not in out and "MAX_TOKENS=100" in out


def test_non_secret_env_values_survive_in_output():
    register_env_redaction_values(PROFILE_ENV)
    text = "image python:3.11-slim-bookworm-hermes at http://192.0.2.10:8080 by alice"
    assert redact_registered_vault_values(text) == text


def test_vault_and_env_buckets_both_apply_longest_first():
    register_vault_redaction_value("hunter2-vault-password")
    register_env_redaction_values({"API_TOKEN": "hunter2-vault-password-plus-suffix"})
    out = redact_registered_vault_values("a hunter2-vault-password-plus-suffix b hunter2-vault-password c")
    assert out == "a «redacted-env-secret:API_TOKEN» b «redacted-vault-secret» c"


def test_registry_is_profile_scoped(monkeypatch, tmp_path):
    register_env_redaction_values(PROFILE_ENV)
    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert redact_registered_vault_values(APP_TOKEN) == APP_TOKEN


def test_reregistration_replaces_rather_than_accumulates():
    register_env_redaction_values(PROFILE_ENV)
    register_env_redaction_values({"APP_TOKEN": "newRotatedValue0987654321"})
    assert redact_registered_vault_values(APP_TOKEN) == APP_TOKEN  # rotated-out value no longer held
    assert "«redacted-env-secret:APP_TOKEN»" in redact_registered_vault_values("x newRotatedValue0987654321 y")


def test_empty_selection_clears_the_bucket():
    register_env_redaction_values(PROFILE_ENV)
    assert register_env_redaction_values({"APP_URL": "http://x"}) == 0
    assert redact_registered_vault_values(APP_TOKEN) == APP_TOKEN


def test_bucket_is_capped_per_profile():
    many = {f"TOKEN_{i:04d}": f"value-{i:04d}-abcdefghijklmnop" for i in range(400)}
    assert register_env_redaction_values(many) == redact_mod._ENV_REDACTION_MAX_PER_PROFILE


# ── Resolution from the profile home (no explicit mapping) ──────────────────


def test_none_resolves_the_isolated_profile_env_file(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    (home / ".env").write_text(
        f'APP_TOKEN="{APP_TOKEN}"\nAPP_URL=http://app.example:8080\n# comment\nexport SERVICE_PASS={SERVICE_PASS}\n',
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert register_env_redaction_values() == 2
    assert "«redacted-env-secret:APP_TOKEN»" in redact_registered_vault_values(f"got `{APP_TOKEN}`")


def test_installed_secret_scope_wins_over_the_env_file(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    home = tmp_path / "profile-home"
    home.mkdir()
    (home / ".env").write_text(f"APP_TOKEN={APP_TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_secret_scope({"OTHER_API_KEY": "scopeOnlyValue1234567890"}, profile_home=str(home))
    try:
        assert register_env_redaction_values() == 1
    finally:
        reset_secret_scope(token)
    assert redact_registered_vault_values(APP_TOKEN) == APP_TOKEN
    assert "«redacted-env-secret:OTHER_API_KEY»" in redact_registered_vault_values("x scopeOnlyValue1234567890")


def test_missing_env_file_registers_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nowhere"))
    assert register_env_redaction_values() == 0


# ── Executor choke point ────────────────────────────────────────────────────


def test_scrub_covers_string_results():
    register_env_redaction_values(PROFILE_ENV)
    out = _scrub_registered_secrets(f"[Subdirectory context discovered: x/CLAUDE.md]\ntoken `{APP_TOKEN}`")
    assert APP_TOKEN not in out and "APP_TOKEN»" in out


def test_scrub_preserves_marker_subclass_when_nothing_matches():
    class _Marker(str):
        pass
    register_env_redaction_values(PROFILE_ENV)
    result = _scrub_registered_secrets(_Marker("timed out"))
    assert isinstance(result, _Marker)


def test_scrub_covers_multimodal_text_parts_only():
    register_env_redaction_values(PROFILE_ENV)
    envelope = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": f"stdout {APP_TOKEN}"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
        "text_summary": f"summary {SERVICE_PASS}",
    }
    out = _scrub_registered_secrets(envelope)
    assert out is envelope
    assert APP_TOKEN not in envelope["content"][0]["text"]
    assert envelope["content"][1]["image_url"]["url"].endswith("AAAA")
    assert SERVICE_PASS not in envelope["text_summary"]


def test_scrub_leaves_non_text_results_alone():
    register_env_redaction_values(PROFILE_ENV)
    assert _scrub_registered_secrets(None) is None
    payload = {"a": 1}
    assert _scrub_registered_secrets(payload) is payload


# ── Cost ────────────────────────────────────────────────────────────────────


def test_scrub_cost_on_100kb_result_with_full_bucket():
    """Budget for the per-result pass: 128 registered values over a 100 KB result (no hits, the
    common case) must stay in the low milliseconds — it runs on every tool result."""
    many = {f"TOKEN_{i:04d}": f"value-{i:04d}-abcdefghijklmnop" for i in range(redact_mod._ENV_REDACTION_MAX_PER_PROFILE)}
    register_env_redaction_values(many)
    text = ("line of ordinary tool output with paths /usr/local/bin and ids 1234567890\n" * 1400)[:100_000]
    assert len(text) == 100_000
    best = min(_timed(lambda: redact_registered_vault_values(text)) for _ in range(5))
    assert best < 0.05, f"exact-value scrub took {best * 1000:.1f} ms on 100 KB"


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start
