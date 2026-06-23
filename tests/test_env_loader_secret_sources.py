"""Tests for the secret-source tracking in ``hermes_cli.env_loader``.

These cover the small public surface that lets `hermes model` / `hermes setup`
label detected credentials with their origin ("from Bitwarden") so users
don't see an unexplained "credentials ✓" line when their .env is empty.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sources():
    """Each test starts with a clean source map and applied-home guard."""
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()


def test_get_secret_source_returns_none_for_untracked_var():
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_get_secret_source_returns_label_for_tracked_var():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"


def test_format_secret_source_suffix_empty_for_untracked():
    # Credentials from .env or the shell shouldn't add noise — the
    # implicit case stays unlabeled.
    assert env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY") == ""


def test_format_secret_source_suffix_bitwarden_uses_proper_name():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_format_secret_source_suffix_onepassword_uses_proper_name():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "1password"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from 1Password)"
    )


def test_format_secret_source_suffix_keychain_uses_proper_name():
    env_loader._SECRET_SOURCES["TELEGRAM_BOT_TOKEN"] = "keychain"
    assert (
        env_loader.format_secret_source_suffix("TELEGRAM_BOT_TOKEN")
        == " (from macOS Keychain)"
    )


def test_format_secret_source_suffix_generic_label_for_future_sources():
    # Future-proofing: a new secret source (e.g. "vault") should still
    # produce a sensible label without needing to edit every call site.
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "vault"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from vault)"
    )


def test_reset_secret_source_cache_clears_origin_labels():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "1password"

    env_loader.reset_secret_source_cache()

    assert env_loader.format_secret_source_suffix("OPENAI_API_KEY") == ""


def test_apply_external_secret_sources_records_bitwarden_origin(tmp_path, monkeypatch):
    """End-to-end: when ``apply_bitwarden_secrets`` returns applied keys,
    they end up in ``_SECRET_SOURCES`` so the UI can label them."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    # Stub apply_bitwarden_secrets to return a synthetic FetchResult.
    from agent.secret_sources.bitwarden import FetchResult

    fake_result = FetchResult(
        secrets={"ANTHROPIC_API_KEY": "sk-ant-test"},
        applied=["ANTHROPIC_API_KEY"],
    )

    def _fake_apply(**_kwargs):
        return fake_result

    # The import inside _apply_external_secret_sources is lazy, so we
    # patch the *module attribute* it will pull in.
    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "apply_bitwarden_secrets", _fake_apply)

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_apply_external_secret_sources_records_onepassword_and_keychain_origins(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  '1password':\n"
        "    enabled: true\n"
        "    timeout_seconds: 3\n"
        "    override_existing: true\n"
        "    resolve_env_references: false\n"
        "    env:\n"
        "      OP_API_KEY: 'op://Vault/Item/key'\n"
        "  keychain:\n"
        "    enabled: true\n"
        "    timeout_seconds: 4\n"
        "    env:\n"
        "      KEYCHAIN_TOKEN:\n"
        "        service: halvo-shared\n"
        "        account: HERMES_MBP_TELEGRAM_BOT_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OP_API_KEY", raising=False)
    monkeypatch.delenv("KEYCHAIN_TOKEN", raising=False)

    from agent.secret_sources.keychain import FetchResult as KeychainFetchResult
    from agent.secret_sources.onepassword import FetchResult as OnePasswordFetchResult

    calls = []

    def fake_op_apply(**kwargs):
        calls.append(("op", kwargs))
        os.environ["OP_API_KEY"] = "from-op"
        return OnePasswordFetchResult(
            secrets={"OP_API_KEY": "from-op"},
            applied=["OP_API_KEY"],
        )

    def fake_keychain_apply(**kwargs):
        calls.append(("keychain", kwargs))
        os.environ["KEYCHAIN_TOKEN"] = "from-keychain"
        return KeychainFetchResult(
            secrets={"KEYCHAIN_TOKEN": "from-keychain"},
            applied=["KEYCHAIN_TOKEN"],
        )

    monkeypatch.setattr(
        "agent.secret_sources.onepassword.apply_onepassword_secrets",
        fake_op_apply,
    )
    monkeypatch.setattr(
        "agent.secret_sources.keychain.apply_keychain_secrets",
        fake_keychain_apply,
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ["OP_API_KEY"] == "from-op"
    assert os.environ["KEYCHAIN_TOKEN"] == "from-keychain"
    assert env_loader.get_secret_source("OP_API_KEY") == "1password"
    assert env_loader.get_secret_source("KEYCHAIN_TOKEN") == "keychain"
    assert env_loader.format_secret_source_suffix("OP_API_KEY") == " (from 1Password)"
    assert env_loader.format_secret_source_suffix("KEYCHAIN_TOKEN") == " (from macOS Keychain)"
    assert [kind for kind, _kwargs in calls] == ["op", "keychain"]
    assert calls[0][1]["timeout_seconds"] == 3
    assert calls[0][1]["override_existing"] is True
    assert calls[0][1]["resolve_env_references"] is False
    assert calls[1][1]["timeout_seconds"] == 4

    stderr = capsys.readouterr().err
    assert "1Password CLI: applied 1 secret (OP_API_KEY)" in stderr
    assert "macOS Keychain: applied 1 secret (KEYCHAIN_TOKEN)" in stderr
    assert "from-op" not in stderr
    assert "from-keychain" not in stderr


def test_apply_external_secret_sources_noop_when_disabled(tmp_path, monkeypatch):
    """Disabled Bitwarden config must not touch the source map."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_apply_external_secret_sources_dedupes_within_process(tmp_path, monkeypatch):
    """``load_hermes_dotenv()`` is called at module-import time from several
    hot modules (cli.py, hermes_cli/main.py, run_agent.py, ...).  The
    Bitwarden status line previously printed once per call — 3-5x per
    startup.  The applied-home guard must short-circuit subsequent calls
    so the heavy work (config re-parse, Bitwarden lookup, status print)
    runs exactly once per HERMES_HOME per process.
    """

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    from agent.secret_sources.bitwarden import FetchResult

    call_count = {"n": 0}

    def _fake_apply(**_kwargs):
        call_count["n"] += 1
        return FetchResult(
            secrets={"ANTHROPIC_API_KEY": "sk-ant-test"},
            applied=["ANTHROPIC_API_KEY"],
        )

    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "apply_bitwarden_secrets", _fake_apply)

    # Five calls in a row, simulating module-import-time invocations from
    # cli.py, hermes_cli/main.py, run_agent.py, trajectory_compressor.py,
    # gateway/run.py.  Only the first should actually call the backend.
    for _ in range(5):
        env_loader._apply_external_secret_sources(tmp_path)

    assert call_count["n"] == 1, (
        "Bitwarden backend was called {} time(s); expected exactly 1 — "
        "the applied-home guard is broken.".format(call_count["n"])
    )

    # Source tracking still works after dedup.
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"

    # reset_secret_source_cache() forces a fresh pull on the next call.
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert call_count["n"] == 2
