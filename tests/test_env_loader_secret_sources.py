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
    env_loader._SECRET_OWNERSHIP.clear()
    env_loader._PENDING_SECRET_OWNERSHIP.clear()
    env_loader._APPLIED_HOMES.clear()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_OWNERSHIP.clear()
    env_loader._PENDING_SECRET_OWNERSHIP.clear()
    env_loader._APPLIED_HOMES.clear()


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


def test_format_secret_source_suffix_generic_label_for_future_sources():
    # Future-proofing: a new secret source (e.g. "vault") should still
    # produce a sensible label without needing to edit every call site.
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "vault"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from vault)"
    )


def test_format_secret_source_suffix_onepassword_uses_proper_name():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "onepassword"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from 1Password)"
    )


def test_apply_external_secret_sources_records_bitwarden_origin(tmp_path, monkeypatch):
    """End-to-end: when the Bitwarden source fetches keys, applied vars
    end up in ``_SECRET_SOURCES`` so the UI can label them."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    # Stub the fetch layer under the SecretSource adapter.
    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module,
        "fetch_bitwarden_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "sk-ant-test"}, []),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


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


def test_bitwarden_backend_failure_output_is_safe_and_startup_continues(
    tmp_path, monkeypatch, capsys, caplog
):
    access_token = "BWS-ENV-TOKEN-SENTINEL"
    stdout_sentinel = "BWS-ENV-STDOUT-SENTINEL"
    stderr_sentinel = "BWS-ENV-STDERR-SENTINEL"
    fetched_value = "BWS-ENV-VALUE-SENTINEL"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    cache_ttl_seconds: 0\n"
        "    auto_install: false\n",
        encoding="utf-8",
    )

    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module.subprocess,
        "run",
        lambda *a, **kw: bw_module.subprocess.CompletedProcess(
            args=a[0],
            returncode=1,
            stdout=f"{stdout_sentinel} {fetched_value}",
            stderr=f"{stderr_sentinel} {access_token}",
        ),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)

    captured = capsys.readouterr()
    user_and_log_output = captured.out + captured.err + caplog.text
    assert "Bitwarden secret fetch failed" in user_and_log_output
    assert "hermes secrets bitwarden setup" in user_and_log_output
    for sentinel in (access_token, stdout_sentinel, stderr_sentinel, fetched_value):
        assert sentinel not in user_and_log_output


def test_bitwarden_fetch_exception_is_safe_and_startup_continues(
    tmp_path, monkeypatch, capsys, caplog
):
    access_token = "BWS-EXC-TOKEN-SENTINEL"
    fetched_value = "BWS-EXC-VALUE-SENTINEL"
    exception_sentinel = "BWS-EXCEPTION-SENTINEL"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", access_token)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    auto_install: false\n",
        encoding="utf-8",
    )

    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))

    def _raise_fetch_error(**_kwargs):
        raise RuntimeError(
            f"{exception_sentinel}: {access_token} {fetched_value}"
        )

    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", _raise_fetch_error)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)

    captured = capsys.readouterr()
    user_and_log_output = captured.out + captured.err + caplog.text
    assert "Bitwarden secret fetch failed" in user_and_log_output
    assert "hermes secrets bitwarden setup" in user_and_log_output
    for sentinel in (access_token, fetched_value, exception_sentinel):
        assert sentinel not in user_and_log_output


def test_apply_external_secret_sources_dedupes_within_process(tmp_path, monkeypatch):
    """``load_hermes_dotenv()`` is called at module-import time from several
    hot modules (cli.py, hermes_cli/main.py, run_agent.py, ...).  The
    Bitwarden status line previously printed once per call — 3-5x per
    startup.  The applied-home guard must short-circuit subsequent calls
    so the heavy work (config re-parse, Bitwarden lookup, status print)
    runs exactly once per HERMES_HOME per process.
    """

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    call_count = {"n": 0}

    def _fake_fetch(**_kwargs):
        call_count["n"] += 1
        return {"ANTHROPIC_API_KEY": "sk-ant-test"}, []

    import agent.secret_sources.bitwarden as bw_module
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", _fake_fetch)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

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
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None
    env_loader._apply_external_secret_sources(tmp_path)
    assert call_count["n"] == 2


def test_apply_external_secret_sources_records_onepassword_origin(tmp_path, monkeypatch):
    """When the 1Password source resolves refs, applied vars end up in
    ``_SECRET_SOURCES`` labeled ``onepassword``."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        "      ANTHROPIC_API_KEY: 'op://Private/Anthropic/credential'\n",
        encoding="utf-8",
    )

    import agent.secret_sources.onepassword as op_module

    monkeypatch.setattr(op_module, "find_op", lambda *_a, **_kw: Path("/fake/op"))
    monkeypatch.setattr(
        op_module,
        "fetch_onepassword_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "sk-ant-test"}, []),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "onepassword"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from 1Password)"
    )


def test_apply_external_secret_sources_survives_non_dict_section(tmp_path, monkeypatch):
    """A malformed `secrets:` section must not abort startup (fail-open).

    Both `onepassword: true` (non-dict) and a bad bitwarden section must be
    coerced to empty config instead of raising AttributeError up through
    load_hermes_dotenv().
    """

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden: true\n"
        "  onepassword: true\n",
        encoding="utf-8",
    )

    # Must not raise and must not record anything.
    env_loader._apply_external_secret_sources(tmp_path)
    assert env_loader.get_secret_source("ANYTHING") is None


def test_apply_external_secret_sources_bad_ttl_does_not_crash(tmp_path, monkeypatch):
    """A non-numeric cache_ttl_seconds must be coerced, not crash startup."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    cache_ttl_seconds: not-a-number\n"
        "    env:\n"
        "      K: 'op://V/I/F'\n",
        encoding="utf-8",
    )

    captured = {}

    def _fake_fetch(**kwargs):
        captured.update(kwargs)
        return {}, []

    import agent.secret_sources.onepassword as op_module
    monkeypatch.setattr(op_module, "find_op", lambda *_a, **_kw: Path("/fake/op"))
    monkeypatch.setattr(op_module, "fetch_onepassword_secrets", _fake_fetch)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    # Coerced to the 300s default rather than raising ValueError.
    assert captured["cache_ttl_seconds"] == 300


def _configure_bitwarden_refresh(home: Path) -> None:
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: refresh-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n"
        "    cache_ttl_seconds: 0\n"
        "    override_existing: false\n"
        "    auto_install: false\n",
        encoding="utf-8",
    )


def _install_bitwarden_fetch_sequence(monkeypatch, outcomes):
    import agent.secret_sources.bitwarden as bw_module

    remaining = iter(outcomes)
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))

    def _fetch(**_kwargs):
        outcome = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return dict(outcome), []

    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", _fetch)


def _prime_bitwarden_refresh(tmp_path, monkeypatch, key, outcomes):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(monkeypatch, outcomes)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)


def test_refresh_malformed_config_retains_known_good_value_and_provenance(
    tmp_path, monkeypatch, capsys, caplog
):
    key = "HERMES_TEST_MALFORMED_CONFIG_SECRET"
    known_good = "known-good-malformed-config-value"
    config_sentinel = "MALFORMED-CONFIG-SENTINEL"
    _prime_bitwarden_refresh(tmp_path, monkeypatch, key, [{key: known_good}])

    env_loader.reset_secret_source_cache()
    (tmp_path / "config.yaml").write_text(
        f"secrets:\n  bitwarden: [{config_sentinel}\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == known_good
    assert env_loader.get_secret_source(key) == "bitwarden"
    assert key in env_loader._PENDING_SECRET_OWNERSHIP
    assert str(tmp_path.resolve()) not in env_loader._APPLIED_HOMES
    captured = capsys.readouterr()
    output = captured.out + captured.err + caplog.text
    assert config_sentinel not in output
    assert known_good not in output


def test_refresh_unreadable_config_retains_known_good_value_and_provenance(
    tmp_path, monkeypatch, capsys, caplog
):
    key = "HERMES_TEST_UNREADABLE_CONFIG_SECRET"
    known_good = "known-good-unreadable-config-value"
    error_sentinel = "UNREADABLE-CONFIG-SENTINEL"
    config_path = tmp_path / "config.yaml"
    _prime_bitwarden_refresh(tmp_path, monkeypatch, key, [{key: known_good}])

    env_loader.reset_secret_source_cache()
    real_open = open

    def _deny_config_read(path, *args, **kwargs):
        if path == config_path:
            raise PermissionError(error_sentinel)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _deny_config_read)
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == known_good
    assert env_loader.get_secret_source(key) == "bitwarden"
    assert key in env_loader._PENDING_SECRET_OWNERSHIP
    assert str(tmp_path.resolve()) not in env_loader._APPLIED_HOMES
    captured = capsys.readouterr()
    output = captured.out + captured.err + caplog.text
    assert error_sentinel not in output
    assert known_good not in output


def test_refresh_non_mapping_secrets_retains_known_good_value_and_provenance(
    tmp_path, monkeypatch
):
    key = "HERMES_TEST_NON_MAPPING_CONFIG_SECRET"
    known_good = "known-good-non-mapping-config-value"
    _prime_bitwarden_refresh(tmp_path, monkeypatch, key, [{key: known_good}])

    env_loader.reset_secret_source_cache()
    (tmp_path / "config.yaml").write_text(
        "secrets: [not, a, mapping]\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == known_good
    assert env_loader.get_secret_source(key) == "bitwarden"
    assert key in env_loader._PENDING_SECRET_OWNERSHIP
    assert str(tmp_path.resolve()) not in env_loader._APPLIED_HOMES


@pytest.mark.parametrize("config_text", ["false\n", "0\n", "[]\n", '""\n'])
def test_refresh_falsy_non_mapping_config_retains_known_good_value(
    tmp_path, monkeypatch, config_text
):
    key = "HERMES_TEST_FALSY_NON_MAPPING_CONFIG_SECRET"
    known_good = "known-good-falsy-non-mapping-config-value"
    _prime_bitwarden_refresh(tmp_path, monkeypatch, key, [{key: known_good}])

    env_loader.reset_secret_source_cache()
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == known_good
    assert env_loader.get_secret_source(key) == "bitwarden"
    assert key in env_loader._PENDING_SECRET_OWNERSHIP
    assert str(tmp_path.resolve()) not in env_loader._APPLIED_HOMES


def test_refresh_recovers_after_malformed_config_becomes_valid(tmp_path, monkeypatch):
    key = "HERMES_TEST_CONFIG_RECOVERY_SECRET"
    _prime_bitwarden_refresh(
        tmp_path,
        monkeypatch,
        key,
        [{key: "known-good-value"}, {key: "recovered-value"}],
    )

    env_loader.reset_secret_source_cache()
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  bitwarden: [unterminated\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)

    _configure_bitwarden_refresh(tmp_path)
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "recovered-value"
    assert env_loader.get_secret_source(key) == "bitwarden"
    assert key in env_loader._SECRET_OWNERSHIP
    assert key not in env_loader._PENDING_SECRET_OWNERSHIP


def test_refresh_intentional_disable_removes_source_owned_value(
    tmp_path, monkeypatch
):
    key = "HERMES_TEST_DISABLED_SOURCE_SECRET"
    _prime_bitwarden_refresh(
        tmp_path,
        monkeypatch,
        key,
        [{key: "value-to-remove"}],
    )

    env_loader.reset_secret_source_cache()
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    enabled: false\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)

    assert key not in os.environ
    assert env_loader.get_secret_source(key) is None
    assert key not in env_loader._SECRET_OWNERSHIP
    assert key not in env_loader._PENDING_SECRET_OWNERSHIP


def test_refresh_rotates_source_owned_value(tmp_path, monkeypatch):
    key = "HERMES_TEST_ROTATING_SECRET"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "first-vault-value"}, {key: "second-vault-value"}],
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    assert os.environ[key] == "first-vault-value"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "second-vault-value"
    assert env_loader.get_secret_source(key) == "bitwarden"


def test_refresh_removes_source_owned_value_missing_from_successful_fetch(
    tmp_path, monkeypatch
):
    key = "HERMES_TEST_REMOVED_SECRET"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "removed-vault-value"}, {}],
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    assert os.environ[key] == "removed-vault-value"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert key not in os.environ
    assert env_loader.get_secret_source(key) is None


def test_refresh_tracks_sanitized_source_value_for_later_removal(
    tmp_path, monkeypatch
):
    key = "HERMES_TEST_REMOVED_API_KEY"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "source—value"}, {}],
    )

    env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    assert os.environ[key] == "sourcevalue"

    env_loader.reset_secret_source_cache()
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)

    assert key not in os.environ
    assert env_loader.get_secret_source(key) is None


def test_refresh_failure_retains_previous_known_good_source_value(
    tmp_path, monkeypatch
):
    key = "HERMES_TEST_RETAINED_SECRET"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "known-good-vault-value"}, RuntimeError("backend unavailable")],
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "known-good-vault-value"
    assert env_loader.get_secret_source(key) == "bitwarden"


def test_refresh_does_not_delete_later_shell_replacement(tmp_path, monkeypatch):
    key = "HERMES_TEST_SHELL_REPLACEMENT"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "vault-value"}, {key: "rotated-vault-value"}],
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    os.environ[key] = "later-shell-value"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "later-shell-value"
    assert env_loader.get_secret_source(key) is None


def test_refresh_does_not_delete_later_dotenv_replacement(tmp_path, monkeypatch):
    key = "HERMES_TEST_DOTENV_REPLACEMENT"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.delenv(key, raising=False)
    _configure_bitwarden_refresh(tmp_path)
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [{key: "vault-value"}, {}],
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    (tmp_path / ".env").write_text(f"{key}=later-dotenv-value\n", encoding="utf-8")

    env_loader.reset_secret_source_cache()
    env_loader.load_hermes_dotenv(hermes_home=tmp_path)

    assert os.environ[key] == "later-dotenv-value"
    assert env_loader.get_secret_source(key) is None


def test_refresh_source_collision_still_uses_first_winner(tmp_path, monkeypatch):
    key = "HERMES_TEST_COLLIDING_SECRET"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "test-bootstrap-token")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "test-op-token")
    monkeypatch.delenv(key, raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  sources: [bitwarden, onepassword]\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: refresh-project\n"
        "    override_existing: false\n"
        "    auto_install: false\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    override_existing: false\n"
        "    env:\n"
        f"      {key}: op://Private/Test/credential\n",
        encoding="utf-8",
    )
    _install_bitwarden_fetch_sequence(
        monkeypatch,
        [
            {key: "first-bitwarden-value"},
            {key: "second-bitwarden-value"},
            {key: "third-bitwarden-value"},
        ],
    )

    import agent.secret_sources.onepassword as op_module

    op_values = iter(
        (
            "first-onepassword-value",
            "second-onepassword-value",
            RuntimeError("onepassword unavailable"),
        )
    )
    monkeypatch.setattr(op_module, "find_op", lambda *_a, **_kw: Path("/fake/op"))

    def _fetch_onepassword(**_kw):
        value = next(op_values)
        if isinstance(value, Exception):
            raise value
        return {key: value}, []

    monkeypatch.setattr(
        op_module,
        "fetch_onepassword_secrets",
        _fetch_onepassword,
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(tmp_path)
    assert os.environ[key] == "first-onepassword-value"
    assert env_loader.get_secret_source(key) == "onepassword"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "second-onepassword-value"
    assert env_loader.get_secret_source(key) == "onepassword"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ[key] == "second-onepassword-value"
    assert env_loader.get_secret_source(key) == "onepassword"
