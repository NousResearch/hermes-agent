"""Tests for secret-source tracking in ``hermes_cli.env_loader``.

These cover the small public surface that lets `hermes model` / `hermes setup`
label detected credentials with their origin ("from Bitwarden") so users
don't see an unexplained "credentials ok" line when their .env is empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sources():
    """Each test starts with a clean source map, registry, and home guard."""
    from agent.secret_sources import registry as reg_module

    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    reg_module._reset_registry_for_tests()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    reg_module._reset_registry_for_tests()


def test_get_secret_source_returns_none_for_untracked_var():
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_get_secret_source_returns_label_for_tracked_var():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"


def test_format_secret_source_suffix_empty_for_untracked():
    assert env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY") == ""


def test_format_secret_source_suffix_bitwarden_uses_proper_name():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_format_secret_source_suffix_onepassword_uses_registry_label():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "onepassword"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from 1Password)"
    )


def test_format_secret_source_suffix_protonpass_uses_registry_label():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "protonpass"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Proton Pass)"
    )


def test_format_secret_source_suffix_generic_label_for_future_sources():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "vault"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from vault)"
    )


def test_apply_external_secret_sources_records_bitwarden_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module,
        "fetch_bitwarden_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "sk-ant-test"}, []),
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_apply_external_secret_sources_records_onepassword_origin(tmp_path, monkeypatch):
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

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "onepassword"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from 1Password)"
    )


def test_apply_external_secret_sources_records_protonpass_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  protonpass:\n"
        "    enabled: true\n"
        "    env:\n"
        "      ANTHROPIC_API_KEY: 'pass://SHARE/ITEM/api_key'\n",
        encoding="utf-8",
    )

    import agent.secret_sources.protonpass as pp_module

    captured = {}

    def _fake_fetch(**kwargs):
        captured.update(kwargs)
        return {"ANTHROPIC_API_KEY": "sk-ant-test"}, []

    monkeypatch.setattr(pp_module, "find_pass_cli", lambda **_kw: Path("/fake/pass-cli"))
    monkeypatch.setattr(pp_module, "fetch_protonpass_secrets", _fake_fetch)

    env_loader._apply_external_secret_sources(tmp_path)

    assert captured["bootstrap_env"] == "PROTON_PASS_PERSONAL_ACCESS_TOKEN"
    assert captured["env_refs"] == {
        "ANTHROPIC_API_KEY": "pass://SHARE/ITEM/api_key"
    }
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "protonpass"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Proton Pass)"
    )


def test_apply_external_secret_sources_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: false\n"
        "  protonpass:\n"
        "    enabled: false\n"
        "    env:\n"
        "      ANTHROPIC_API_KEY: 'pass://SHARE/ITEM/api_key'\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_apply_external_secret_sources_dedupes_within_process(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
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

    for _ in range(5):
        env_loader._apply_external_secret_sources(tmp_path)

    assert call_count["n"] == 1
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"

    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert call_count["n"] == 2


def test_apply_external_secret_sources_survives_non_dict_section(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden: true\n"
        "  onepassword: true\n"
        "  protonpass: true\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)
    assert env_loader.get_secret_source("ANYTHING") is None


def test_apply_external_secret_sources_onepassword_bad_ttl_does_not_crash(
    tmp_path, monkeypatch
):
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

    env_loader._apply_external_secret_sources(tmp_path)

    assert captured["cache_ttl_seconds"] == 300


def test_apply_external_secret_sources_protonpass_bad_ttl_does_not_crash(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_test-token")
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  protonpass:\n"
        "    enabled: true\n"
        "    cache_ttl_seconds: not-a-number\n"
        "    env:\n"
        "      K: 'pass://SHARE/ITEM/api_key'\n",
        encoding="utf-8",
    )

    captured = {}

    def _fake_fetch(**kwargs):
        captured.update(kwargs)
        return {}, []

    import agent.secret_sources.protonpass as pp_module

    monkeypatch.setattr(pp_module, "find_pass_cli", lambda **_kw: Path("/fake/pass-cli"))
    monkeypatch.setattr(pp_module, "fetch_protonpass_secrets", _fake_fetch)

    env_loader._apply_external_secret_sources(tmp_path)

    assert captured["cache_ttl_seconds"] == 300.0


def test_protonpass_setup_never_accepts_token_via_argv():
    import argparse

    from hermes_cli import protonpass_secrets_cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pp_parser = sub.add_parser("protonpass")
    protonpass_secrets_cli.register_protonpass_cli(pp_parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["protonpass", "setup", "--service-token", "pst_secret"])

    def _walk_options(p):
        opts = []
        for action in p._actions:
            opts.extend(action.option_strings)
            if isinstance(action, argparse._SubParsersAction):
                for choice in action.choices.values():
                    opts.extend(_walk_options(choice))
        return opts

    all_opts = _walk_options(parser)
    assert "--service-token" not in all_opts
    assert not any("token" in o and o != "--token-env" for o in all_opts)
