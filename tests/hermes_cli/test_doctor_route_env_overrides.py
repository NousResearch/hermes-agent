"""``hermes doctor`` names environment variables that silently override the configured model route.

An inherited ``ANTHROPIC_BASE_URL`` sent every turn of a daily-driver profile to another host while
config.yaml still said Anthropic, and doctor reported nothing. The check reports the variable, where it
came from, what it overrides (scheme + host, or auth mode) and a one-line fix — never its value — and
stays silent for variables the resolver ignores for the configured provider.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from hermes_cli import doctor_env_overrides as deo
from hermes_cli.doctor_connectivity import _model_cfg

_ROUTE_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
               "ANTHROPIC_AUTH_TOKEN", "OPENAI_BASE_URL", "OPENAI_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY",
               "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY_3", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "CUSTOM_BASE_URL",
               "HERMES_INFERENCE_PROVIDER")
_API_KEY = "sk-ant-api03-SECRETKEYVALUE0000"


@pytest.fixture
def home(monkeypatch):
    for var in _ROUTE_VARS:
        monkeypatch.delenv(var, raising=False)
    from hermes_cli.config import get_hermes_home
    return get_hermes_home()


def _configure(home, model: dict, dotenv: str = "") -> None:
    (home / "config.yaml").write_text(json.dumps({"model": model}), encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")


def _run_check(capsys):
    """Run the registered doctor check the way ``run_doctor`` does; (stdout, Finding)."""
    finding = deo._check_route_env_overrides(False)
    return capsys.readouterr().out, finding


def test_inherited_base_url_redirecting_the_anthropic_route_warns(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic", "default": "claude-sonnet-4-5"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test/v1")
    out, finding = _run_check(capsys)
    assert "⚠ ANTHROPIC_BASE_URL from the process environment" in out
    assert "https://api.anthropic.com → https://proxy.example.test" in out
    assert len(finding.manual_issues) == 1
    fix = finding.manual_issues[0]
    assert fix.startswith("ANTHROPIC_BASE_URL (the process environment)") and "unset ANTHROPIC_BASE_URL" in fix
    assert "providers:" in fix
    assert _API_KEY not in out + fix


def test_profile_dotenv_is_named_as_the_source(home, capsys):
    _configure(home, {"provider": "anthropic"},
               f"ANTHROPIC_API_KEY={_API_KEY}\nANTHROPIC_BASE_URL=https://proxy.example.test\n")
    out, finding = _run_check(capsys)
    assert "ANTHROPIC_BASE_URL from " in out and "/.env" in out
    assert "⚠" not in out and finding.manual_issues == [] and finding.issues == []


def test_project_dotenv_is_info_only(home, monkeypatch, capsys):
    from hermes_cli.env_loader import load_hermes_dotenv
    _configure(home, {"provider": "anthropic"})
    project = home / "checkout"
    project.mkdir()
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: project)
    (project / ".env").write_text(
        f"ANTHROPIC_API_KEY={_API_KEY}\nANTHROPIC_BASE_URL=https://proxy.example.test\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home, project_env=project / ".env", load_external_secrets=False)
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "ANTHROPIC_BASE_URL")
    assert "from the project .env" in out
    assert "remove ANTHROPIC_BASE_URL from the project .env" in deo._unset_hint("ANTHROPIC_BASE_URL")


@pytest.mark.parametrize("stored_login", [False, True])
def test_persisted_env_api_key_does_not_hide_anthropic_overrides(home, monkeypatch, capsys, stored_login):
    _configure(home, {"provider": "anthropic"})
    if stored_login:
        _store_hermes_login(home)
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "env-key", "provider": "anthropic", "source": "env:ANTHROPIC_API_KEY",
        "auth_type": "api_key", "base_url": "https://proxy.example.test/v1"}]}}), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test/v1")
    before = (home / "auth.json").read_bytes()
    out, finding = _run_check(capsys)
    assert "⚠ ANTHROPIC_BASE_URL from the process environment" in out
    if stored_login:
        assert "⚠ ANTHROPIC_API_KEY from the process environment" in out
    assert len(finding.manual_issues) == (2 if stored_login else 1)
    assert (home / "auth.json").read_bytes() == before
    assert _API_KEY not in out + "".join(finding.manual_issues)


@pytest.mark.parametrize("source", ["hermes_pkce", "claude_code"])
def test_pruned_oauth_row_does_not_hide_an_api_key_redirect(home, monkeypatch, capsys, source):
    # With ANTHROPIC_API_KEY and no OAuth variable the pool drops autodiscovered OAuth rows, so the
    # env-seeded key and its inherited base URL carry every turn.
    _configure(home, {"provider": "anthropic"})
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "stored-login", "provider": "anthropic", "source": source,
        "auth_type": "oauth", "access_token": "sk-ant-oat01-STOREDLOGIN"}]}}), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test/v1")
    out, finding = _run_check(capsys)
    assert "⚠ ANTHROPIC_BASE_URL from the process environment" in out
    assert "may override if the stored credential is unavailable" not in out
    assert finding.manual_issues and _API_KEY not in out


def test_no_finding_when_nothing_overrides(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    out, finding = _run_check(capsys)
    assert "No environment variable overrides the configured model route" in out
    assert finding.manual_issues == [] and finding.issues == []


def test_base_url_matching_the_configured_route_is_info_only(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/")
    out, finding = _run_check(capsys)
    assert "matches the configured route" in out and "⚠" not in out
    assert finding.manual_issues == []


def test_configured_anthropic_base_url_shadows_the_variable(home, monkeypatch, capsys):
    # A trustworthy model.base_url wins over the env-seeded pool row, so the variable takes no effect.
    _configure(home, {"provider": "anthropic", "base_url": "https://gw.example.test/anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test")
    assert [o.var for o in deo.find_route_env_overrides(_model_cfg())] == ["ANTHROPIC_API_KEY"]
    _out, finding = _run_check(capsys)
    assert finding.manual_issues == []


def test_base_url_without_an_env_credential_is_not_applied(home, monkeypatch):
    # Only rows seeded from an env API key carry the env URL; a stored login keeps the default host.
    _configure(home, {"provider": "anthropic"})
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_other_configured_provider_makes_anthropic_variables_irrelevant(home, monkeypatch, capsys):
    _configure(home, {"provider": "deepseek", "default": "deepseek-chat"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-SECRET")
    for var, value in (("ANTHROPIC_API_KEY", _API_KEY), ("ANTHROPIC_BASE_URL", "https://proxy.example.test"),
                       ("ANTHROPIC_AUTH_TOKEN", "auth-token-SECRET"), ("OPENAI_BASE_URL", "https://o.example.test/v1")):
        monkeypatch.setenv(var, value)
    out, finding = _run_check(capsys)
    assert "No environment variable overrides the configured model route" in out
    assert finding.manual_issues == []


def test_anthropic_auth_token_is_never_reported(home, monkeypatch):
    # The Anthropic client omits the SDK's ANTHROPIC_AUTH_TOKEN fallback, so it never authenticates a turn.
    _configure(home, {"provider": "anthropic"})
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token-SECRET")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_generic_provider_env_url_beats_configured_base_url(home, monkeypatch, capsys):
    _configure(home, {"provider": "deepseek", "base_url": "https://cfg.example.test/v1"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-SECRET")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    out, finding = _run_check(capsys)
    assert "deepseek route https://cfg.example.test → https://env.example.test" in out
    assert len(finding.manual_issues) == 1 and "sk-deepseek-SECRET" not in out


def test_openai_alias_follows_openai_base_url(home, monkeypatch, capsys):
    _configure(home, {"provider": "openai", "default": "gpt-5"})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
    out, finding = _run_check(capsys)
    assert "openai route https://api.openai.com → https://env.example.test" in out
    assert len(finding.manual_issues) == 1


def test_openai_base_url_does_not_steer_openrouter(home, monkeypatch, capsys):
    _configure(home, {"provider": "openrouter"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
    _out, finding = _run_check(capsys)
    assert finding.manual_issues == []


def test_openai_app_server_runtime_ignores_openai_base_url(home, monkeypatch):
    # model.openai_runtime: codex_app_server hands the turn to the app-server, which never uses the endpoint.
    _configure(home, {"provider": "openai", "openai_runtime": "codex_app_server"})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_openrouter_base_url_redirects_openrouter_without_a_configured_mirror(home, monkeypatch, capsys):
    _configure(home, {"provider": "openrouter"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SECRETVALUE")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://mirror.example.test/api/v1")
    out, finding = _run_check(capsys)
    assert "openrouter route https://openrouter.ai → https://mirror.example.test" in out
    assert len(finding.manual_issues) == 1


def test_configured_openrouter_mirror_outranks_openrouter_base_url(home, monkeypatch):
    _configure(home, {"provider": "openrouter", "base_url": "https://cfgmirror.example.test/api/v1"})
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://mirror.example.test/api/v1")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_custom_base_url_outranks_the_configured_custom_endpoint(home, monkeypatch, capsys):
    _configure(home, {"provider": "custom", "base_url": "http://127.0.0.1:11434/v1"})
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://custom.example.test/v1")
    out, finding = _run_check(capsys)
    assert "custom route http://127.0.0.1 → https://custom.example.test" in out
    assert len(finding.manual_issues) == 1


def test_numbered_env_key_still_carries_the_base_url(home, monkeypatch):
    # _seed_from_env also seeds DEEPSEEK_API_KEY_2, and that row carries DEEPSEEK_BASE_URL too.
    _configure(home, {"provider": "deepseek"})
    monkeypatch.setenv("DEEPSEEK_API_KEY_2", "sk-deepseek-SECRETVALUE00")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    (override,) = deo.find_route_env_overrides(_model_cfg())
    assert override.warn and "DEEPSEEK_API_KEY_2" in override.overrides


def test_suppressed_env_credential_does_not_carry_the_base_url(home, monkeypatch):
    # `hermes auth remove` suppresses the env source: no pool row is seeded from it, so the URL is unused.
    _configure(home, {"provider": "deepseek"})
    (home / "auth.json").write_text(json.dumps({"providers": {}, "suppressed_sources": {"deepseek": ["env:DEEPSEEK_API_KEY"]}}),
                                    encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-SECRETVALUE00")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_openrouter_pool_answers_before_custom_base_url(home, monkeypatch):
    # No custom endpoint for the pool gate (OPENAI/OPENROUTER_BASE_URL, config mirror): the env-seeded
    # OpenRouter pool row resolves first, at openrouter.ai, so CUSTOM_BASE_URL is never consulted.
    _configure(home, {"provider": "openrouter"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SECRETVALUE")
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://custom.example.test/v1")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_numbered_key_beyond_a_suppressed_sibling_still_carries_the_base_url(home, monkeypatch):
    _configure(home, {"provider": "deepseek"})
    (home / "auth.json").write_text(json.dumps({"providers": {}, "suppressed_sources": {"deepseek": ["env:DEEPSEEK_API_KEY_2"]}}),
                                    encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY_2", "sk-deepseek-SECRETVALUE02")
    monkeypatch.setenv("DEEPSEEK_API_KEY_3", "sk-deepseek-SECRETVALUE03")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    (override,) = deo.find_route_env_overrides(_model_cfg())
    assert override.warn and "DEEPSEEK_API_KEY_3" in override.overrides


def test_suppressed_env_oauth_token_leaves_the_stored_login_in_charge(home, monkeypatch):
    _configure(home, {"provider": "anthropic"})
    _store_hermes_login(home)
    (home / "auth.json").write_text(json.dumps({"providers": {}, "suppressed_sources": {"anthropic": ["env:ANTHROPIC_TOKEN"]}}),
                                    encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-ENVTOKEN")
    assert deo.find_route_env_overrides(_model_cfg()) == []


def test_secrets_in_urls_and_credentials_never_reach_output(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://user:URLPASSWORD@proxy.example.test:8443/tenant-SECRETPATH?key=QUERYSECRET")
    out, finding = _run_check(capsys)
    text = out + "\n".join(finding.manual_issues)
    assert "https://proxy.example.test" in text
    for secret in (_API_KEY, "URLPASSWORD", "SECRETPATH", "QUERYSECRET", "8443", "user:"):
        assert secret not in text


def _store_hermes_login(home, token: str = "sk-ant-oat01-STOREDLOGIN") -> None:
    expires = int(time.time() * 1000) + 86_400_000
    (home / ".anthropic_oauth.json").write_text(
        json.dumps({"accessToken": token, "refreshToken": "refresh-SECRET", "expiresAt": expires}), encoding="utf-8")


def test_api_key_over_a_stored_oauth_login_warns_about_auth_mode(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    _store_hermes_login(home)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    out, finding = _run_check(capsys)
    assert "⚠ ANTHROPIC_API_KEY from the process environment" in out
    assert "API key (Console, per-token billing) instead of the stored Hermes OAuth login" in out
    assert len(finding.manual_issues) == 1
    assert "hermes model" in finding.manual_issues[0]
    assert "hermes auth remove" not in finding.manual_issues[0]
    for secret in (_API_KEY, "STOREDLOGIN", "refresh-SECRET"):
        assert secret not in out + finding.manual_issues[0]


def test_env_oauth_token_with_a_stored_login_is_uncertain(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    _store_hermes_login(home)
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-ENVTOKEN")
    out, finding = _run_check(capsys)
    assert "static OAuth token from ANTHROPIC_TOKEN ahead of the stored" in out
    assert "⚠" not in out and "ENVTOKEN" not in out
    assert finding.manual_issues == [] and finding.issues == []


@pytest.mark.parametrize("var, source, warns", [
    ("ANTHROPIC_TOKEN", "hermes_pkce", True),
    ("CLAUDE_CODE_OAUTH_TOKEN", "hermes_pkce", True),
    ("ANTHROPIC_API_KEY", "hermes_pkce", True),  # an API-key-only environment prunes PKCE rows
    ("ANTHROPIC_API_KEY", "claude_code", True),
    ("ANTHROPIC_TOKEN", "manual", False),
])
def test_anthropic_stored_pool_source_rank(home, monkeypatch, capsys, var, source, warns):
    _configure(home, {"provider": "anthropic"})
    _store_hermes_login(home)
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "stored-login", "provider": "anthropic", "source": source,
        "auth_type": "oauth", "access_token": "sk-ant-oat01-STOREDLOGIN"}]}}), encoding="utf-8")
    monkeypatch.setenv(var, _API_KEY if var == "ANTHROPIC_API_KEY" else "sk-ant-oat01-ENVTOKEN")
    out, finding = _run_check(capsys)
    assert (f"⚠ {var} from the process environment" in out) is warns
    assert bool(finding.manual_issues) is warns


def test_env_credential_without_a_stored_login_is_info_only(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    out, finding = _run_check(capsys)
    assert "ANTHROPIC_API_KEY from the process environment: Anthropic auth: API key from ANTHROPIC_API_KEY" in out
    assert finding.manual_issues == [] and _API_KEY not in out


def _assert_info(out, finding, var):
    assert f"{var} from " in out and "⚠" not in out
    assert finding.manual_issues == [] and finding.issues == []


def test_saved_api_key_after_pkce_login_is_info_only(home, capsys):
    _configure(home, {"provider": "anthropic"}, f"ANTHROPIC_API_KEY={_API_KEY}\n")
    _store_hermes_login(home)
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "ANTHROPIC_API_KEY")
    assert "/.env" in out and _API_KEY not in out


def test_saved_openai_proxy_is_info_only(home, capsys):
    _configure(home, {"provider": "openai"}, "OPENAI_BASE_URL=https://proxy.example.test/v1\n")
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "OPENAI_BASE_URL")
    assert "/.env" in out


def _configure_root_pool(home, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: home)
    root = home / ".hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openrouter": [{
        "id": "root-key", "provider": "openrouter", "source": "manual",
        "auth_type": "api_key", "access_token": "sk-or-ROOTFIXTURE", "priority": 0,
        "base_url": "https://openrouter.ai/api/v1"}]}}), encoding="utf-8")
    _configure(profile, {"provider": "openrouter", "default": "some-model"})
    return profile


def test_root_pool_makes_custom_url_uncertain(home, monkeypatch, capsys):
    profile = _configure_root_pool(home, monkeypatch)
    monkeypatch.setenv("CUSTOM_BASE_URL", "https://env.example.test/v1")
    before = (profile.parent.parent / "auth.json").read_bytes()
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "CUSTOM_BASE_URL")
    assert "may override if the stored credential is unavailable" in out
    assert not (profile / "auth.json").exists()
    assert (profile.parent.parent / "auth.json").read_bytes() == before
    from hermes_cli.runtime_provider import resolve_runtime_provider
    assert resolve_runtime_provider()["base_url"] == "https://openrouter.ai/api/v1"


def test_suppressed_oauth_with_api_key_and_login_is_info_only(home, monkeypatch, capsys):
    _configure(home, {"provider": "anthropic", "default": "some-model"})
    _store_hermes_login(home)
    (home / "auth.json").write_text(json.dumps({"providers": {}, "suppressed_sources": {
        "anthropic": ["env:ANTHROPIC_TOKEN"]}}), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-ENVTOKEN")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _API_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.test/v1")
    before = {p.name: p.read_bytes() for p in (home / "auth.json", home / ".anthropic_oauth.json")}
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "ANTHROPIC_API_KEY")
    assert {name: (home / name).read_bytes() for name in before} == before
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider()
    assert runtime["base_url"].rstrip("/") == "https://api.anthropic.com"
    assert runtime["api_key"] == "sk-ant-oat01-STOREDLOGIN"


def test_persisted_numbered_key_across_a_gap_warns(home, monkeypatch, capsys):
    _configure(home, {"provider": "deepseek"})
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"deepseek": [{
        "id": "third-key", "provider": "deepseek", "source": "env:DEEPSEEK_API_KEY_3"}]}}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY_3", "sk-deepseek-THIRDFIXTURE")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    out, finding = _run_check(capsys)
    assert "⚠ DEEPSEEK_BASE_URL from the process environment" in out
    assert len(finding.manual_issues) == 1
    assert "DEEPSEEK_API_KEY_3" in out


def test_deepseek_route_without_v1_matches_the_default(home, monkeypatch, capsys):
    _configure(home, {"provider": "deepseek"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-FIXTURE")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "DEEPSEEK_BASE_URL")
    assert "matches the configured route" in out


def test_profile_pool_makes_generic_provider_redirect_uncertain(home, monkeypatch, capsys):
    _configure(home, {"provider": "deepseek"})
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"deepseek": [{
        "id": "stored-key", "provider": "deepseek", "source": "manual", "access_token": "sk-STORED"}, {
        "id": "env-key", "provider": "deepseek", "source": "env:DEEPSEEK_API_KEY"}]}}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-FIXTURE")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.example.test/v1")
    out, finding = _run_check(capsys)
    _assert_info(out, finding, "DEEPSEEK_BASE_URL")
    assert "may override if the stored credential is unavailable" in out


@pytest.mark.parametrize("model, env, warns", [
    ({"provider": "anthropic"}, {"ANTHROPIC_API_KEY": _API_KEY, "ANTHROPIC_BASE_URL": "https://proxy.example.test"}, True),
    ({"provider": "anthropic", "base_url": "https://gw.example.test/anthropic"},
     {"ANTHROPIC_API_KEY": _API_KEY, "ANTHROPIC_BASE_URL": "https://proxy.example.test"}, False),
    ({"provider": "deepseek", "base_url": "https://cfg.example.test/v1"},
     {"DEEPSEEK_API_KEY": "sk-deepseek-SECRETVALUE00", "DEEPSEEK_BASE_URL": "https://env.example.test/v1"}, True),
    ({"provider": "deepseek"}, {"DEEPSEEK_API_KEY": "sk-deepseek-SECRETVALUE00", "ANTHROPIC_BASE_URL": "https://proxy.example.test"}, False),
    ({"provider": "openrouter"},
     {"OPENROUTER_API_KEY": "sk-or-v1-SECRETVALUE", "OPENROUTER_BASE_URL": "https://env.example.test/api/v1"}, True),
    ({"provider": "openrouter", "base_url": "https://cfg.example.test/api/v1"},
     {"OPENROUTER_API_KEY": "sk-or-v1-SECRETVALUE", "OPENROUTER_BASE_URL": "https://env.example.test/api/v1"}, False),
    ({"provider": "custom", "base_url": "http://127.0.0.1:11434/v1"}, {"CUSTOM_BASE_URL": "https://env.example.test/v1"}, True),
    ({"provider": "openrouter"}, {"OPENROUTER_API_KEY": "sk-or-v1-SECRETVALUE", "CUSTOM_BASE_URL": "https://env.example.test/v1"}, False),
])
def test_doctor_agrees_with_the_runtime_resolver(home, monkeypatch, model, env, warns):
    """Inherited route warnings agree after the real resolver persists its env rows."""
    _configure(home, {**model, "default": "some-model"})
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider()
    route_warns = [o for o in deo.find_route_env_overrides(_model_cfg()) if o.warn and "route" in o.overrides]
    redirected = "env.example.test" in runtime["base_url"] or "proxy.example.test" in runtime["base_url"]
    assert redirected is warns
    assert bool(route_warns) is warns
    assert os.environ.get("HERMES_HOME")  # sandboxed home, never the real one
