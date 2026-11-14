"""Vault helper CLIs (``bws`` / ``op``) get an allowlisted env, never the post-dotenv
provider credentials. Each test drives the real spawn site and inspects the env the
child would receive."""

from pathlib import Path
from unittest import mock

import pytest

from agent.secret_sources import base, bitwarden as bw
from hermes_cli import onepassword_secrets_cli as op_cli
from hermes_cli import secrets_cli as bw_cli

PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GH_TOKEN")


def _capture(monkeypatch, module, stdout="[]"):
    seen: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen.append(dict(kwargs["env"]))
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return seen


@pytest.fixture
def provider_env(monkeypatch):
    for key in PROVIDER_KEYS:
        monkeypatch.setenv(key, f"sentinel-{key}")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")


def _assert_no_provider_keys(env):
    assert not set(PROVIDER_KEYS) & set(env)


@pytest.mark.usefixtures("provider_env")
@pytest.mark.parametrize("multiplex", [False, True], ids=["single-profile", "multiplex"])
def test_bws_fetch_child_gets_only_allowlisted_env(monkeypatch, tmp_path, multiplex):
    import os

    # bws global options that pick a self-hosted server must survive the allowlist.
    monkeypatch.setenv("BWS_CONFIG_FILE", "/etc/bws/config")
    monkeypatch.setenv("BWS_PROFILE", "selfhosted")
    seen = _capture(monkeypatch, base)
    fake = tmp_path / "bws"
    fake.write_text("", encoding="utf-8")
    token = base.set_source_environment(dict(os.environ)) if multiplex else None
    try:
        bw.fetch_bitwarden_secrets(access_token="0.t", project_id="12345678-1234-1234-1234-123456789abc", binary=fake, use_cache=False)
    finally:
        if token is not None:
            base.reset_source_environment(token)
    env = seen[-1]
    _assert_no_provider_keys(env)
    assert env["BWS_ACCESS_TOKEN"] == "0.t"
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["PATH"] == os.environ["PATH"]
    assert env["BWS_CONFIG_FILE"] == "/etc/bws/config"
    assert env["BWS_PROFILE"] == "selfhosted"


@pytest.mark.usefixtures("provider_env")
def test_bws_fetch_child_env_comes_from_the_per_fetch_view(monkeypatch, tmp_path):
    # Under multiplex the per-fetch view is the profile's env; the process env
    # belongs to the launch profile and must not leak into the child.
    monkeypatch.setenv("BWS_CONFIG_FILE", "/launch/bws/config")
    monkeypatch.setenv("BWS_PROFILE", "launch")
    view = {
        "PATH": "/profile/bin",
        "HTTPS_PROXY": "http://profile-proxy.example:8080",
        "BWS_CONFIG_FILE": "/profile/bws/config",
        "BWS_PROFILE": "profile-b",
        "OPENAI_API_KEY": "view-provider-key",
    }
    seen = _capture(monkeypatch, base)
    fake = tmp_path / "bws"
    fake.write_text("", encoding="utf-8")
    token = base.set_source_environment(view)
    try:
        bw.fetch_bitwarden_secrets(access_token="0.t", project_id="12345678-1234-1234-1234-123456789abc", binary=fake, use_cache=False)
    finally:
        base.reset_source_environment(token)
    env = seen[-1]
    _assert_no_provider_keys(env)
    assert env["PATH"] == "/profile/bin"
    assert env["HTTPS_PROXY"] == "http://profile-proxy.example:8080"
    assert env["BWS_CONFIG_FILE"] == "/profile/bws/config"
    assert env["BWS_PROFILE"] == "profile-b"


@pytest.mark.usefixtures("provider_env")
def test_bws_project_list_probe_gets_only_allowlisted_env(monkeypatch):
    monkeypatch.setenv("BWS_SERVER_URL", "https://vault.bitwarden.eu")
    monkeypatch.setenv("BWS_CONFIG_FILE", "/etc/bws/config")
    monkeypatch.setenv("BWS_PROFILE", "selfhosted")
    seen = _capture(monkeypatch, bw_cli)
    assert bw_cli._list_projects(Path("/fake/bws"), "0.probe", bw_cli.Console()) == []
    _assert_no_provider_keys(seen[-1])
    assert seen[-1]["BWS_ACCESS_TOKEN"] == "0.probe"
    assert seen[-1]["BWS_SERVER_URL"] == "https://vault.bitwarden.eu"
    assert seen[-1]["BWS_CONFIG_FILE"] == "/etc/bws/config"
    assert seen[-1]["BWS_PROFILE"] == "selfhosted"


@pytest.mark.usefixtures("provider_env")
def test_op_whoami_probe_gets_only_allowlisted_env(monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops.ambient")
    monkeypatch.setenv("OP_SESSION_demo", "session")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/corp-ca.pem")
    seen = _capture(monkeypatch, op_cli, stdout="me@example")
    assert op_cli._op_whoami(Path("/fake/op"), "") == "me@example"
    assert op_cli._op_whoami(Path("/fake/op"), "", token_value="ops.candidate") == "me@example"
    ambient, candidate = seen
    for env in seen:
        _assert_no_provider_keys(env)
        assert env["OP_SESSION_demo"] == "session"
        # Proxy / CA-bundle settings reach op, as they did before the allowlist.
        assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
        assert env["SSL_CERT_FILE"] == "/etc/corp-ca.pem"
    assert ambient["OP_SERVICE_ACCOUNT_TOKEN"] == "ops.ambient"
    assert candidate["OP_SERVICE_ACCOUNT_TOKEN"] == "ops.candidate"


@pytest.mark.usefixtures("provider_env")
def test_version_probes_get_no_credentials(monkeypatch):
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.ambient")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops.ambient")
    from hermes_cli import _secrets_common

    seen = _capture(monkeypatch, _secrets_common, stdout="tool 1.0")
    bw_cli._bws_version(Path("/fake/bws"))
    op_cli._op_version(Path("/fake/op"))
    for env in seen:
        _assert_no_provider_keys(env)
        assert "BWS_ACCESS_TOKEN" not in env and "OP_SERVICE_ACCOUNT_TOKEN" not in env
