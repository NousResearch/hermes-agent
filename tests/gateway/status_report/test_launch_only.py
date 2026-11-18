"""Only the launch environment can switch status reports on: a profile ``.env``, a secret source
and the ``.env`` writers cannot set ``GATEWAY_STATUS_REPORT_URL``."""

from __future__ import annotations

import os

import pytest

from agent.secret_sources import registry
from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource
from gateway.status_report.reporter import StatusReporter, status_reporter_from_launch_env
from gateway.status_report.target import STATUS_REPORT_URL_ENV
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import is_launch_only_env_key
from tests.gateway.status_report.support import new_credential, stamped_url


@pytest.mark.parametrize("name", [STATUS_REPORT_URL_ENV, STATUS_REPORT_URL_ENV.lower()])
def test_the_key_is_launch_only(name):
    assert is_launch_only_env_key(name)


def test_a_profile_env_file_cannot_switch_reports_on(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        f"{STATUS_REPORT_URL_ENV}={stamped_url(new_credential())}\nSTATUS_REPORT_CONTROL=loaded\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(STATUS_REPORT_URL_ENV, raising=False)
    monkeypatch.delenv("STATUS_REPORT_CONTROL", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.environ["STATUS_REPORT_CONTROL"] == "loaded"  # control: the file was read
    assert STATUS_REPORT_URL_ENV not in os.environ
    assert status_reporter_from_launch_env() is None


def test_a_secret_source_cannot_switch_reports_on(tmp_path, monkeypatch):
    registry._reset_registry_for_tests()
    monkeypatch.setattr(registry, "_ensure_builtin_sources", lambda: None)

    class Vault(SecretSource):
        name, label, shape, scheme, api_version = "vault", "Vault", "mapped", None, SECRET_SOURCE_API_VERSION

        def fetch(self, cfg, home_path):
            result = FetchResult()
            result.secrets = {STATUS_REPORT_URL_ENV: stamped_url(new_credential()), "VAULT_CONTROL": "1"}
            return result

        def override_existing(self, cfg):
            return False

        def protected_env_vars(self, cfg):
            return frozenset()

    try:
        registry.register_source(Vault())
        environ: dict = {}
        report = registry.apply_all({"vault": {"enabled": True}}, tmp_path, environ=environ)
    finally:
        registry._reset_registry_for_tests()

    assert environ == {"VAULT_CONTROL": "1"}  # control: the source's other secret still applies
    assert STATUS_REPORT_URL_ENV in report.sources[0].skipped_protected
    assert status_reporter_from_launch_env(environ) is None


def test_the_env_writers_refuse_the_key(tmp_path, monkeypatch):
    from hermes_cli.config import save_env_value

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(ValueError):
        save_env_value(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))

    assert not (home / ".env").exists() or STATUS_REPORT_URL_ENV not in (home / ".env").read_text(encoding="utf-8")


def test_the_launch_environment_switches_reports_on(monkeypatch):
    monkeypatch.setenv(STATUS_REPORT_URL_ENV, stamped_url(new_credential()))

    assert isinstance(status_reporter_from_launch_env(), StatusReporter)
