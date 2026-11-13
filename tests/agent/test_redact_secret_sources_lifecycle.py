"""Secret-source mask registration is per profile and follows revocation (#77165 review).

Split from test_redact_secret_sources.py: shape classification must not read another profile's
registered values, and a value a source no longer supplies stops being masked.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from agent import redact_secret_sources
from agent.redact import redact_sensitive_text, register_vault_redaction_value
from agent.secret_sources import registry as reg_module
from agent.secret_sources.base import FetchResult
from agent.secret_sources.registry import AppliedVar, ApplyReport, SourceReport
from hermes_cli import env_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

OPAQUE = "mxq7Tpl2vRw9Kd"
PATH = "/srv/workspace/project"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(redact_secret_sources, "_VALUES_BY_HOME", {})
    monkeypatch.setattr("agent.redact._VAULT_REDACTION_VALUES", {})
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _home(tmp_path, name, config="secrets:\n  cmd:\n    enabled: true\n"):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize("registry", ["secret_source", "vault"])
def test_another_profiles_values_do_not_classify_this_profiles_config(tmp_path, registry):
    home_a, home_b = _home(tmp_path, "a"), _home(tmp_path, "b")
    with _profile(home_a):
        if registry == "vault":
            register_vault_redaction_value("workspace")
        else:
            redact_secret_sources.publish_secret_source_values(home_a, {"API_SECRET": "workspace"})
        assert "workspace" not in redact_sensitive_text(PATH)  # A's registry is live while B publishes
        redact_secret_sources.publish_secret_source_values(home_b, {"PROJECT_DIR": PATH})
    with _profile(home_b):
        assert redact_secret_sources.redact_secret_source_values(PATH) == PATH


def _source(monkeypatch, pulls):
    """Each apply pass supplies the next dict from ``pulls``."""
    pulls = iter(pulls)

    def _apply_all(_cfg, _home, environ=None):
        secrets = next(pulls)
        environ.update(secrets)
        return ApplyReport(
            sources=[SourceReport(name="cmd", label="cmd", result=FetchResult(secrets=dict(secrets)),
                                  applied=list(secrets))],
            provenance={n: AppliedVar(name=n, source="cmd", shape="bulk", overrode_env=False) for n in secrets},
        )

    monkeypatch.setattr(reg_module, "apply_all", _apply_all)
    monkeypatch.setattr(reg_module, "enabled_source_names", lambda _cfg, _home: frozenset({"cmd"}))


def test_value_the_source_stops_supplying_is_unmasked(tmp_path, monkeypatch):
    home = _home(tmp_path, "a")
    monkeypatch.delenv("VENDOR_TOKEN", raising=False)
    _source(monkeypatch, [{"VENDOR_TOKEN": OPAQUE}, {}])
    env_loader._apply_external_secret_sources(home)
    with _profile(home):
        assert OPAQUE not in redact_sensitive_text(f"x {OPAQUE}")

    env_loader.reset_secret_source_cache(home)
    env_loader._apply_external_secret_sources(home)

    with _profile(home):
        assert redact_secret_sources.redact_secret_source_values(f"x {OPAQUE}") == f"x {OPAQUE}"


@pytest.mark.parametrize("config", ["", "secrets:\n  cmd:\n    enabled: false\n"])
def test_removing_or_disabling_every_source_unmasks_its_values(tmp_path, monkeypatch, config):
    home = _home(tmp_path, "a")
    monkeypatch.delenv("VENDOR_TOKEN", raising=False)
    _source(monkeypatch, [{"VENDOR_TOKEN": OPAQUE}])
    env_loader._apply_external_secret_sources(home)

    (home / "config.yaml").write_text(config, encoding="utf-8")
    env_loader.reset_secret_source_cache(home)
    env_loader._apply_external_secret_sources(home)

    with _profile(home):
        assert redact_secret_sources.redact_secret_source_values(f"x {OPAQUE}") == f"x {OPAQUE}"

