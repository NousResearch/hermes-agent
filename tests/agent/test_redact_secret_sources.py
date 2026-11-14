"""Values applied from external secret sources are masked by exact value on model egress, per profile.

A vault value with no credential shape (an opaque 1Password field, a DSN under ``DATABASE_URL``)
is invisible to the pattern passes, so before this every tool result that echoed it reached the
provider verbatim (#77165, #77162).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest

from agent import redact_secret_sources
from agent.redact import redact_sensitive_text, redact_terminal_output
from agent.secret_sources import registry as reg_module
from agent.secret_sources.base import FetchResult
from agent.secret_sources.registry import AppliedVar, ApplyReport, SourceReport
from agent.tool_dispatch_helpers import make_tool_result_message
from hermes_cli import env_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

OPAQUE = "mxq7Tpl2vRw9Kd"  # no vendor prefix: the pattern passes keep it bare
DSN = "postgres://reporting:Vb8wqLr2@db.internal:5432/prod"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(redact_secret_sources, "_VALUES_BY_HOME", {})
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


def _fake_source(monkeypatch, secrets: dict[str, str]) -> None:
    def _apply_all(_cfg, _home, environ=None):
        assert environ is not None
        environ.update(secrets)
        return ApplyReport(
            sources=[SourceReport(name="cmd", label="cmd", result=FetchResult(secrets=dict(secrets)),
                                  applied=list(secrets))],
            provenance={n: AppliedVar(name=n, source="cmd", shape="bulk", overrode_env=False) for n in secrets},
        )

    monkeypatch.setattr(reg_module, "apply_all", _apply_all)
    monkeypatch.setattr(reg_module, "enabled_source_names", lambda _cfg, _home: frozenset({"cmd"}))


def _home(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text("secrets:\n  cmd:\n    enabled: true\n", encoding="utf-8")
    return home


def test_process_apply_masks_values_on_every_egress_path_in_its_profile_only(tmp_path, monkeypatch):
    home_a, home_b = _home(tmp_path, "a"), _home(tmp_path, "b")
    for name in ("VENDOR_TOKEN", "DATABASE_URL", "DEBUG"):
        monkeypatch.delenv(name, raising=False)
    _fake_source(monkeypatch, {"VENDOR_TOKEN": OPAQUE, "DATABASE_URL": DSN, "DEBUG": "true"})
    env_loader._apply_external_secret_sources(home_a)

    dump = f"VENDOR_TOKEN={OPAQUE}\nDATABASE_URL={DSN}\nDEBUG=true\nvalue alone: {OPAQUE}"
    with _profile(home_a):
        # The user's redact_secrets preference does not apply: like vault fills, this is exact bytes.
        assert OPAQUE not in redact_sensitive_text(dump)
        assert DSN not in redact_terminal_output(dump, command="printenv")
        msg = make_tool_result_message("mcp_db_query", [{"type": "text", "text": dump}], "call_1")
        wire = msg["content"][0]["text"]
        assert OPAQUE not in wire and DSN not in wire
        assert OPAQUE not in make_tool_result_message("custom_plugin_tool", dump, "call_2")["content"]
        # A bulk source's mundane values never become global substring scrubs.
        assert "DEBUG=true" in redact_sensitive_text(dump)
    with _profile(home_b):
        assert OPAQUE in make_tool_result_message("custom_plugin_tool", OPAQUE, "call_3")["content"]


def test_routed_profile_hydration_masks_under_that_profile(tmp_path, monkeypatch):
    home_a, home_b = _home(tmp_path, "a"), _home(tmp_path, "b")
    _fake_source(monkeypatch, {"VENDOR_TOKEN": OPAQUE})
    assert env_loader.hydrate_profile_secret_sources(home_b) == {"VENDOR_TOKEN": OPAQUE}

    with _profile(home_b):
        assert OPAQUE not in make_tool_result_message("web_extract", f"page body {OPAQUE}", "c")["content"]
    with _profile(home_a):
        assert OPAQUE in redact_sensitive_text(f"page body {OPAQUE}")


def test_republish_replaces_the_profile_set(tmp_path):
    home = _home(tmp_path, "a")
    redact_secret_sources.publish_secret_source_values(home, {"VENDOR_TOKEN": OPAQUE})
    redact_secret_sources.publish_secret_source_values(home, {"OTHER_SECRET": "Zr4kWq8mNp2x"})
    with _profile(home):
        assert redact_sensitive_text(f"{OPAQUE} Zr4kWq8mNp2x") == f"{OPAQUE} «redacted:secret-source»"


def test_ordinary_config_from_a_bulk_source_is_not_a_global_scrub(tmp_path):
    home = _home(tmp_path, "a")
    ordinary = {"AWS_REGION": "eu-central-1", "DB_HOST": "db01.internal.example",
                "PROJECT_DIR": "/srv/app2/work", "MODEL": "claude-opus-4-5"}
    redact_secret_sources.publish_secret_source_values(home, {**ordinary, "VENDOR_TOKEN": OPAQUE})
    text = " ".join(ordinary.values())
    with _profile(home):
        assert redact_secret_sources.redact_secret_source_values(text) == text
        assert OPAQUE not in redact_sensitive_text(f"x {OPAQUE}")


def test_secret_named_settings_holding_a_word_or_host_are_not_a_global_scrub(tmp_path):
    # Settings ABOUT auth, named like secrets (teknium1 review on #155274).
    home = _home(tmp_path, "a")
    settings = {"AUTH_MODE": "password", "AUTH_PROVIDER": "localhost", "SECRET_BACKEND": "bitwarden",
                "API_KEY_HEADER": "Authorization", "AUTH_HOST": "auth-server-01.example.com"}
    secrets = {"VENDOR_TOKEN": OPAQUE, "DB_PASSWORD": "correcthorse"}
    redact_secret_sources.publish_secret_source_values(home, {**settings, **secrets})
    text = "use password auth on localhost via bitwarden; Authorization at auth-server-01.example.com"
    with _profile(home):
        assert redact_secret_sources.redact_secret_source_values(text) == text
        for value in secrets.values():
            assert value not in redact_secret_sources.redact_secret_source_values(f"x {value}")


def test_json_serialised_value_is_masked(tmp_path):
    home = _home(tmp_path, "a")
    value = 'Qw3"rT9\\yU7iO1p'
    redact_secret_sources.publish_secret_source_values(home, {"API_SECRET": value})
    with _profile(home):
        wire = redact_secret_sources.redact_secret_source_values(json.dumps({"out": f"pw {value}"}))
    assert "yU7iO1p" not in wire and "«redacted:secret-source»" in wire


def test_failed_repull_keeps_the_still_live_value_masked(tmp_path, monkeypatch):
    home = _home(tmp_path, "a")
    for name in ("VENDOR_TOKEN", "OTHER_SECRET"):
        monkeypatch.delenv(name, raising=False)
    second = "Zr4kWq8mNp2x"
    pulls = iter([None, "vault unreachable"])

    def _apply_all(_cfg, _home, environ=None):
        error = next(pulls)
        environ["OTHER_SECRET"] = second
        if not error:
            environ["VENDOR_TOKEN"] = OPAQUE
        ok = SourceReport(name="cmd", label="cmd", result=FetchResult(secrets={"OTHER_SECRET": second}),
                          applied=["OTHER_SECRET"])
        flaky = SourceReport(name="vault", label="vault",
                             result=FetchResult(error=error) if error else FetchResult(secrets={"VENDOR_TOKEN": OPAQUE}),
                             applied=[] if error else ["VENDOR_TOKEN"])
        provenance = {"OTHER_SECRET": AppliedVar(name="OTHER_SECRET", source="cmd", shape="bulk", overrode_env=False)}
        if not error:
            provenance["VENDOR_TOKEN"] = AppliedVar(name="VENDOR_TOKEN", source="vault", shape="bulk",
                                                    overrode_env=False)
        return ApplyReport(sources=[ok, flaky], provenance=provenance)

    monkeypatch.setattr(reg_module, "apply_all", _apply_all)
    monkeypatch.setattr(reg_module, "enabled_source_names", lambda _cfg, _home: frozenset({"cmd", "vault"}))
    env_loader._apply_external_secret_sources(home)
    env_loader.reset_secret_source_cache(home)  # what a cron per-fire re-pull does
    env_loader._apply_external_secret_sources(home)

    assert os.environ["VENDOR_TOKEN"] == OPAQUE  # a transient outage is not a revocation
    with _profile(home):
        assert OPAQUE not in redact_sensitive_text(f"printenv: {OPAQUE} {second}")
        assert second not in redact_sensitive_text(f"printenv: {OPAQUE} {second}")
