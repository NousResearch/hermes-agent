"""Tests for ``agent.secret_sources.protonpass.apply`` + the CLI / env_loader
glue that consumes it.

Covers the single application planner (``plan_application``), the env_loader
entry point ``apply_protonpass_secrets`` (which NEVER raises — B4), the CLI
``sync``/``setup`` honoring ``auto_install=false`` (B3), the ``_pp_version``
empty-output guard, and the env_loader registry-level bool coercion (B9).

v3 round:
* V4 — the CLI ``_pp_version`` routes through ``install._read_pass_cli_version``
  (the scrubbed-env probe) instead of a bespoke full-env subprocess call.
* V5 — ``cmd_pp_setup`` normalizes a scalar ``protonpass: true`` config to a
  dict before mutating, so a hand-edited scalar config doesn't crash the wizard.
"""

from __future__ import annotations

import argparse
import json
import os
from unittest import mock

from tests._protonpass_helpers import (  # noqa: F401
    _fail,
    _ok,
    _patch_run,
    _reset_caches,
    hermes_home,
    pp,
    pp_apply,
    pp_install,
)


# ---------------------------------------------------------------------------
# plan_application — the single application planner / skip reasons
# ---------------------------------------------------------------------------


def test_plan_skips_bootstrap_token():
    plan = pp_apply.plan_application(
        {"TOKEN_ENV": "x", "OTHER": "y"},
        {},
        override_existing=True,
        token_env="TOKEN_ENV",
    )
    by_name = {i.name: i for i in plan}
    assert by_name["TOKEN_ENV"].applied is False
    assert by_name["TOKEN_ENV"].reason == pp_apply.SKIP_BOOTSTRAP_TOKEN
    assert by_name["OTHER"].applied is True


def test_plan_skips_already_set_without_override():
    plan = pp_apply.plan_application(
        {"A": "new"},
        {"A": "existing"},
        override_existing=False,
        token_env="TOKEN_ENV",
    )
    assert plan[0].applied is False
    assert plan[0].reason == pp_apply.SKIP_ALREADY_SET


def test_plan_overrides_when_requested():
    plan = pp_apply.plan_application(
        {"A": "new"},
        {"A": "existing"},
        override_existing=True,
        token_env="TOKEN_ENV",
    )
    assert plan[0].applied is True
    assert plan[0].overrides is True


def test_plan_applies_new_value():
    plan = pp_apply.plan_application(
        {"A": "new"},
        {},
        override_existing=False,
        token_env="TOKEN_ENV",
    )
    assert plan[0].applied is True
    assert plan[0].overrides is False
    assert plan[0].reason is None


# ---------------------------------------------------------------------------
# apply_protonpass_secrets — public entry point used by env_loader
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = pp.apply_protonpass_secrets(enabled=False)
    assert result.ok
    assert not result.applied
    assert not result.error


def test_apply_missing_token(monkeypatch):
    monkeypatch.delenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", raising=False)
    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False
    )
    assert not result.ok
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" in result.error


def test_apply_no_mode_errors(monkeypatch):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    result = pp.apply_protonpass_secrets(enabled=True, auto_install=False)
    assert not result.ok
    assert "neither a vault" in result.error


def test_apply_no_binary_errors(monkeypatch):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: None)
    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False
    )
    assert not result.ok
    assert "pass-cli binary not available" in result.error


def test_apply_does_not_override_existing(hermes_home, monkeypatch, tmp_path):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    monkeypatch.setenv("PROBE_LOGIN_PASSWORD", "existing-value")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Probe Login",
                    "content": {"Login": {"password": "pp-value"}},
                    "extra_fields": [{"name": "New", "content": {"Text": "new-value"}}],
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", override_existing=False,
        auto_install=False, home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert result.ok
    assert "PROBE_LOGIN_NEW" in result.applied
    assert "PROBE_LOGIN_PASSWORD" in result.skipped
    assert os.environ["PROBE_LOGIN_PASSWORD"] == "existing-value"
    assert os.environ["PROBE_LOGIN_NEW"] == "new-value"


def test_apply_override_existing(hermes_home, monkeypatch, tmp_path):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    monkeypatch.setenv("PROBE_LOGIN_PASSWORD", "stale")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    payload = json.dumps({
        "items": [
            {"content": {"title": "Probe Login",
                         "content": {"Login": {"password": "fresh"}}}}
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", override_existing=True,
        auto_install=False, home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert result.ok
    assert os.environ["PROBE_LOGIN_PASSWORD"] == "fresh"


def test_apply_skips_bootstrap_token_env_name(hermes_home, monkeypatch, tmp_path):
    """Even with override_existing, the service-token var is preserved."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "original")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout="malicious-replacement\n", stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    result = pp.apply_protonpass_secrets(
        enabled=True,
        env_refs={"PROTON_PASS_PERSONAL_ACCESS_TOKEN": "pass://S/I/token"},
        override_existing=True, auto_install=False,
        home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert os.environ["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] == "original"
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" in result.skipped


def test_apply_mode_b_bootstrap_ref_skipped_single_warning(
    hermes_home, monkeypatch, tmp_path
):
    """C1: ``apply`` strips the MODE B bootstrap ref via the centralized helper,
    records it in ``result.skipped`` + ONE warning, and the sibling ref still
    applies.  fetch's own strip is a no-op on the already-filtered refs, so there
    is no duplicate warning."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout="resolved\n", stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    result = pp.apply_protonpass_secrets(
        enabled=True,
        env_refs={
            "PROTON_PASS_PERSONAL_ACCESS_TOKEN": "pass://S/I/token",
            "OPENAI_API_KEY": "pass://S/I/api_key",
        },
        override_existing=True, auto_install=False,
        home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert result.ok
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" in result.skipped
    assert "OPENAI_API_KEY" in result.applied
    boot_warnings = [
        w for w in result.warnings if "PROTON_PASS_PERSONAL_ACCESS_TOKEN" in w
    ]
    assert len(boot_warnings) == 1


def test_apply_mode_a_bootstrap_named_value_not_applied(
    hermes_home, monkeypatch, tmp_path
):
    """C1 (MODE A through apply): a vault item whose DERIVED name equals the
    bootstrap service-token env var must never be applied — fetch drops it before
    it reaches the planner, so the live token is preserved."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "original")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    # Title "Proton Pass Personal Access" / field "Token" does NOT derive the
    # exact var, so use a custom service_token_env that the title/field DO derive.
    payload = json.dumps({
        "items": [
            {
                "content": {
                    "title": "Proton",
                    "content": {"Login": {"token": "malicious", "other": "keep"}},
                }
            }
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    cfg = pp.ProtonPassConfig.from_mapping({
        "enabled": True,
        "vault": "V",
        "service_token_env": "PROTON_TOKEN",
        "auto_install": False,
    })
    monkeypatch.setenv("PROTON_TOKEN", "live-token")

    result = pp.apply_protonpass_secrets(
        enabled=True, config=cfg, override_existing=True,
        home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert result.ok
    # The live token var is untouched, the sibling field is applied.
    assert os.environ["PROTON_TOKEN"] == "live-token"
    assert "PROTON_TOKEN" not in result.applied
    assert "PROTON_OTHER" in result.applied
    assert os.environ["PROTON_OTHER"] == "keep"


def test_apply_never_raises_on_fetch_failure(hermes_home, monkeypatch, tmp_path):
    """A session/auth failure produces an error, NOT an exception."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    # login always fails → _establish_session raises → apply swallows it.
    _patch_run(monkeypatch, lambda cmd, env: _fail())
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False,
        home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert not result.ok
    assert result.error
    assert not result.applied


def test_apply_never_raises_on_non_runtimeerror(hermes_home, monkeypatch, tmp_path):
    """B4: ``apply_*`` must catch ANY Exception (not just RuntimeError) from
    fetch and degrade to an error result — never crash startup."""
    token = "svc-tok"
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", token)
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    def fetch_explodes(**kwargs):
        # A non-RuntimeError (e.g. an unexpected bug in fetch/session/install).
        raise ValueError(f"unexpected bug leaking {token}")

    monkeypatch.setattr(pp_apply, "fetch_protonpass_secrets", fetch_explodes)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False,
        home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert not result.ok
    assert result.error
    # The token must be redacted even in the degraded non-RuntimeError path.
    assert token not in result.error


def test_apply_never_raises_on_keyerror(hermes_home, monkeypatch, tmp_path):
    """B4: even a KeyError (clearly a bug, not a RuntimeError) is swallowed."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    def fetch_keyerror(**kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(pp_apply, "fetch_protonpass_secrets", fetch_keyerror)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False, home_path=hermes_home,
    )
    assert not result.ok
    assert result.error


def test_apply_error_redacts_token(hermes_home, monkeypatch, tmp_path):
    """If a token were ever interpolated into an error, it's scrubbed."""
    token = "leak-me-if-you-can"
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", token)
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")

    def fetch_raises(**kwargs):
        raise RuntimeError(f"auth failed with {token}")

    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(pp_apply, "fetch_protonpass_secrets", fetch_raises)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False,
        home_path=hermes_home,
    )
    assert not result.ok
    assert token not in result.error


def test_apply_threads_config_object(hermes_home, monkeypatch, tmp_path):
    """B6: a ProtonPassConfig can be threaded in via ``config=`` and supplies
    vault / env_refs / auto_install (env_loader path)."""
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    payload = json.dumps({
        "items": [
            {"content": {"title": "K", "content": {"Login": {"password": "v"}}}}
        ]
    })

    def fake_run(cmd, env):
        verb = cmd[1]
        if verb in ("login", "info"):
            return _ok()
        return mock.Mock(returncode=0, stdout=payload, stderr="")

    _patch_run(monkeypatch, fake_run)
    monkeypatch.setattr(pp_apply, "find_pass_cli", lambda **kw: binary)

    cfg = pp.ProtonPassConfig.from_mapping(
        {"enabled": True, "vault": "V", "auto_install": False}
    )
    result = pp.apply_protonpass_secrets(
        enabled=True, config=cfg, home_path=hermes_home, cache_ttl_seconds=0,
    )
    assert result.ok
    assert "K_PASSWORD" in result.applied


# ---------------------------------------------------------------------------
# auto_install=false must never trigger a download (apply side)
# ---------------------------------------------------------------------------


def test_apply_auto_install_false_no_download(hermes_home, monkeypatch):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")
    monkeypatch.setattr(pp_install.shutil, "which", lambda name: None)

    def boom():  # pragma: no cover - must not be called
        raise AssertionError("no download when auto_install=False")

    monkeypatch.setattr(pp_install, "install_pass_cli", boom)

    result = pp.apply_protonpass_secrets(
        enabled=True, vault="V", auto_install=False, home_path=hermes_home,
    )
    assert not result.ok
    assert "pass-cli binary not available" in result.error


# ---------------------------------------------------------------------------
# CLI sync / setup honor auto_install=false (B3): install_pass_cli not called
# ---------------------------------------------------------------------------


def _pp_cli():
    from hermes_cli import protonpass_secrets_cli
    return protonpass_secrets_cli


def test_cli_sync_honors_auto_install_false(hermes_home, monkeypatch):
    """B3: ``cmd_pp_sync``'s test fetch must thread ``auto_install`` from config
    so an ``auto_install: false`` config never triggers a download."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    monkeypatch.setattr(cli, "load_config", lambda: {
        "secrets": {"protonpass": {
            "enabled": True, "vault": "V", "auto_install": False,
        }}
    })

    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return ({}, [])

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fake_fetch)

    def boom_install(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("install_pass_cli must not run when auto_install=false")

    monkeypatch.setattr(cli.pp, "install_pass_cli", boom_install)

    args = argparse.Namespace(apply=False)
    rc = cli.cmd_pp_sync(args)
    assert rc == 0
    assert captured["auto_install"] is False


def test_cli_setup_honors_auto_install_false(hermes_home, monkeypatch):
    """B3: ``cmd_pp_setup``'s test fetch must pass ``auto_install`` from config."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    # An existing config with auto_install disabled.
    cfg = {"secrets": {"protonpass": {
        "enabled": True, "vault": "V", "auto_install": False,
        "service_token_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    }}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: None)
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: None)
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    # A binary already exists (no install needed); version probe is cheap.
    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")

    def boom_install(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("install_pass_cli must not run when auto_install=false")

    monkeypatch.setattr(cli.pp, "install_pass_cli", boom_install)

    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return ({}, [])

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fake_fetch)

    # Token comes from --token-env so no interactive prompt is needed.
    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)
    assert rc == 0
    assert captured["auto_install"] is False


def test_cli_setup_scalar_protonpass_config_does_not_crash(hermes_home, monkeypatch):
    """V5: a hand-edited scalar config (``secrets.protonpass: true``) must NOT
    crash the setup wizard.  ``setdefault`` would hand back the bool ``True`` and
    the later ``secrets_cfg["enabled"] = ...`` write would raise; the wizard now
    normalizes the scalar to a clean dict first."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    # A scalar (non-dict) protonpass config — the V5 crash case.
    cfg = {"secrets": {"protonpass": True}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    saved = {}

    def fake_save(c):
        saved["cfg"] = c

    monkeypatch.setattr(cli, "save_config", fake_save)
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: None)
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")
    monkeypatch.setattr(cli.pp, "install_pass_cli", lambda *a, **kw: binary)

    def fake_fetch(**kwargs):
        return ({"K": "v"}, [])

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fake_fetch)

    # Vault supplied so the config has a fetch target; token via --token-env.
    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)  # must not raise
    assert rc == 0
    # The scalar was normalized to a dict and persisted with assembled fields.
    pp_section = saved["cfg"]["secrets"]["protonpass"]
    assert isinstance(pp_section, dict)
    assert pp_section["enabled"] is True
    assert pp_section["vault"] == "V"


def test_cli_setup_defers_token_save_until_fetch_succeeds(hermes_home, monkeypatch):
    """C6: ``cmd_pp_setup`` must set the token in os.environ (so the test fetch
    can authenticate) but DEFER ``save_env_value`` until AFTER a successful test
    fetch.  When the test fetch raises, ``save_env_value`` must NOT be called —
    otherwise the token is left in .env with no matching config entry."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    cfg = {"secrets": {"protonpass": {
        "enabled": True, "vault": "V", "auto_install": False,
        "service_token_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    }}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    saved_calls = {"env": 0, "config": 0}
    monkeypatch.setattr(cli, "save_config", lambda c: saved_calls.__setitem__("config", saved_calls["config"] + 1))
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: saved_calls.__setitem__("env", saved_calls["env"] + 1))
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")

    def fetch_raises(**kwargs):
        raise RuntimeError("auth failed")

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fetch_raises)

    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)
    assert rc == 1
    # The token was NOT persisted and the config was NOT saved on a failed test.
    assert saved_calls["env"] == 0
    assert saved_calls["config"] == 0


def test_cli_setup_saves_token_after_successful_fetch(hermes_home, monkeypatch):
    """C6: on a successful test fetch the token IS persisted (exactly once),
    just deferred to the save step."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    cfg = {"secrets": {"protonpass": {
        "enabled": True, "vault": "V", "auto_install": False,
        "service_token_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    }}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    env_saves = []
    monkeypatch.setattr(cli, "save_config", lambda c: None)
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: env_saves.append((k, v)))
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")
    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", lambda **kw: ({"K": "v"}, []))

    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)
    assert rc == 0
    assert env_saves == [("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")]


def test_cli_setup_passes_bootstrap_env_to_fetch(hermes_home, monkeypatch):
    """V8-B parity: ``cmd_pp_setup``'s Step-4 test fetch must pass
    ``bootstrap_env=token_env`` so the wizard protects the token env var exactly
    like sync/apply."""
    cli = _pp_cli()
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "svc")

    cfg = {"secrets": {"protonpass": {
        "enabled": True, "vault": "V", "auto_install": False,
        "service_token_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    }}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: None)
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: None)
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")

    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return ({"K": "v"}, [])

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fake_fetch)

    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)
    assert rc == 0
    assert captured["bootstrap_env"] == "PROTON_PASS_PERSONAL_ACCESS_TOKEN"


def test_cli_setup_redacts_token_in_fetch_failure(hermes_home, monkeypatch, capsys):
    """V8-E: a Step-4 test-fetch failure whose error string somehow embeds the
    token must be redacted before it is printed."""
    cli = _pp_cli()
    token = "pst_LEAK_ME_SETUP"
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", token)

    cfg = {"secrets": {"protonpass": {
        "enabled": True, "vault": "V", "auto_install": False,
        "service_token_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN",
    }}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "save_config", lambda c: None)
    monkeypatch.setattr(cli, "save_env_value", lambda k, v: None)
    monkeypatch.setattr(cli, "get_env_path", lambda: "/tmp/.env")

    binary = hermes_home / "bin" / "pass-cli"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: binary)
    monkeypatch.setattr(cli, "_pp_version", lambda b: "pass-cli 2.1.1")

    def fetch_raises(**kwargs):
        raise RuntimeError(f"auth failed leaking {token}")

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fetch_raises)

    args = argparse.Namespace(vault="V", token_env="PROTON_PASS_PERSONAL_ACCESS_TOKEN")
    rc = cli.cmd_pp_setup(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert token not in out
    assert "Fetch failed" in out


def test_cli_sync_redacts_token_in_fetch_failure(hermes_home, monkeypatch, capsys):
    """V8-E: a ``sync`` test-fetch failure whose error string somehow embeds the
    token must be redacted before it is printed."""
    cli = _pp_cli()
    token = "pst_LEAK_ME_SYNC"
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", token)

    monkeypatch.setattr(cli, "load_config", lambda: {
        "secrets": {"protonpass": {
            "enabled": True, "vault": "V", "auto_install": False,
        }}
    })

    def fetch_raises(**kwargs):
        raise RuntimeError(f"auth failed leaking {token}")

    monkeypatch.setattr(cli.pp, "fetch_protonpass_secrets", fetch_raises)

    args = argparse.Namespace(apply=False)
    rc = cli.cmd_pp_sync(args)
    assert rc == 1
    out = capsys.readouterr().out
    assert token not in out
    assert "Fetch failed" in out


def test_cli_disable_scalar_protonpass_config_does_not_crash(monkeypatch):
    """V5: ``cmd_pp_disable`` likewise normalizes a scalar config before
    mutating, so ``protonpass: true`` doesn't crash the disable path."""
    cli = _pp_cli()
    cfg = {"secrets": {"protonpass": True}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)

    saved = {}
    monkeypatch.setattr(cli, "save_config", lambda c: saved.update(cfg=c))

    args = argparse.Namespace()
    rc = cli.cmd_pp_disable(args)  # must not raise
    assert rc == 0
    assert saved["cfg"]["secrets"]["protonpass"]["enabled"] is False


def test_cli_status_scalar_secrets_does_not_crash(hermes_home, monkeypatch):
    """codex-sec/code regression: a top-level scalar ``secrets: true`` must NOT
    crash ``status``.  The old ``(cfg.get('secrets') or {}).get('protonpass')``
    called ``.get`` on a bool (``True or {}`` is ``True``); the read-only helper
    returns ``None`` → a safe disabled config."""
    cli = _pp_cli()
    monkeypatch.setattr(cli, "load_config", lambda: {"secrets": True})
    monkeypatch.setattr(cli.pp, "find_pass_cli", lambda **kw: None)

    rc = cli.cmd_pp_status(argparse.Namespace())  # must not raise
    assert rc == 0


def test_cli_sync_scalar_secrets_does_not_crash(monkeypatch):
    """codex-sec/code regression: a top-level scalar ``secrets: true`` must NOT
    crash ``sync``; with no parseable config it reports disabled and returns 1
    instead of raising ``AttributeError`` on the bool."""
    cli = _pp_cli()
    monkeypatch.setattr(cli, "load_config", lambda: {"secrets": True})

    rc = cli.cmd_pp_sync(argparse.Namespace())  # must not raise
    assert rc == 1


# ---------------------------------------------------------------------------
# _pp_version — delegates to the scrubbed-env probe (V4) + empty-output guard
# ---------------------------------------------------------------------------


def test_pp_version_empty_output_no_indexerror(monkeypatch):
    cli = _pp_cli()
    # The scrubbed probe yields None / empty → "version unknown", never IndexError.
    monkeypatch.setattr(cli._install, "_read_pass_cli_version", lambda b: None)
    assert cli._pp_version("/bin/pass-cli") == "version unknown"
    monkeypatch.setattr(cli._install, "_read_pass_cli_version", lambda b: "")
    assert cli._pp_version("/bin/pass-cli") == "version unknown"


def test_pp_version_reads_first_line(monkeypatch):
    cli = _pp_cli()
    monkeypatch.setattr(
        cli._install, "_read_pass_cli_version",
        lambda b: "pass-cli 2.1.1\nextra",
    )
    assert cli._pp_version("/bin/pass-cli") == "pass-cli 2.1.1"


def test_pp_version_probe_failure_returns_unknown(monkeypatch):
    cli = _pp_cli()
    # The probe is best-effort and never raises; a failure surfaces as None.
    monkeypatch.setattr(cli._install, "_read_pass_cli_version", lambda b: None)
    assert cli._pp_version("/bin/pass-cli") == "version unknown"


def test_pp_version_uses_scrubbed_env_probe(monkeypatch):
    """V4: the CLI version probe must go through ``install._read_pass_cli_version``
    (the scrubbed-env probe) rather than running its own full-env subprocess —
    so loaded secrets / the token are never leaked to a PATH binary."""
    cli = _pp_cli()
    # The token + a secret are present in the process env.
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "leak-me")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")

    captured = {}

    def fake_run(cmd, *, capture_output, text, timeout, env, **kwargs):
        # This is the subprocess call INSIDE _read_pass_cli_version — it must be
        # handed a scrubbed env.  Accept extra kwargs (e.g. ``errors=``) so the
        # probe's exact call shape can evolve without breaking this assertion.
        captured["env"] = env
        return _CompletedVersion()

    # Patch at the install module's subprocess so we observe the real probe path
    # (proving _pp_version routes through it, not a bespoke call).
    monkeypatch.setattr(cli._install.subprocess, "run", fake_run)

    result = cli._pp_version("/usr/bin/pass-cli")
    assert result == "pass-cli 2.1.1"
    env = captured["env"]
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert env.get("NO_COLOR") == "1"  # the minimal-env marker


class _CompletedVersion:
    def __init__(self, returncode=0, stdout="pass-cli 2.1.1", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# SecretSource bool coercion (B9): enabled:"false" disables the source
# ---------------------------------------------------------------------------


def test_protonpass_source_coerces_enabled_string_false():
    from agent.secret_sources.protonpass import ProtonPassSource

    src = ProtonPassSource()
    assert src.is_enabled({"enabled": "false"}) is False
    assert src.is_enabled({"enabled": "true"}) is True
    assert src.is_enabled({"enabled": "maybe"}) is False
    assert src.is_enabled({"enabled": True}) is True
    assert src.is_enabled({"enabled": 0}) is False


def test_env_loader_string_false_disables_protonpass(tmp_path, monkeypatch):
    """B9 end-to-end: ``enabled: "false"`` (a string) must DISABLE the source so
    the fetch layer is never called."""
    from hermes_cli import env_loader
    from agent.secret_sources import registry as reg_module

    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    reg_module._reset_registry_for_tests()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "pst_test-token")
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  protonpass:\n"
        "    enabled: \"false\"\n"
        "    env:\n"
        "      ANTHROPIC_API_KEY: 'pass://SHARE/ITEM/api_key'\n",
        encoding="utf-8",
    )

    called = {"n": 0}

    def _fake_fetch(**_kwargs):  # pragma: no cover - must not be called
        called["n"] += 1
        return {"ANTHROPIC_API_KEY": "sk-ant"}, []

    import agent.secret_sources.protonpass as pp_module
    monkeypatch.setattr(pp_module, "find_pass_cli", lambda **_kw: "/fake/pass-cli")
    monkeypatch.setattr(pp_module, "fetch_protonpass_secrets", _fake_fetch)

    env_loader._apply_external_secret_sources(tmp_path)

    assert called["n"] == 0
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    reg_module._reset_registry_for_tests()
