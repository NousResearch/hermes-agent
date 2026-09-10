"""Behavior contracts for standalone external-process model providers."""
from __future__ import annotations

import importlib
import os
import stat
import threading
import time
from types import SimpleNamespace


def _profile(**overrides):
    from providers.base import ProviderProfile

    values = {
        "name": "test-process",
        "display_name": "Test Process",
        "description": "Test subprocess provider",
        "base_url": "process://test",
        "auth_type": "external_process",
        "process_command": "test-process-cli",
        "process_args": ("serve", "--stdio"),
        "process_command_env_vars": ("TEST_PROCESS_COMMAND",),
        "process_args_env_var": "TEST_PROCESS_ARGS",
        "fallback_models": ("test-model-a", "test-model-b"),
    }
    values.update(overrides)
    return ProviderProfile(**values)


def test_external_process_status_uses_the_selected_profile_not_copilot_defaults(
    tmp_path, monkeypatch
):
    from hermes_cli import auth

    executable = tmp_path / "test-process-cli"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    profile = _profile(process_command=str(executable))

    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)

    status = auth.get_external_process_provider_status(profile.name)

    assert status["configured"] is True
    assert status["resolved_command"] == str(executable)
    assert status["args"] == ["serve", "--stdio"]
    assert status["base_url"] == "process://test"


def test_configured_command_and_args_override_profile_defaults(tmp_path, monkeypatch):
    from hermes_cli import auth
    from hermes_cli import config as config_mod

    executable = tmp_path / "configured-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    profile = _profile(process_command="missing-profile-default")
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "providers": {
                profile.name: {
                    "command": str(executable),
                    "args": ["--tenant", "test"],
                }
            }
        },
    )

    status = auth.get_external_process_provider_status(profile.name)
    creds = auth.resolve_external_process_provider_credentials(profile.name)

    assert status["resolved_command"] == str(executable)
    assert status["args"] == ["--tenant", "test"]
    assert creds["command"] == str(executable)
    assert creds["args"] == ["--tenant", "test"]


def test_status_honors_optional_profile_auth_probe(tmp_path, monkeypatch):
    from hermes_cli import auth

    executable = tmp_path / "test-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    profile = _profile(process_command=str(executable))
    profile.external_process_auth_status = lambda **_: {
        "logged_in": False,
        "auth_evidence": "provider probe reported authentication required",
    }
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)

    status = auth.get_external_process_provider_status(profile.name)

    assert status["configured"] is True
    assert status["logged_in"] is False
    assert status["auth_verified"] is False
    assert status["auth_evidence"] == "provider probe reported authentication required"


def test_status_hard_bounds_a_probe_that_ignores_timeout(tmp_path, monkeypatch):
    from hermes_cli import auth

    executable = tmp_path / "test-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    blocked = threading.Event()
    calls = 0
    profile = _profile(process_command=str(executable))

    def blocked_probe(**_):
        nonlocal calls
        calls += 1
        return blocked.wait(5)

    setattr(profile, "external_process_auth_status", blocked_probe)
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(auth, "_EXTERNAL_PROCESS_PROBE_JOIN_TIMEOUT", 0.02)

    started = time.monotonic()
    status = auth.get_external_process_provider_status(profile.name)
    for _ in range(3):
        auth.get_external_process_provider_status(profile.name)

    assert time.monotonic() - started < 1.5
    assert calls == 1
    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["auth_verified"] is False
    assert status["auth_source"] == "provider_probe_timeout"
    blocked.set()
    state = auth._EXTERNAL_PROCESS_PROBES.pop(profile.name)
    state["worker"].join(timeout=1)


def test_status_concurrent_callers_share_one_started_probe(tmp_path, monkeypatch):
    from hermes_cli import auth

    executable = tmp_path / "test-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    release = threading.Event()
    calls = 0
    profile = _profile(process_command=str(executable))

    def blocked_probe(**_):
        nonlocal calls
        calls += 1
        release.wait(1)
        return {"logged_in": True}

    setattr(profile, "external_process_auth_status", blocked_probe)
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(auth, "_EXTERNAL_PROCESS_PROBE_JOIN_TIMEOUT", 0.05)
    errors = []

    def status_call():
        try:
            auth.get_external_process_provider_status(profile.name)
        except Exception as exc:
            errors.append(exc)

    callers = [threading.Thread(target=status_call) for _ in range(6)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=1)
    release.set()

    assert not errors
    assert all(not caller.is_alive() for caller in callers)
    assert calls == 1
    state = auth._EXTERNAL_PROCESS_PROBES.pop(profile.name)
    state["worker"].join(timeout=1)


def test_status_reports_malformed_configured_args(tmp_path, monkeypatch):
    from hermes_cli import auth

    executable = tmp_path / "test-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    profile = _profile(process_command=str(executable))
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"providers": {profile.name: {"args": "\""}}},
    )

    status = auth.get_external_process_provider_status(profile.name)

    assert status["configured"] is False
    assert status["auth_source"] == "invalid_configuration"
    assert status["error"] == "invalid_external_process_args"


def test_status_rejects_nul_in_structured_argv(tmp_path, monkeypatch):
    from hermes_cli import auth

    executable = tmp_path / "test-agent"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    profile = _profile(process_command=str(executable))
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"providers": {profile.name: {"args": ["--ok", "\0"]}}},
    )

    status = auth.get_external_process_provider_status(profile.name)

    assert status["configured"] is False
    assert status["error"] == "invalid_external_process_args"


def test_external_process_profile_is_listed_as_a_canonical_model_provider(monkeypatch):
    import providers
    import hermes_cli.models_catalog_static as models

    profile = _profile()
    original = providers.list_providers
    monkeypatch.setattr(providers, "list_providers", lambda: [profile])
    try:
        reloaded = importlib.reload(models)
        matching = [entry for entry in reloaded.CANONICAL_PROVIDERS if entry.slug == profile.name]
        assert len(matching) == 1
        assert matching[0].label == profile.display_name
    finally:
        monkeypatch.setattr(providers, "list_providers", original)
        importlib.reload(models)


def test_external_process_model_choices_prefer_live_catalog_and_fall_back(monkeypatch):
    from hermes_cli.model_setup_flows import _external_process_model_choices

    profile = _profile()
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    seen = {}

    def fetch_models(**kwargs):
        seen.update(kwargs)
        return ["live-b", "live-a", "live-b"]
    monkeypatch.setattr(profile, "fetch_models", fetch_models)
    assert _external_process_model_choices(profile.name) == ["live-b", "live-a"]
    assert "command" in seen
    assert "args" in seen

    monkeypatch.setattr(profile, "fetch_models", lambda **kwargs: None)
    assert _external_process_model_choices(profile.name) == ["test-model-a", "test-model-b"]


def test_model_catalog_internal_type_error_is_not_retried(monkeypatch):
    from hermes_cli.model_setup_flows import _external_process_model_choices

    profile = _profile()
    calls = 0

    def broken_fetch(**kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("provider implementation defect")

    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(profile, "fetch_models", broken_fetch)

    assert _external_process_model_choices(profile.name) == ["test-model-a", "test-model-b"]
    assert calls == 1


def test_runtime_client_kwargs_include_generic_process_launch_settings(monkeypatch):
    from agent.agent_init import _explicit_client_kwargs

    profile = _profile()
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    agent = SimpleNamespace(
        provider=profile.name,
        acp_command="/configured/process-cli",
        acp_args=["--tenant", "test"],
    )

    kwargs = _explicit_client_kwargs(agent, "process-provider", profile.base_url, 30)

    assert kwargs["command"] == "/configured/process-cli"
    assert kwargs["args"] == ["--tenant", "test"]


def test_external_process_setup_stops_on_failed_provider_auth_probe(monkeypatch, capsys):
    from hermes_cli import auth
    from hermes_cli.model_setup_flows import _model_flow_external_process

    profile = _profile()
    monkeypatch.setitem(
        auth.PROVIDER_REGISTRY,
        profile.name,
        auth.ProviderConfig(
            id=profile.name,
            name=profile.display_name,
            auth_type="external_process",
            inference_base_url=profile.base_url,
        ),
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr(
        auth,
        "get_external_process_provider_status",
        lambda provider_id: {
            "configured": True,
            "logged_in": False,
            "auth_source": "provider_probe",
            "auth_evidence": "provider probe reported authentication required",
            "command": profile.process_command,
            "base_url": profile.base_url,
        },
    )
    monkeypatch.setattr(
        auth,
        "resolve_external_process_provider_credentials",
        lambda provider_id: (_ for _ in ()).throw(AssertionError("must not continue")),
    )

    _model_flow_external_process({}, profile.name)

    assert "authentication" in capsys.readouterr().out.lower()


def test_main_classifies_profile_external_process_providers(monkeypatch):
    from hermes_cli.main_provider_setup import _is_profile_external_process_provider

    profile = _profile()
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile if name == profile.name else None)

    assert _is_profile_external_process_provider(profile.name) is True
    assert _is_profile_external_process_provider("missing") is False
