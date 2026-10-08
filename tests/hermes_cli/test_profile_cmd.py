from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, profile_cmd
from hermes_cli.profiles import ProfileInfo
from hermes_constants import (
    _get_platform_default_hermes_home,
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(params=["native", "custom"])
def profile_homes(tmp_path, monkeypatch, request):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.delenv("HERMES_DATA_DIR_SUFFIX", raising=False)
    monkeypatch.delenv("SUDO_USER", raising=False)
    native = _get_platform_default_hermes_home()
    custom = tmp_path / "data"
    root = native if request.param == "native" else custom
    monkeypatch.setenv("HERMES_HOME", str(root))
    # Keep Linux's legacy bare-unit ownership lookup inside the disposable tree too.
    monkeypatch.setattr(gateway, "_SYSTEM_UNIT_DIR", tmp_path / "system-units")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    homes = [
        ("default", native, True, "hermes-gateway"),
        ("demo", native / "profiles" / "demo", False, "hermes-gateway-demo" if root == native else None),
        ("default", custom, True, None),
        ("demo", custom / "profiles" / "demo", False, "hermes-gateway-demo" if root == custom else None),
        ("demo", tmp_path / "standalone" / "demo", False, None),
    ]
    profiles = []
    for name, path, is_default, service in homes:
        path.mkdir(parents=True)
        (path / "config.yaml").write_text("model: test\n", encoding="utf-8")
        service = service or "hermes-gateway-" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        profiles.append((ProfileInfo(name, path, is_default, gateway_running=True), service))
    return profiles


def test_profile_gateway_service_identity_restores_caller_home(profile_homes):
    ambient = get_hermes_home()
    token = set_hermes_home_override(profile_homes[-1][0].path)
    try:
        caller_home = get_hermes_home()
        caller_override = get_hermes_home_override()
        # A -> B -> A, covering native/custom roots and noncanonical named homes.
        for profile, expected in profile_homes + profile_homes[:1]:
            assert profile_cmd._profile_gateway_service_name(profile) == expected
            assert get_hermes_home() == caller_home
            assert get_hermes_home_override() == caller_override
    finally:
        reset_hermes_home_override(token)
    assert get_hermes_home() == ambient


@pytest.mark.parametrize("bound_caller", [False, True])
def test_profile_gateway_service_identity_restores_home_on_error(profile_homes, monkeypatch, bound_caller):
    def fail_resolution():
        assert get_hermes_home() == profile_homes[2][0].path
        raise RuntimeError("service resolution failed")

    monkeypatch.setattr(gateway, "get_service_name", fail_resolution)
    token = set_hermes_home_override(profile_homes[-1][0].path) if bound_caller else None
    try:
        caller_home = get_hermes_home()
        caller_override = get_hermes_home_override()
        with pytest.raises(RuntimeError, match="service resolution failed"):
            profile_cmd._profile_gateway_service_name(profile_homes[2][0])
        assert get_hermes_home() == caller_home
        assert get_hermes_home_override() == caller_override
    finally:
        if token is not None:
            reset_hermes_home_override(token)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("scope", ["user", "system", "missing", "unavailable", "timeout"])
def test_profile_gateway_scope_queries_resolved_unit(profile_homes, monkeypatch, scope):
    for profile, expected in profile_homes:
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            assert get_hermes_home_override() is None
            assert argv[argv.index("--all") + 1] == f"{expected}.service"
            if scope == "unavailable":
                raise OSError("no systemctl")
            if scope == "timeout":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            queried_scope = "user" if "--user" in argv else "system"
            output = f"{expected}.service loaded inactive dead\n" if queried_scope == scope else "other.service loaded active running\n"
            return SimpleNamespace(stdout=output)

        monkeypatch.setattr(profile_cmd.subprocess, "run", fake_run)
        assert profile_cmd._profile_gateway_scope(profile) == (scope if scope in {"user", "system"} else "—")
        assert calls == [
            prefix + ["list-units", "--all", f"{expected}.service", "--no-legend", "--plain", "--no-pager"]
            for prefix in ([["systemctl", "--user"]] if scope == "user" else [["systemctl", "--user"], ["systemctl"]])
        ]


@pytest.mark.platforms("not linux")
def test_profile_gateway_scope_is_not_applicable_off_linux(profile_homes, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("non-Linux scope must not resolve or query a systemd unit")

    monkeypatch.setattr(gateway, "get_service_name", unexpected_call)
    monkeypatch.setattr(profile_cmd.subprocess, "run", unexpected_call)
    assert profile_cmd._profile_gateway_scope(profile_homes[0][0]) == "—"


@pytest.mark.parametrize("long_models", [False, True], ids=["short-and-missing", "long-suffixes"])
def test_profile_list_preserves_models_and_column_alignment(tmp_path, monkeypatch, capsys, long_models):
    """Regression for #129827 part 3: real config discovery must retain model suffixes."""
    from hermes_cli import profiles

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes-root"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gateway, "_SYSTEM_UNIT_DIR", tmp_path / "system-units")
    models = {"default": "short-model", "missing": None}
    if long_models:
        prefix = "provider-with-a-long-namespace/model-family-with-a-long-name-"
        models.update({"alpha": prefix + "thinking", "beta": prefix + "instruct-v2"})
    for name, model in models.items():
        home = root if name == "default" else root / "profiles" / name
        home.mkdir(parents=True)
        config = f"model:\n  default: {model}\n  provider: openrouter\n" if model else "{}\n"
        (home / "config.yaml").write_text(config, encoding="utf-8")

    discovered = profiles.list_profiles()
    assert {p.name: p.model for p in discovered} == models
    profile_cmd.cmd_profile(SimpleNamespace(profile_action="list"))
    lines = capsys.readouterr().out.strip("\n").splitlines()
    header, rule, *rows = lines
    headers = ("Profile", "Model", "Gateway", "Scope", "Alias", "Distribution")
    starts = [header.index(label) for label in headers]
    rule_starts = [i for i, ch in enumerate(rule) if ch == "─" and rule[i - 1] == " "]
    assert rule_starts == starts
    assert len(rows) == len(discovered)
    for row, profile in zip(rows, discovered):
        if models[profile.name]:
            assert models[profile.name] in row
    for row, profile in zip(rows, discovered):
        assert row[starts[0]:starts[1]].strip() == profile.name
        assert row[starts[1]:starts[2]].strip() == (models[profile.name] or "—")
        assert row[starts[2]:starts[3]].strip() == "stopped"
        assert row[starts[3]:starts[4]].strip() == "—"
        assert row[starts[4]:starts[5]].strip() == "—"
        assert row[starts[5]:].strip() == "—"
        assert row.index("stopped") == starts[2]
    assert rows[0][1] == "◆"
    assert all(row[1] == " " for row in rows[1:])
