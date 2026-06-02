"""Tests for the Photon CLI and setup flow."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from plugins.platforms.photon import cli as photon_cli
from plugins.platforms.photon import phone_management


PHONE = "+15105550123"
OTHER_PHONE = "+15105550124"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    photon_cli.register_cli(parser)
    return parser


def _ctx(tmp_path: Path, phone: str = PHONE) -> photon_cli._PhotonSetupContext:
    return photon_cli._PhotonSetupContext(
        args=argparse.Namespace(
            phone=phone,
            first_name=None,
            last_name=None,
            email=None,
            no_browser=True,
            skip_adapter_install=True,
            verbose=False,
        ),
        hermes_home=tmp_path,
        env_path=tmp_path / ".env",
        project_name=photon_cli._FIXED_PROJECT_NAME,
    )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["setup", PHONE], {"photon_command": "setup", "phone": PHONE}),
        (["reset", "--all"], {"photon_command": "reset", "all": True}),
        (
            ["phones", "list"],
            {"photon_command": "phones", "photon_phones_command": "list"},
        ),
        (
            ["phones", "add", PHONE],
            {
                "photon_command": "phones",
                "photon_phones_command": "add",
                "phone": PHONE,
            },
        ),
        (
            ["phones", "remove", PHONE],
            {
                "photon_command": "phones",
                "photon_phones_command": "remove",
                "phone": PHONE,
            },
        ),
        (
            ["projects", "list"],
            {"photon_command": "projects", "photon_projects_command": "list"},
        ),
    ],
)
def test_parser_accepts_supported_photon_commands(
    argv: list[str],
    expected: dict[str, Any],
) -> None:
    args = _parser().parse_args(argv)

    for key, value in expected.items():
        assert getattr(args, key) == value


@pytest.mark.parametrize(
    "argv",
    [
        ["setup", "my-project", PHONE],
        ["reset", "all"],
        ["phones", "add"],
        ["phones", "remove"],
        ["phones", "add", "555-1234"],
        ["phones", "remove", "not-a-phone"],
    ],
)
def test_parser_rejects_invalid_photon_command_inputs(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)


def test_fixed_project_existing_match_is_used(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _ctx(tmp_path)
    env: dict[str, str] = {}
    stored: list[dict[str, Any]] = []

    project = {
        "name": photon_cli._FIXED_PROJECT_NAME,
        "dashboard_project_id": "dashboard-1",
        "spectrum_project_id": "project-1",
        "project_secret": "secret-1",
        "spectrum_enabled": True,
        "imessage_enabled": True,
    }

    monkeypatch.setattr(photon_cli.photon_auth, "list_projects", lambda _token: [project])
    monkeypatch.setattr(photon_cli, "_refresh_project_details", lambda _token, _project: project)
    monkeypatch.setattr(photon_cli.photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_cli.photon_auth, "list_project_users", lambda *_args: [])
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: env.get(key))

    def fake_store(project_id: str, project_secret: str, **extra: Any) -> None:
        stored.append({"id": project_id, "secret": project_secret, **extra})
        env["PHOTON_PROJECT_ID"] = project_id
        env["PHOTON_PROJECT_SECRET"] = project_secret
        env["PHOTON_PROJECT_NAME"] = str(extra.get("name") or "")

    monkeypatch.setattr(photon_cli.photon_auth, "store_project_credentials", fake_store)

    photon_cli._ensure_fixed_spectrum_project(ctx, "token")

    assert ctx.project_id == "project-1"
    assert ctx.project_secret == "secret-1"
    assert stored[0]["name"] == photon_cli._FIXED_PROJECT_NAME
    out = capsys.readouterr().out
    assert "found existing 'hermes-agent' project" in out
    assert "project ready: hermes-agent (project-1)" in out


def test_fixed_project_missing_is_created(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _ctx(tmp_path)
    created: list[tuple[str, str]] = []

    monkeypatch.setattr(photon_cli.photon_auth, "list_projects", lambda _token: [])
    monkeypatch.setattr(photon_cli.photon_auth, "load_project_credentials", lambda: (None, None))
    monkeypatch.setattr(photon_cli.photon_auth, "list_project_users", lambda *_args: [])
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    def fake_create(_token: str, *, name: str, source: str) -> tuple[str, str]:
        created.append((name, source))
        return "project-new", "secret-new"

    monkeypatch.setattr(photon_cli, "_create_and_store_project", fake_create)
    monkeypatch.setattr(photon_cli.photon_auth, "store_project_credentials", lambda *_args, **_kw: None)

    photon_cli._ensure_fixed_spectrum_project(ctx, "token")

    assert created == [(photon_cli._FIXED_PROJECT_NAME, "fixed-project-created")]
    assert ctx.project_id == "project-new"
    assert ctx.project_secret == "secret-new"
    out = capsys.readouterr().out
    assert "no 'hermes-agent' project found; creating it" in out
    assert "project ready: hermes-agent (project-new)" in out


def test_duplicate_fixed_projects_fail(tmp_path: Path, monkeypatch: Any) -> None:
    ctx = _ctx(tmp_path)
    projects = [
        {
            "name": photon_cli._FIXED_PROJECT_NAME,
            "spectrumProjectId": "project-1",
            "projectSecret": "secret-1",
            "spectrum": True,
            "platforms": ["imessage"],
        },
        {
            "name": photon_cli._FIXED_PROJECT_NAME,
            "spectrumProjectId": "project-2",
            "projectSecret": "secret-2",
            "spectrum": True,
            "platforms": ["imessage"],
        },
    ]
    monkeypatch.setattr(photon_cli.photon_auth, "list_projects", lambda _token: projects)

    with pytest.raises(photon_cli._FailedInvariant) as raised:
        photon_cli._ensure_fixed_spectrum_project(ctx, "token")

    assert "duplicate" in raised.value.summary


def test_stale_local_project_id_fails(tmp_path: Path, monkeypatch: Any) -> None:
    ctx = _ctx(tmp_path)
    project = {
        "name": photon_cli._FIXED_PROJECT_NAME,
        "dashboard_project_id": "dashboard-1",
        "spectrum_project_id": "project-1",
        "project_secret": "secret-1",
        "spectrum_enabled": True,
        "imessage_enabled": True,
    }

    monkeypatch.setattr(photon_cli.photon_auth, "list_projects", lambda _token: [project])
    monkeypatch.setattr(photon_cli, "_refresh_project_details", lambda _token, _project: project)
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("stale-project", "old-secret"),
    )

    with pytest.raises(photon_cli._FailedInvariant) as raised:
        photon_cli._ensure_fixed_spectrum_project(ctx, "token")

    assert "stale" in raised.value.summary


def test_existing_project_user_is_reused_without_create(tmp_path: Path, monkeypatch: Any) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-1"
    ctx.project_secret = "secret-1"
    saved: dict[str, str] = {}

    monkeypatch.setattr(
        photon_cli.photon_auth,
        "find_project_user_by_phone",
        lambda *_args: {
            "phone_number": PHONE,
            "assigned_phone_number": "+15550001111",
            "phone_numbers": [PHONE, "+15550001111"],
        },
    )
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "create_user",
        lambda *_args, **_kw: pytest.fail("existing users must not be recreated"),
    )
    monkeypatch.setattr(photon_cli, "_ensure_operator_phone_allowed", lambda *_args, **_kw: True)
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: saved.get(key))
    monkeypatch.setattr(
        photon_cli,
        "_save_photon_env_value_checked",
        lambda _ctx, key, value: saved.setdefault(key, value) is None,
    )

    photon_cli._ensure_operator_phone(ctx)

    assert ctx.assigned_phone_number == "+15550001111"
    assert saved["PHOTON_OPERATOR_PHONE"] == PHONE
    assert saved[photon_cli._PHOTON_ASSIGNED_PHONE_ENV] == "+15550001111"


def test_missing_project_user_is_created(tmp_path: Path, monkeypatch: Any) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-1"
    ctx.project_secret = "secret-1"
    created: list[str] = []
    saved: dict[str, str] = {}

    monkeypatch.setattr(photon_cli.photon_auth, "find_project_user_by_phone", lambda *_args: None)

    def fake_create(*_args: Any, phone_number: str, **_kw: Any) -> dict[str, Any]:
        created.append(phone_number)
        return {"phoneNumber": phone_number, "assignedPhoneNumber": "+15550002222"}

    monkeypatch.setattr(photon_cli.photon_auth, "create_user", fake_create)
    monkeypatch.setattr(photon_cli, "_ensure_operator_phone_allowed", lambda *_args, **_kw: True)
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: saved.get(key))
    monkeypatch.setattr(
        photon_cli,
        "_save_photon_env_value_checked",
        lambda _ctx, key, value: saved.setdefault(key, value) is None,
    )

    photon_cli._ensure_operator_phone(ctx)

    assert created == [PHONE]
    assert ctx.assigned_phone_number == "+15550002222"
    assert saved["PHOTON_OPERATOR_PHONE"] == PHONE


def test_missing_project_user_prechecks_shared_number_availability(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-1"
    ctx.project_secret = "secret-1"
    ctx.dashboard_token = "dashboard-token"
    availability_calls: list[str] = []
    created: list[str] = []
    saved: dict[str, str] = {}

    monkeypatch.setattr(photon_cli.photon_auth, "find_project_user_by_phone", lambda *_args: None)

    def fake_check(_token: str, phone: str) -> bool:
        availability_calls.append(phone)
        return True

    def fake_create(*_args: Any, phone_number: str, **_kw: Any) -> dict[str, Any]:
        created.append(phone_number)
        return {"phoneNumber": phone_number, "assignedPhoneNumber": "+15550003333"}

    monkeypatch.setattr(photon_cli.photon_auth, "check_phone_availability", fake_check)
    monkeypatch.setattr(photon_cli.photon_auth, "create_user", fake_create)
    monkeypatch.setattr(photon_cli, "_ensure_operator_phone_allowed", lambda *_args, **_kw: True)
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: saved.get(key))
    monkeypatch.setattr(
        photon_cli,
        "_save_photon_env_value_checked",
        lambda _ctx, key, value: saved.setdefault(key, value) is None,
    )

    photon_cli._ensure_operator_phone(ctx)

    assert availability_calls == [PHONE]
    assert created == [PHONE]
    assert ctx.assigned_phone_number == "+15550003333"


@pytest.mark.parametrize(
    ("initial_env", "expected_saves", "runtime_changed"),
    [
        (
            {},
            [
                ("PHOTON_HOME_CHANNEL", f"any;-;{PHONE}"),
                ("PHOTON_HOME_CHANNEL_NAME", "You (iMessage)"),
            ],
            True,
        ),
        (
            {
                "PHOTON_HOME_CHANNEL": "custom-space",
                "PHOTON_HOME_CHANNEL_NAME": "Custom",
            },
            [],
            False,
        ),
        (
            {"PHOTON_HOME_CHANNEL": "custom-space"},
            [("PHOTON_HOME_CHANNEL_NAME", "You (iMessage)")],
            True,
        ),
    ],
)
def test_setup_home_channel_defaults(
    tmp_path: Path,
    monkeypatch: Any,
    initial_env: dict[str, str],
    expected_saves: list[tuple[str, str]],
    runtime_changed: bool,
) -> None:
    ctx = _ctx(tmp_path)
    env = dict(initial_env)
    save_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: env.get(key))

    def fake_save(_ctx: photon_cli._PhotonSetupContext, key: str, value: str) -> bool:
        save_calls.append((key, value))
        env[key] = value
        return True

    monkeypatch.setattr(photon_cli, "_save_photon_env_value_checked", fake_save)

    photon_cli._ensure_home_channel_default(ctx, PHONE)

    assert save_calls == expected_saves
    assert ctx.runtime_secrets_changed is runtime_changed


def test_unavailable_shared_number_stops_before_user_create(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-1"
    ctx.project_secret = "secret-1"
    ctx.dashboard_token = "dashboard-token"

    monkeypatch.setattr(photon_cli.photon_auth, "find_project_user_by_phone", lambda *_args: None)
    monkeypatch.setattr(photon_cli.photon_auth, "check_phone_availability", lambda *_args: False)
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "create_user",
        lambda *_args, **_kw: pytest.fail("unavailable phones must not create users"),
    )

    with pytest.raises(photon_cli._FailedInvariant) as raised:
        photon_cli._ensure_operator_phone(ctx)

    assert "no shared iMessage number" in raised.value.summary


def test_existing_user_create_conflict_must_verify_current_project(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-1"
    ctx.project_secret = "secret-1"

    monkeypatch.setattr(photon_cli.photon_auth, "find_project_user_by_phone", lambda *_args: None)

    with pytest.raises(photon_cli._FailedInvariant) as raised:
        photon_cli._require_project_user_by_phone(
            ctx,
            PHONE,
            cause=RuntimeError("user already exists"),
        )

    assert "not in the current Photon project" in raised.value.summary


def test_new_setup_reconciler_uses_adapter_flow(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ctx = _ctx(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(photon_cli, "_ensure_dashboard_auth", lambda _ctx: "token")
    monkeypatch.setattr(
        photon_cli,
        "_ensure_fixed_spectrum_project",
        lambda _ctx, _token: calls.append("project"),
    )
    monkeypatch.setattr(photon_cli, "_ensure_operator_phone", lambda _ctx: calls.append("phone"))
    monkeypatch.setattr(photon_cli, "_ensure_sidecar_ready", lambda _ctx: calls.append("deps"))
    monkeypatch.setattr(photon_cli, "_ensure_photon_gateway_platform_enabled", lambda _ctx: calls.append("platform"))
    monkeypatch.setattr(photon_cli, "_report_gateway_handoff", lambda _ctx: calls.append("handoff"))

    photon_cli._run_setup_reconciler(ctx)

    assert ctx.dashboard_token == "token"
    assert calls == [
        "project",
        "phone",
        "deps",
        "platform",
        "handoff",
    ]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "pid": 123,
                "health": {
                    "healthy": True,
                    "pid": 123,
                    "sdk": {"connected": True},
                },
            },
            "✓ connected (pid 123)",
        ),
        (
            {
                "health": {
                    "healthy": False,
                    "state": "failed",
                    "last_error": {"message": "Spectrum inbound stream ended"},
                },
            },
            "✗ failed (Spectrum inbound stream ended)",
        ),
    ],
)
def test_adapter_runtime_status_formatting(
    state: dict[str, Any],
    expected: str,
) -> None:
    assert photon_cli._format_adapter_runtime_status(state) == expected


def test_primary_setup_reconciler_stays_inside_photon_boundary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    ctx = _ctx(tmp_path)
    calls: list[str] = []

    def record(name: str):
        def _inner(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(name)
            if name == "dashboard-auth":
                return "dashboard-token"
            return None

        return _inner

    monkeypatch.setattr(photon_cli, "_ensure_dashboard_auth", record("dashboard-auth"))
    monkeypatch.setattr(
        photon_cli,
        "_ensure_fixed_spectrum_project",
        record("fixed-spectrum-project"),
    )
    monkeypatch.setattr(photon_cli, "_ensure_operator_phone", record("operator-phone"))
    monkeypatch.setattr(
        photon_cli,
        "_ensure_sidecar_ready",
        record("sidecar-ready"),
    )
    monkeypatch.setattr(
        photon_cli,
        "_ensure_photon_gateway_platform_enabled",
        record("platform-enabled"),
    )
    monkeypatch.setattr(photon_cli, "_report_gateway_handoff", record("gateway-handoff"))

    photon_cli._run_setup_reconciler(ctx)

    assert calls == [
        "dashboard-auth",
        "fixed-spectrum-project",
        "operator-phone",
        "sidecar-ready",
        "platform-enabled",
        "gateway-handoff",
    ]
    assert ctx.runtime_secrets_changed is False


def test_interactive_setup_runs_setup_without_existing_token_or_project(
    monkeypatch: Any,
) -> None:
    captured: list[argparse.Namespace] = []

    monkeypatch.setattr(photon_cli, "_interactive_setup_already_configured", lambda: False)

    def fake_cmd_setup(args: argparse.Namespace) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(photon_cli, "_cmd_setup", fake_cmd_setup)

    photon_cli.interactive_setup()

    assert len(captured) == 1
    assert captured[0].phone is None
    assert captured[0].no_browser is False


def test_interactive_setup_failure_does_not_print_duplicate_guidance(
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(photon_cli, "_interactive_setup_already_configured", lambda: False)

    def fake_cmd_setup(_args: argparse.Namespace) -> int:
        print("Photon setup stopped: operator phone number is required")
        print("  repair   : rerun `hermes photon setup '+<country-code><number>'`")
        return 1

    monkeypatch.setattr(photon_cli, "_cmd_setup", fake_cmd_setup)

    photon_cli.interactive_setup()

    out = capsys.readouterr().out
    assert "Photon setup stopped: operator phone number is required" in out
    assert "hermes photon setup '+<country-code><number>'" in out
    assert "Photon iMessage setup is not complete yet." not in out
    assert "Current local state:" not in out
    assert "Login-only fallback" not in out


def test_interactive_setup_reconfigure_prompt_shows_current_binding(
    monkeypatch: Any,
) -> None:
    import hermes_cli.cli_output as cli_output

    prompts: list[str] = []
    info_messages: list[str] = []
    captured: list[argparse.Namespace] = []

    monkeypatch.setattr(photon_cli, "_interactive_setup_already_configured", lambda: True)
    monkeypatch.setattr(photon_cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-1", "secret-1"),
    )
    monkeypatch.setattr(
        photon_cli,
        "_get_env_value",
        lambda key: {
            "PHOTON_OPERATOR_PHONE": PHONE,
            photon_cli._PHOTON_ASSIGNED_PHONE_ENV: "+15550001111",
        }.get(key),
    )
    monkeypatch.setattr(cli_output, "print_info", lambda message: info_messages.append(message))

    def fake_prompt_yes_no(question: str, default: bool = True) -> bool:
        prompts.append(question)
        assert default is False
        return True

    monkeypatch.setattr(cli_output, "prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr(
        photon_cli,
        "_cmd_setup",
        lambda args: captured.append(args) or 0,
    )

    photon_cli.interactive_setup()

    assert info_messages == [
        (
            "Photon iMessage is already configured: operator +15105550123; "
            "assigned Photon number +15550001111."
        ),
        "Reusing existing operator phone +15105550123.",
    ]
    assert prompts == [
        (
            "Reconfigure Photon iMessage bound to operator +15105550123; "
            "assigned Photon number +15550001111?"
        )
    ]
    assert len(captured) == 1
    assert captured[0].phone == PHONE


def test_interactive_setup_decline_reconfigure_does_not_run_setup(
    monkeypatch: Any,
) -> None:
    import hermes_cli.cli_output as cli_output

    prompts: list[str] = []
    info_messages: list[str] = []

    monkeypatch.setattr(photon_cli, "_interactive_setup_already_configured", lambda: True)
    monkeypatch.setattr(photon_cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-1", "secret-1"),
    )
    monkeypatch.setattr(
        photon_cli,
        "_get_env_value",
        lambda key: {
            "PHOTON_OPERATOR_PHONE": PHONE,
            photon_cli._PHOTON_ASSIGNED_PHONE_ENV: "+15550001111",
        }.get(key),
    )
    monkeypatch.setattr(cli_output, "print_info", lambda message: info_messages.append(message))

    def fake_prompt_yes_no(question: str, default: bool = True) -> bool:
        prompts.append(question)
        assert default is False
        return False

    monkeypatch.setattr(cli_output, "prompt_yes_no", fake_prompt_yes_no)
    monkeypatch.setattr(
        photon_cli,
        "_cmd_setup",
        lambda _args: pytest.fail("declining reconfigure must not run setup"),
    )

    photon_cli.interactive_setup()

    assert info_messages == [
        (
            "Photon iMessage is already configured: operator +15105550123; "
            "assigned Photon number +15550001111."
        ),
        (
            "Leaving existing Photon iMessage configuration unchanged. "
            "Run `hermes photon status` to inspect it."
        ),
    ]
    assert prompts == [
        (
            "Reconfigure Photon iMessage bound to operator +15105550123; "
            "assigned Photon number +15550001111?"
        )
    ]


def test_registered_platform_setup_fn_enters_interactive_setup(
    monkeypatch: Any,
) -> None:
    from plugins.platforms.photon import adapter as photon_adapter

    platforms: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    captured: list[argparse.Namespace] = []

    class FakePluginContext:
        def register_platform(self, **kwargs: Any) -> None:
            platforms.append(kwargs)

        def register_cli_command(self, **kwargs: Any) -> None:
            commands.append(kwargs)

    monkeypatch.setattr(photon_cli, "_interactive_setup_already_configured", lambda: False)

    def fake_cmd_setup(args: argparse.Namespace) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(photon_cli, "_cmd_setup", fake_cmd_setup)

    photon_adapter.register(FakePluginContext())
    platforms[0]["setup_fn"]()

    assert platforms[0]["name"] == "photon"
    assert platforms[0]["label"] == "iMessage (via Photon)"
    assert platforms[0]["setup_fn"] is photon_cli.interactive_setup
    assert platforms[0]["cron_deliver_env_var"] == "PHOTON_HOME_CHANNEL"
    assert callable(platforms[0]["standalone_sender_fn"])
    assert commands[0]["name"] == "photon"
    assert commands[0]["setup_fn"] is photon_cli.register_cli
    assert commands[0]["handler_fn"] is photon_cli.dispatch
    assert len(captured) == 1
    assert captured[0].phone is None


@pytest.mark.parametrize(
    (
        "verbose",
        "stdout_present",
        "stdout_absent",
        "stderr_present",
        "stderr_absent",
    ),
    [
        (
            False,
            [],
            ["[logs] Existing logs"],
            [
                "Photon setup stopped: operator phone number is required",
                "repair   : rerun `hermes photon setup '+<country-code><number>'`",
            ],
            ["step     :", "expected :", "observed :", "evidence :"],
        ),
        (
            True,
            ["[logs] Existing logs for this setup:"],
            [],
            [
                "Photon setup stopped: operator phone number is required",
                "step     : operator phone",
                "expected : an E.164 phone number",
                "observed :",
                "evidence :",
            ],
            [],
        ),
    ],
)
def test_cmd_setup_failure_output_respects_verbose(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
    stdout_present: list[str],
    stdout_absent: list[str],
    stderr_present: list[str],
    stderr_absent: list[str],
) -> None:
    args = argparse.Namespace(
        phone=None,
        first_name=None,
        last_name=None,
        email=None,
        no_browser=True,
        skip_adapter_install=True,
        verbose=verbose,
    )

    monkeypatch.setattr(photon_cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(photon_cli.photon_auth, "_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(photon_cli.photon_auth, "setup_lock", lambda: nullcontext())

    def fake_log_paths(_ctx: photon_cli._PhotonSetupContext) -> None:
        print("[logs] Existing logs for this setup:")

    if verbose:
        monkeypatch.setattr(photon_cli, "_print_setup_log_paths", fake_log_paths)
    else:
        monkeypatch.setattr(
            photon_cli,
            "_print_setup_log_paths",
            lambda _ctx: pytest.fail("log paths should be verbose-only"),
        )

    def fail_setup(ctx: photon_cli._PhotonSetupContext) -> None:
        raise photon_cli._failed_invariant(
            ctx,
            step="operator phone",
            summary="operator phone number is required",
            expected="an E.164 phone number",
            observed="missing phone",
            repair=f"rerun `hermes photon setup {photon_cli._PHONE_ARG_PLACEHOLDER}`",
        )

    monkeypatch.setattr(photon_cli, "_run_setup_reconciler", fail_setup)

    rc = photon_cli._cmd_setup(args)

    captured = capsys.readouterr()
    assert rc == 1
    for text in stdout_present:
        assert text in captured.out
    for text in stdout_absent:
        assert text not in captured.out
    for text in stderr_present:
        assert text in captured.err
    for text in stderr_absent:
        assert text not in captured.err


def test_fresh_status_points_to_guided_setup(monkeypatch: Any) -> None:
    monkeypatch.setattr(photon_cli.photon_auth, "load_photon_token", lambda: None)
    monkeypatch.setattr(photon_cli.photon_auth, "load_project_credentials", lambda: (None, None))

    assert (
        photon_cli._next_status_step("✗ missing")
        == "hermes photon setup '+<country-code><number>'"
    )


def test_status_points_to_home_channel_when_missing(monkeypatch: Any) -> None:
    monkeypatch.setattr(photon_cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "secret"),
    )
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    assert (
        photon_cli._next_status_step("✓ installed (spectrum-ts 1.17.1)")
        == "rerun `hermes photon setup '+<country-code><number>'` or set PHOTON_HOME_CHANNEL"
    )


def test_status_prints_home_channel_state(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {
        "PHOTON_PROJECT_NAME": photon_cli._FIXED_PROJECT_NAME,
        "PHOTON_OPERATOR_PHONE": PHONE,
        "PHOTON_HOME_CHANNEL": f"any;-;{PHONE}",
        "PHOTON_HOME_CHANNEL_NAME": "You (iMessage)",
    }
    monkeypatch.setattr(photon_cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(photon_cli.photon_auth, "_env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(photon_cli.photon_auth, "print_credential_summary", lambda emit: emit("credentials"))
    monkeypatch.setattr(photon_cli.photon_auth, "load_photon_token", lambda: "token")
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "secret"),
    )
    monkeypatch.setattr(photon_cli.photon_auth, "list_project_users", lambda *_args: [])
    monkeypatch.setattr(photon_cli.photon_auth, "load_allowed_phone_numbers", lambda: [PHONE])
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda key: env.get(key, ""))
    monkeypatch.setattr(
        photon_cli,
        "_sidecar_dependency_status",
        lambda: "✓ installed (spectrum-ts 1.17.1)",
    )
    monkeypatch.setattr(
        photon_cli,
        "_read_adapter_runtime_state",
        lambda _home: {"health": {"healthy": True, "pid": 123}},
    )
    monkeypatch.setattr(photon_cli, "_dashboard_token_status", lambda: "✓ valid")

    rc = photon_cli._cmd_status(argparse.Namespace())

    out = capsys.readouterr().out
    assert rc == 0
    assert f"home channel        : any;-;{PHONE}" in out
    assert "home channel name   : You (iMessage)" in out
    assert "adapter runtime" not in out
    assert "next step           : send an iMessage to the Photon number" in out


def test_setup_summary_omits_adapter_health(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ctx = _ctx(tmp_path)
    ctx.project_id = "project-id"
    ctx.operator_phone = PHONE
    ctx.assigned_phone_number = "+15550001111"
    monkeypatch.setattr(
        photon_cli,
        "_read_adapter_runtime_state",
        lambda _home: {"health": {"healthy": True, "pid": 123}},
    )
    monkeypatch.setattr(
        photon_cli,
        "_get_env_value",
        lambda key: {
            "PHOTON_HOME_CHANNEL": f"any;-;{PHONE}",
            "PHOTON_HOME_CHANNEL_NAME": "You (iMessage)",
        }.get(key, ""),
    )

    photon_cli._print_setup_reconciled(ctx)

    out = capsys.readouterr().out
    assert "Photon setup complete." in out
    assert "adapter health" not in out
    assert "gateway lifecycle : managed by Hermes core" in out


def test_photon_cli_does_not_import_gateway_lifecycle_internals() -> None:
    plugin_root = Path(photon_cli.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in plugin_root.rglob("*")
        if path.suffix in {".py", ".mjs"} and "node_modules" not in path.parts
    )
    forbidden = [
        "from hermes_cli import gateway",
        "gateway_cli.",
        "load_gateway_config",
        "from gateway import status",
        "from gateway.status import",
        "_get_runtime_status_path",
        "_run_systemctl",
        "launchd_restart",
        "systemd_restart",
        "launchd_start",
        "systemd_start",
        "gateway_windows",
        "is_gateway_running",
        "read_runtime_status",
    ]

    for needle in forbidden:
        assert needle not in source


def test_phone_management_does_not_introduce_local_http_endpoint_code() -> None:
    plugin_root = Path(photon_cli.__file__).parent
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in plugin_root.rglob("*")
        if path.suffix in {".py", ".mjs"} and "node_modules" not in path.parts
    )
    forbidden = [
        "createServer(",
        ".listen(",
        "localhost:",
        "127.0.0.1:",
    ]

    for needle in forbidden:
        assert needle not in sources


@pytest.mark.parametrize(
    ("verbose", "expected", "unexpected"),
    [
        (
            False,
            [
                "Photon setup stopped: test failure",
                "repair   : rerun with --verbose for logs",
            ],
            [
                "step     :",
                "expected :",
                "observed :",
                "evidence :",
                "relevant logs",
                "gateway log line",
            ],
        ),
        (
            True,
            [
                "Photon setup stopped: test failure",
                "step     : test",
                "expected : log tail in verbose mode",
                "observed :",
                "evidence :",
                "repair   : inspect logs",
                "relevant logs",
                "gateway log line",
            ],
            [],
        ),
    ],
)
def test_failed_invariant_log_output_respects_verbose(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
    expected: list[str],
    unexpected: list[str],
) -> None:
    ctx = _ctx(tmp_path)
    ctx.verbose = verbose
    error = photon_cli._failed_invariant(
        ctx,
        step="test",
        summary="test failure",
        expected="log tail in verbose mode" if verbose else "no log tail by default",
        observed="boom",
        repair="inspect logs" if verbose else "rerun with --verbose for logs",
    )

    monkeypatch.setattr(photon_cli, "_stream_setup_logs", lambda _ctx: None)
    monkeypatch.setattr(
        photon_cli,
        "_collect_relevant_log_tail",
        lambda _ctx: {"gateway": ["gateway log line"]},
    )

    photon_cli._finalize_failed_invariant_logs(error, ctx)
    photon_cli._print_failed_invariant(error)

    err = capsys.readouterr().err
    for text in expected:
        assert text in err
    for text in unexpected:
        assert text not in err


def test_phones_list_renders_project_users_and_auth_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )
    monkeypatch.setattr(
        photon_cli.phone_management,
        "list_phones",
        lambda *_args: {
            "users": [
                {
                    "phone": PHONE,
                    "assigned_phone_number": "+15550001111",
                    "user_id": "user-1",
                    "raw": {},
                },
                {
                    "phone": OTHER_PHONE,
                    "assigned_phone_number": "",
                    "user_id": "user-2",
                    "raw": {},
                },
            ]
        },
    )
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_allowed_phone_numbers",
        lambda: [PHONE],
    )
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    rc = photon_cli.dispatch(_parser().parse_args(["phones", "list"]))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Photon phones" in out
    assert f"phone={PHONE}" in out
    assert "assigned=+15550001111" in out
    assert "authorized=yes" in out
    assert f"phone={OTHER_PHONE}" in out
    assert "authorized=no" in out


def test_phones_add_creates_sdk_user_then_appends_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli import config as hermes_config

    calls: list[str] = []
    env = {"PHOTON_ALLOWED_USERS": OTHER_PHONE}

    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )

    def fake_add(_project_id: str, _project_secret: str, phone: str) -> dict[str, Any]:
        calls.append(f"remote:{phone}")
        assert env["PHOTON_ALLOWED_USERS"] == OTHER_PHONE
        return {
            "user": {
                "phone": phone,
                "assigned_phone_number": "+15550001111",
                "user_id": "user-1",
            }
        }

    def fake_save(key: str, value: str) -> None:
        calls.append(f"save:{value}")
        env[key] = value

    monkeypatch.setattr(photon_cli.phone_management, "add_phone", fake_add)
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_allowed_phone_numbers",
        lambda: [item for item in env.get("PHOTON_ALLOWED_USERS", "").split(",") if item],
    )
    monkeypatch.setattr(hermes_config, "save_env_value", fake_save)
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    rc = photon_cli.dispatch(_parser().parse_args(["phones", "add", PHONE]))

    assert rc == 0
    assert calls == [f"remote:{PHONE}", f"save:{OTHER_PHONE},{PHONE}"]
    assert env["PHOTON_ALLOWED_USERS"] == f"{OTHER_PHONE},{PHONE}"
    assert "assigned number" in capsys.readouterr().out


@pytest.mark.parametrize(
    (
        "argv",
        "remote_method",
        "remote_error",
        "expected_err",
        "unexpected_err",
    ),
    [
        (
            ["phones", "add", PHONE],
            "add_phone",
            phone_management.PhotonPhoneManagementError(
                code="PHONE_EXISTS",
                message="already exists",
            ),
            ["already exists"],
            [],
        ),
        (
            ["phones", "add", PHONE],
            "add_phone",
            phone_management.PhotonPhoneManagementError(
                code="SHARED_USER_LIMIT",
                message="Maximum number of shared users (10) reached",
                detail='{"message":"Maximum number of shared users (10) reached"}',
                status=409,
            ),
            [
                "Photon free plan shared-user limit reached",
                "https://app.photon.codes/dashboard",
                "hermes-agent",
                "Billing -> Upgrade plan",
            ],
            ["Maximum number of shared users"],
        ),
        (
            ["phones", "add", PHONE, "--verbose"],
            "add_phone",
            phone_management.PhotonPhoneManagementError(
                code="SHARED_USER_LIMIT",
                message="Maximum number of shared users (10) reached",
                detail='{"message":"Maximum number of shared users (10) reached"}',
                status=409,
            ),
            [
                "Photon free plan shared-user limit reached",
                "Maximum number of shared users",
            ],
            [],
        ),
        (
            ["phones", "remove", PHONE],
            "remove_phone",
            phone_management.PhotonPhoneManagementError(
                code="PHONE_NOT_FOUND",
                message="does not exist",
            ),
            ["does not exist"],
            [],
        ),
    ],
)
def test_phones_management_errors_do_not_mutate_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    remote_method: str,
    remote_error: phone_management.PhotonPhoneManagementError,
    expected_err: list[str],
    unexpected_err: list[str],
) -> None:
    from hermes_cli import config as hermes_config

    save_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )

    def fail_remote(*_args: Any, **_kwargs: Any) -> None:
        raise remote_error

    monkeypatch.setattr(photon_cli.phone_management, remote_method, fail_remote)
    monkeypatch.setattr(hermes_config, "save_env_value", lambda *args: save_calls.append(args))

    rc = photon_cli.dispatch(_parser().parse_args(argv))

    err = capsys.readouterr().err
    assert rc == 1
    for text in expected_err:
        assert text in err
    for text in unexpected_err:
        assert text not in err
    assert save_calls == []


def test_phones_remove_removes_sdk_user_then_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as hermes_config

    calls: list[str] = []
    env = {"PHOTON_ALLOWED_USERS": f"{PHONE},{OTHER_PHONE}"}
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_project_credentials",
        lambda: ("project-id", "project-secret"),
    )

    def fake_remove(_project_id: str, _project_secret: str, phone: str) -> dict[str, Any]:
        calls.append(f"remote:{phone}")
        assert env["PHOTON_ALLOWED_USERS"] == f"{PHONE},{OTHER_PHONE}"
        return {"user": {"phone": phone, "user_id": "user-1"}}

    def fake_save(key: str, value: str) -> None:
        calls.append(f"save:{value}")
        env[key] = value

    monkeypatch.setattr(photon_cli.phone_management, "remove_phone", fake_remove)
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_allowed_phone_numbers",
        lambda: [item for item in env.get("PHOTON_ALLOWED_USERS", "").split(",") if item],
    )
    monkeypatch.setattr(hermes_config, "save_env_value", fake_save)
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    rc = photon_cli.dispatch(_parser().parse_args(["phones", "remove", PHONE]))

    assert rc == 0
    assert calls == [f"remote:{PHONE}", f"save:{OTHER_PHONE}"]
    assert env["PHOTON_ALLOWED_USERS"] == OTHER_PHONE


def test_status_phone_summary_does_not_use_operator_phone_source_of_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        photon_cli.photon_auth,
        "load_allowed_phone_numbers",
        lambda: [OTHER_PHONE],
    )
    monkeypatch.setattr(photon_cli, "_get_env_value", lambda _key: "")

    summary = photon_cli._format_phones_summary([
        {"phoneNumber": PHONE, "assignedPhoneNumber": "+15550001111"},
        {"phoneNumber": OTHER_PHONE, "assignedPhoneNumber": "+15550002222"},
    ])

    assert summary == "2 project user(s); 1 authorized in Hermes"
