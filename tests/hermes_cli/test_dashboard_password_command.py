from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.subcommands.dashboard import build_dashboard_parser
from plugins.dashboard_auth.basic import _verify_password


@pytest.fixture
def hermes_home(tmp_path: Path):
    home = tmp_path / "profile"
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _parser(handler):
    parser = argparse.ArgumentParser()
    build_dashboard_parser(
        parser.add_subparsers(dest="command"),
        cmd_dashboard=lambda _args: None,
        cmd_dashboard_register=lambda _args: None,
        cmd_dashboard_password=handler,
    )
    return parser


def test_password_rotate_parser_never_accepts_plaintext_argv():
    handler = object()
    parsed = _parser(handler).parse_args(
        ["dashboard", "password", "rotate", "--generate", "--username", "operator"]
    )

    assert parsed.func is handler
    assert parsed.generate is True
    assert parsed.username == "operator"
    with pytest.raises(SystemExit):
        _parser(handler).parse_args(
            ["dashboard", "password", "rotate", "--password", "leak"]
        )


def test_generate_prints_once_and_persists_only_hash(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from hermes_cli.dashboard_password import cmd_dashboard_password

    generated = "generated-password-value"
    generated_values = iter([generated, "generated-session-secret"])
    monkeypatch.setattr(
        "hermes_cli.dashboard_password.secrets.token_urlsafe",
        lambda _n: next(generated_values),
    )
    args = argparse.Namespace(generate=True, username="admin", force_logout=False)

    cmd_dashboard_password(args)

    output = capsys.readouterr().out
    assert output.count(generated) == 1
    env_text = (hermes_home / ".env").read_text(encoding="utf-8")
    assert generated not in env_text
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=" not in env_text
    values = _load_dotenv(hermes_home)
    assert values["HERMES_DASHBOARD_BASIC_AUTH_USERNAME"] == "admin"
    assert _verify_password(generated, values["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH"])
    assert values["HERMES_DASHBOARD_BASIC_AUTH_SECRET"]


def test_interactive_rotation_confirms_password_and_preserves_sessions(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
):
    from hermes_cli.dashboard_password import cmd_dashboard_password
    from hermes_cli.config import save_env_values

    save_env_values(
        {
            "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "admin",
            "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH": "old-hash",
            "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "stale-plaintext",
            "HERMES_DASHBOARD_BASIC_AUTH_SECRET": "stable-session-secret",
        }
    )
    answers = iter(["new-password", "new-password"])
    monkeypatch.setattr("hermes_cli.dashboard_password.getpass.getpass", lambda _prompt: next(answers))
    monkeypatch.setattr("hermes_cli.dashboard_password._is_interactive", lambda: True)

    cmd_dashboard_password(
        argparse.Namespace(generate=False, username=None, force_logout=False)
    )

    values = _load_dotenv(hermes_home)
    assert values["HERMES_DASHBOARD_BASIC_AUTH_SECRET"] == "stable-session-secret"
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" not in values
    assert _verify_password("new-password", values["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH"])
    assert not _verify_password("stale-plaintext", values["HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH"])


def test_force_logout_rotates_existing_session_secret(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
):
    from hermes_cli.dashboard_password import cmd_dashboard_password
    from hermes_cli.config import save_env_values

    save_env_values(
        {
            "HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "admin",
            "HERMES_DASHBOARD_BASIC_AUTH_SECRET": "old-session-secret",
        }
    )
    generated_values = iter(["new-password", "new-session-secret"])
    monkeypatch.setattr(
        "hermes_cli.dashboard_password.secrets.token_urlsafe",
        lambda _n: next(generated_values),
    )

    cmd_dashboard_password(
        argparse.Namespace(generate=True, username=None, force_logout=True)
    )

    assert _load_dotenv(hermes_home)["HERMES_DASHBOARD_BASIC_AUTH_SECRET"] == "new-session-secret"


def test_noninteractive_prompt_is_rejected_without_generate(hermes_home: Path, monkeypatch):
    from hermes_cli.dashboard_password import cmd_dashboard_password

    monkeypatch.setattr("hermes_cli.dashboard_password._is_interactive", lambda: False)
    with pytest.raises(SystemExit, match="--generate"):
        cmd_dashboard_password(
            argparse.Namespace(generate=False, username=None, force_logout=False)
        )
    assert not (hermes_home / ".env").exists()


def test_env_batch_lock_preserves_concurrent_updates(hermes_home: Path):
    script = (
        "import sys; from hermes_cli.config import save_env_values; "
        "save_env_values({sys.argv[1]: sys.argv[2]})"
    )
    env = dict(os.environ, HERMES_HOME=str(hermes_home))
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, f"DASHBOARD_TEST_{index}", str(index)],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
        )
        for index in range(8)
    ]

    assert [process.wait(timeout=30) for process in processes] == [0] * 8
    values = _load_dotenv(hermes_home)
    assert {values[f"DASHBOARD_TEST_{index}"] for index in range(8)} == {
        str(index) for index in range(8)
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_rotation_preserves_existing_env_permissions(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
):
    from hermes_cli.dashboard_password import cmd_dashboard_password

    hermes_home.mkdir(parents=True)
    env_path = hermes_home / ".env"
    env_path.write_text("OTHER=value\n", encoding="utf-8")
    env_path.chmod(0o640)
    monkeypatch.setattr(
        "hermes_cli.dashboard_password.secrets.token_urlsafe", lambda _n: "generated"
    )

    cmd_dashboard_password(
        argparse.Namespace(generate=True, username="admin", force_logout=False)
    )

    assert stat.S_IMODE(env_path.stat().st_mode) == 0o640


def _load_dotenv(home: Path) -> dict[str, str]:
    from hermes_cli.config import load_env

    assert home.exists()
    return load_env()
