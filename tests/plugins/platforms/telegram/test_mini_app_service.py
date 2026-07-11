from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from plugins.platforms.telegram.mini_app import cli, run, service


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        cli.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
    )


def test_owner_validation_fails_closed() -> None:
    assert cli.validate_owner_ids(["123", "456,123"]) == ["123", "456"]
    for owners in ([], ["*"], ["-100123"], ["@owner"], ["group"]):
        with pytest.raises(cli.MiniAppSetupError):
            cli.validate_owner_ids(owners)


def test_public_url_rejects_non_https_paths_and_quick_tunnels() -> None:
    assert (
        cli.validate_public_url("https://mini.example.com/")
        == "https://mini.example.com"
    )
    for value in (
        "http://mini.example.com",
        "https://mini.example.com/path",
        "https://random.trycloudflare.com",
        "https://user:secret@mini.example.com",
    ):
        with pytest.raises(cli.MiniAppSetupError):
            cli.validate_public_url(value)


def test_public_url_rejects_ip_literals_and_non_public_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(cli.MiniAppSetupError, match="IP literal"):
        cli.validate_public_url("https://127.0.0.1")
    monkeypatch.setattr(
        cli.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("10.0.0.8", port))],
    )
    with pytest.raises(cli.MiniAppSetupError, match="non-public"):
        cli.validate_public_url("https://internal.example.com")


def test_default_port_is_profile_specific(tmp_path: Path) -> None:
    assert cli.default_listen_port(tmp_path / "one") == cli.default_listen_port(
        tmp_path / "one"
    )
    assert cli.default_listen_port(tmp_path / "one") != cli.default_listen_port(
        tmp_path / "two"
    )


def test_setup_writes_only_dedicated_credentials_and_probes_before_menu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {})
    monkeypatch.setattr(
        cli,
        "get_env_value_prefer_dotenv",
        lambda key: "123456:abcdefghijklmnopqrstuvwxyzABCDE",
    )
    monkeypatch.setattr(
        cli,
        "_menu_snapshot",
        lambda token, owners: {
            "global": {"type": "commands"},
            **{owner: {"type": "commands"} for owner in owners},
        },
    )
    monkeypatch.setattr(cli.service, "status", lambda home: (False, "not installed"))
    monkeypatch.setattr(cli.service, "_platform", lambda: "systemd")
    monkeypatch.setattr(
        cli.service, "systemd_unit_path", lambda home: tmp_path / "unit"
    )
    monkeypatch.setattr(
        cli, "_persist_behavior", lambda **kwargs: events.append("config")
    )
    monkeypatch.setattr(cli.service, "require_install_support", lambda: None)
    monkeypatch.setattr(cli.service, "install", lambda home: events.append("install"))
    monkeypatch.setattr(cli.service, "start", lambda home: events.append("start"))
    monkeypatch.setattr(cli, "_probe", lambda url: events.append("probe"))
    monkeypatch.setattr(
        cli,
        "_set_menu_button",
        lambda token, url, chat_id=None: events.append(
            "menu" if chat_id is None else f"menu:{chat_id}"
        ),
    )

    cli.setup(
        public_url="https://mini.example.com",
        owner_values=["111", "222"],
        listen_port=8787,
    )

    env_path = service.paths_for(tmp_path).env
    assert env_path.read_text() == (
        "TELEGRAM_BOT_TOKEN=123456:abcdefghijklmnopqrstuvwxyzABCDE\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111,222\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
    )
    assert env_path.stat().st_mode & 0o777 == 0o600
    assert service.paths_for(tmp_path).state.stat().st_mode & 0o777 == 0o600
    assert events == [
        "config",
        "install",
        "start",
        "probe",
        "menu",
        "menu:111",
        "menu:222",
    ]


def test_setup_failure_rolls_back_service_files_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    previous = {"platforms": {"telegram": {"enabled": True}}}
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: previous)
    monkeypatch.setattr(
        cli,
        "get_env_value_prefer_dotenv",
        lambda key: (
            "123456:abcdefghijklmnopqrstuvwxyzABCDE"
            if key == "TELEGRAM_BOT_TOKEN"
            else None
        ),
    )
    monkeypatch.setattr(cli.service, "require_install_support", lambda: None)
    monkeypatch.setattr(cli.service, "_platform", lambda: "systemd")
    monkeypatch.setattr(
        cli,
        "_menu_snapshot",
        lambda token, owners: {
            "global": {"type": "commands"},
            **{owner: {"type": "commands"} for owner in owners},
        },
    )
    monkeypatch.setattr(cli.service, "status", lambda home: (False, "not installed"))
    monkeypatch.setattr(
        cli.service, "systemd_unit_path", lambda home: tmp_path / "unit"
    )
    monkeypatch.setattr(
        cli, "_restore_menus", lambda token, snapshot: events.append("restore-menus")
    )
    monkeypatch.setattr(
        cli, "_persist_behavior", lambda **kwargs: events.append("config")
    )
    monkeypatch.setattr(cli.service, "install", lambda home: events.append("install"))
    monkeypatch.setattr(cli.service, "start", lambda home: events.append("start"))
    monkeypatch.setattr(cli.service, "stop", lambda home: events.append("stop"))
    monkeypatch.setattr(
        cli.service, "uninstall", lambda home: events.append("uninstall")
    )
    monkeypatch.setattr(
        cli, "_probe", lambda url: (_ for _ in ()).throw(RuntimeError("unsafe"))
    )
    monkeypatch.setattr(
        cli, "save_config", lambda config: events.append(("restore", config))
    )

    with pytest.raises(RuntimeError, match="unsafe"):
        cli.setup(
            public_url="https://mini.example.com",
            owner_values=["111"],
            listen_port=8787,
        )

    paths = service.paths_for(tmp_path)
    assert not paths.env.exists()
    assert not paths.state.exists()
    assert events == [
        "config",
        "install",
        "start",
        "restore-menus",
        "stop",
        "uninstall",
        ("restore", previous),
    ]


def test_failed_active_reconfiguration_restores_files_and_restarts_previous_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    old_token = "654321:ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"
    new_token = "123456:abcdefghijklmnopqrstuvwxyzABCDE"
    old_env = (
        f"TELEGRAM_BOT_TOKEN={old_token}\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://old.example.com\n"
    )
    old_state = json.dumps({
        "owners": ["111"],
        "listen_port": 9001,
        "menu_backups": {
            "global": {"type": "commands"},
            "111": {"type": "commands"},
        },
    })
    paths.env.write_text(old_env)
    paths.state.write_text(old_state)
    unit = tmp_path / "installed.service"
    unit.touch()
    events: list[str] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {"old": True})
    monkeypatch.setattr(
        cli,
        "get_env_value_prefer_dotenv",
        lambda key: new_token,
    )
    monkeypatch.setattr(cli.service, "require_install_support", lambda: None)
    monkeypatch.setattr(cli.service, "_platform", lambda: "systemd")
    monkeypatch.setattr(cli.service, "systemd_unit_path", lambda home: unit)
    monkeypatch.setattr(cli.service, "status", lambda home: (True, "active"))
    monkeypatch.setattr(
        cli,
        "_menu_snapshot",
        lambda token, owners: {
            "global": {"type": "commands"},
            **{owner: {"type": "commands"} for owner in owners},
        },
    )
    monkeypatch.setattr(
        cli,
        "_restore_menus",
        lambda token, menus: events.append(f"restore-menus:{token}"),
    )
    monkeypatch.setattr(
        cli, "_persist_behavior", lambda **kwargs: events.append("config")
    )
    monkeypatch.setattr(cli.service, "install", lambda home: events.append("install"))
    monkeypatch.setattr(cli.service, "restart", lambda home: events.append("restart"))
    monkeypatch.setattr(cli.service, "start", lambda home: events.append("start-old"))
    monkeypatch.setattr(cli.service, "stop", lambda home: events.append("stop"))
    monkeypatch.setattr(
        cli.service, "uninstall", lambda home: events.append("uninstall")
    )
    monkeypatch.setattr(
        cli, "save_config", lambda config: events.append("restore-config")
    )
    monkeypatch.setattr(
        cli, "_probe", lambda url: (_ for _ in ()).throw(RuntimeError("probe failed"))
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        cli.setup(
            public_url="https://new.example.com",
            owner_values=["111"],
            listen_port=9002,
        )

    assert paths.env.read_text() == old_env
    assert paths.state.read_text() == old_state
    assert events == [
        "config",
        "install",
        "restart",
        f"restore-menus:{new_token}",
        f"restore-menus:{old_token}",
        "stop",
        "uninstall",
        "restore-config",
        "install",
        "start-old",
    ]


def test_unsupported_setup_configures_foreground_without_native_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    events: list[str] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {})
    monkeypatch.setattr(
        cli,
        "get_env_value_prefer_dotenv",
        lambda key: "123456:abcdefghijklmnopqrstuvwxyzABCDE",
    )
    monkeypatch.setattr(cli.service, "_platform", lambda: "unsupported")
    monkeypatch.setattr(
        cli,
        "_menu_snapshot",
        lambda token, owners: {
            "global": {"type": "commands"},
            **{owner: {"type": "commands"} for owner in owners},
        },
    )
    monkeypatch.setattr(
        cli, "_persist_behavior", lambda **kwargs: events.append("config")
    )
    monkeypatch.setattr(
        cli,
        "_set_menu_button",
        lambda token, url, chat_id=None: events.append(
            "menu" if chat_id is None else f"menu:{chat_id}"
        ),
    )
    for name in ("require_install_support", "install", "start", "restart"):
        monkeypatch.setattr(
            cli.service,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"foreground setup called {_name}"
            ),
        )
    monkeypatch.setattr(
        cli, "_probe", lambda *args, **kwargs: pytest.fail("foreground setup probed")
    )

    cli.setup(
        public_url="https://mini.example.com",
        owner_values=["111"],
        listen_port=8787,
    )

    assert events == ["config", "menu", "menu:111"]
    assert service.paths_for(tmp_path).env.is_file()
    output = capsys.readouterr().out
    assert "configured" in output
    assert "hermes gateway mini-app serve" in output


def test_setup_rejects_platform_without_clean_foreground_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {})
    monkeypatch.setattr(
        cli,
        "get_env_value_prefer_dotenv",
        lambda key: "123456:abcdefghijklmnopqrstuvwxyzABCDE",
    )
    monkeypatch.setattr(cli.service, "_platform", lambda: "unsupported")
    monkeypatch.setattr(cli, "_supports_foreground", lambda: False)

    with pytest.raises(cli.MiniAppSetupError, match="Windows is not supported"):
        cli.setup(
            public_url="https://mini.example.com",
            owner_values=["111"],
            listen_port=8787,
        )

    assert not service.paths_for(tmp_path).root.exists()


def test_bot_token_rotation_restores_old_bot_and_keeps_new_bot_backups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    old_token = "654321:ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"
    new_token = "123456:abcdefghijklmnopqrstuvwxyzABCDE"
    old_backups = {
        "global": {"type": "commands"},
        "111": {"type": "commands"},
    }
    paths.env.write_text(
        f"TELEGRAM_BOT_TOKEN={old_token}\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://old.example.com\n"
    )
    paths.state.write_text(
        json.dumps({
            "owners": ["111"],
            "listen_port": 9001,
            "menu_backups": old_backups,
        })
    )
    unit = tmp_path / "installed.service"
    unit.touch()
    events: list[object] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {})
    monkeypatch.setattr(cli, "get_env_value_prefer_dotenv", lambda key: new_token)
    monkeypatch.setattr(cli.service, "_platform", lambda: "systemd")
    monkeypatch.setattr(cli.service, "require_install_support", lambda: None)
    monkeypatch.setattr(cli.service, "systemd_unit_path", lambda home: unit)
    monkeypatch.setattr(cli.service, "status", lambda home: (True, "active"))
    monkeypatch.setattr(cli.service, "install", lambda home: events.append("install"))
    monkeypatch.setattr(cli.service, "restart", lambda home: events.append("restart"))
    monkeypatch.setattr(
        cli, "_persist_behavior", lambda **kwargs: events.append("config")
    )
    monkeypatch.setattr(cli, "_probe", lambda url: events.append("probe"))

    def snapshot(token, owners):
        events.append(("snapshot", token, tuple(owners)))
        return {
            "global": {"type": "default", "bot": token},
            **{owner: {"type": "default", "bot": token} for owner in owners},
        }

    monkeypatch.setattr(cli, "_menu_snapshot", snapshot)
    monkeypatch.setattr(
        cli,
        "_set_menu_button",
        lambda token, url, chat_id=None: events.append(("set", token, chat_id)),
    )
    monkeypatch.setattr(
        cli,
        "_restore_menus",
        lambda token, menus: events.append(("restore", token, menus)),
    )

    cli.setup(
        public_url="https://new.example.com",
        owner_values=["222"],
        listen_port=9002,
    )

    new_state = json.loads(paths.state.read_text())
    assert new_state["menu_backups"]["global"]["bot"] == new_token
    assert new_state["menu_backups"]["222"]["bot"] == new_token
    assert ("restore", old_token, old_backups) in events
    assert ("set", new_token, None) in events
    assert ("set", new_token, "222") in events


def test_canonical_config_location_and_runtime_port() -> None:
    config: dict = {}
    mini_app = cli._mini_app_config(config)
    mini_app["listen_port"] = 9443
    assert config == {
        "platforms": {"telegram": {"extra": {"mini_app": {"listen_port": 9443}}}}
    }
    assert run._listen_port({"listen_port": 9443}) == 9443


def test_dedicated_env_loader_rejects_inherited_configuration(tmp_path: Path) -> None:
    env_path = tmp_path / "service.env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=token\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=123\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
        "ANTHROPIC_API_KEY=must-not-load\n"
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        run._load_dedicated_env(env_path)


def test_runner_sanitizes_inherited_credentials_before_loading_dedicated_env(
    tmp_path: Path,
) -> None:
    original = dict(os.environ)
    try:
        os.environ.update({
            "HERMES_HOME": str(tmp_path),
            "ANTHROPIC_API_KEY": "drop-me",
            "OPENAI_API_KEY": "drop-me-too",
            "MCP_SECRET_TOKEN": "also-drop-me",
            "PATH": original.get("PATH", "/usr/bin"),
        })
        env_path = tmp_path / "service.env"
        env_path.write_text(
            "TELEGRAM_BOT_TOKEN=dedicated-token\n"
            "TELEGRAM_MINI_APP_OWNER_IDS=123\n"
            "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
        )
        assert run._sanitize_environment() == tmp_path.resolve()
        run._load_dedicated_env(env_path)
        assert "ANTHROPIC_API_KEY" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ
        assert "MCP_SECRET_TOKEN" not in os.environ
        assert os.environ["TELEGRAM_BOT_TOKEN"] == "dedicated-token"
        assert os.environ["TELEGRAM_MINI_APP_PUBLIC_URL"] == "https://mini.example.com"
    finally:
        os.environ.clear()
        os.environ.update(original)


def test_clean_runner_drops_inherited_secrets_and_ignores_home_shadow_package(
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "home"
    shadow = hermes_home / "plugins" / "platforms" / "telegram" / "mini_app"
    shadow.mkdir(parents=True)
    for parent in (shadow, *shadow.parents[:3]):
        (parent / "__init__.py").write_text("")
    bad_marker = tmp_path / "shadow-imported"
    (shadow / "run.py").write_text(
        f"from pathlib import Path\nPath({str(bad_marker)!r}).write_text('unsafe')\n"
    )

    repo = Path(__file__).resolve().parents[4]
    python = (
        repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    assert python.is_file(), "canonical test runner virtualenv is required"
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "drop-me",
        "OPENAI_API_KEY": "drop-me-too",
        "MCP_SECRET_TOKEN": "also-drop-me",
        "PYTHONPATH": str(hermes_home),
    }
    result = subprocess.run(
        service.service_command(hermes_home, python_executable=python),
        cwd=hermes_home,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Mini App credentials are missing" in result.stderr
    assert not bad_marker.exists()


def test_probe_checks_existing_protected_route(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    class Healthy:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://mini.example.com/health"

    def urlopen(request, timeout):
        url = request.full_url if hasattr(request, "full_url") else request
        requested.append(url)
        if url.endswith("/api/me"):
            raise urllib.error.HTTPError(url, 401, "unauthorized", {}, None)
        return Healthy()

    class Opener:
        open = staticmethod(urlopen)

    monkeypatch.setattr(cli, "_direct_opener", lambda: Opener())
    cli._probe("https://mini.example.com", attempts=1, delay=0)
    assert requested == [
        "https://mini.example.com/health",
        "https://mini.example.com/api/me",
    ]


def test_probe_rejects_redirects_and_cross_origin_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        cli._NoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "https://evil.example"
        )
        is None
    )

    class CrossOrigin:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://evil.example/health"

    class Opener:
        def open(self, request, timeout):
            return CrossOrigin()

    monkeypatch.setattr(cli, "_direct_opener", lambda: Opener())
    with pytest.raises(cli.MiniAppSetupError, match="health probe failed"):
        cli._probe("https://mini.example.com", attempts=1, delay=0)


def test_menu_button_registers_global_and_owner_specific_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict] = []

    class Accepted:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return Accepted()

    class Opener:
        open = staticmethod(urlopen)

    monkeypatch.setattr(cli, "_direct_opener", lambda: Opener())
    cli._set_menu_button(
        "123:abcdefghijklmnopqrstuvwxyzABCDEFG", "https://mini.example.com"
    )
    cli._set_menu_button(
        "123:abcdefghijklmnopqrstuvwxyzABCDEFG",
        "https://mini.example.com",
        chat_id="111",
    )

    assert "chat_id" not in payloads[0]
    assert payloads[1]["chat_id"] == 111
    assert payloads[0]["menu_button"] == payloads[1]["menu_button"]


def test_native_install_fails_clearly_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    paths.env.write_text("TELEGRAM_BOT_TOKEN=x\nTELEGRAM_MINI_APP_OWNER_IDS=1\n")
    monkeypatch.setattr(service, "_platform", lambda: "unsupported")
    with pytest.raises(service.MiniAppServiceError, match="foreground"):
        service.install(tmp_path)


def test_systemd_uninstall_preserves_unit_when_failed_disable_is_still_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit = tmp_path / "mini-app.service"
    unit.write_text("unit")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(service, "_platform", lambda: "systemd")
    monkeypatch.setattr(service, "systemd_unit_path", lambda home: unit)

    def run(argv, *, check=True):
        calls.append(tuple(argv))
        if "disable" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "stop failed")
        if "is-active" in argv:
            return subprocess.CompletedProcess(argv, 0, "active\n", "")
        pytest.fail(f"unexpected systemctl call: {argv}")

    monkeypatch.setattr(service, "_run", run)

    with pytest.raises(service.MiniAppServiceError, match="stop failed"):
        service.uninstall(tmp_path)

    assert unit.read_text() == "unit"
    assert not any("daemon-reload" in call for call in calls)


@pytest.mark.parametrize(("state", "returncode"), (("inactive", 3), ("unknown", 4)))
def test_systemd_uninstall_tolerates_proven_inactive_or_absent_unit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: str,
    returncode: int,
) -> None:
    unit = tmp_path / "mini-app.service"
    unit.write_text("unit")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(service, "_platform", lambda: "systemd")
    monkeypatch.setattr(service, "systemd_unit_path", lambda home: unit)

    def run(argv, *, check=True):
        calls.append(tuple(argv))
        if "disable" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "already stopped")
        if "is-active" in argv:
            return subprocess.CompletedProcess(argv, returncode, f"{state}\n", "")
        if "daemon-reload" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        pytest.fail(f"unexpected systemctl call: {argv}")

    monkeypatch.setattr(service, "_run", run)

    service.uninstall(tmp_path)

    assert not unit.exists()
    assert any("daemon-reload" in call for call in calls)


def test_generated_services_are_profile_scoped_and_do_not_source_gateway_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service, "_profile_suffix", lambda home: "work")
    default_root = tmp_path
    (default_root / "profiles" / "one").mkdir(parents=True)
    (default_root / "profiles" / "two").mkdir(parents=True)
    memories = tmp_path / "memories"
    memories.mkdir()
    (memories / "USER.md").write_text("user")
    (memories / "MEMORY.md").write_text("memory")
    (memories / "SECRET.md").write_text("must stay hidden")
    skill = tmp_path / "skills" / "safe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: safe\n---\n")
    (skill / "secret.txt").write_text("must stay hidden")
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: default_root
    )
    unit = service._systemd_unit(tmp_path)
    assert (
        "hermes-telegram-mini-app-work" not in unit
    )  # name belongs to unit file, not ExecStart
    assert '"plugins.platforms.telegram.mini_app.run"' in unit
    assert '"-i"' in unit
    assert '"-I"' in unit
    assert "hermes_cli.main" not in unit
    assert "EnvironmentFile" not in unit
    assert "Environment=" not in unit
    assert "WorkingDirectory=/" in unit
    assert "0.0.0.0" not in unit
    assert "UMask=0077" in unit
    assert "ProtectControlGroups=true" in unit
    assert "ProtectKernelModules=true" in unit
    assert "ProtectKernelTunables=true" in unit
    assert "ProtectKernelLogs=true" in unit
    assert "ProtectClock=true" in unit
    assert "ProtectHostname=true" in unit
    assert "RestrictSUIDSGID=true" in unit
    assert "RestrictRealtime=true" in unit
    assert "RestrictNamespaces=true" in unit
    assert "LockPersonality=true" in unit
    assert "RemoveIPC=true" in unit
    assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6" in unit
    assert "IPAddressDeny=any" in unit
    assert "IPAddressAllow=localhost" in unit
    assert "SystemCallFilter=@system-service" in unit
    assert "PrivateDevices=true" in unit
    assert "ProtectHome=tmpfs" in unit
    assert "InaccessiblePaths=" not in unit
    assert f'TemporaryFileSystem="{default_root.resolve()}:ro"' in unit
    assert "BindReadOnlyPaths=" in unit
    for safe_path in (
        memories / "USER.md",
        memories / "MEMORY.md",
        skill / "SKILL.md",
        tmp_path / "state.db",
        tmp_path / "state.db-wal",
        tmp_path / "state.db-shm",
        tmp_path / "gateway_state.json",
        default_root / "kanban" / "current",
        default_root / "kanban.db",
        default_root / "kanban.db-wal",
        default_root / "kanban.db-shm",
    ):
        assert str(safe_path.resolve()) in unit
    bind_line = next(
        line for line in unit.splitlines() if line.startswith("BindReadOnlyPaths=")
    )
    bind_targets = {
        token.removeprefix("-") for token in shlex.split(bind_line.split("=", 1)[1])
    }
    assert str(memories.resolve()) not in bind_targets
    assert str((tmp_path / "skills").resolve()) not in bind_targets
    assert str((memories / "SECRET.md").resolve()) not in bind_targets
    assert str((skill / "secret.txt").resolve()) not in bind_targets
    # Credential-bearing parents and leaf paths are absent by construction;
    # the empty home/custom-root namespaces hide existing and future secrets.
    for secret_path in (
        default_root / ".env",
        default_root / "auth.json",
        default_root / "platforms" / "pairing",
        default_root / "slack_tokens.json",
        default_root / "google_chat_user_tokens",
        default_root / "workspace" / "meetings" / "auth.json",
        default_root / "profiles" / "one",
        Path.home() / ".claude",
        Path.home() / ".modal.toml",
    ):
        assert str(secret_path.resolve()) not in unit
    assert f'BindPaths="{service.paths_for(tmp_path).root.resolve()}"' in unit
    assert f'ReadWritePaths="{service.paths_for(tmp_path).root.resolve()}"' in unit
    assert service.service_name(tmp_path) == "hermes-telegram-mini-app-work"


def test_custom_root_credentials_created_after_start_remain_outside_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "custom-root"
    root.mkdir()
    home = root / "profiles" / "work"
    home.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    unit = service._systemd_unit(home)
    (root / "auth.json").write_text('{"access_token":"later-secret"}')
    (home / "slack_tokens.json").write_text('{"token":"later-secret"}')

    assert f'TemporaryFileSystem="{root.resolve()}:ro"' in unit
    assert str((root / "auth.json").resolve()) not in unit
    assert str((home / "slack_tokens.json").resolve()) not in unit
    assert str((home / "state.db").resolve()) in unit
    assert str(service.paths_for(home).root.resolve()) in unit


def test_runtime_bind_cannot_reexpose_entire_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "data-root"
    root.mkdir()
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(service.sys, "prefix", str(root))
    monkeypatch.setattr(service.sys, "base_prefix", "/usr")

    with pytest.raises(
        service.MiniAppServiceError, match="contains the Hermes data root"
    ):
        service._runtime_read_paths(root)


@pytest.mark.parametrize("relative", ("state.db", "state.db-wal", "state.db-shm"))
def test_session_storage_symlinks_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative: str
) -> None:
    external = tmp_path / "external-secret"
    external.write_text("secret")
    home = tmp_path / "home"
    home.mkdir()
    (home / relative).symlink_to(external)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: home)

    with pytest.raises(service.MiniAppServiceError, match="must not contain symlinks"):
        service._systemd_unit(home)


def test_kanban_selection_and_database_symlinks_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    home = root / "profiles" / "work"
    home.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "current").write_text("secret")
    kanban = root / "kanban"
    kanban.mkdir()
    (kanban / "current").symlink_to(external / "current")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    with pytest.raises(service.MiniAppServiceError, match="must not contain symlinks"):
        service._systemd_unit(home)

    (kanban / "current").unlink()
    (kanban / "current").write_text("board-one")
    board = root / "kanban" / "boards" / "board-one"
    board.mkdir(parents=True)
    (external / "kanban.db").write_text("secret")
    (board / "kanban.db").symlink_to(external / "kanban.db")

    with pytest.raises(service.MiniAppServiceError, match="must not contain symlinks"):
        service._systemd_unit(home)


@pytest.mark.parametrize("relative", ("kanban.db-wal", "kanban.db-shm"))
def test_kanban_sidecar_symlinks_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    external = tmp_path / "external-secret"
    external.write_text("secret")
    (root / relative).symlink_to(external)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    with pytest.raises(service.MiniAppServiceError, match="must not contain symlinks"):
        service._systemd_unit(root)


@pytest.mark.parametrize(
    "relative",
    (
        "state.db",
        "state.db-wal",
        "state.db-shm",
        "memories/USER.md",
        "memories/MEMORY.md",
        "skills/safe/SKILL.md",
        "kanban/current",
        "kanban.db",
        "kanban.db-wal",
        "kanban.db-shm",
    ),
)
def test_allowlisted_data_leaves_must_be_regular_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, relative: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / relative).mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: home)

    with pytest.raises(service.MiniAppServiceError, match="must be a regular file"):
        service._systemd_unit(home)


def test_memory_and_skill_symlink_escapes_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "USER.md").write_text("secret")
    (home / "memories").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: home)

    with pytest.raises(service.MiniAppServiceError, match="must not contain symlinks"):
        service._systemd_unit(home)

    (home / "memories").unlink()
    (home / "memories").mkdir()
    skills = home / "skills"
    skills.mkdir()
    external_skill = external / "skill"
    external_skill.mkdir()
    (external_skill / "SKILL.md").write_text("secret")
    (skills / "escape").symlink_to(external_skill, target_is_directory=True)

    with pytest.raises(
        service.MiniAppServiceError, match="skill directories must not be symlinks"
    ):
        service._systemd_unit(home)


def test_cli_command_status_is_non_mutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli.service, "status", lambda home: (False, "not installed"))
    cli.command(argparse.Namespace(mini_app_command="status"))
    assert "not installed" in capsys.readouterr().out


def test_cli_foreground_serve_reexecutes_clean_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli.service, "_platform", lambda: "systemd")
    monkeypatch.setattr(
        cli.service, "exec_clean_runner", lambda home: seen.append(home)
    )

    cli.command(argparse.Namespace(mini_app_command="serve"))

    assert seen == [tmp_path]


def test_uninstall_restores_telegram_menus_before_deleting_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_token = "123456:abcdefghijklmnopqrstuvwxyzABCDE"
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    paths.env.write_text(
        f"TELEGRAM_BOT_TOKEN={expected_token}\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
    )
    paths.state.write_text(
        json.dumps({
            "owners": ["111"],
            "menu_backups": {
                "global": {"type": "commands"},
                "111": {"type": "commands"},
            },
        })
    )
    events: list[str] = []
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)

    def restore(token, menus):
        assert paths.env.exists()
        assert token == expected_token
        events.append("menus")

    def uninstall(home):
        assert paths.env.exists()
        events.append("service")

    monkeypatch.setattr(cli, "_restore_menus", restore)
    monkeypatch.setattr(cli.service, "uninstall", uninstall)
    monkeypatch.setattr(cli, "_remove_behavior", lambda: events.append("config"))
    monkeypatch.setattr(cli, "read_raw_config", lambda: {"configured": True})
    cli.command(argparse.Namespace(mini_app_command="uninstall"))
    assert events == ["config", "menus", "service"]
    assert not paths.env.exists()


def test_uninstall_fails_closed_when_menu_recovery_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    paths.env.write_text(
        "TELEGRAM_BOT_TOKEN=123456:abcdefghijklmnopqrstuvwxyzABCDE\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
    )
    paths.state.write_text(
        json.dumps({
            "owners": ["111"],
            "menu_backups": {"global": {"type": "commands"}},
        })
    )
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "_remove_behavior",
        lambda: pytest.fail("config changed before recovery validation"),
    )
    monkeypatch.setattr(
        cli.service,
        "uninstall",
        lambda home: pytest.fail("service removed without recovery data"),
    )

    with pytest.raises(SystemExit):
        cli.command(argparse.Namespace(mini_app_command="uninstall"))

    assert paths.env.exists()
    assert paths.state.exists()


def test_uninstall_config_failure_precedes_external_or_file_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    paths.env.write_text(
        "TELEGRAM_BOT_TOKEN=123456:abcdefghijklmnopqrstuvwxyzABCDE\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
    )
    paths.state.write_text(
        json.dumps({
            "owners": ["111"],
            "menu_backups": {
                "global": {"type": "commands"},
                "111": {"type": "commands"},
            },
        })
    )
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "read_raw_config",
        lambda: {"platforms": {"telegram": {"extra": {"mini_app": {"enabled": True}}}}},
    )
    monkeypatch.setattr(
        cli,
        "save_config",
        lambda config: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        cli,
        "_restore_menus",
        lambda token, menus: pytest.fail("menus changed after config failure"),
    )
    monkeypatch.setattr(
        cli.service,
        "uninstall",
        lambda home: pytest.fail("service removed after config failure"),
    )

    with pytest.raises(SystemExit):
        cli.command(argparse.Namespace(mini_app_command="uninstall"))

    assert paths.env.exists()
    assert paths.state.exists()


def test_uninstall_preserves_recovery_files_when_service_does_not_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = service.paths_for(tmp_path)
    paths.root.mkdir(parents=True)
    paths.env.write_text(
        "TELEGRAM_BOT_TOKEN=123456:abcdefghijklmnopqrstuvwxyzABCDE\n"
        "TELEGRAM_MINI_APP_OWNER_IDS=111\n"
        "TELEGRAM_MINI_APP_PUBLIC_URL=https://mini.example.com\n"
    )
    paths.state.write_text(
        json.dumps({
            "owners": ["111"],
            "menu_backups": {
                "global": {"type": "commands"},
                "111": {"type": "commands"},
            },
        })
    )
    monkeypatch.setattr(cli, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_raw_config", lambda: {})
    monkeypatch.setattr(cli, "_remove_behavior", lambda: None)
    monkeypatch.setattr(cli, "_restore_menus", lambda token, menus: None)
    monkeypatch.setattr(
        cli.service,
        "uninstall",
        lambda home: (_ for _ in ()).throw(
            service.MiniAppServiceError("service is still active")
        ),
    )

    with pytest.raises(SystemExit):
        cli.command(argparse.Namespace(mini_app_command="uninstall"))

    assert paths.env.exists()
    assert paths.state.exists()


def test_remove_behavior_clears_only_generated_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "platforms": {
            "telegram": {
                "enabled": True,
                "extra": {
                    "mini_app": {
                        "enabled": True,
                        "public_url": "https://mini.example.com",
                        "listen_port": 8787,
                    },
                    "keep": {"value": True},
                },
            }
        }
    }
    saved: list[dict] = []
    monkeypatch.setattr(cli, "read_raw_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(cli, "save_config", lambda value: saved.append(value))

    cli._remove_behavior()

    assert saved == [
        {
            "platforms": {
                "telegram": {
                    "enabled": True,
                    "extra": {"keep": {"value": True}},
                }
            }
        }
    ]
