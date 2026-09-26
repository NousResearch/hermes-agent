from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli import main as cli_main
from hermes_cli import dashboard_procs, main_dashboard, webapp


def _args(**overrides):
    values = {
        "build_only": False,
        "force_build": False,
        "host": "127.0.0.1",
        "insecure": False,
        "isolated": False,
        "no_open": False,
        "open_profile": "",
        "port": 9119,
        "skip_build": False,
        "status": False,
        "stop": False,
        "ui_surface": "webapp",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _workspace_tree(root: Path) -> None:
    (root / "apps" / "desktop").mkdir(parents=True)
    (root / "apps" / "shared").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "ui-tui").mkdir()
    (root / "package.json").write_text('{"private":true}', encoding="utf-8")
    (root / "package-lock.json").write_text("locked\n", encoding="utf-8")
    for path in (
        root / "apps" / "desktop" / "package.json",
        root / "apps" / "shared" / "package.json",
        root / "web" / "package.json",
        root / "ui-tui" / "package.json",
    ):
        path.write_text("{}", encoding="utf-8")


def test_skip_build_requires_the_separate_webapp_bundle(tmp_path: Path):
    _workspace_tree(tmp_path)
    (tmp_path / "apps" / "desktop" / "dist").mkdir()
    (tmp_path / "apps" / "desktop" / "dist" / "index.html").write_text(
        "native", encoding="utf-8"
    )

    with pytest.raises(webapp.WebappBuildError, match="dist-webapp"):
        webapp.prepare_webapp_renderer(tmp_path, skip_build=True)


def _assert_build_lock_excludes_second_open(tmp_path: Path):
    from pm.filesystem import lock_fd

    lock_path = tmp_path / "webapp.lock"

    with webapp._exclusive_build_lock(lock_path):
        with lock_path.open("a+b") as contender:
            assert lock_fd(contender.fileno(), wait=False) is False

    with lock_path.open("a+b") as contender:
        assert lock_fd(contender.fileno(), wait=False) is True
        webapp._unlock_file(contender)


def test_webapp_build_lock_excludes_a_second_open(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


@pytest.mark.platforms("windows")
def test_webapp_build_lock_excludes_a_second_open_on_windows(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


def test_launch_rebuilds_only_stale_or_forced_renderers(tmp_path: Path, monkeypatch):
    from hermes_cli import source_build

    _workspace_tree(tmp_path)
    receipt_current = True
    builds = []
    monkeypatch.setattr(source_build, "source_product_current", lambda *_a: receipt_current)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit=False: {})
    monkeypatch.setattr(source_build, "build_source_webapp", lambda _root, *, env, explicit: builds.append(explicit))

    dist = webapp.prepare_webapp_renderer(tmp_path)
    assert dist == tmp_path / "apps" / "desktop" / "dist-webapp"
    assert builds == []
    webapp.prepare_webapp_renderer(tmp_path, force=True, explicit=True)
    receipt_current = False
    webapp.prepare_webapp_renderer(tmp_path)
    assert builds == [True, False]


@pytest.mark.parametrize("failure", [
    PermissionError("workspace node_modules is locked"),
    subprocess.CalledProcessError(1, ["node", "build-webapp.mjs"]),
    RuntimeError("npm is unavailable and lazy installs are disabled"),
])
def test_renderer_build_failures_are_reported_not_raised_raw(tmp_path: Path, monkeypatch, failure):
    from hermes_cli import source_build

    _workspace_tree(tmp_path)

    def fail(*_a, **_k):
        raise failure

    monkeypatch.setattr(source_build, "source_product_current", lambda *_a: False)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit=False: {})
    monkeypatch.setattr(source_build, "build_source_webapp", fail)

    with pytest.raises(webapp.WebappBuildError, match="build failed"):
        webapp.prepare_webapp_renderer(tmp_path)


def test_only_requested_builds_may_install_renderer_dependencies(tmp_path: Path, monkeypatch):
    """A plain launch honors disabled lazy installs; --build-only / --force-build ask explicitly."""
    requests = []
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        webapp, "prepare_webapp_renderer",
        lambda *_a, **kwargs: requests.append(kwargs["explicit"]) or tmp_path / "dist-webapp")
    monkeypatch.setattr(cli_main, "cmd_dashboard", lambda _args: None)

    cli_main.cmd_webapp(_args())
    cli_main.cmd_webapp(_args(build_only=True))
    cli_main.cmd_webapp(_args(force_build=True))
    assert requests == [False, True, True]


def test_webapp_build_only_prepares_without_starting_server(tmp_path: Path, monkeypatch):
    prepared = tmp_path / "dist-webapp"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda *a, **k: prepared)

    assert cli_main.cmd_webapp(_args(build_only=True)) is None


def test_webapp_runs_through_the_shared_dashboard_server(tmp_path: Path, monkeypatch):
    prepared = tmp_path / "dist-webapp"
    prepared.mkdir()
    delegated = []
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda *a, **k: prepared)
    monkeypatch.setattr(
        cli_main, "cmd_dashboard", lambda args: delegated.append(args) or "running"
    )
    args = _args(host="0.0.0.0", port=9443)

    assert cli_main.cmd_webapp(args) == "running"
    assert delegated == [args]
    assert args.skip_build is True
    assert args.web_dist == prepared


def test_webapp_serves_its_renderer_without_exporting_it_to_children(tmp_path: Path, monkeypatch):
    """The renderer once travelled through os.environ["HERMES_WEB_DIST"], which every host shell
    and PTY child inherited: `hermes dashboard` run from a Webapp terminal then served the Desktop
    renderer as its dashboard (the #52945 class). It is now handed to start_server explicitly."""
    import hermes_cli.config
    import hermes_cli.mcp_startup
    import hermes_cli.plugins
    import hermes_cli.resource_limits
    from fastapi.testclient import TestClient
    from hermes_cli import web_host_terminal, web_server
    from hermes_constants import get_hermes_home

    renderer = tmp_path / "dist-webapp"
    renderer.mkdir()
    (renderer / "index.html").write_text(
        "<html><head></head><body>webapp-renderer</body></html>", encoding="utf-8")
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda *_a, **_k: renderer)
    for owner, name in (
        (cli_main, "_sync_bundled_skills_quietly"),
        (cli_main, "_maybe_setup_dashboard_auth_interactively"),
        (hermes_cli.config, "apply_terminal_config_to_env"),
        (hermes_cli.plugins, "discover_plugins"),
        (hermes_cli.mcp_startup, "defer_background_mcp_discovery"),
        (hermes_cli.resource_limits, "apply_nofile_soft_limit"),
        (web_server, "_configure_auth_gate"),
    ):
        monkeypatch.setattr(owner, name, lambda *_a, **_k: None)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: None)
    # start_server writes these; restore them for the rest of this file's tests.
    for key, value in (("ui_surface", "dashboard"), ("web_dist", None), ("auth_required", False),
                       ("bound_host", ""), ("initial_profile", "")):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", web_server._SESSION_TOKEN)

    class _Bind(Exception):
        pass

    def stop_at_bind(*_a, **_k):
        raise _Bind

    monkeypatch.setattr(web_server, "_build_uvicorn_server", stop_at_bind)
    with pytest.raises(_Bind):
        cli_main.cmd_webapp(_args(no_open=True, isolated=True))

    _argv, _cwd, shell_env, _shell = web_host_terminal.resolve_argv(home=get_hermes_home())
    assert "HERMES_WEB_DIST" not in shell_env
    served = TestClient(web_server.app, base_url="http://127.0.0.1").get("/")
    assert served.status_code == 200 and "webapp-renderer" in served.text


def test_webapp_status_is_scoped_and_does_not_build(monkeypatch):
    reported = []
    monkeypatch.setattr(
        main_dashboard,
        "_report_dashboard_status",
        lambda **kwargs: reported.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        webapp,
        "prepare_webapp_renderer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_webapp(_args(status=True))

    assert exc.value.code == 0
    assert reported == [{"modes": {"webapp"}}]


def test_webapp_stop_never_targets_desktop_serve_backend(monkeypatch):
    from hermes_constants import get_hermes_home

    own_home = str(get_hermes_home())
    monkeypatch.setattr(dashboard_procs, "_hermes_home_for_pid", lambda pid: own_home)
    scans = iter(
        [
            [
                (111, "python -m hermes_cli.main webapp --port 9119"),
                (222, "python -m hermes_cli.main serve --port 0"),
            ],
            [],
        ]
    )
    killed = []
    monkeypatch.setattr(
        dashboard_procs,
        "_scan_dashboard_processes",
        lambda: next(scans),
    )
    monkeypatch.setattr(
        dashboard_procs,
        "_kill_stale_dashboard_processes",
        lambda **kwargs: killed.append(kwargs) or {"killed": [111]},
    )

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_webapp(_args(stop=True))

    assert exc.value.code == 0
    assert killed == [
        {
            "include_pids": {111},
            "scope_home": own_home,
            "reason": "requested via webapp --stop",
        }
    ]


@pytest.mark.parametrize(
    ("own_running", "own_survives", "expected_exit"),
    [(True, False, 0), (True, True, 1), (False, False, 0)],
)
def test_webapp_stop_only_targets_the_invoking_home(
    tmp_path, monkeypatch, own_running, own_survives, expected_exit,
):
    own_home = str(tmp_path / "own")
    foreign_home = str(tmp_path / "foreign")
    monkeypatch.setenv("HERMES_HOME", own_home)
    own_webapp = (111, "hermes webapp --port 9119")
    spared = [
        (222, "hermes serve --port 0"),
        (333, "hermes webapp --port 9120"),
        (444, "hermes webapp --port 9121"),
    ]
    scans = iter([
        ([own_webapp] if own_running else []) + spared,
        ([own_webapp] if own_survives else []) + spared,
    ])
    monkeypatch.setattr(dashboard_procs, "_scan_dashboard_processes", lambda: next(scans))
    monkeypatch.setattr(
        dashboard_procs, "_hermes_home_for_pid",
        lambda pid: {111: own_home, 222: own_home, 333: foreign_home, 444: None}[pid],
    )
    killed = []
    monkeypatch.setattr(
        dashboard_procs, "_kill_stale_dashboard_processes",
        lambda **kwargs: killed.append(kwargs),
    )

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_webapp(_args(stop=True))

    assert exc.value.code == expected_exit
    assert killed == ([{
        "include_pids": {111},
        "scope_home": own_home,
        "reason": "requested via webapp --stop",
    }] if own_running else [])


def test_web_server_commands_cannot_be_shadowed_by_profile_aliases():
    from hermes_cli.profiles import check_alias_collision

    for command in ("dashboard", "serve", "webapp"):
        assert check_alias_collision(command) == (
            f"'{command}' conflicts with a hermes subcommand"
        )


def test_webapp_process_identity_uses_the_existing_web_server_lifecycle(monkeypatch):
    from hermes_cli.dashboard_procs import (
        _dashboard_subcommand_index,
        _is_hermes_web_server_command,
        _is_dashboard_lifecycle_probe,
        _ledger_web_server_processes,
        _normalize_dashboard_cmdline,
    )

    argv = ["python", "-m", "hermes_cli.main", "-p", "coder", "webapp", "--port", "9443"]

    assert _dashboard_subcommand_index(argv) == 5
    assert _normalize_dashboard_cmdline(argv) == (
        "-p",
        "coder",
        "webapp",
        "--port",
        "9443",
    )
    assert main_dashboard._parse_dashboard_runtime(
        "python -m hermes_cli.main webapp --host 0.0.0.0 --port 9443"
    ) == ("webapp", "0.0.0.0", 9443)
    assert main_dashboard._parse_dashboard_runtime(
        "python -m hermes_cli.main -p default webapp --port 9119"
    ) == ("webapp", "127.0.0.1", 9119)
    assert main_dashboard._parse_dashboard_runtime(
        "python hermes_cli/main.py -p coder webapp --port 9120"
    ) == ("webapp", "127.0.0.1", 9120)
    assert main_dashboard._parse_dashboard_runtime(
        "python nothermes_cli/main.py webapp --port 9121"
    ) is None
    assert main_dashboard._parse_dashboard_runtime(
        "python /tmp/myhermes_cli/main.py webapp --port 9122"
    ) is None
    assert main_dashboard._parse_dashboard_runtime(
        "python -m nothermes_cli.main webapp --port 9123"
    ) is None
    assert _is_dashboard_lifecycle_probe(
        "python -m hermes_cli.main webapp --stop"
    ) is True
    assert _is_dashboard_lifecycle_probe(
        "python -m hermes_cli.main dashboard --status"
    ) is True
    assert _is_dashboard_lifecycle_probe(
        "/bin/sh -c '\"python -m hermes_cli.main webapp --stop\"'"
    ) is True
    assert _is_dashboard_lifecycle_probe(
        "python -m hermes_cli.main webapp --host 127.0.0.1 --port 9443"
    ) is False
    assert _is_hermes_web_server_command(
        "python -m hermes_cli.main -p coder webapp --port 9443"
    ) is True
    assert _is_hermes_web_server_command("/opt/hermes/bin/hermes dashboard --no-open") is True
    assert _is_hermes_web_server_command("hermes chat -q webapp") is False

    monkeypatch.setattr(
        "hermes_cli.process_identity.ledger_entries",
        lambda *, verified_only: [
            {
                "argv": "python -m hermes_cli.main webapp --no-open",
                "pid": 4242,
                "purpose": "webapp",
            }
        ] if verified_only else [],
    )
    assert _ledger_web_server_processes() == {
        4242: "python -m hermes_cli.main webapp --no-open --port 0"
    }


def test_webapp_process_table_fallback_uses_structured_command_identity(monkeypatch):
    from hermes_cli import dashboard_procs

    monkeypatch.setattr(dashboard_procs, "_ledger_web_server_processes", lambda: {})
    monkeypatch.setattr(
        dashboard_procs,
        "_iter_process_table",
        lambda: [
            (4242, "python -m hermes_cli.main webapp --host 127.0.0.1 --port 9119"),
            (4243, "hermes chat -q webapp"),
        ],
    )

    assert dashboard_procs._scan_dashboard_processes() == [
        (4242, "python -m hermes_cli.main webapp --host 127.0.0.1 --port 9119")
    ]


def test_dashboard_and_webapp_builds_share_the_workspace_lock(tmp_path: Path):
    """A Webapp build excludes the Dashboard builder's lock on the same checkout."""
    from pm.filesystem import lock_fd

    _workspace_tree(tmp_path)
    lock_path = tmp_path / webapp._LOCK_NAME
    assert webapp._LOCK_NAME == ".web_ui_build.lock"
    with webapp._exclusive_build_lock(lock_path):
        with lock_path.open("ab") as contender:
            assert lock_fd(contender.fileno(), wait=False) is False
    with lock_path.open("ab") as contender:
        assert lock_fd(contender.fileno(), wait=False) is True


def test_webapp_help_describes_its_scoped_lifecycle(capsys):
    from hermes_cli.subcommands.dashboard import build_dashboard_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=lambda _args: None,
        cmd_dashboard_register=lambda _args: None,
        cmd_webapp=lambda _args: None,
    )

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["webapp", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Stop running Hermes Webapp processes and exit" in output
    assert "Stop all running Hermes web server processes and exit" not in output
