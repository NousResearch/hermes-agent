from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as cli_main
from hermes_cli import dashboard_procs, main_dashboard, main_desktop, main_web_build, webapp


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


def test_desktop_content_hash_tracks_shared_source(tmp_path: Path):
    _workspace_tree(tmp_path)
    desktop_source = tmp_path / "apps" / "desktop" / "src.ts"
    shared_source = tmp_path / "apps" / "shared" / "src.ts"
    desktop_source.write_text("desktop", encoding="utf-8")
    shared_source.write_text("shared-v1", encoding="utf-8")

    before = main_desktop._compute_desktop_content_hash(tmp_path)
    shared_source.write_text("shared-v2", encoding="utf-8")

    assert main_desktop._compute_desktop_content_hash(tmp_path) != before


def _assert_build_lock_excludes_second_open(tmp_path: Path):
    lock_path = tmp_path / "webapp.lock"

    with webapp._exclusive_build_lock(lock_path):
        with lock_path.open("a+b") as contender:
            assert webapp._try_file_lock(contender) is False

    with lock_path.open("a+b") as contender:
        assert webapp._try_file_lock(contender) is True
        webapp._unlock_file(contender)


def test_webapp_build_lock_excludes_a_second_open(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


@pytest.mark.windows_only
def test_webapp_build_lock_excludes_a_second_open_on_windows(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


def test_failed_renderer_publish_restores_previous_generation(tmp_path: Path, monkeypatch):
    dist = tmp_path / "dist-webapp"
    staging = tmp_path / "staging"
    dist.mkdir()
    staging.mkdir()
    (dist / "index.html").write_text("old", encoding="utf-8")
    (staging / "index.html").write_text("new", encoding="utf-8")
    real_replace = os.replace

    def fail_staging_publish(source, destination):
        if Path(source) == staging:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(webapp.os, "replace", fail_staging_publish)

    with pytest.raises(webapp.WebappBuildError, match="publish"):
        webapp._publish_dist(staging, dist)

    assert (dist / "index.html").read_text(encoding="utf-8") == "old"


def test_failed_renderer_restore_preserves_last_known_good_backup(
    tmp_path: Path, monkeypatch
):
    dist = tmp_path / "dist-webapp"
    staging = tmp_path / "staging"
    dist.mkdir()
    staging.mkdir()
    (dist / "index.html").write_text("old", encoding="utf-8")
    (staging / "index.html").write_text("new", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_publish_and_restore(source, destination):
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError("simulated replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(webapp.os, "replace", fail_publish_and_restore)

    with pytest.raises(webapp.WebappBuildError, match="backup preserved at"):
        webapp._publish_dist(staging, dist)

    backups = list(tmp_path.glob(".dist-webapp-backup-*"))
    assert not dist.exists()
    assert len(backups) == 1
    assert (backups[0] / "index.html").read_text(encoding="utf-8") == "old"


def test_browser_build_uses_locked_closure_and_never_replaces_native_dist(
    tmp_path: Path, monkeypatch
):
    _workspace_tree(tmp_path)
    native_index = tmp_path / "apps" / "desktop" / "dist" / "index.html"
    native_index.parent.mkdir()
    native_index.write_text("native-electron", encoding="utf-8")
    stamp = tmp_path / "stamp.json"
    install_calls = []
    build_calls = []

    def install(npm, cwd, **kwargs):
        install_calls.append((npm, cwd, kwargs))
        return SimpleNamespace(returncode=0)

    def build(argv, *, cwd, env):
        build_calls.append((argv, cwd, env))
        staging = Path(argv[-1])
        staging.mkdir()
        (staging / "index.html").write_text("webapp", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    fake_main = SimpleNamespace(
        _compute_desktop_content_hash=lambda _root: "content-hash",
        _npm_lifecycle_env=lambda env: dict(env),
        _resolve_node_runtime_npm=lambda: "/node/npm",
        _run_npm_install_deterministic=install,
        _run_with_idle_timeout=build,
    )
    for name, implementation in vars(fake_main).items():
        monkeypatch.setattr(webapp, name, implementation)
    monkeypatch.setattr(webapp, "_stamp_path", lambda: stamp)
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {"PATH": "/node"}
    )

    result = webapp.prepare_webapp_renderer(tmp_path, force=True)

    assert result == tmp_path / "apps" / "desktop" / "dist-webapp"
    assert native_index.read_text(encoding="utf-8") == "native-electron"
    assert install_calls[0][0:2] == ("/node/npm", tmp_path)
    install_args = install_calls[0][2]["extra_args"]
    assert "--ignore-scripts" in install_args
    assert "--no-save" in install_args
    assert "--workspaces" in install_args
    assert "--include-workspace-root" in install_args
    assert build_calls[0][0] == [
        "/node/npm",
        "run",
        "--workspace",
        "apps/desktop",
        "build:webapp",
        "--",
        "--outDir",
        build_calls[0][0][-1],
    ]
    assert (tmp_path / "package-lock.json").read_text(encoding="utf-8") == "locked\n"
    assert stamp.is_file()


def test_webapp_selects_its_dist_without_leaking_headless_mode(tmp_path: Path, monkeypatch):
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    monkeypatch.setenv("HERMES_WEB_DIST", "restore-after-test")
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")

    webapp.activate_webapp_dist(dist)

    assert Path(os.environ["HERMES_WEB_DIST"]) == dist
    assert "HERMES_SERVE_HEADLESS" not in os.environ


def test_webapp_build_only_prepares_without_starting_server(tmp_path: Path, monkeypatch):
    prepared = tmp_path / "dist-webapp"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda *a, **k: prepared)

    assert cli_main.cmd_webapp(_args(build_only=True)) is None


def test_webapp_runs_through_the_shared_dashboard_server(tmp_path: Path, monkeypatch):
    prepared = tmp_path / "dist-webapp"
    prepared.mkdir()
    delegated = []
    monkeypatch.setenv("HERMES_WEB_DIST", "restore-after-test")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(webapp, "prepare_webapp_renderer", lambda *a, **k: prepared)
    monkeypatch.setattr(
        cli_main, "cmd_dashboard", lambda args: delegated.append(args) or "running"
    )
    args = _args(host="0.0.0.0", port=9443)

    assert cli_main.cmd_webapp(args) == "running"
    assert delegated == [args]
    assert args.skip_build is True
    assert Path(os.environ["HERMES_WEB_DIST"]) == prepared


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
            "reason": "requested via webapp --stop",
        }
    ]


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
        lambda: [
            {
                "argv": "python -m hermes_cli.main webapp --no-open",
                "pid": 4242,
                "purpose": "webapp",
            }
        ],
    )
    assert _ledger_web_server_processes() == {
        4242: "python -m hermes_cli.main webapp --no-open --port 0"
    }


def test_webapp_process_table_fallback_uses_structured_command_identity(monkeypatch):
    from hermes_cli import dashboard_procs

    monkeypatch.setattr(dashboard_procs, "_ledger_web_server_processes", lambda: {})
    monkeypatch.setattr(
        dashboard_procs.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": (
                    "4242 python -m hermes_cli.main webapp --host 127.0.0.1 --port 9119\n"
                    "4243 hermes chat -q webapp\n"
                ),
            },
        )(),
    )

    assert dashboard_procs._scan_dashboard_processes() == [
        (4242, "python -m hermes_cli.main webapp --host 127.0.0.1 --port 9119")
    ]


def test_dashboard_and_webapp_builds_enter_the_same_workspace_lock(tmp_path: Path, monkeypatch):
    _workspace_tree(tmp_path)
    dist = tmp_path / "apps" / "desktop" / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("ready", encoding="utf-8")
    lock_paths = []

    @contextmanager
    def record_lock(path):
        lock_paths.append(path)
        yield

    monkeypatch.setattr(webapp, "_exclusive_build_lock", record_lock)
    monkeypatch.setattr(webapp, "_try_file_lock", lambda _handle: False)
    monkeypatch.setattr(main_web_build, "_do_build_web_ui", lambda *_args, **_kwargs: True)

    assert webapp.prepare_webapp_renderer(tmp_path, skip_build=True) == dist
    assert main_web_build._build_web_ui(tmp_path / "web") is True
    assert lock_paths == [
        tmp_path / ".web_ui_build.lock",
        tmp_path / ".web_ui_build.lock",
    ]


def test_dashboard_serves_existing_dist_while_shared_workspace_lock_is_busy(
    tmp_path: Path, monkeypatch
):
    _workspace_tree(tmp_path)
    dashboard_index = tmp_path / "hermes_cli" / "web_dist" / "index.html"
    dashboard_index.parent.mkdir(parents=True)
    dashboard_index.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(webapp, "_try_file_lock", lambda _handle: False)
    monkeypatch.setattr(
        main_web_build,
        "_do_build_web_ui",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert main_web_build._build_web_ui(tmp_path / "web") is True


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
