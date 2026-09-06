"""A failed Desktop pack must not look like a successful update.

#88251: ``hermes update`` treated a failed desktop pack as non-fatal, printed
an early warning, then still ended with ``✓ Update complete!``. The Python
side moved on; the Electron app stayed on the previous build.

``_rebuild_desktop_after_update`` returns False only when a rebuild was
attempted and failed. The final banner then prints ``⚠ Update partially
complete`` instead of the success line, and gateway mode writes ``1`` to
``.update_exit_code``.
"""

import pytest

from hermes_cli import update_cmd
import hermes_cli.update_cmd_maint as update_cmd_maint
from hermes_cli.update_cmd import (
    _print_update_summary,
    _rebuild_desktop_after_update,
    _write_gateway_update_exit_code,
)


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture()
def desktop_env(tmp_path, monkeypatch):
    """A desktop dir that looks installed and a faked CLI main module."""
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")

    calls = {"builds": 0, "build_needed": True}

    class _FakeMain:
        PROJECT_ROOT = tmp_path

        @staticmethod
        def _resolve_node_runtime_npm():
            return "/fake/npm"

        @staticmethod
        def _desktop_build_needed(*_a, **_kw):
            return calls["build_needed"]

        @staticmethod
        def _run_logged_subprocess(cmd, cwd=None, env=None):
            calls["builds"] += 1
            return _Result(1, stdout="Error: [stage-native-deps] boom")

    monkeypatch.setattr(update_cmd, "_m", lambda: _FakeMain)
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {}, raising=False
    )
    monkeypatch.setattr(
        "hermes_constants.display_hermes_home", lambda: str(tmp_path), raising=False
    )
    return desktop_dir, calls


def _run(desktop_dir):
    return _rebuild_desktop_after_update(
        desktop_dir, had_desktop_app_before_update=True
    )


def test_failed_rebuild_returns_false_and_keeps_the_retry_hint(desktop_env, capsys):
    desktop_dir, calls = desktop_env
    assert _run(desktop_dir) is False
    assert calls["builds"] == 2
    out = capsys.readouterr().out
    assert "Desktop build failed" in out
    assert "stage-native-deps" in out
    assert "Update complete" not in out


def test_successful_rebuild_returns_true(desktop_env, monkeypatch, capsys):
    desktop_dir, _calls = desktop_env
    builds = []
    monkeypatch.setattr(
        update_cmd._m(),
        "_run_logged_subprocess",
        staticmethod(lambda cmd, cwd=None, env=None: builds.append(cmd) or _Result(0)),
    )
    assert _run(desktop_dir) is True
    assert len(builds) == 1
    assert "Desktop app up to date" in capsys.readouterr().out


def test_up_to_date_desktop_returns_true_without_spawning(desktop_env):
    desktop_dir, calls = desktop_env
    calls["build_needed"] = False
    assert _run(desktop_dir) is True
    assert calls["builds"] == 0


def test_desktop_never_installed_returns_true(tmp_path, monkeypatch):
    spawned = []
    monkeypatch.setattr(
        update_cmd,
        "_m",
        lambda: type(
            "_M",
            (),
            {
                "PROJECT_ROOT": tmp_path,
                "_resolve_node_runtime_npm": staticmethod(lambda: "/fake/npm"),
                "_run_logged_subprocess": staticmethod(
                    lambda *a, **k: spawned.append(1) or _Result(0)
                ),
            },
        ),
    )
    missing = tmp_path / "apps" / "desktop"
    missing.mkdir(parents=True)
    assert _run(missing) is True
    assert spawned == []


def test_summary_omits_success_banner_when_desktop_rebuild_failed(capsys):
    _print_update_summary(
        node_failures=[],
        desktop_build_ok=False,
        pre_update_version="0.20.1",
    )
    out = capsys.readouterr().out
    assert "Update complete" not in out
    assert "partially complete" in out
    assert "desktop app was not rebuilt" in out
    assert "hermes desktop" in out


def test_summary_keeps_success_banner_when_desktop_ok(capsys, monkeypatch):
    monkeypatch.setattr(
        update_cmd, "_update_complete_message", lambda _v: "✓ Update complete! (v0.20.2)"
    )
    monkeypatch.setattr(
        update_cmd_maint, "_update_complete_message", lambda _v: "✓ Update complete! (v0.20.2)"
    )
    monkeypatch.setattr(update_cmd, "_branch_head_suffix", lambda *a, **k: "")
    monkeypatch.setattr(
        update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, None)
    )
    monkeypatch.setattr(
        update_cmd_maint, "_post_update_sqlite_runtime_status", lambda: (True, None)
    )
    _print_update_summary(
        node_failures=[],
        desktop_build_ok=True,
        pre_update_version="0.20.1",
    )
    out = capsys.readouterr().out
    assert "✓ Update complete!" in out
    assert "partially complete" not in out


def test_summary_combines_node_and_desktop_failures(capsys):
    _print_update_summary(
        node_failures=["dashboard"],
        desktop_build_ok=False,
        pre_update_version="0.20.1",
    )
    out = capsys.readouterr().out
    assert "Update complete" not in out
    assert "dashboard" in out
    assert "desktop app was not rebuilt" in out


def test_gateway_exit_code_file_tracks_desktop_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    _write_gateway_update_exit_code(True)
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8") == "0"
    _write_gateway_update_exit_code(False)
    assert (tmp_path / ".update_exit_code").read_text(encoding="utf-8") == "1"
