"""Regression tests: `hermes dashboard` validates HERMES_WEB_DIST before serving.

A custom HERMES_WEB_DIST without --skip-build previously skipped BOTH the
build and any validation, so the server started and served 404s with no
obvious cause (same failure mode as issue #23817, reached via the env-var
path instead of --skip-build). The env-var branch must now fail fast when
the dist has no index.html, and proceed when it does.

Design credit: PR #17845 (@Caelier).
"""

import os
import sys
import types

import pytest


@pytest.fixture()
def main_mod():
    import hermes_cli.main as main
    return main


def _args(**over):
    base = {
        "host": "127.0.0.1",
        "port": 0,
        "no_open": True,
        "open_profile": None,
        "skip_build": False,
        "headless_backend": False,
        "tui": False,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


def _wire_common(main_mod, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_k: None,
    )


def test_env_dist_without_index_exits(main_mod, monkeypatch, tmp_path, capsys):
    """HERMES_WEB_DIST pointing at a dist with no index.html must exit 1,
    not start a server that 404s."""
    _wire_common(main_mod, monkeypatch)
    empty_dist = tmp_path / "empty_dist"
    empty_dist.mkdir()
    monkeypatch.setenv("HERMES_WEB_DIST", str(empty_dist))

    started = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: started.append(k)),
    )
    builds = []
    monkeypatch.setattr(
        "hermes_cli.main_web_build._build_web_ui", lambda *a, **k: builds.append(a) or True
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_dashboard(_args())

    assert exc.value.code == 1
    assert started == []
    assert builds == []  # env var set -> build skipped, validation is the gate
    out = capsys.readouterr().out
    assert "HERMES_WEB_DIST" in out and str(empty_dist) in out




# ---------------------------------------------------------------------------
# --skip-build recovery (issue #59288): a missing dist under --skip-build
# should warn and attempt ONE recovery build via _build_web_ui before the
# fatal exit, instead of hard-failing immediately.
# ---------------------------------------------------------------------------


def test_skip_build_missing_dist_attempts_one_recovery_build(
    main_mod, monkeypatch, tmp_path, capsys
):
    """--skip-build + missing index.html triggers exactly one recovery build;
    when the build produces a dist, the server starts."""
    _wire_common(main_mod, monkeypatch)
    monkeypatch.delenv("HERMES_WEB_DIST", raising=False)
    project_root = tmp_path / "proj"
    dist = project_root / "hermes_cli" / "web_dist"
    dist.mkdir(parents=True)
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", project_root)

    started = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: started.append(k)),
    )

    builds = []

    def fake_build(web_dir, *, fatal=False):
        builds.append((web_dir, fatal))
        (dist / "index.html").write_text("<html></html>", encoding="utf-8")
        return True

    monkeypatch.setattr("hermes_cli.main_web_build._build_web_ui", fake_build)

    main_mod.cmd_dashboard(_args(skip_build=True))

    assert len(builds) == 1  # exactly ONE recovery build
    assert builds[0][0] == project_root / "web"
    assert len(started) == 1
    out = capsys.readouterr().out
    assert "recovery build" in out.lower()




# ---------------------------------------------------------------------------
# Desktop-inherited env isolation (issue #52945 / supersedes #52948, #67402)
# ---------------------------------------------------------------------------


PACKAGED_DIST = "/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist"
CUSTOM_DIST = "/srv/hermes/renderer-build"


def test_dashboard_from_desktop_spawned_shell_strips_packaged_dist(main_mod, monkeypatch):
    """#116107: a desktop-spawned shell inherits HERMES_DESKTOP=1 together
    with the packaged dist; a `hermes dashboard` launched from it must not
    keep serving the desktop renderer ("Desktop IPC bridge is unavailable")."""
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", PACKAGED_DIST)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=False)

    assert "HERMES_WEB_DIST" not in os.environ


def test_desktop_headless_backend_keeps_packaged_dist(main_mod, monkeypatch):
    """The real Electron backend (`serve` entry point + HERMES_DESKTOP=1)
    still serves the packaged dist it was spawned with."""
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", PACKAGED_DIST)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=True)

    assert os.environ.get("HERMES_WEB_DIST") == PACKAGED_DIST


def test_dashboard_without_desktop_marker_still_strips(main_mod, monkeypatch):
    """Pre-#116107 behavior unchanged: a packaged dist inherited without
    HERMES_DESKTOP set is stripped for a standalone dashboard."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setenv("HERMES_WEB_DIST", PACKAGED_DIST)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=False)

    assert "HERMES_WEB_DIST" not in os.environ


def test_desktop_spawned_custom_dist_is_kept(main_mod, monkeypatch):
    """Caller-managed overrides survive: a non-Electron-packaged
    HERMES_WEB_DIST is never stripped, even under HERMES_DESKTOP=1."""
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", CUSTOM_DIST)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=False)

    assert os.environ.get("HERMES_WEB_DIST") == CUSTOM_DIST










