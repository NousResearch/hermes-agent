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
        "profiles.current.get_active_profile_name", lambda: "default"
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


def test_desktop_child_dashboard_drops_packaged_renderer(main_mod, monkeypatch):
    """#116107: every desktop-spawned process inherits HERMES_DESKTOP=1 with the
    packaged dist; a browser `hermes dashboard` from it must not keep serving the
    IPC-only desktop renderer."""
    packaged = "/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist"
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", packaged)
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")

    main_mod._dashboard_sanitize_desktop_env(headless_backend=False)

    assert "HERMES_WEB_DIST" not in os.environ
    assert "HERMES_SERVE_HEADLESS" not in os.environ


def test_desktop_headless_serve_keeps_packaged_renderer(main_mod, monkeypatch):
    """The real Desktop backend remains distinguished by the `serve` entry path."""
    packaged = "/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist"
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_WEB_DIST", packaged)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=True)

    assert os.environ["HERMES_WEB_DIST"] == packaged


def test_desktop_owned_fallback_dashboard_keeps_packaged_renderer(main_mod, monkeypatch):
    """The Desktop's own legacy `dashboard --no-open` fallback spawn (serve probe
    timed out) is not headless but carries the per-spawn session token; stripping
    its dist would send a packaged install into `_build_web_ui(fatal=True)`."""
    packaged = "/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist"
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-spawn-token")
    monkeypatch.setenv("HERMES_WEB_DIST", packaged)

    main_mod._dashboard_sanitize_desktop_env(headless_backend=False)

    assert os.environ["HERMES_WEB_DIST"] == packaged
