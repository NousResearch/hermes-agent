"""Regression tests: `hermes dashboard` validates HERMES_WEB_DIST before serving.

A custom HERMES_WEB_DIST without --skip-build previously skipped BOTH the
build and any validation, so the server started and served 404s with no
obvious cause (same failure mode as issue #23817, reached via the env-var
path instead of --skip-build). The env-var branch must now fail fast when
the dist has no index.html, and proceed when it does.

Design credit: PR #17845 (@Caelier).
"""

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
        main_mod, "_build_web_ui", lambda *a, **k: builds.append(a) or True
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_dashboard(_args())

    assert exc.value.code == 1
    assert started == []
    assert builds == []  # env var set -> build skipped, validation is the gate
    out = capsys.readouterr().out
    assert "HERMES_WEB_DIST" in out and str(empty_dist) in out


def test_env_dist_with_index_starts_server(main_mod, monkeypatch, tmp_path):
    """A valid HERMES_WEB_DIST (has index.html) proceeds to start_server
    without building."""
    _wire_common(main_mod, monkeypatch)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("HERMES_WEB_DIST", str(dist))

    started = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: started.append(k)),
    )
    builds = []
    monkeypatch.setattr(
        main_mod, "_build_web_ui", lambda *a, **k: builds.append(a) or True
    )

    main_mod.cmd_dashboard(_args())

    assert len(started) == 1
    assert builds == []


def test_env_dist_tilde_expanded_for_web_server(main_mod, monkeypatch, tmp_path):
    """A '~/...' HERMES_WEB_DIST must be written back expanded so
    web_server's raw os.environ read serves the validated path."""
    _wire_common(main_mod, monkeypatch)
    home = tmp_path / "home"
    dist = home / "mydist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_WEB_DIST", "~/mydist")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **k: None),
    )

    main_mod.cmd_dashboard(_args())

    import os
    assert os.environ["HERMES_WEB_DIST"] == str(dist)
