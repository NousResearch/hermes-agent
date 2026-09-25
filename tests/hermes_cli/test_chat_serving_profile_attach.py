"""Explicit names for the serving profile must reuse its chat backend (#122365)."""

from pathlib import Path

import pytest
from fastapi import HTTPException


@pytest.fixture
def chat_homes(tmp_path, monkeypatch):
    from hermes_cli import main_tui_launch, web_server_chat

    root = tmp_path / "home"
    other = root / "profiles" / "worker"
    other.mkdir(parents=True)
    for home, backend in [(root, "docker"), (other, "ssh")]:
        (home / "config.yaml").write_text(
            f"terminal:\n  backend: {backend}\n", encoding="utf-8"
        )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", lambda *a, **k: (["node", "fixture.js"], tmp_path))
    monkeypatch.setattr(main_tui_launch, "_apply_tui_python_env", lambda env: None)
    monkeypatch.setattr(web_server_chat, "_build_gateway_ws_url", lambda: "ws://127.0.0.1:1234/api/ws?token=fixture")
    return root, other


@pytest.mark.parametrize("serving", ["default", "worker"])
def test_chat_attaches_only_to_its_serving_home(chat_homes, monkeypatch, serving):
    from hermes_cli import web_server_chat, web_server_profiles

    root, other = chat_homes
    own, foreign = (root, other) if serving == "default" else (other, root)
    foreign_name = "worker" if serving == "default" else "default"
    monkeypatch.setenv("HERMES_HOME", str(own))
    assert web_server_profiles.serving_profile_name() == serving
    # A -> B -> A: the explicit SPA name, another home, and own aliases.
    for requested in [serving, foreign_name, serving, None, "", "current"]:
        _, _, env = web_server_chat._resolve_chat_argv(profile=requested)
        expected_home = foreign if requested == foreign_name else own
        assert Path(env["HERMES_HOME"]).resolve() == expected_home.resolve()
        assert ("HERMES_TUI_GATEWAY_URL" in env) == (expected_home == own)
        assert env["TERMINAL_ENV"] == ("docker" if expected_home == root else "ssh")


def test_unbound_chat_fallback_and_invalid_profile(chat_homes, monkeypatch):
    from hermes_cli import web_server_chat

    monkeypatch.setattr(web_server_chat, "_build_gateway_ws_url", lambda: None)
    for requested in [None, "current", "default"]:
        _, _, env = web_server_chat._resolve_chat_argv(profile=requested)
        assert "HERMES_TUI_GATEWAY_URL" not in env
    with pytest.raises(HTTPException) as exc:
        web_server_chat._resolve_chat_argv(profile="missing")
    assert exc.value.status_code == 404
