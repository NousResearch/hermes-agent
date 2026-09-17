"""Dashboard routers that read a named profile's config/credentials run under that profile's FULL
scope (home + secrets) and never mutate the dashboard process environment.

Regression for the cross-profile leak class in ``hermes dashboard`` / ``hermes serve``:
``_config_profile_scope`` bound only HERMES_HOME, so ``GET /api/config?profile=B`` expanded B's
``${VAR}`` refs to the DEFAULT profile's plaintext credentials (its ``os.environ``), and console
``send`` for B (``send_cmd._load_hermes_env``) copied B's ``.env`` into the shared process env with
``override=True``, so every later default-profile read saw B's tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402

A_VAL = "a-only-secret-0001"
B_VAL = "b-only-secret-0002"


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes_home"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    (root / ".env").write_text(f"A_ONLY_TOKEN={A_VAL}\n", encoding="utf-8")
    (b / ".env").write_text(f"B_ONLY_TOKEN={B_VAL}\nTELEGRAM_BOT_TOKEN=b-telegram-token\n", encoding="utf-8")
    for home in (root, b):
        (home / "config.yaml").write_text(
            "model:\n  default: openai/gpt-4o-mini\n  api_key: ${A_ONLY_TOKEN}\n"
            "custom_probe:\n  a_ref: ${A_ONLY_TOKEN}\n  b_ref: ${B_ONLY_TOKEN}\n",
            encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("A_ONLY_TOKEN", A_VAL)  # the dashboard process loaded its own .env
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    from agent import secret_scope
    from tui_gateway import launch_profile_policy as lpp
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(lpp, "_snapshot", None)
    from hermes_cli import config as cfg_mod
    for attr in ("_CONFIG_CACHE", "_config_cache"):
        if hasattr(cfg_mod, attr):
            monkeypatch.setattr(cfg_mod, attr, None if not isinstance(getattr(cfg_mod, attr), dict) else {})
    return root, b


@pytest.fixture
def client(two_homes):
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_get_config_for_named_profile_expands_only_its_own_secrets(client, two_homes):
    from agent.secret_scope import is_multiplex_active

    resp = client.get("/api/config?profile=b")
    assert resp.status_code == 200, resp.text
    probe = resp.json()["custom_probe"]
    assert probe["b_ref"] == B_VAL
    assert probe["a_ref"] == "${A_ONLY_TOKEN}"  # not the dashboard profile's value
    # Hosting a second profile flipped the process to fail-closed; environ is untouched.
    assert is_multiplex_active()
    assert os.environ["A_ONLY_TOKEN"] == A_VAL and "B_ONLY_TOKEN" not in os.environ

    # The dashboard's own profile still resolves its own value (frozen launch env).
    probe_a = client.get("/api/config").json()["custom_probe"]
    assert probe_a["a_ref"] == A_VAL and probe_a["b_ref"] == "${B_ONLY_TOKEN}"


def test_console_send_for_named_profile_does_not_write_process_env(two_homes, monkeypatch):
    """``send`` loads the target profile's ``.env`` for the gateway config loader; inside a
    multi-profile host that must land in the request's scope, never ``os.environ``."""
    from hermes_cli.web_routers.chat_ws import _execute_console_line

    root, b = two_homes
    seen = {}

    def fake_send(args):
        import hermes_cli.send_cmd as send_cmd
        from gateway.config import _getenv
        send_cmd._load_hermes_env()
        seen["loader_sees"] = _getenv("TELEGRAM_BOT_TOKEN")
        seen["environ_has"] = "TELEGRAM_BOT_TOKEN" in os.environ
        seen["home"] = Path(os.environ.get("HERMES_HOME", ""))
        return '{"success": true}'

    class Engine:
        def execute(self, line, *, confirmed=False):
            import argparse
            return fake_send(argparse.Namespace())

    _execute_console_line(Engine(), "send --to telegram hi", confirmed=False, profile="b")
    assert seen["loader_sees"] == "b-telegram-token"  # the loader gets B's token through the scope
    assert seen["environ_has"] is False  # and the dashboard process env never learns it
    assert "B_ONLY_TOKEN" not in os.environ


def test_memory_provider_discovery_reads_schema_under_launch_secret_scope(two_homes, monkeypatch):
    """``GET /api/memory`` and the plugin hub reach provider discovery themselves, with no request
    scope of their own. Once another profile has flipped the process fail-closed, a provider whose
    schema reads a secret (mem0 resolves ``MEM0_MODE``) must still see the launch profile's value;
    an unscoped read raised and the schema was dropped with a logged traceback."""
    from agent import secret_scope

    from hermes_cli import web_server_memory as mem_mod

    monkeypatch.setenv("MEM0_MODE", "oss")

    seen = {}

    class _Provider:
        def get_config_schema(self):
            try:
                seen["mode"] = secret_scope.get_secret("MEM0_MODE")
            except secret_scope.UnscopedSecretError as exc:  # pragma: no cover - the regression
                seen["error"] = exc
                raise
            return [{"key": "mode", "label": "Mode", "required": True, "default": "platform"}]

    monkeypatch.setattr(mem_mod, "_load_memory_provider", lambda name: _Provider())
    monkeypatch.setattr(mem_mod, "_memory_provider_setup_info", lambda name: {"dependencies_installed": True})
    monkeypatch.setattr("plugins.memory.discover_memory_providers", lambda: [("mem0", "Mem0 memory", True)])

    secret_scope.set_secret_scope(None)  # discovery is called with no scope installed...
    secret_scope.set_multiplex_active(True)  # ...after a secondary profile flipped us fail-closed

    rows = mem_mod._discover_memory_provider_statuses()

    assert "error" not in seen, f"schema read stayed unscoped: {seen.get('error')}"
    assert seen["mode"] == "oss"
    assert [row["name"] for row in rows] == ["mem0"]
