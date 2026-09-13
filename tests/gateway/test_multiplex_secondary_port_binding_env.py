"""A secondary profile's port-binding credential wires the key without claiming the listener (#100397).

The docs require ``API_SERVER_KEY`` in a secondary's ``.env`` for ``/p/<profile>/`` auth, yet the env pass
turned it into ``api_server.enabled = True`` and ``_load_secondary_profile_config`` skipped the whole profile.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def multiplex_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: x}\n")
    (root / "profiles" / "coder" / "config.yaml").write_text("model: {default: x}\n")
    (root / "profiles" / "coder" / ".env").write_text("API_SERVER_KEY=abcdefghijklmnopqrstuvwxyz123456\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    return root


def _api_server(home):
    from gateway.config import Platform, load_gateway_config
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(home):
        return load_gateway_config().platforms.get(Platform.API_SERVER)


def test_secondary_api_server_key_wires_key_but_does_not_enable_listener(multiplex_root):
    pc = _api_server(multiplex_root / "profiles" / "coder")
    assert pc is not None and pc.extra.get("key")
    assert pc.enabled is False


def test_default_profile_api_server_key_still_enables_listener(multiplex_root):
    (multiplex_root / ".env").write_text("API_SERVER_KEY=abcdefghijklmnopqrstuvwxyz123456\n")
    pc = _api_server(multiplex_root)
    assert pc is not None and pc.enabled is True and pc.extra.get("key")


@pytest.mark.parametrize(
    ("cmdline", "expected"),
    [
        ("/v/python -m hermes_cli.main -p ops-2 gateway run", False),
        ("/v/python -m hermes_cli.main --profile ops2 gateway run", False),
        ("/v/python -m hermes_cli.main -p ops gateway run", True),
        ("/v/python -m hermes_cli.main --profile=ops gateway run", True),
    ],
)
def test_profile_match_is_token_equality_not_substring(tmp_path, cmdline, expected):
    """``-p ops`` must never claim (or let ``gateway stop`` SIGTERM) an ``-p ops-2`` gateway."""
    from gateway.status import _command_line_belongs_to_profile
    assert _command_line_belongs_to_profile(cmdline, tmp_path / "profiles" / "ops") is expected
