"""Regression tests for ``cmd_whatsapp`` env-var write ordering.

Before the fix, ``hermes whatsapp`` wrote ``WHATSAPP_ENABLED=true`` at
step 2 — before npm install (step 4) and before QR pairing (step 6).
If the user Ctrl+C'd at any later step, ``.env`` claimed WhatsApp was
ready when the bridge still had no ``creds.json``.  Every subsequent
``hermes gateway`` then paid a 30s bridge-bootstrap timeout and queued
WhatsApp for indefinite retries — looking like "the gateway is broken."

The fix: only set ``WHATSAPP_ENABLED=true`` once pairing actually
succeeds (creds.json exists).  Aborted setup leaves no enabled state.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    # Ensure get_env_value cache doesn't carry stale state.
    for key in list(os.environ):
        if key.startswith("WHATSAPP_"):
            monkeypatch.delenv(key, raising=False)
    return hermes


def _env_value(hermes_home: Path, key: str) -> str | None:
    env_file = hermes_home / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def test_aborted_setup_does_not_enable_whatsapp(isolated_home, monkeypatch):
    """User picks mode 1, then Ctrl+C's at the allowed-users prompt.

    WHATSAPP_ENABLED must NOT be present in .env after abort.
    """
    from hermes_cli.main_platform_setup import cmd_whatsapp

    # First input() = mode choice, second input() = allowed-users prompt
    # We raise KeyboardInterrupt on the second call to simulate abort.
    inputs = iter(["1"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    # _require_tty calls sys.stdin.isatty — make it pass.
    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *_a, **_kw: None)
    # No node, no bridge script — we shouldn't reach those steps anyway.

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cmd_whatsapp(MagicMock())
        except KeyboardInterrupt:
            pass

    assert _env_value(isolated_home, "WHATSAPP_ENABLED") is None, (
        "Setup aborted before pairing — WHATSAPP_ENABLED must not be set. "
        f"Got .env: {(isolated_home / '.env').read_text() if (isolated_home / '.env').exists() else '(missing)'}"
    )


def test_existing_pairing_skip_branch_enables_whatsapp(isolated_home, monkeypatch):
    """User runs ``hermes whatsapp`` with an existing paired session and
    chooses "no, keep my session" at the re-pair prompt.  The env var
    should be (re-)written to true so the gateway picks WhatsApp back up,
    even if the var was lost since the original pairing.
    """
    from hermes_cli.main_platform_setup import cmd_whatsapp

    # Pre-create a paired session WITHOUT WHATSAPP_ENABLED in .env.
    session = isolated_home / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / "creds.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WHATSAPP_MODE", "bot")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15551234567")

    # mode already set → skip mode prompt; users already set → skip update
    # prompt with "no"; pairing exists → "no, keep session" → return.
    inputs = iter(["n", "n"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            return "n"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "gateway.platforms.whatsapp_common.whatsapp_bridge_dependencies_fresh",
        lambda _bridge_dir: True,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_whatsapp(MagicMock())

    # The skip-rebar branch should have set the env var on its way out.
    assert _env_value(isolated_home, "WHATSAPP_ENABLED") == "true"
