"""BasicAuthProvider must not sign sessions with a secret that is unexpanded shell text.

A ``HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)`` line written through
a quoted heredoc lands in ``.env`` literally, so every install that copied it would share
one known signing key. ``_resolve_secret`` must ignore such a value and fall back to a
random per-process key.
"""

from __future__ import annotations

import base64
import logging
import secrets

import pytest

from plugins.dashboard_auth import basic

LITERAL = "$(openssl rand -base64 32)"


@pytest.fixture(autouse=True)
def _clear_basic_env(monkeypatch):
    for var in (
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH",
        "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
        "HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("value", [
    LITERAL, "`openssl rand -base64 32`", "prefix-$(head -c 32 /dev/urandom)",
    # The web-dashboard.md placeholder pasted as-is, a spaced "$ (", and a bare command.
    "<32+ random bytes; openssl rand -base64 32>", "$ (openssl rand -base64 32)", "openssl rand -hex 32",
])
def test_env_secret_with_shell_text_is_not_used(monkeypatch, caplog, value):
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", value)
    with caplog.at_level(logging.ERROR, logger=basic.logger.name):
        first = basic._resolve_secret({})
        second = basic._resolve_secret({})
    assert first != value.encode("utf-8")
    assert len(first) == 32
    assert first != second  # freshly random, not derived from the text
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "unexpanded shell" in errors[0].getMessage()


def test_shell_text_secret_is_reported_once_on_stderr(monkeypatch, capsys):
    # The log files are not the console of `hermes serve` / `hermes dashboard`.
    monkeypatch.setattr(basic, "_shell_text_secret_reported", False, raising=False)
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", "$(openssl rand -hex 32)")
    basic._resolve_secret({})
    basic._resolve_secret({})
    err = capsys.readouterr().err
    assert err.count("HERMES_DASHBOARD_BASIC_AUTH_SECRET") == 1
    assert len(err.strip().splitlines()) == 1
    assert "$(openssl rand -hex 32)" not in err


def test_config_secret_with_shell_text_is_not_used():
    assert basic._resolve_secret({"secret": LITERAL}) != LITERAL.encode("utf-8")


def test_generated_secret_still_resolves(monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", base64.b64encode(key).decode())
    assert basic._resolve_secret({}) == key


def test_token_signed_with_the_literal_text_is_rejected(monkeypatch):
    monkeypatch.setattr(basic, "_load_config_basic_auth_section", dict)
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "admin")
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "choose-a-strong-password")
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", LITERAL)
    provider = basic.BasicAuthProvider(**basic._settings())

    forged = basic._sign(
        {"sub": "admin", "kind": "access", "exp": 2**31 - 1, "jti": "x"}, LITERAL.encode("utf-8"))
    assert provider.verify_session(access_token=forged) is None

    # The provider still registers and its own sessions work.
    session = provider.complete_password_login(username="admin", password="choose-a-strong-password")
    assert provider.verify_session(access_token=session.access_token) is not None
