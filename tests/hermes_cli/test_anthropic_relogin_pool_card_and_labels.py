"""Regression tests for #133328 (Anthropic re-login): pool logins invisible to the
Accounts card, and re-logins colliding on an auto-numbered label.

1. ``hermes auth add anthropic`` lands the login in the credential pool, NOT in the
   ``.anthropic_oauth.json`` singleton. ``_anthropic_oauth_status`` only read the singleton
   and env vars, so after a terminal login the "Anthropic Account" card stayed signed out
   while ``hermes auth list`` showed the credential — and the claude-code card (backed by
   ``~/.claude/.credentials.json``) looked like the only connected Anthropic entry.
2. The auto label was ``<provider>-oauth-{len(entries) + 1}``: after removing a mid-pool
   row (say #2 of three) the count re-issues the removed number, so the next login lands on
   the same label as the surviving #3 row.
"""

import json
import time


def _opaque_token(tag: str) -> str:
    # Assembled, not a literal: these are fake placeholders, never real credentials.
    return "-".join(("sk", "ant", "oat01", tag))


def _fake_refresh_token(tag: str) -> str:
    return "-".join(("rt", tag))


def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return home


def _write_pool(home, rows):
    store_path = home / "auth.json"
    store = json.loads(store_path.read_text()) if store_path.exists() else {}
    pool = store.setdefault("credential_pool", {})
    pool["anthropic"] = rows
    store_path.write_text(json.dumps(store))


def _pkce_row(label, *, expires_at_ms):
    return {
        "id": f"id-{label}",
        "label": label,
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:hermes_pkce",
        "access_token": _opaque_token(label),
        "refresh_token": _fake_refresh_token(label),
        "expires_at_ms": expires_at_ms,
    }


def test_pool_login_marks_anthropic_card_connected(tmp_path, monkeypatch):
    home = _hermes_home(tmp_path, monkeypatch)
    _write_pool(
        home,
        [
            _pkce_row(
                "anthropic-oauth-1", expires_at_ms=int(time.time() * 1000) + 3_600_000
            )
        ],
    )

    from hermes_cli.web_server_oauth import _anthropic_oauth_status

    status = _anthropic_oauth_status()
    assert status["logged_in"] is True
    assert status["source"] == "hermes_pkce"
    assert "anthropic-oauth-1" in status["source_label"]


def test_expired_pool_row_is_not_a_login(tmp_path, monkeypatch):
    home = _hermes_home(tmp_path, monkeypatch)
    _write_pool(
        home,
        [_pkce_row("anthropic-oauth-1", expires_at_ms=int(time.time() * 1000) - 1_000)],
    )

    from hermes_cli.web_server_oauth import _anthropic_oauth_status

    assert _anthropic_oauth_status()["logged_in"] is False


def test_non_pkce_pool_rows_do_not_connect_the_card(tmp_path, monkeypatch):
    home = _hermes_home(tmp_path, monkeypatch)
    _write_pool(
        home,
        [
            {
                **_pkce_row(
                    "seeded", expires_at_ms=int(time.time() * 1000) + 3_600_000
                ),
                "source": "env:ANTHROPIC_API_KEY",
            }
        ],
    )

    from hermes_cli.web_server_oauth import _anthropic_oauth_status

    assert _anthropic_oauth_status()["logged_in"] is False


def test_singleton_file_still_takes_precedence(tmp_path, monkeypatch):
    home = _hermes_home(tmp_path, monkeypatch)
    _write_pool(
        home,
        [
            _pkce_row(
                "anthropic-oauth-1", expires_at_ms=int(time.time() * 1000) + 3_600_000
            )
        ],
    )
    (home / ".anthropic_oauth.json").write_text(
        json.dumps({
            "accessToken": _opaque_token("singleton"),
            "refreshToken": _fake_refresh_token("singleton"),
            "expiresAt": int(time.time() * 1000) + 3_600_000,
        })
    )

    from hermes_cli.web_server_oauth import _anthropic_oauth_status

    status = _anthropic_oauth_status()
    assert status["logged_in"] is True
    assert ".anthropic_oauth.json" in status["source_label"]


def test_add_credential_skips_taken_label_numbers(tmp_path, monkeypatch):
    """Two surviving rows labeled #1 and #3 (the user removed #2): the next auto label
    must be #4, not the old count-based #3 that collides with the survivor."""
    from dataclasses import replace

    home = _hermes_home(tmp_path, monkeypatch)
    now_ms = int(time.time() * 1000)
    _write_pool(
        home,
        [
            _pkce_row("anthropic-oauth-1", expires_at_ms=now_ms + 3_600_000),
            _pkce_row("anthropic-oauth-3", expires_at_ms=now_ms + 3_600_000),
        ],
    )

    import hermes_cli.auth_commands as auth_commands
    from agent.credential_pool import load_pool

    spec = auth_commands._OAUTH_ADD_SPECS["anthropic"]
    monkeypatch.setitem(
        auth_commands._OAUTH_ADD_SPECS,
        "anthropic",
        replace(
            spec,
            login=lambda args: {
                "access_token": _opaque_token("fresh"),
                "refresh_token": _fake_refresh_token("fresh"),
                "expires_at_ms": now_ms + 3_600_000,
            },
        ),
    )

    class _Args:
        label = None

    entry = auth_commands._add_credential(
        _Args(), "anthropic", load_pool("anthropic"), "oauth"
    )

    assert entry.label == "anthropic-oauth-4"
    labels = [e.label for e in load_pool("anthropic").entries()]
    assert len(labels) == len(set(labels)) == 3
