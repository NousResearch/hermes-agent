"""Regression tests for the Tools & Keys GitHub PAT save/remove path (#40041).

Users following generic docs add ``export GITHUB_TOKEN=ghp_...`` to
``~/.hermes/.env``. ``load_env()`` parses the export prefix (#6659), so every
UI shows the token as set (green light) — but ``save_env_value`` /
``remove_env_value`` only matched plain ``KEY=`` lines. Result: the UI could
neither replace nor remove the token (delete 404s as "not found in .env";
save appends a duplicate line that a later delete removes while the export
line silently resurrects the old value).

Fake tokens are constructed at runtime — no key-shaped literals on disk.
"""

import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

# Classic-PAT-shaped token, constructed at runtime (36 chars after prefix).
OLD_PAT = "ghp_" + "A" * 36
NEW_PAT = "ghp_" + "B" * 36


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / "pat_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()
    return home


def _write_env_raw(home, text):
    home.joinpath(".env").write_text(text, encoding="utf-8")
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()


def test_classic_pat_save_via_endpoint_succeeds(hermes_home):
    """Saving a ghp_* classic PAT through the env endpoint must not 500 —
    GITHUB_TOKEN is a REST/Skills-Hub credential, not a Copilot one."""
    resp = client.put(
        "/api/env", json={"key": "GITHUB_TOKEN", "value": NEW_PAT}, headers=HEADERS
    )
    assert resp.status_code == 200, resp.text

    from hermes_cli.config import load_env

    assert load_env()["GITHUB_TOKEN"] == NEW_PAT


def test_remove_export_prefixed_token(hermes_home):
    """DELETE must clear an ``export KEY=...`` line, not 404 on it."""
    _write_env_raw(hermes_home, f"export GITHUB_TOKEN={OLD_PAT}\n")

    resp = client.request(
        "DELETE", "/api/env", json={"key": "GITHUB_TOKEN"}, headers=HEADERS
    )
    assert resp.status_code == 200, (
        "export-prefixed lines are parsed by load_env (UI shows the token as "
        "set) so the delete path must recognise them too (#40041)"
    )

    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert OLD_PAT not in env_text

    from hermes_cli.config import load_env

    assert "GITHUB_TOKEN" not in load_env()


def test_update_export_prefixed_token_does_not_duplicate(hermes_home):
    """Saving over an ``export KEY=`` line must replace it in place."""
    _write_env_raw(hermes_home, f"export GITHUB_TOKEN={OLD_PAT}\n")

    resp = client.put(
        "/api/env", json={"key": "GITHUB_TOKEN", "value": NEW_PAT}, headers=HEADERS
    )
    assert resp.status_code == 200

    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert OLD_PAT not in env_text, "old exported token line must be replaced"
    assert env_text.count("GITHUB_TOKEN") == 1, (
        "save must not append a duplicate GITHUB_TOKEN line alongside the "
        "export-prefixed one"
    )

    from hermes_cli.config import load_env

    assert load_env()["GITHUB_TOKEN"] == NEW_PAT


def test_plain_line_save_and_remove_still_work(hermes_home):
    """Sanity: the ordinary KEY= path is unchanged."""
    from hermes_cli.config import load_env, remove_env_value, save_env_value

    save_env_value("GITHUB_TOKEN", OLD_PAT)
    assert load_env()["GITHUB_TOKEN"] == OLD_PAT
    save_env_value("GITHUB_TOKEN", NEW_PAT)
    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert env_text.count("GITHUB_TOKEN") == 1
    assert remove_env_value("GITHUB_TOKEN") is True
    assert "GITHUB_TOKEN" not in load_env()


def test_export_line_with_comment_untouched(hermes_home):
    """Commented-out export lines are not live assignments — leave them."""
    _write_env_raw(
        hermes_home,
        f"# export GITHUB_TOKEN={OLD_PAT}\nOTHER_KEY=value\n",
    )

    resp = client.request(
        "DELETE", "/api/env", json={"key": "GITHUB_TOKEN"}, headers=HEADERS
    )
    assert resp.status_code == 404
    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert "# export GITHUB_TOKEN=" in env_text
    assert "OTHER_KEY=value" in env_text


# -- whitespace around '=' (same resurrection hole, other line form) ----------


def test_remove_token_written_with_spaces_around_equals(hermes_home):
    """DELETE must clear a ``KEY = value`` line, not 404 on it.

    ``load_env()`` splits on the first ``=`` and strips the name, so this is a
    live assignment and every UI shows the token as set — the writers have to
    recognise it for the same reason they recognise ``export`` (#40041).
    """
    _write_env_raw(hermes_home, f"GITHUB_TOKEN = {OLD_PAT}\n")

    from hermes_cli.config import load_env

    assert load_env()["GITHUB_TOKEN"] == OLD_PAT, "precondition: token is live"

    resp = client.request(
        "DELETE", "/api/env", json={"key": "GITHUB_TOKEN"}, headers=HEADERS
    )
    assert resp.status_code == 200, resp.text

    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert OLD_PAT not in env_text
    assert "GITHUB_TOKEN" not in load_env()


def test_spaced_token_rotate_then_delete_does_not_resurrect(hermes_home):
    """Rotating over a spaced line must replace it, not shadow it.

    Otherwise the save appends a second line and a later delete removes only
    that one — silently restoring the key the user rotated away from.
    """
    _write_env_raw(hermes_home, f"GITHUB_TOKEN = {OLD_PAT}\n")

    resp = client.put(
        "/api/env", json={"key": "GITHUB_TOKEN", "value": NEW_PAT}, headers=HEADERS
    )
    assert resp.status_code == 200
    env_text = hermes_home.joinpath(".env").read_text(encoding="utf-8")
    assert OLD_PAT not in env_text, "the rotated-away token must not survive"
    assert env_text.count("GITHUB_TOKEN") == 1

    from hermes_cli.config import load_env

    assert load_env()["GITHUB_TOKEN"] == NEW_PAT

    client.request("DELETE", "/api/env", json={"key": "GITHUB_TOKEN"}, headers=HEADERS)
    assert "GITHUB_TOKEN" not in load_env(), "the old token must not resurrect"


def test_writer_matches_exactly_what_load_env_accepts(hermes_home):
    """Parity: the writers must recognise a line iff ``load_env()`` does."""
    from hermes_cli.config import _env_line_defines_key, load_env

    for line, expected in [
        (f"GITHUB_TOKEN={OLD_PAT}", True),
        (f"export GITHUB_TOKEN={OLD_PAT}", True),
        (f"GITHUB_TOKEN = {OLD_PAT}", True),
        (f"GITHUB_TOKEN\t=\t{OLD_PAT}", True),
        (f"export GITHUB_TOKEN = {OLD_PAT}", True),
        (f"# GITHUB_TOKEN={OLD_PAT}", False),
        (f"GITHUB_TOKEN_EXTRA={OLD_PAT}", False),
        (f"MY_GITHUB_TOKEN={OLD_PAT}", False),
    ]:
        _write_env_raw(hermes_home, line + "\n")
        assert _env_line_defines_key(line, "GITHUB_TOKEN") is expected, line
        assert ("GITHUB_TOKEN" in load_env()) is expected, (
            f"writer and load_env disagree on: {line!r}"
        )
