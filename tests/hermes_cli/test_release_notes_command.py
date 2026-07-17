"""Tests for `hermes release-notes` command (upstream issue #64133)."""

import types
from unittest.mock import MagicMock, patch


def _resp(data, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        r.raise_for_status = MagicMock()
    return r


def _releases():
    return [
        {
            "tag_name": "v0.18.2",
            "published_at": "2026-07-07",
            "body": "## v0.18.2\n- boot_count fix\n- CLI added",
        },
        {
            "tag_name": "v0.18.1",
            "published_at": "2026-07-01",
            "body": "## v0.18.1\n- initial release",
        },
    ]


def _args(latest=False):
    return types.SimpleNamespace(latest=latest)


def test_latest_shows_newest_body(capsys):
    from hermes_cli.subcommands.release_notes import cmd_release_notes

    with patch("hermes_cli.subcommands.release_notes.httpx") as mock_http:
        mock_http.get.return_value = _resp(_releases())
        mock_http.HTTPError = Exception
        rc = cmd_release_notes(_args(latest=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "v0.18.2" in out
    assert "boot_count fix" in out


def test_network_error_returns_nonzero_with_hint(capsys):
    from hermes_cli.subcommands.release_notes import cmd_release_notes

    with patch("hermes_cli.subcommands.release_notes.httpx") as mock_http:
        mock_http.get.side_effect = Exception("boom")
        mock_http.HTTPError = Exception
        rc = cmd_release_notes(_args(latest=True))
    captured = capsys.readouterr()
    assert rc != 0
    combined = (captured.out + captured.err).lower()
    assert (
        "github_token" in combined
        or "gh auth" in combined
        or "network" in combined
        or "online" in combined
        or "fetch" in combined
    )


def test_picker_invoked_then_shows_selected(capsys):
    from hermes_cli.subcommands.release_notes import cmd_release_notes

    with patch("hermes_cli.subcommands.release_notes.httpx") as mock_http, patch(
        "hermes_cli.subcommands.release_notes.curses_single_select"
    ) as mock_pick:
        mock_http.get.return_value = _resp(_releases())
        mock_http.HTTPError = Exception
        mock_pick.return_value = 1  # user selects v0.18.1
        rc = cmd_release_notes(_args(latest=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert mock_pick.called
    assert "v0.18.1" in out
    assert "initial release" in out
