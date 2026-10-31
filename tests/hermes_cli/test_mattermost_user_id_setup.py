"""Mattermost setup must name a user-id lookup Mattermost actually documents.

The profile dialog does not show the id. System Console → User Management →
Users, and GET /api/v4/users/username/<username>, do. /users/me with the bot
token is the bot.
"""

import io


def test_gateway_setup_points_at_a_documented_user_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import gateway as gw

    platform = next(p for p in gw._PLATFORMS if p["key"] == "mattermost")
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("https://mm.example.com\ntok-abc\nuserid26\n")
    )
    gw._setup_standard_platform(dict(platform))

    out = capsys.readouterr().out
    assert "User Management" in out
    assert "/api/v4/users/username/" in out
    assert "displayed there" not in out
    assert "click your avatar" not in out
    assert "bot's id" in out


def test_plugin_setup_points_at_a_documented_user_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.mattermost.adapter import interactive_setup

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO("https://mm.example.com\ntok-abc\nuserid26\n\n"),
    )
    interactive_setup()

    out = capsys.readouterr().out
    assert "User Management" in out
    assert "/api/v4/users/username/" in out
    assert "click your avatar" not in out
    assert "bot's id" in out
