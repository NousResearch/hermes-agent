"""``hermes feishu login|status|logout`` reaches the plugin's own argparse tree (#11540).

Feishu is a *deferred* bundled platform, so the command only exists if
``_resolve_deferred_platform_cli_command`` imports the plugin and its ``register_cli_command`` side
effect runs. Building the real parser is the only way to catch the argparse ``invalid choice``
failure mode that bit ``hermes photon`` in #54678.
"""

import sys

import pytest

from tests.tools.feishu_user_helpers import configure_app, store_grant


def _parse(argv):
    from hermes_cli import main

    original = sys.argv
    sys.argv = ["hermes", *argv]
    try:
        parser, _subparsers = main._build_cli_parser()
        return parser.parse_args(argv)
    finally:
        sys.argv = original


@pytest.mark.parametrize("verb", ["login", "status", "logout"])
def test_every_verb_resolves_through_the_real_parser(verb):
    args = _parse(["feishu", verb])
    assert args.feishu_command == verb
    # Identity would compare the plugin loader's own module object against a plain import, which is
    # a different object by design; the handler's origin is what this pins.
    assert args.func.__name__ == "dispatch"
    assert args.func.__module__.endswith("feishu_cli")


def test_a_bare_hermes_feishu_reports_status_rather_than_erroring(capsys):
    from plugins.platforms.feishu import feishu_cli

    args = _parse(["feishu"])
    assert getattr(args, "feishu_command", None) is None
    assert feishu_cli.dispatch(args) == 0
    out = capsys.readouterr().out
    assert "not authorized" in out
    assert "hermes feishu login" in out


def test_login_prints_scopes_and_expiry_but_never_token_material(monkeypatch, capsys):
    """A login confirmation is read over shoulders and pasted into issues; keep tokens out of it."""
    from plugins.platforms.feishu import feishu_cli
    from tools import feishu_user_auth

    monkeypatch.setattr(feishu_user_auth, "login", lambda **_kwargs: {
        "granted_scope": "offline_access search:message", "expires_at": "2099-01-01T00:00:00+00:00",
        "access_token": "u-secret-access", "refresh_token": "u-secret-refresh",
    })
    args = _parse(["feishu", "login"])
    assert feishu_cli.dispatch(args) == 0
    out = capsys.readouterr().out
    assert "offline_access search:message" in out
    assert "u-secret-access" not in out
    assert "u-secret-refresh" not in out


def test_login_failure_is_a_nonzero_exit_with_the_reason_on_stderr(monkeypatch, capsys):
    from plugins.platforms.feishu import feishu_cli
    from tools import feishu_user_auth

    def _boom(**_kwargs):
        raise RuntimeError("redirect URI is not allow-listed")

    monkeypatch.setattr(feishu_user_auth, "login", _boom)
    assert feishu_cli.dispatch(_parse(["feishu", "login"])) == 1
    assert "redirect URI is not allow-listed" in capsys.readouterr().err


def test_status_then_logout_reflects_the_stored_grant(monkeypatch, capsys):
    from plugins.platforms.feishu import feishu_cli

    configure_app(monkeypatch)
    store_grant()
    assert feishu_cli.dispatch(_parse(["feishu", "status"])) == 0
    assert "authorized" in capsys.readouterr().out

    assert feishu_cli.dispatch(_parse(["feishu", "logout"])) == 0
    assert "Forgot" in capsys.readouterr().out
    assert feishu_cli.dispatch(_parse(["feishu", "logout"])) == 0
    assert "No Feishu / Lark user grant" in capsys.readouterr().out
