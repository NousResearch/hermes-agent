from argparse import ArgumentParser
from unittest.mock import patch

from hermes_cli.mobile import build_mobile_parser


def test_mobile_pair_prints_qr_and_fallback_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_mobile_parser(subparsers)
    args = parser.parse_args(["mobile", "pair", "--url", "https://host.example"])

    with patch("gateway.mobile_notifications.mobile_extension_enabled", return_value=True), patch(
        "hermes_cli.dingtalk_auth.render_qr_to_terminal", return_value=True
    ) as render:
        args.func(args)

    payload = render.call_args.args[0]
    output = capsys.readouterr().out
    assert payload.startswith("hermes://pair?url=https%3A%2F%2Fhost.example&grant=")
    assert "Fallback code:" in output
    assert "Expires in 5 minutes" in output
