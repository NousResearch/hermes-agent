import argparse
import os

import pytest
from hermes_constants import set_hermes_home_override, reset_hermes_home_override

from hermes_cli.main import _read_ssh_session_token_file, cmd_dashboard
from hermes_cli.subcommands.dashboard import build_dashboard_parser


def dashboard_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_dashboard_parser(
        subparsers,
        cmd_dashboard=lambda _args: None,
        cmd_dashboard_register=lambda _args: None,
    )
    return parser


def test_serve_help_advertises_secure_ssh_bootstrap_flags(capsys):
    with pytest.raises(SystemExit) as exit_info:
        dashboard_parser().parse_args(["serve", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--ssh-session-token-file PATH" in output
    assert "--ssh-owner-nonce NONCE" in output








@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink contract")
def test_token_file_rejects_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    token_dir = home / ".hermes" / "desktop-ssh" / ("a" * 32)
    token_dir.mkdir(parents=True, mode=0o700)
    target = tmp_path / "token"
    target.write_text("b" * 64)
    target.chmod(0o600)
    token_path = token_dir / "0123456789abcdef.token"
    token_path.symlink_to(target)
    override = set_hermes_home_override(home / ".hermes")
    try:
        with pytest.raises(SystemExit, match="symlink|not accessible"):
            _read_ssh_session_token_file(str(token_path))
        assert not token_path.exists()
        assert target.read_text() == "b" * 64
    finally:
        reset_hermes_home_override(override)


