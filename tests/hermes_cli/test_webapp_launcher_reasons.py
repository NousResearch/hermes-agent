"""``hermes webapp`` says why the Webapp is off.

When the ``webapp:`` block asks for the Webapp but is malformed, the command prints the block's own
problem (an invalid host, port, public URL or a non-boolean ``enabled``), never "Set webapp.enabled:
true" when it is already set; that advice is kept for a config that does not turn the Webapp on.
"""
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import webapp_launcher as launcher


def _args(**overrides) -> Namespace:
    values = {"no_open": False, "status": False, "sign_out": False, "build_only": False,
              "force_build": False, "skip_build": False}
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def home(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(launcher, "_build", lambda args: pytest.fail("built a Webapp that is off"))
    monkeypatch.setattr(launcher, "_ensure", lambda home: pytest.fail("started a gateway for a Webapp that is off"))
    return root


def _write(home: Path, config: str) -> None:
    (home / "config.yaml").write_text(config, encoding="utf-8")


@pytest.mark.parametrize(("block", "key"), [
    ("webapp:\n  enabled: true\n  port: 70000\n", "webapp.port"),
    ("webapp:\n  enabled: true\n  host: 192.168.1.20\n", "webapp.host"),
    ("webapp:\n  enabled: 'yes'\n", "webapp.enabled"),
])
def test_the_launcher_prints_why_a_malformed_webapp_block_keeps_it_off(home, capsys, block, key):
    _write(home, block)
    assert launcher.cmd_webapp(_args()) == 1
    out = capsys.readouterr().out
    assert key in out
    assert "Set webapp.enabled: true" not in out


def test_the_launcher_still_says_how_to_turn_an_absent_webapp_on(home, capsys):
    _write(home, "")
    assert launcher.cmd_webapp(_args()) == 1
    assert "Set webapp.enabled: true" in capsys.readouterr().out


def test_status_without_a_gateway_prints_why_the_webapp_is_off(home, monkeypatch, capsys):
    def unreachable(*_args, **_kwargs):
        raise launcher.GatewayUnreachable("no gateway")

    monkeypatch.setattr(launcher, "_control", unreachable)
    _write(home, "webapp:\n  enabled: true\n  port: 70000\n")
    assert launcher.cmd_webapp(_args(status=True)) == 1
    out = capsys.readouterr().out
    assert "webapp.port" in out and "Set webapp.enabled: true" not in out


@pytest.mark.parametrize(("block", "expected"), [
    ("webapp:\n  enabled: true\n  port: 70000\n", "webapp.port"),
    # Valid and on: the gateway predates the change and only needs a restart.
    ("webapp:\n  enabled: true\n", launcher.RESTART_COMMAND),
])
def test_status_of_a_gateway_that_serves_no_webapp_reads_the_current_config(home, monkeypatch, capsys, block, expected):
    answer = {"ok": True, "result": {"state": "off", "url": None, "reason": None}}
    monkeypatch.setattr(launcher, "_control", lambda *_args, **_kwargs: answer)
    _write(home, block)
    launcher.cmd_webapp(_args(status=True))
    out = capsys.readouterr().out
    assert expected in out and "Set webapp.enabled: true" not in out
