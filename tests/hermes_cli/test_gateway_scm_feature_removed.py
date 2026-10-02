"""Pins the REMOVAL of the MSIX SCM Windows service feature (audit C06/C07).

Settled user decision (2026-09-03): Windows gateway supervision stays on the
existing user-logon Scheduled Task. The desktop6:Service manifest fragment
and the `gateway run --service` / `hermes gateway service` frontend were
removed because the desktop6 schema cannot grant the promised installing-user
account model (StartAccount is required and limited to
localSystem|localService|networkService). These tests pin the surface the
feature used to own so it cannot silently come back.
"""

from __future__ import annotations

import argparse

import hermes_cli.config_defaults as config_defaults
from hermes_cli.subcommands import gateway as gateway_subcommands


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    gateway_subcommands.build_gateway_parser(
        subparsers, cmd_gateway=lambda *_: None, cmd_proxy=lambda *_: None,
        cmd_gateway_enroll=lambda *_: None,
    )
    return parser


def test_gateway_subparser_has_no_service_subcommand(capsys):
    parser = _build_parser()
    try:
        parser.parse_args(["gateway", "service", "on"])
    except SystemExit:
        pass
    else:
        raise AssertionError("`hermes gateway service` must stay removed")
    finally:
        capsys.readouterr()


def test_gateway_run_has_no_service_flag():
    parser = _build_parser()
    args = parser.parse_args(["gateway", "run"])
    assert not hasattr(args, "service"), "gateway run --service must stay removed"


def test_config_defaults_have_no_gateway_service_key():
    assert "service" not in config_defaults.DEFAULT_CONFIG["gateway"], (
        "gateway.service (the SCM posture key) must stay removed"
    )
