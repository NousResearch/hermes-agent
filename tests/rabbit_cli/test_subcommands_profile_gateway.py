"""Unit tests for extracted subcommand parser builders (profile, gateway).

Confirms the builders attach the same subactions and ``func=`` dispatch that
lived inline in ``main()`` before the god-file Phase 2 extraction.
"""

from __future__ import annotations

import argparse

from rabbit_cli.subcommands.gateway import build_gateway_parser


def _h_gateway(args):  # pragma: no cover - identity only
    return "gateway"


def _h_proxy(args):  # pragma: no cover - identity only
    return "proxy"






def _gateway_parser():
    p = argparse.ArgumentParser(prog="rabbit")
    sub = p.add_subparsers(dest="command")
    build_gateway_parser(
        sub,
        cmd_gateway=_h_gateway,
        cmd_proxy=_h_proxy,
    )
    return p










