"""Smoke tests for the batch-extracted subcommand parser builders.

Each ``build_<group>_parser`` should attach its subcommand to a subparsers
group and wire ``func`` to the injected handler. These are intentionally
light — the byte-identical ``--help`` verification done at extraction time is
the real behavioral guarantee; this just guards against a module failing to
import or a builder raising.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.auth import build_auth_parser
from hermes_cli.subcommands.backup import build_backup_parser
from hermes_cli.subcommands.config import build_config_parser
from hermes_cli.subcommands.dashboard import build_dashboard_parser
from hermes_cli.subcommands.debug import build_debug_parser
from hermes_cli.subcommands.doctor import build_doctor_parser
from hermes_cli.subcommands.dump import build_dump_parser
from hermes_cli.subcommands.gui import build_gui_parser
from hermes_cli.subcommands.hooks import build_hooks_parser
from hermes_cli.subcommands.import_cmd import build_import_cmd_parser
from hermes_cli.subcommands.login import build_login_parser
from hermes_cli.subcommands.logout import build_logout_parser
from hermes_cli.subcommands.logs import build_logs_parser
from hermes_cli.subcommands.model import build_model_parser
from hermes_cli.subcommands.postinstall import build_postinstall_parser
from hermes_cli.subcommands.prompt_size import build_prompt_size_parser
from hermes_cli.subcommands.security import build_security_parser
from hermes_cli.subcommands.setup import build_setup_parser
from hermes_cli.subcommands.slack import build_slack_parser
from hermes_cli.subcommands.status import build_status_parser
from hermes_cli.subcommands.uninstall import build_uninstall_parser
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.subcommands.version import build_version_parser
from hermes_cli.subcommands.webhook import build_webhook_parser
from hermes_cli.subcommands.whatsapp import build_whatsapp_parser


def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler


# (subcommand_name, builder, handler_kwargs, sample_argv)
SINGLE_HANDLER_CASES = [
    ("model", build_model_parser, "cmd_model", ["model"]),
    ("setup", build_setup_parser, "cmd_setup", ["setup"]),
    ("postinstall", build_postinstall_parser, "cmd_postinstall", ["postinstall"]),
    ("whatsapp", build_whatsapp_parser, "cmd_whatsapp", ["whatsapp"]),
    ("slack", build_slack_parser, "cmd_slack", ["slack"]),
    ("login", build_login_parser, "cmd_login", ["login"]),
    ("logout", build_logout_parser, "cmd_logout", ["logout"]),
    ("auth", build_auth_parser, "cmd_auth", ["auth"]),
    ("status", build_status_parser, "cmd_status", ["status"]),
    ("webhook", build_webhook_parser, "cmd_webhook", ["webhook"]),
    ("hooks", build_hooks_parser, "cmd_hooks", ["hooks"]),
    ("doctor", build_doctor_parser, "cmd_doctor", ["doctor"]),
    ("security", build_security_parser, "cmd_security", ["security"]),
    ("dump", build_dump_parser, "cmd_dump", ["dump"]),
    ("debug", build_debug_parser, "cmd_debug", ["debug"]),
    ("backup", build_backup_parser, "cmd_backup", ["backup"]),
    ("import", build_import_cmd_parser, "cmd_import", ["import", "/tmp/x.zip"]),
    ("config", build_config_parser, "cmd_config", ["config"]),
    ("version", build_version_parser, "cmd_version", ["version"]),
    ("update", build_update_parser, "cmd_update", ["update"]),
    ("uninstall", build_uninstall_parser, "cmd_uninstall", ["uninstall"]),
    ("gui", build_gui_parser, "cmd_gui", ["gui"]),
    ("logs", build_logs_parser, "cmd_logs", ["logs"]),
    ("prompt-size", build_prompt_size_parser, "cmd_prompt_size", ["prompt-size"]),
]


@pytest.mark.parametrize("name,builder,kw,argv", SINGLE_HANDLER_CASES, ids=[c[0] for c in SINGLE_HANDLER_CASES])
def test_single_handler_builders(name, builder, kw, argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    handler = _h(name)
    builder(sub, **{kw: handler})
    ns = parser.parse_args(argv)
    assert ns.func is handler


def test_dashboard_builder_two_handlers():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    dash, reg = _h("dashboard"), _h("dashboard_register")
    build_dashboard_parser(sub, cmd_dashboard=dash, cmd_dashboard_register=reg)
    # bare dashboard -> launch handler
    assert parser.parse_args(["dashboard"]).func is dash
    # dashboard register -> register handler
    assert parser.parse_args(["dashboard", "register"]).func is reg
