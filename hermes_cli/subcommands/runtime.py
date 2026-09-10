"""``hermes runtime`` subcommand parser."""

from __future__ import annotations

from typing import Callable

from hermes_cli.subcommands._shared import add_json_flag, add_yes_flag


def build_runtime_parser(subparsers, *, cmd_runtime: Callable) -> None:
    """Attach the ``runtime`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "runtime",
        help="Assign isolated local/docker/ssh execution runtimes to bots",
        description=(
            "Register named execution targets (local, Docker, or SSH), assign one "
            "to the current profile, test connectivity without starting a bot, "
            "and stop only that profile's containers. Does not provision cloud "
            "resources or store secrets in config.yaml."
        ),
    )
    subs = parser.add_subparsers(dest="runtime_action")

    add = subs.add_parser("add", help="Register a named runtime target")
    add.add_argument("name", help="Runtime name (letters, digits, hyphen, underscore)")
    add.add_argument(
        "--kind", required=True, choices=("local", "docker", "ssh"),
        help="Execution target kind (cloud kinds are not provisioned here)",
    )
    add.add_argument("--workspace", default="", help="Project/workspace directory for this bot")
    add.add_argument("--image", default="", help="Container image (docker kind)")
    add.add_argument(
        "--volume", action="append", default=[],
        help="Docker volume HOST:CONTAINER (repeatable)",
    )
    add.add_argument("--cpu", type=int, default=None, help="container_cpu limit")
    add.add_argument("--memory", type=int, default=None, help="container_memory in MB")
    add.add_argument("--ssh-host", default="", dest="ssh_host", help="SSH host (not a secret)")
    add.add_argument("--ssh-user", default="", dest="ssh_user", help="SSH user")
    add.add_argument("--ssh-port", type=int, default=None, dest="ssh_port", help="SSH port")

    lst = subs.add_parser("list", aliases=["ls"], help="List registered runtimes")
    add_json_flag(lst, "Print catalog as JSON")

    rm = subs.add_parser("remove", aliases=["rm"], help="Unregister a runtime (does not stop other bots)")
    rm.add_argument("name", help="Runtime name to remove")

    assign = subs.add_parser("assign", help="Assign a runtime to this profile and apply terminal.*")
    assign.add_argument("name", help="Registered runtime name")

    subs.add_parser("unassign", help="Clear the assigned runtime name")

    test = subs.add_parser("test", help="Test connectivity without starting a bot")
    test.add_argument("name", nargs="?", default="", help="Runtime name (default: assigned)")

    status = subs.add_parser("status", help="Show assigned runtime, probe, and this profile's containers")
    add_json_flag(status, "Print status as JSON")

    stop = subs.add_parser("stop", help="Stop this profile's runtime containers only")
    add_yes_flag(stop, "Confirm stop/remove for this profile's containers")

    parser.set_defaults(func=cmd_runtime)
