"""``hermes completion`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_completion_parser(subparsers, *, cmd_completion: Callable, parser) -> None:
    """Attach the ``completion`` subcommand to ``subparsers``."""
    completion_parser = subparsers.add_parser(
        "completion", help="Set up or print shell completions (bash, zsh, or fish)")
    completion_parser.add_argument(
        "shell", nargs="?", choices=["bash", "zsh", "fish"],
        help="Legacy raw-script output; may also be used with an explicit mode")
    completion_parser.add_argument(
        "--shell", dest="shell_option", choices=["bash", "zsh", "fish"],
        help="Shell to configure")
    modes = completion_parser.add_mutually_exclusive_group()
    modes.add_argument("--install", dest="mode", action="store_const", const="install",
                       help="Install static per-user completions without prompts when --shell is given")
    modes.add_argument("--manual", dest="mode", action="store_const", const="manual",
                       help="Print setup instructions without writing files")
    modes.add_argument("--raw", dest="mode", action="store_const", const="raw",
                       help="Print only the completion script; requires a shell")
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))
