"""Base option parser setup"""

from __future__ import annotations

import logging
import optparse
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Generator
from contextlib import suppress
from typing import Any, NoReturn

from pip._vendor.rich.markup import escape
from pip._vendor.rich.theme import Theme

from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.logging import PipConsole
from pip._internal.utils.misc import redact_auth_from_url, strtobool

logger = logging.getLogger(__name__)


class PrettyHelpFormatter(optparse.IndentedHelpFormatter):
    """A prettier/less verbose help formatter for optparse."""

    styles = {
        "optparse.shortargs": "green",
        "optparse.longargs": "cyan",
        "optparse.groups": "bold blue",
        "optparse.metavar": "yellow",
    }
    highlights = {
        r"\s(-{1}[\w]+[\w-]*)": "shortargs",  # highlight -letter as short args
        r"\s(-{2}[\w]+[\w-]*)": "longargs",  # highlight --words as long args
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # help position must be aligned with __init__.parseopts.description
        kwargs["max_help_position"] = 30
        kwargs["indent_increment"] = 1
        kwargs["width"] = shutil.get_terminal_size()[0] - 2
        super().__init__(*args, **kwargs)

    def format_option_strings(self, option: optparse.Option) -> str:
        """Return a comma-separated list of option strings and metavars."""
        opts = []

        if option._short_opts:
            opts.append(f"[optparse.shortargs]{option._short_opts[0]}[/]")
        if option._long_opts:
            opts.append(f"[optparse.longargs]{option._long_opts[0]}[/]")
        if len(opts) > 1:
            opts.insert(1, ", ")

        if option.takes_value():
            assert option.dest is not None
            metavar = option.metavar or option.dest.lower()
            opts.append(f" [optparse.metavar]<{escape(metavar.lower())}>[/]")

        return "".join(opts)

    def format_option(self, option: optparse.Option) -> str:
        """Overridden method with Rich support."""
        # fmt: off
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        # Remove the rich style tags before calculating width during
        # text wrap calculations. Also store the length removed to adjust
        # the padding in the else branch.
        stripped = re.sub(r"(\[[a-z.]+\])|(\[\/\])", "", opts)
        style_tag_length = len(opts) - len(stripped)
        if len(stripped) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)  # noqa: UP031
            indent_first = self.help_position
        else:                       # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "",      # noqa: UP031
                                  opt_width + style_tag_length, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            help_lines = textwrap.wrap(help_text, self.help_width)
            result.append("%*s%s\n" % (indent_first, "", help_lines[0]))  # noqa: UP031
            result.extend(["%*s%s\n" % (self.help_position, "", line)     # noqa: UP031
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)
        # fmt: on

    def format_heading(self, heading: str) -> str:
        if heading == "Options":
            return ""
        return "[optparse.groups]" + escape(heading) + ":[/]\n"

    def format_usage(self, usage: str) -> str:
        """
        Ensure there is only one newline between usage and the first heading
        if there is no description.
        """
        contents = self.indent_lines(textwrap.dedent(usage), "  ")
        msg = f"\n[optparse.groups]Usage:[/] {escape(contents)}\n"
        return msg

    def format_description(self, description: str | None) -> str:
        # leave full control over description to us
        if description:
            if hasattr(self.parser, "main"):
                label = "[optparse.groups]Commands:[/]"
            else:
                label = "[optparse.groups]Description:[/]"

            # some doc strings have initial newlines, some don't
            description = description.lstrip("\n")
            # some doc strings have final newlines and spaces, some don't
            description = description.rstrip()
            # dedent, then reindent
            description = self.indent_lines(textwrap.dedent(description), "  ")
            description = f"{label}\n{description}\n"
            return description
        else:
            return ""

    def format_epilog(self, epilog: str | None) -> str:
        # leave full control over epilog to us
        if epilog:
            return escape(epilog)
        else:
            return ""

    def expand_default(self, option: optparse.Option) -> str:
        """Overridden HelpFormatter.expand_default() which colorizes flags."""
        help = escape(super().expand_default(option))
        for regex, style in self.highlights.items():
            help = re.sub(regex, rf"[optparse.{style}] \1[/]", help)
        return help

    def indent_lines(self, text: str, indent: str) -> str:
        new_lines = [indent + line for line in text.split("\n")]
        return "\n".join(new_lines)


class UpdatingDefaultsHelpFormatter(PrettyHelpFormatter):
    """Custom help formatter for use in ConfigOptionParser.

    This is updates the defaults before expanding them, allowing
    them to show up correctly in the help listing.

    Also redact auth from url type options
    """

    def expand_default(self, option: optparse.Option) -> str:
        default_values = None
        if self.parser is not None:
            assert isinstance(self.parser, ConfigOptionParser)
            self.parser._update_defaults(self.parser.defaults)
            assert option.dest is not None
            default_values = self.parser.defaults.get(option.dest)
        help_text = super().expand_default(option)

        if default_values and option.metavar == "URL":
            if isinstance(default_values, str):
                default_values = [default_values]

            # If its not a list, we should abort and just return the help text
            if not isinstance(default_values, list):
                default_values = []

            for val in default_values:
                help_text = help_text.replace(val, redact_auth_from_url(val))

        return help_text


class CustomOptionParser(optparse.OptionParser):
    def insert_option_group(
        self, idx: int, *args: Any, **kwargs: Any
    ) -> optparse.OptionGroup:
        """Insert an OptionGroup at a given position."""
        group = self.add_option_group(*args, **kwargs)

        self.option_groups.pop()
        self.option_groups.insert(idx, group)

        return group

    @property
    def option_list_all(self) -> list[optparse.Option]:
        """Get a list of all options, including those in option groups."""
        res = self.option_list[:]
        for i in self.option_groups:
            res.extend(i.option_list)

        return res


class ConfigOptionParser(CustomOptionParser):
    """Custom option parser which updates its defaults by checking the
    configuration files and environmental variables"""

    def __init__(
        self,
        *args: Any,
        name: str,
        isolated: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.config = Configuration(isolated)

        assert self.name
        super().__init__(*args, **kwargs)

    def check_default(self, option: optparse.Option, key: str, val: Any) -> Any:
        try:
            return option.check_value(key, val)
        except optparse.OptionValueError as exc:
            print(f"An error occurred during configuration: {exc}")
            sys.exit(3)

    def _get_ordered_configuration_items(
        self,
    ) -> Generator[tuple[str, Any], None, None]:
        # Configuration gives keys in an unordered manner. Order them.
        override_order = ["global", self.name, ":env:"]

        # Pool the options into different groups
        # Use a dict because we need to implement the fallthrough logic after PR 12201
        # was merged which removed the fallthrough logic for options
        section_items_dict: dict[str, dict[str, Any]] = {
            name: {} for name in override_order
        }

        for _, value in self.config.items():
            for section_key, val in value.items():

                section, key = section_key.split(".", 1)
                if section in override_order:
                    section_items_dict[section][key] = val

        # Now that we a dict of items per section, convert to list of tuples
        # Make sure we completely remove empty values again
        section_items = {
            name: [(k, v) for k, v in section_items_dict[name].items() if v]
            for name in override_order
        }

        # Yield each group in their override order
        for section in override_order:
            yield from section_items[section]

    def _update_defaults(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Updates the given defaults with values from the config files and
        the environ. Does a little special handling for certain types of
        options (lists)."""

        # Accumulate complex default state.
        self.values = optparse.Values(self.defaults)
        late_eval = set()
        # Then set the options with those values
        for key, val in self._get_ordered_configuration_items():
            # '--' because configuration supports only long names
            option = self.get_option("--" + key)

            # Ignore options not present in this parser. E.g. non-globals put
            # in [global] by users that want them to apply to all applicable
            # commands.
            if option is None:
                continue

            assert option.dest is not None

            if option.action in ("store_true", "store_false"):
                try:
                    val = strtobool(val)
                except ValueError:
                    self.error(
                        f"{val} is not a valid value for {key} option, "
                        "please specify a boolean value like yes/no, "
                        "true/false or 1/0 instead."
                    )
            elif option.action == "count":
                with suppress(ValueError):
                    val = strtobool(val)
                with suppress(ValueError):
                    val = int(val)
                if not isinstance(val, int) or val < 0:
                    self.error(
                        f"{val} is not a valid value for {key} option, "
                        "please instead specify either a non-negative integer "
                        "or a boolean value like yes/no or false/true "
                        "which is equivalent to 1/0."
                    )
            elif option.action == "append":
                val = val.split()
                val = [self.check_default(option, key, v) for v in val]
            elif option.action == "callback":
                assert option.callback is not None
                late_eval.add(option.dest)
                opt_str = option.get_opt_string()
                val = option.convert_value(opt_str, val)
                # From take_action
                args = option.callback_args or ()
                kwargs = option.callback_kwargs or {}
                option.callback(option, opt_str, val, self, *args, **kwargs)
            else:
                val = self.check_default(option, key, val)

            defaults[option.dest] = val

        for key in late_eval:
            defaults[key] = getattr(self.values, key)
        self.values = None
        return defaults

    def get_default_values(self) -> optparse.Values:
        """Overriding to make updating the defaults after instantiation of
        the option parser possible, _update_defaults() does the dirty work."""
        if not self.process_default_values:
            # Old, pre-Optik 1.5 behaviour.
            return optparse.Values(self.defaults)

        # Load the configuration, or error out in case of an error
        try:
            self.config.load()
        except ConfigurationError as err:
            self.exit(UNKNOWN_ERROR, str(err))

        defaults = self._update_defaults(self.defaults.copy())  # ours
        for option in self._get_all_options():
            assert option.dest is not None
            default = defaults.get(option.dest)
            if isinstance(default, str):
                opt_str = option.get_opt_string()
                defaults[option.dest] = option.check_value(opt_str, default)
        return optparse.Values(defaults)

    def error(self, msg: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(UNKNOWN_ERROR, f"{msg}\n")

    def print_help(self, file: Any = None) -> None:
        # This is unfortunate but necessary since arguments may have not been
        # parsed yet at this point, so detect --no-color manually.
        no_color = (
            "--no-color" in sys.argv
            or bool(strtobool(os.environ.get("PIP_NO_COLOR", "no") or "no"))
            or "NO_COLOR" in os.environ
        )
        console = PipConsole(
            theme=Theme(PrettyHelpFormatter.styles), no_color=no_color, file=file
        )
        console.print(self.format_help().rstrip(), highlight=False)
