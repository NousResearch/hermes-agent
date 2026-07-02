"""distutils.cmd

Provides the Command class, the base class for the command classes
in the distutils.command package.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from abc import abstractmethod
from collections.abc import Callable, MutableSequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload

from . import _modified, archive_util, dir_util, file_util, util
from ._log import log
from .errors import DistutilsOptionError

if TYPE_CHECKING:
    # type-only import because of mutual dependence between these classes
    from distutils.dist import Distribution

    from typing_extensions import TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

_StrPathT = TypeVar("_StrPathT", bound="str | os.PathLike[str]")
_BytesPathT = TypeVar("_BytesPathT", bound="bytes | os.PathLike[bytes]")
_CommandT = TypeVar("_CommandT", bound="Command")


class Command:
    """Abstract base class for defining command classes, the "worker bees"
    of the Distutils.  A useful analogy for command classes is to think of
    them as subroutines with local variables called "options".  The options
    are "declared" in 'initialize_options()' and "defined" (given their
    final values, aka "finalized") in 'finalize_options()', both of which
    must be defined by every command class.  The distinction between the
    two is necessary because option values might come from the outside
    world (command line, config file, ...), and any options dependent on
    other options must be computed *after* these outside influences have
    been processed -- hence 'finalize_options()'.  The "body" of the
    subroutine, where it does all its work based on the values of its
    options, is the 'run()' method, which must also be implemented by every
    command class.
    """

    # 'sub_commands' formalizes the notion of a "family" of commands,
    # eg. "install" as the parent with sub-commands "install_lib",
    # "install_headers", etc.  The parent of a family of commands
    # defines 'sub_commands' as a class attribute; it's a list of
    #    (command_name : string, predicate : unbound_method | string | None)
    # tuples, where 'predicate' is a method of the parent command that
    # determines whether the corresponding command is applicable in the
    # current situation.  (Eg. we "install_headers" is only applicable if
    # we have any C header files to install.)  If 'predicate' is None,
    # that command is always applicable.
    #
    # 'sub_commands' is usually defined at the *end* of a class, because
    # predicates can be unbound methods, so they must already have been
    # defined.  The canonical example is the "install" command.
    sub_commands: ClassVar[  # Any to work around variance issues
        list[tuple[str, Callable[[Any], bool] | None]]
    ] = []

    user_options: ClassVar[
        # Specifying both because list is invariant. Avoids mypy override assignment issues
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = []

    # -- Creation/initialization methods -------------------------------

    def __init__(self, dist: Distribution) -> None:
        """Create and initialize a new Command object.  Most importantly,
        invokes the 'initialize_options()' method, which is the real
        initializer and depends on the actual command being
        instantiated.
        """
        # late import because of mutual dependence between these classes
        from distutils.dist import Distribution

        if not isinstance(dist, Distribution):
            raise TypeError("dist must be a Distribution instance")
        if self.__class__ is Command:
            raise RuntimeError("Command is an abstract class")

        self.distribution = dist
        self.initialize_options()

        # Per-command versions of the global flags, so that the user can
        # customize Distutils' behaviour command-by-command and let some
        # commands fall back on the Distribution's behaviour. None means
        # "not defined, check self.distribution's copy".

        # verbose is largely ignored, but needs to be set for
        # backwards compatibility (I think)?
        self.verbose = dist.verbose

        # Some commands define a 'self.force' option to ignore file
        # timestamps, but methods defined *here* assume that
        # 'self.force' exists for all commands.  So define it here
        # just to be safe.
        self.force = None

        # The 'help' flag is just used for command-line parsing, so
        # none of that complicated bureaucracy is needed.
        self.help = False

        # 'finalized' records whether or not 'finalize_options()' has been
        # called.  'finalize_options()' itself should not pay attention to
        # this flag: it is the business of 'ensure_finalized()', which
        # always calls 'finalize_options()', to respect/update it.
        self.finalized = False

    def ensure_finalized(self) -> None:
        if not self.finalized:
            self.finalize_options()
        self.finalized = True

    # Subclasses must define:
    #   initialize_options()
    #     provide default values for all options; may be customized by
    #     setup script, by options from config file(s), or by command-line
    #     options
    #   finalize_options()
    #     decide on the final values for all options; this is called
    #     after all possible intervention from the outside world
    #     (command-line, option file, etc.) has been processed
    #   run()
    #     run the command: do whatever it is we're here to do,
    #     controlled by the command's various option values

    @abstractmethod
    def initialize_options(self) -> None:
        """Set default values for all the options that this command
        supports.  Note that these defaults may be overridden by other
        commands, by the setup script, by config files, or by the
        command-line.  Thus, this is not the place to code dependencies
        between options; generally, 'initialize_options()' implementations
        are just a bunch of "self.foo = None" assignments.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    @abstractmethod
    def finalize_options(self) -> None:
        """Set final values for all the options that this command supports.
        This is always called as late as possible, ie.  after any option
        assignments from the command-line or from other commands have been
        done.  Thus, this is the place to code option dependencies: if
        'foo' depends on 'bar', then it is safe to set 'foo' from 'bar' as
        long as 'foo' still has the same value it was assigned in
        'initialize_options()'.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    def dump_options(self, header=None, indent=""):
        from distutils.fancy_getopt import longopt_xlate

        if header is None:
            header = f"command options for '{self.get_command_name()}':"
        self.announce(indent + header, level=logging.INFO)
        indent = indent + "  "
        for option, _, _ in self.user_options:
            option = option.translate(longopt_xlate)
            if option[-1] == "=":
                option = option[:-1]
            value = getattr(self, option)
            self.announce(indent + f"{option} = {value}", level=logging.INFO)

    @abstractmethod
    def run(self) -> None:
        """A command's raison d'etre: carry out the action it exists to
        perform, controlled by the options initialized in
        'initialize_options()', customized by other commands, the setup
        script, the command-line, and config files, and finalized in
        'finalize_options()'.  All terminal output and filesystem
        interaction should be done by 'run()'.

        This method must be implemented by all command classes.
        """
        raise RuntimeError(
            f"abstract method -- subclass {self.__class__} must override"
        )

    def announce(self, msg: object, level: int = logging.DEBUG) -> None:
        log.log(level, msg)

    def debug_print(self, msg: object) -> None:
        """Print 'msg' to stdout if the global DEBUG (taken from the
        DISTUTILS_DEBUG environment variable) flag is true.
        """
        from distutils.debug import DEBUG

        if DEBUG:
            print(msg)
            sys.stdout.flush()

    # -- Option validation methods -------------------------------------
    # (these are very handy in writing the 'finalize_options()' method)
    #
    # NB. the general philosophy here is to ensure that a particular option
    # value meets certain type and value constraints.  If not, we try to
    # force it into conformance (eg. if we expect a list but have a string,
    # split the string on comma and/or whitespace).  If we can't force the
    # option into conformance, raise DistutilsOptionError.  Thus, command
    # classes need do nothing more than (eg.)
    #   self.ensure_string_list('foo')
    # and they can be guaranteed that thereafter, self.foo will be
    # a list of strings.

    def _ensure_stringlike(self, option, what, default=None):
        val = getattr(self, option)
        if val is None:
            setattr(self, option, default)
            return default
        elif not isinstance(val, str):
            raise DistutilsOptionError(f"'{option}' must be a {what} (got `{val}`)")
        return val

    def ensure_string(self, option: str, default: str | None = None) -> None:
        """Ensure that 'option' is a string; if not defined, set it to
        'default'.
        """
        self._ensure_stringlike(option, "string", default)

    def ensure_string_list(self, option: str) -> None:
        r"""Ensure that 'option' is a list of strings.  If 'option' is
        currently a string, we split it either on /,\s*/ or /\s+/, so
        "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become
        ["foo", "bar", "baz"].
        """
        val = getattr(self, option)
        if val is None:
            return
        elif isinstance(val, str):
            setattr(self, option, re.split(r',\s*|\s+', val))
        else:
            if isinstance(val, list):
                ok = all(isinstance(v, str) for v in val)
            else:
                ok = False
            if not ok:
                raise DistutilsOptionError(
                    f"'{option}' must be a list of strings (got {val!r})"
                )

    def _ensure_tested_string(self, option, tester, what, error_fmt, default=None):
        val = self._ensure_stringlike(option, what, default)
        if val is not None and not tester(val):
            raise DistutilsOptionError(
                ("error in '%s' option: " + error_fmt) % (option, val)
            )

    def ensure_filename(self, option: str) -> None:
        """Ensure that 'option' is the name of an existing file."""
        self._ensure_tested_string(
            option, os.path.isfile, "filename", "'%s' does not exist or is not a file"
        )

    def ensure_dirname(self, option: str) -> None:
        self._ensure_tested_string(
            option,
            os.path.isdir,
            "directory name",
            "'%s' does not exist or is not a directory",
        )

    # -- Convenience methods for commands ------------------------------

    def get_command_name(self) -> str:
        if hasattr(self, 'command_name'):
            return self.command_name
        else:
            return self.__class__.__name__

    def set_undefined_options(
        self, src_cmd: str, *option_pairs: tuple[str, str]
    ) -> None:
        """Set the values of any "undefined" options from corresponding
        option values in some other command object.  "Undefined" here means
        "is None", which is the convention used to indicate that an option
        has not been changed between 'initialize_options()' and
        'finalize_options()'.  Usually called from 'finalize_options()' for
        options that depend on some other command rather than another
        option of the same command.  'src_cmd' is the other command from
        which option values will be taken (a command object will be created
        for it if necessary); the remaining arguments are
        '(src_option,dst_option)' tuples which mean "take the value of
        'src_option' in the 'src_cmd' command object, and copy it to
        'dst_option' in the current command object".
        """
        # Option_pairs: list of (src_option, dst_option) tuples
        src_cmd_obj = self.distribution.get_command_obj(src_cmd)
        src_cmd_obj.ensure_finalized()
        for src_option, dst_option in option_pairs:
            if getattr(self, dst_option) is None:
                setattr(self, dst_option, getattr(src_cmd_obj, src_option))

    # NOTE: Because distutils is private to Setuptools and not all commands are exposed here,
    # not every possible command is enumerated in the signature.
    def get_finalized_command(self, command: str, create: bool = True) -> Command:
        """Wrapper around Distribution's 'get_command_obj()' method: find
        (create if necessary and 'create' is true) the command object for
        'command', call its 'ensure_finalized()' method, and return the
        finalized command object.
        """
        cmd_obj = self.distribution.get_command_obj(command, create)
        cmd_obj.ensure_finalized()
        return cmd_obj

    # XXX rename to 'get_reinitialized_command()'? (should do the
    # same in dist.py, if so)
    @overload
    def reinitialize_command(
        self, command: str, reinit_subcommands: bool = False
    ) -> Command: ...
    @overload
    def reinitialize_command(
        self, command: _CommandT, reinit_subcommands: bool = False
    ) -> _CommandT: ...
    def reinitialize_command(
        self, command: str | Command, reinit_subcommands=False
    ) -> Command:
        return self.distribution.reinitialize_command(command, reinit_subcommands)

    def run_command(self, command: str) -> None:
        """Run some other command: uses the 'run_command()' method of
        Distribution, which creates and finalizes the command object if
        necessary and then invokes its 'run()' method.
        """
        self.distribution.run_command(command)

    def get_sub_commands(self) -> list[str]:
        """Determine the sub-commands that are relevant in the current
        distribution (ie., that need to be run).  This is based on the
        'sub_commands' class attribute: each tuple in that list may include
        a method that we call to determine if the subcommand needs to be
        run for the current distribution.  Return a list of command names.
        """
        commands = []
        for cmd_name, method in self.sub_commands:
            if method is None or method(self):
                commands.append(cmd_name)
        return commands

    # -- External world manipulation -----------------------------------

    def warn(self, msg: object) -> None:
        log.warning("warning: %s: %s\n", self.get_command_name(), msg)

    def execute(
        self,
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        msg: object = None,
        level: int = 1,
    ) -> None:
        util.execute(func, args, msg)

    def mkpath(self, name: str, mode: int = 0o777) -> None:
        dir_util.mkpath(name, mode)

    @overload
    def copy_file(
        self,
        infile: str | os.PathLike[str],
        outfile: _StrPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[_StrPathT | str, bool]: ...
    @overload
    def copy_file(
        self,
        infile: bytes | os.PathLike[bytes],
        outfile: _BytesPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[_BytesPathT | bytes, bool]: ...
    def copy_file(
        self,
        infile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        outfile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: int = 1,
    ) -> tuple[str | os.PathLike[str] | bytes | os.PathLike[bytes], bool]:
        """Copy a file respecting verbose, dry-run and force flags.  (The
        former two default to whatever is in the Distribution object, and
        the latter defaults to false for commands that don't define it.)"""
        return file_util.copy_file(
            infile,
            outfile,
            preserve_mode,
            preserve_times,
            not self.force,
            link,
        )

    def copy_tree(
        self,
        infile: str | os.PathLike[str],
        outfile: str,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        preserve_symlinks: bool = False,
        level: int = 1,
    ) -> list[str]:
        """Copy an entire directory tree respecting verbose, dry-run,
        and force flags.
        """
        return dir_util.copy_tree(
            infile,
            outfile,
            preserve_mode,
            preserve_times,
            preserve_symlinks,
            not self.force,
        )

    @overload
    def move_file(
        self, src: str | os.PathLike[str], dst: _StrPathT, level: int = 1
    ) -> _StrPathT | str: ...
    @overload
    def move_file(
        self, src: bytes | os.PathLike[bytes], dst: _BytesPathT, level: int = 1
    ) -> _BytesPathT | bytes: ...
    def move_file(
        self,
        src: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        dst: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        level: int = 1,
    ) -> str | os.PathLike[str] | bytes | os.PathLike[bytes]:
        """Move a file respecting dry-run flag."""
        return file_util.move_file(src, dst)

    def spawn(
        self, cmd: MutableSequence[str], search_path: bool = True, level: int = 1
    ) -> None:
        """Spawn an external command respecting dry-run flag."""
        from distutils.spawn import spawn

        spawn(cmd, search_path)

    @overload
    def make_archive(
        self,
        base_name: str,
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str: ...
    @overload
    def make_archive(
        self,
        base_name: str | os.PathLike[str],
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str: ...
    def make_archive(
        self,
        base_name: str | os.PathLike[str],
        format: str,
        root_dir: str | os.PathLike[str] | bytes | os.PathLike[bytes] | None = None,
        base_dir: str | None = None,
        owner: str | None = None,
        group: str | None = None,
    ) -> str:
        return archive_util.make_archive(
            base_name,
            format,
            root_dir,
            base_dir,
            owner=owner,
            group=group,
        )

    def make_file(
        self,
        infiles: str | list[str] | tuple[str, ...],
        outfile: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        exec_msg: object = None,
        skip_msg: object = None,
        level: int = 1,
    ) -> None:
        """Special case of 'execute()' for operations that process one or
        more input files and generate one output file.  Works just like
        'execute()', except the operation is skipped and a different
        message printed if 'outfile' already exists and is newer than all
        files listed in 'infiles'.  If the command defined 'self.force',
        and it is true, then the command is unconditionally run -- does no
        timestamp checks.
        """
        if skip_msg is None:
            skip_msg = f"skipping {outfile} (inputs unchanged)"

        # Allow 'infiles' to be a single string
        if isinstance(infiles, str):
            infiles = (infiles,)
        elif not isinstance(infiles, (list, tuple)):
            raise TypeError("'infiles' must be a string, or a list or tuple of strings")

        if exec_msg is None:
            exec_msg = "generating {} from {}".format(outfile, ', '.join(infiles))

        # If 'outfile' must be regenerated (either because it doesn't
        # exist, is out-of-date, or the 'force' flag is true) then
        # perform the action that presumably regenerates it
        if self.force or _modified.newer_group(infiles, outfile):
            self.execute(func, args, exec_msg, level)
        # Otherwise, print the "skip" message
        else:
            log.debug(skip_msg)
