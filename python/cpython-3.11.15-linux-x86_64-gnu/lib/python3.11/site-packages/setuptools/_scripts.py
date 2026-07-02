from __future__ import annotations

import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from ._importlib import metadata, resources

if TYPE_CHECKING:
    from typing_extensions import Self

from .warnings import SetuptoolsWarning

from distutils.command.build_scripts import first_line_re
from distutils.util import get_platform


class _SplitArgs(TypedDict, total=False):
    comments: bool
    posix: bool


class CommandSpec(list):
    """
    A command spec for a #! header, specified as a list of arguments akin to
    those passed to Popen.
    """

    options: list[str] = []
    split_args = _SplitArgs()

    @classmethod
    def best(cls):
        """
        Choose the best CommandSpec class based on environmental conditions.
        """
        return cls

    @classmethod
    def _sys_executable(cls):
        _default = os.path.normpath(sys.executable)
        return os.environ.get('__PYVENV_LAUNCHER__', _default)

    @classmethod
    def from_param(cls, param: Self | str | Iterable[str] | None) -> Self:
        """
        Construct a CommandSpec from a parameter to build_scripts, which may
        be None.
        """
        if isinstance(param, cls):
            return param
        if isinstance(param, str):
            return cls.from_string(param)
        if isinstance(param, Iterable):
            return cls(param)
        if param is None:
            return cls.from_environment()
        raise TypeError(f"Argument has an unsupported type {type(param)}")

    @classmethod
    def from_environment(cls):
        return cls([cls._sys_executable()])

    @classmethod
    def from_string(cls, string: str) -> Self:
        """
        Construct a command spec from a simple string representing a command
        line parseable by shlex.split.
        """
        items = shlex.split(string, **cls.split_args)
        return cls(items)

    def install_options(self, script_text: str):
        self.options = shlex.split(self._extract_options(script_text))
        cmdline = subprocess.list2cmdline(self)
        if not isascii(cmdline):
            self.options[:0] = ['-x']

    @staticmethod
    def _extract_options(orig_script):
        """
        Extract any options from the first line of the script.
        """
        first = (orig_script + '\n').splitlines()[0]
        match = _first_line_re().match(first)
        options = match.group(1) or '' if match else ''
        return options.strip()

    def as_header(self):
        return self._render(self + list(self.options))

    @staticmethod
    def _strip_quotes(item):
        _QUOTES = '"\''
        for q in _QUOTES:
            if item.startswith(q) and item.endswith(q):
                return item[1:-1]
        return item

    @staticmethod
    def _render(items):
        cmdline = subprocess.list2cmdline(
            CommandSpec._strip_quotes(item.strip()) for item in items
        )
        return '#!' + cmdline + '\n'


class WindowsCommandSpec(CommandSpec):
    split_args = _SplitArgs(posix=False)


class ScriptWriter:
    """
    Encapsulates behavior around writing entry point scripts for console and
    gui apps.
    """

    template = textwrap.dedent(
        r"""
        # EASY-INSTALL-ENTRY-SCRIPT: %(spec)r,%(group)r,%(name)r
        import re
        import sys

        # for compatibility with easy_install; see #2198
        __requires__ = %(spec)r

        try:
            from importlib.metadata import distribution
        except ImportError:
            try:
                from importlib_metadata import distribution
            except ImportError:
                from pkg_resources import load_entry_point


        def importlib_load_entry_point(spec, group, name):
            dist_name, _, _ = spec.partition('==')
            matches = (
                entry_point
                for entry_point in distribution(dist_name).entry_points
                if entry_point.group == group and entry_point.name == name
            )
            return next(matches).load()


        globals().setdefault('load_entry_point', importlib_load_entry_point)


        if __name__ == '__main__':
            sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
            sys.exit(load_entry_point(%(spec)r, %(group)r, %(name)r)())
        """
    ).lstrip()

    command_spec_class = CommandSpec

    @classmethod
    def get_args(cls, dist, header=None):
        """
        Yield write_script() argument tuples for a distribution's
        console_scripts and gui_scripts entry points.
        """

        # If distribution is not an importlib.metadata.Distribution, assume
        # it's a pkg_resources.Distribution and transform it.
        if not hasattr(dist, 'entry_points'):
            SetuptoolsWarning.emit("Unsupported distribution encountered.")
            dist = metadata.Distribution.at(dist.egg_info)

        if header is None:
            header = cls.get_header()
        spec = f'{dist.name}=={dist.version}'
        for type_ in 'console', 'gui':
            group = f'{type_}_scripts'
            for ep in dist.entry_points.select(group=group):
                name = ep.name
                cls._ensure_safe_name(ep.name)
                script_text = cls.template % locals()
                args = cls._get_script_args(type_, ep.name, header, script_text)
                yield from args

    @staticmethod
    def _ensure_safe_name(name):
        """
        Prevent paths in *_scripts entry point names.
        """
        has_path_sep = re.search(r'[\\/]', name)
        if has_path_sep:
            raise ValueError("Path separators not allowed in script names")

    @classmethod
    def best(cls):
        """
        Select the best ScriptWriter for this environment.
        """
        if sys.platform == 'win32' or (os.name == 'java' and os._name == 'nt'):
            return WindowsScriptWriter.best()
        else:
            return cls

    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        # Simply write the stub with no extension.
        yield (name, header + script_text)

    @classmethod
    def get_header(
        cls,
        script_text: str = "",
        executable: str | CommandSpec | Iterable[str] | None = None,
    ) -> str:
        """Create a #! line, getting options (if any) from script_text"""
        cmd = cls.command_spec_class.best().from_param(executable)
        cmd.install_options(script_text)
        return cmd.as_header()


class WindowsScriptWriter(ScriptWriter):
    command_spec_class = WindowsCommandSpec

    @classmethod
    def best(cls):
        """
        Select the best ScriptWriter suitable for Windows
        """
        writer_lookup = dict(
            executable=WindowsExecutableLauncherWriter,
            natural=cls,
        )
        # for compatibility, use the executable launcher by default
        launcher = os.environ.get('SETUPTOOLS_LAUNCHER', 'executable')
        return writer_lookup[launcher]

    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        "For Windows, add a .py extension"
        ext = dict(console='.pya', gui='.pyw')[type_]
        if ext not in os.environ['PATHEXT'].lower().split(';'):
            msg = (
                "{ext} not listed in PATHEXT; scripts will not be "
                "recognized as executables."
            ).format(**locals())
            SetuptoolsWarning.emit(msg)
        old = ['.pya', '.py', '-script.py', '.pyc', '.pyo', '.pyw', '.exe']
        old.remove(ext)
        header = cls._adjust_header(type_, header)
        blockers = [name + x for x in old]
        yield name + ext, header + script_text, 't', blockers

    @classmethod
    def _adjust_header(cls, type_, orig_header):
        """
        Make sure 'pythonw' is used for gui and 'python' is used for
        console (regardless of what sys.executable is).
        """
        pattern = 'pythonw.exe'
        repl = 'python.exe'
        if type_ == 'gui':
            pattern, repl = repl, pattern
        pattern_ob = re.compile(re.escape(pattern), re.IGNORECASE)
        new_header = pattern_ob.sub(string=orig_header, repl=repl)
        return new_header if cls._use_header(new_header) else orig_header

    @staticmethod
    def _use_header(new_header):
        """
        Should _adjust_header use the replaced header?

        On non-windows systems, always use. On
        Windows systems, only use the replaced header if it resolves
        to an executable on the system.
        """
        clean_header = new_header[2:-1].strip('"')
        return sys.platform != 'win32' or shutil.which(clean_header)


class WindowsExecutableLauncherWriter(WindowsScriptWriter):
    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        """
        For Windows, add a .py extension and an .exe launcher
        """
        if type_ == 'gui':
            launcher_type = 'gui'
            ext = '-script.pyw'
            old = ['.pyw']
        else:
            launcher_type = 'cli'
            ext = '-script.py'
            old = ['.py', '.pyc', '.pyo']
        hdr = cls._adjust_header(type_, header)
        blockers = [name + x for x in old]
        yield (name + ext, hdr + script_text, 't', blockers)
        yield (
            name + '.exe',
            get_win_launcher(launcher_type),
            'b',  # write in binary mode
        )
        if not is_64bit():
            # install a manifest for the launcher to prevent Windows
            # from detecting it as an installer (which it will for
            #  launchers like easy_install.exe). Consider only
            #  adding a manifest for launchers detected as installers.
            #  See Distribute #143 for details.
            m_name = name + '.exe.manifest'
            yield (m_name, load_launcher_manifest(name), 't')


def get_win_launcher(type):
    """
    Load the Windows launcher (executable) suitable for launching a script.

    `type` should be either 'cli' or 'gui'

    Returns the executable as a byte string.
    """
    launcher_fn = f'{type}.exe'
    if is_64bit():
        if get_platform() == "win-arm64":
            launcher_fn = launcher_fn.replace(".", "-arm64.")
        else:
            launcher_fn = launcher_fn.replace(".", "-64.")
    else:
        launcher_fn = launcher_fn.replace(".", "-32.")
    return resources.files('setuptools').joinpath(launcher_fn).read_bytes()


def load_launcher_manifest(name):
    res = resources.files(__name__).joinpath('launcher manifest.xml')
    return res.read_text(encoding='utf-8') % vars()


def _first_line_re():
    """
    Return a regular expression based on first_line_re suitable for matching
    strings.
    """
    if isinstance(first_line_re.pattern, str):
        return first_line_re

    # first_line_re in Python >=3.1.4 and >=3.2.1 is a bytes pattern.
    return re.compile(first_line_re.pattern.decode())


def is_64bit():
    return struct.calcsize("P") == 8


def isascii(s):
    try:
        s.encode('ascii')
    except UnicodeError:
        return False
    return True
