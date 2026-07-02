"""Provide access to Python's configuration information.  The specific
configuration variables available depend heavily on the platform and
configuration.  The values may be retrieved using
get_config_var(name), and the list of variables is available via
get_config_vars().keys().  Additional convenience functions are also
available.

Written by:   Fred L. Drake, Jr.
Email:        <fdrake@acm.org>
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import sys
import sysconfig
from typing import TYPE_CHECKING, Literal, overload

from jaraco.functools import pass_none

from .ccompiler import CCompiler
from .compat import py39
from .errors import DistutilsPlatformError
from .util import is_mingw

if TYPE_CHECKING:
    from typing_extensions import deprecated
else:

    def deprecated(message):
        return lambda fn: fn


IS_PYPY = '__pypy__' in sys.builtin_module_names

# These are needed in a couple of spots, so just compute them once.
PREFIX = os.path.normpath(sys.prefix)
EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
BASE_PREFIX = os.path.normpath(sys.base_prefix)
BASE_EXEC_PREFIX = os.path.normpath(sys.base_exec_prefix)

# Path to the base directory of the project. On Windows the binary may
# live in project/PCbuild/win32 or project/PCbuild/amd64.
# set for cross builds
if "_PYTHON_PROJECT_BASE" in os.environ:
    project_base = os.path.abspath(os.environ["_PYTHON_PROJECT_BASE"])
else:
    if sys.executable:
        project_base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # sys.executable can be empty if argv[0] has been changed and Python is
        # unable to retrieve the real program name
        project_base = os.getcwd()


def _is_python_source_dir(d):
    """
    Return True if the target directory appears to point to an
    un-installed Python.
    """
    modules = pathlib.Path(d).joinpath('Modules')
    return any(modules.joinpath(fn).is_file() for fn in ('Setup', 'Setup.local'))


_sys_home = getattr(sys, '_home', None)


def _is_parent(dir_a, dir_b):
    """
    Return True if a is a parent of b.
    """
    return os.path.normcase(dir_a).startswith(os.path.normcase(dir_b))


if os.name == 'nt':

    @pass_none
    def _fix_pcbuild(d):
        # In a venv, sys._home will be inside BASE_PREFIX rather than PREFIX.
        prefixes = PREFIX, BASE_PREFIX
        matched = (
            prefix
            for prefix in prefixes
            if _is_parent(d, os.path.join(prefix, "PCbuild"))
        )
        return next(matched, d)

    project_base = _fix_pcbuild(project_base)
    _sys_home = _fix_pcbuild(_sys_home)


def _python_build():
    if _sys_home:
        return _is_python_source_dir(_sys_home)
    return _is_python_source_dir(project_base)


python_build = _python_build()


# Calculate the build qualifier flags if they are defined.  Adding the flags
# to the include and lib directories only makes sense for an installation, not
# an in-source build.
build_flags = ''
try:
    if not python_build:
        build_flags = sys.abiflags
except AttributeError:
    # It's not a configure-based build, so the sys module doesn't have
    # this attribute, which is fine.
    pass


def get_python_version():
    """Return a string containing the major and minor Python version,
    leaving off the patchlevel.  Sample return values could be '1.5'
    or '2.2'.
    """
    return f'{sys.version_info.major}.{sys.version_info.minor}'


def get_python_inc(plat_specific: bool = False, prefix: str | None = None) -> str:
    """Return the directory containing installed Python header files.

    If 'plat_specific' is false (the default), this is the path to the
    non-platform-specific header files, i.e. Python.h and so on;
    otherwise, this is the path to platform-specific header files
    (namely pyconfig.h).

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """
    default_prefix = BASE_EXEC_PREFIX if plat_specific else BASE_PREFIX
    resolved_prefix = prefix if prefix is not None else default_prefix
    # MinGW imitates posix like layout, but os.name != posix
    os_name = "posix" if is_mingw() else os.name
    try:
        getter = globals()[f'_get_python_inc_{os_name}']
    except KeyError:
        raise DistutilsPlatformError(
            "I don't know where Python installs its C header files "
            f"on platform '{os.name}'"
        )
    return getter(resolved_prefix, prefix, plat_specific)


@pass_none
def _extant(path):
    """
    Replace path with None if it doesn't exist.
    """
    return path if os.path.exists(path) else None


def _get_python_inc_posix(prefix, spec_prefix, plat_specific):
    return (
        _get_python_inc_posix_python(plat_specific)
        or _extant(_get_python_inc_from_config(plat_specific, spec_prefix))
        or _get_python_inc_posix_prefix(prefix)
    )


def _get_python_inc_posix_python(plat_specific):
    """
    Assume the executable is in the build directory. The
    pyconfig.h file should be in the same directory. Since
    the build directory may not be the source directory,
    use "srcdir" from the makefile to find the "Include"
    directory.
    """
    if not python_build:
        return
    if plat_specific:
        return _sys_home or project_base
    incdir = os.path.join(get_config_var('srcdir'), 'Include')
    return os.path.normpath(incdir)


def _get_python_inc_from_config(plat_specific, spec_prefix):
    """
    If no prefix was explicitly specified, provide the include
    directory from the config vars. Useful when
    cross-compiling, since the config vars may come from
    the host
    platform Python installation, while the current Python
    executable is from the build platform installation.

    >>> monkeypatch = getfixture('monkeypatch')
    >>> gpifc = _get_python_inc_from_config
    >>> monkeypatch.setitem(gpifc.__globals__, 'get_config_var', str.lower)
    >>> gpifc(False, '/usr/bin/')
    >>> gpifc(False, '')
    >>> gpifc(False, None)
    'includepy'
    >>> gpifc(True, None)
    'confincludepy'
    """
    if spec_prefix is None:
        return get_config_var('CONF' * plat_specific + 'INCLUDEPY')


def _get_python_inc_posix_prefix(prefix):
    implementation = 'pypy' if IS_PYPY else 'python'
    python_dir = implementation + get_python_version() + build_flags
    return os.path.join(prefix, "include", python_dir)


def _get_python_inc_nt(prefix, spec_prefix, plat_specific):
    if python_build:
        # Include both include dirs to ensure we can find pyconfig.h
        return (
            os.path.join(prefix, "include")
            + os.path.pathsep
            + os.path.dirname(sysconfig.get_config_h_filename())
        )
    return os.path.join(prefix, "include")


# allow this behavior to be monkey-patched. Ref pypa/distutils#2.
def _posix_lib(standard_lib, libpython, early_prefix, prefix):
    if standard_lib:
        return libpython
    else:
        return os.path.join(libpython, "site-packages")


def get_python_lib(
    plat_specific: bool = False, standard_lib: bool = False, prefix: str | None = None
) -> str:
    """Return the directory containing the Python library (standard or
    site additions).

    If 'plat_specific' is true, return the directory containing
    platform-specific modules, i.e. any module from a non-pure-Python
    module distribution; otherwise, return the platform-shared library
    directory.  If 'standard_lib' is true, return the directory
    containing standard Python library modules; otherwise, return the
    directory for site-specific modules.

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """

    early_prefix = prefix

    if prefix is None:
        if standard_lib:
            prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
        else:
            prefix = plat_specific and EXEC_PREFIX or PREFIX

    if os.name == "posix" or is_mingw():
        if plat_specific or standard_lib:
            # Platform-specific modules (any module from a non-pure-Python
            # module distribution) or standard Python library modules.
            libdir = getattr(sys, "platlibdir", "lib")
        else:
            # Pure Python
            libdir = "lib"
        implementation = 'pypy' if IS_PYPY else 'python'
        libpython = os.path.join(prefix, libdir, implementation + get_python_version())
        return _posix_lib(standard_lib, libpython, early_prefix, prefix)
    elif os.name == "nt":
        if standard_lib:
            return os.path.join(prefix, "Lib")
        else:
            return os.path.join(prefix, "Lib", "site-packages")
    else:
        raise DistutilsPlatformError(
            f"I don't know where Python installs its library on platform '{os.name}'"
        )


@functools.lru_cache
def _customize_macos():
    """
    Perform first-time customization of compiler-related
    config vars on macOS. Use after a compiler is known
    to be needed. This customization exists primarily to support Pythons
    from binary installers. The kind and paths to build tools on
    the user system may vary significantly from the system
    that Python itself was built on.  Also the user OS
    version and build tools may not support the same set
    of CPU architectures for universal builds.
    """

    sys.platform == "darwin" and __import__('_osx_support').customize_compiler(
        get_config_vars()
    )


def customize_compiler(compiler: CCompiler) -> None:
    """Do any platform-specific customization of a CCompiler instance.

    Mainly needed on Unix, so we can plug in the information that
    varies across Unices and is stored in Python's Makefile.
    """
    if compiler.compiler_type in ["unix", "cygwin"] or (
        compiler.compiler_type == "mingw32" and is_mingw()
    ):
        _customize_macos()

        (
            cc,
            cxx,
            cflags,
            ccshared,
            ldshared,
            ldcxxshared,
            shlib_suffix,
            ar,
            ar_flags,
        ) = get_config_vars(
            'CC',
            'CXX',
            'CFLAGS',
            'CCSHARED',
            'LDSHARED',
            'LDCXXSHARED',
            'SHLIB_SUFFIX',
            'AR',
            'ARFLAGS',
        )

        cxxflags = cflags

        if 'CC' in os.environ:
            newcc = os.environ['CC']
            if 'LDSHARED' not in os.environ and ldshared.startswith(cc):
                # If CC is overridden, use that as the default
                #       command for LDSHARED as well
                ldshared = newcc + ldshared[len(cc) :]
            cc = newcc
        cxx = os.environ.get('CXX', cxx)
        ldshared = os.environ.get('LDSHARED', ldshared)
        ldcxxshared = os.environ.get('LDCXXSHARED', ldcxxshared)
        cpp = os.environ.get(
            'CPP',
            cc + " -E",  # not always
        )

        ldshared = _add_flags(ldshared, 'LD')
        ldcxxshared = _add_flags(ldcxxshared, 'LD')
        cflags = os.environ.get('CFLAGS', cflags)
        ldshared = _add_flags(ldshared, 'C')
        cxxflags = os.environ.get('CXXFLAGS', cxxflags)
        ldcxxshared = _add_flags(ldcxxshared, 'CXX')
        cpp = _add_flags(cpp, 'CPP')
        cflags = _add_flags(cflags, 'CPP')
        cxxflags = _add_flags(cxxflags, 'CPP')
        ldshared = _add_flags(ldshared, 'CPP')
        ldcxxshared = _add_flags(ldcxxshared, 'CPP')

        ar = os.environ.get('AR', ar)

        archiver = ar + ' ' + os.environ.get('ARFLAGS', ar_flags)
        cc_cmd = cc + ' ' + cflags
        cxx_cmd = cxx + ' ' + cxxflags

        compiler.set_executables(
            preprocessor=cpp,
            compiler=cc_cmd,
            compiler_so=cc_cmd + ' ' + ccshared,
            compiler_cxx=cxx_cmd,
            compiler_so_cxx=cxx_cmd + ' ' + ccshared,
            linker_so=ldshared,
            linker_so_cxx=ldcxxshared,
            linker_exe=cc,
            linker_exe_cxx=cxx,
            archiver=archiver,
        )

        if 'RANLIB' in os.environ and compiler.executables.get('ranlib', None):
            compiler.set_executables(ranlib=os.environ['RANLIB'])

        compiler.shared_lib_extension = shlib_suffix


def get_config_h_filename() -> str:
    """Return full pathname of installed pyconfig.h file."""
    return sysconfig.get_config_h_filename()


def get_makefile_filename() -> str:
    """Return full pathname of installed Makefile from the Python build."""
    return sysconfig.get_makefile_filename()


def parse_config_h(fp, g=None):
    """Parse a config.h-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    return sysconfig.parse_config_h(fp, vars=g)


# Regexes needed for parsing Makefile (and similar syntaxes,
# like old-style Setup files).
_variable_rx = re.compile(r"([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
_findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
_findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")


def parse_makefile(fn, g=None):  # noqa: C901
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    from distutils.text_file import TextFile

    fp = TextFile(
        fn,
        strip_comments=True,
        skip_blanks=True,
        join_lines=True,
        errors="surrogateescape",
    )

    if g is None:
        g = {}
    done = {}
    notdone = {}

    while True:
        line = fp.readline()
        if line is None:  # eof
            break
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            # `$$' is a literal `$' in make
            tmpv = v.replace('$$', '')

            if "$" in tmpv:
                notdone[n] = v
            else:
                try:
                    v = int(v)
                except ValueError:
                    # insert literal `$'
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v

    # Variables with a 'PY_' prefix in the makefile. These need to
    # be made available without that prefix through sysconfig.
    # Special care is needed to ensure that variable expansion works, even
    # if the expansion uses the name without a prefix.
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')

    # do variable interpolation here
    while notdone:
        for name in list(notdone):
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    # get it on a subsequent round
                    found = False
                elif n in os.environ:
                    # do it like make: fall back to environment
                    item = os.environ[n]

                elif n in renamed_variables:
                    if name.startswith('PY_') and name[3:] in renamed_variables:
                        item = ""

                    elif 'PY_' + n in notdone:
                        found = False

                    else:
                        item = str(done['PY_' + n])
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end() :]
                    value = value[: m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        del notdone[name]

                        if name.startswith('PY_') and name[3:] in renamed_variables:
                            name = name[3:]
                            if name not in done:
                                done[name] = value
            else:
                # bogus variable reference; just drop it since we can't deal
                del notdone[name]

    fp.close()

    # strip spurious spaces
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()

    # save the results in the global dictionary
    g.update(done)
    return g


def expand_makefile_vars(s, vars):
    """Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
    'string' according to 'vars' (a dictionary mapping variable names to
    values).  Variables not present in 'vars' are silently expanded to the
    empty string.  The variable values in 'vars' should not contain further
    variable expansions; if 'vars' is the output of 'parse_makefile()',
    you're fine.  Returns a variable-expanded version of 's'.
    """

    # This algorithm does multiple expansion, so if vars['foo'] contains
    # "${bar}", it will expand ${foo} to ${bar}, and then expand
    # ${bar}... and so forth.  This is fine as long as 'vars' comes from
    # 'parse_makefile()', which takes care of such expansions eagerly,
    # according to make's variable expansion semantics.

    while True:
        m = _findvar1_rx.search(s) or _findvar2_rx.search(s)
        if m:
            (beg, end) = m.span()
            s = s[0:beg] + vars.get(m.group(1)) + s[end:]
        else:
            break
    return s


_config_vars = None


@overload
def get_config_vars() -> dict[str, str | int]: ...
@overload
def get_config_vars(arg: str, /, *args: str) -> list[str | int]: ...
def get_config_vars(*args: str) -> list[str | int] | dict[str, str | int]:
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.  Generally this includes
    everything needed to build extensions and install both pure modules and
    extensions.  On Unix, this means every variable defined in Python's
    installed Makefile; on Windows it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _config_vars
    if _config_vars is None:
        _config_vars = sysconfig.get_config_vars().copy()
        py39.add_ext_suffix(_config_vars)

    return [_config_vars.get(name) for name in args] if args else _config_vars


@overload
@deprecated(
    "SO is deprecated, use EXT_SUFFIX. Support will be removed when this module is synchronized with stdlib Python 3.11"
)
def get_config_var(name: Literal["SO"]) -> int | str | None: ...
@overload
def get_config_var(name: str) -> int | str | None: ...
def get_config_var(name: str) -> int | str | None:
    """Return the value of a single variable using the dictionary
    returned by 'get_config_vars()'.  Equivalent to
    get_config_vars().get(name)
    """
    if name == 'SO':
        import warnings

        warnings.warn('SO is deprecated, use EXT_SUFFIX', DeprecationWarning, 2)
    return get_config_vars().get(name)


@pass_none
def _add_flags(value: str, type: str) -> str:
    """
    Add any flags from the environment for the given type.

    type is the prefix to FLAGS in the environment key (e.g. "C" for "CFLAGS").
    """
    flags = os.environ.get(f'{type}FLAGS')
    return f'{value} {flags}' if flags else value
