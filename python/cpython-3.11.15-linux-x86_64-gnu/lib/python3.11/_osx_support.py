"""Shared OS X support functions."""

import os
import re
import sys

__all__ = [
    'compiler_fixup',
    'customize_config_vars',
    'customize_compiler',
    'get_platform_osx',
]

# configuration variables that may contain universal build flags,
# like "-arch" or "-isdkroot", that may need customization for
# the user environment
_UNIVERSAL_CONFIG_VARS = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS', 'BASECFLAGS',
                            'BLDSHARED', 'LDSHARED', 'CC', 'CXX',
                            'PY_CFLAGS', 'PY_LDFLAGS', 'PY_CPPFLAGS',
                            'PY_CORE_CFLAGS', 'PY_CORE_LDFLAGS')

# configuration variables that may contain compiler calls
_COMPILER_CONFIG_VARS = ('BLDSHARED', 'LDSHARED', 'CC', 'CXX')

# prefix added to original configuration variable names
_INITPRE = '_OSX_SUPPORT_INITIAL_'


def _find_executable(executable, path=None):
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    if path is None:
        path = os.environ['PATH']

    paths = path.split(os.pathsep)
    base, ext = os.path.splitext(executable)

    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'

    if not os.path.isfile(executable):
        for p in paths:
            f = os.path.join(p, executable)
            if os.path.isfile(f):
                # the file exists, we have a shot at spawn working
                return f
        return None
    else:
        return executable


def _read_output(commandstring, capture_stderr=False):
    """Output from successful command execution or None"""
    # Similar to os.popen(commandstring, "r").read(),
    # but without actually using os.popen because that
    # function is not usable during python bootstrap.
    # tempfile is also not available then.
    import contextlib
    try:
        import tempfile
        fp = tempfile.NamedTemporaryFile()
    except ImportError:
        fp = open("/tmp/_osx_support.%s"%(
            os.getpid(),), "w+b")

    with contextlib.closing(fp) as fp:
        if capture_stderr:
            cmd = "%s >'%s' 2>&1" % (commandstring, fp.name)
        else:
            cmd = "%s 2>/dev/null >'%s'" % (commandstring, fp.name)
        return fp.read().decode('utf-8').strip() if not os.system(cmd) else None


def _find_build_tool(toolname):
    """Find a build tool on current path or using xcrun"""
    return (_find_executable(toolname)
                or _read_output("/usr/bin/xcrun -find %s" % (toolname,))
                or ''
            )

_SYSTEM_VERSION = None

def _get_system_version():
    """Return the OS X system version as a string"""
    # Reading this plist is a documented way to get the system
    # version (see the documentation for the Gestalt Manager)
    # We avoid using platform.mac_ver to avoid possible bootstrap issues during
    # the build of Python itself (distutils is used to build standard library
    # extensions).

    global _SYSTEM_VERSION

    if _SYSTEM_VERSION is None:
        _SYSTEM_VERSION = ''
        try:
            f = open('/System/Library/CoreServices/SystemVersion.plist', encoding="utf-8")
        except OSError:
            # We're on a plain darwin box, fall back to the default
            # behaviour.
            pass
        else:
            try:
                m = re.search(r'<key>ProductUserVisibleVersion</key>\s*'
                              r'<string>(.*?)</string>', f.read())
            finally:
                f.close()
            if m is not None:
                _SYSTEM_VERSION = '.'.join(m.group(1).split('.')[:2])
            # else: fall back to the default behaviour

    return _SYSTEM_VERSION

_SYSTEM_VERSION_TUPLE = None
def _get_system_version_tuple():
    """
    Return the macOS system version as a tuple

    The return value is safe to use to compare
    two version numbers.
    """
    global _SYSTEM_VERSION_TUPLE
    if _SYSTEM_VERSION_TUPLE is None:
        osx_version = _get_system_version()
        if osx_version:
            try:
                _SYSTEM_VERSION_TUPLE = tuple(int(i) for i in osx_version.split('.'))
            except ValueError:
                _SYSTEM_VERSION_TUPLE = ()

    return _SYSTEM_VERSION_TUPLE


def _remove_original_values(_config_vars):
    """Remove original unmodified values for testing"""
    # This is needed for higher-level cross-platform tests of get_platform.
    for k in list(_config_vars):
        if k.startswith(_INITPRE):
            del _config_vars[k]

def _save_modified_value(_config_vars, cv, newvalue):
    """Save modified and original unmodified value of configuration var"""

    oldvalue = _config_vars.get(cv, '')
    if (oldvalue != newvalue) and (_INITPRE + cv not in _config_vars):
        _config_vars[_INITPRE + cv] = oldvalue
    _config_vars[cv] = newvalue


_cache_default_sysroot = None
def _default_sysroot(cc):
    """ Returns the root of the default SDK for this system, or '/' """
    global _cache_default_sysroot

    if _cache_default_sysroot is not None:
        return _cache_default_sysroot

    contents = _read_output('%s -c -E -v - </dev/null' % (cc,), True)
    in_incdirs = False
    for line in contents.splitlines():
        if line.startswith("#include <...>"):
            in_incdirs = True
        elif line.startswith("End of search list"):
            in_incdirs = False
        elif in_incdirs:
            line = line.strip()
            if line == '/usr/include':
                _cache_default_sysroot = '/'
            elif line.endswith(".sdk/usr/include"):
                _cache_default_sysroot = line[:-12]
    if _cache_default_sysroot is None:
        _cache_default_sysroot = '/'

    return _cache_default_sysroot

def _supports_universal_builds():
    """Returns True if universal builds are supported on this system"""
    # As an approximation, we assume that if we are running on 10.4 or above,
    # then we are running with an Xcode environment that supports universal
    # builds, in particular -isysroot and -arch arguments to the compiler. This
    # is in support of allowing 10.4 universal builds to run on 10.3.x systems.

    osx_version = _get_system_version_tuple()
    return bool(osx_version >= (10, 4)) if osx_version else False

def _supports_arm64_builds():
    """Returns True if arm64 builds are supported on this system"""
    # There are two sets of systems supporting macOS/arm64 builds:
    # 1. macOS 11 and later, unconditionally
    # 2. macOS 10.15 with Xcode 12.2 or later
    # For now the second category is ignored.
    osx_version = _get_system_version_tuple()
    return osx_version >= (11, 0) if osx_version else False


def _find_appropriate_compiler(_config_vars):
    """Find appropriate C compiler for extension module builds"""

    # Issue #13590:
    #    The OSX location for the compiler varies between OSX
    #    (or rather Xcode) releases.  With older releases (up-to 10.5)
    #    the compiler is in /usr/bin, with newer releases the compiler
    #    can only be found inside Xcode.app if the "Command Line Tools"
    #    are not installed.
    #
    #    Furthermore, the compiler that can be used varies between
    #    Xcode releases. Up to Xcode 4 it was possible to use 'gcc-4.2'
    #    as the compiler, after that 'clang' should be used because
    #    gcc-4.2 is either not present, or a copy of 'llvm-gcc' that
    #    miscompiles Python.

    # skip checks if the compiler was overridden with a CC env variable
    if 'CC' in os.environ:
        return _config_vars

    # The CC config var might contain additional arguments.
    # Ignore them while searching.
    cc = oldcc = _config_vars['CC'].split()[0]
    if not _find_executable(cc):
        # Compiler is not found on the shell search PATH.
        # Now search for clang, first on PATH (if the Command LIne
        # Tools have been installed in / or if the user has provided
        # another location via CC).  If not found, try using xcrun
        # to find an uninstalled clang (within a selected Xcode).

        # NOTE: Cannot use subprocess here because of bootstrap
        # issues when building Python itself (and os.popen is
        # implemented on top of subprocess and is therefore not
        # usable as well)

        cc = _find_build_tool('clang')

    elif os.path.basename(cc).startswith('gcc'):
        # Compiler is GCC, check if it is LLVM-GCC
        data = _read_output("'%s' --version"
                             % (cc.replace("'", "'\"'\"'"),))
        if data and 'llvm-gcc' in data:
            # Found LLVM-GCC, fall back to clang
            cc = _find_build_tool('clang')

    if not cc:
        raise SystemError(
               "Cannot locate working compiler")

    if cc != oldcc:
        # Found a replacement compiler.
        # Modify config vars using new compiler, if not already explicitly
        # overridden by an env variable, preserving additional arguments.
        for cv in _COMPILER_CONFIG_VARS:
            if cv in _config_vars and cv not in os.environ:
                cv_split = _config_vars[cv].split()
                cv_split[0] = cc if cv != 'CXX' else cc + '++'
                _save_modified_value(_config_vars, cv, ' '.join(cv_split))

    return _config_vars


def _remove_universal_flags(_config_vars):
    """Remove all universal build arguments from config vars"""

    for cv in _UNIVERSAL_CONFIG_VARS:
        # Do not alter a config var explicitly overridden by env var
        if cv in _config_vars and cv not in os.environ:
            flags = _config_vars[cv]
            flags = re.sub(r'-arch\s+\w+\s', ' ', flags, flags=re.ASCII)
            flags = re.sub(r'-isysroot\s*\S+', ' ', flags)
            _save_modified_value(_config_vars, cv, flags)

    return _config_vars


def _remove_unsupported_archs(_config_vars):
    """Remove any unsupported archs from config vars"""
    # Different Xcode releases support different sets for '-arch'
    # flags. In particular, Xcode 4.x no longer supports the
    # PPC architectures.
    #
    # This code automatically removes '-arch ppc' and '-arch ppc64'
    # when these are not supported. That makes it possible to
    # build extensions on OSX 10.7 and later with the prebuilt
    # 32-bit installer on the python.org website.

    # skip checks if the compiler was overridden with a CC env variable
    if 'CC' in os.environ:
        return _config_vars

    if re.search(r'-arch\s+ppc', _config_vars['CFLAGS']) is not None:
        # NOTE: Cannot use subprocess here because of bootstrap
        # issues when building Python itself
        status = os.system(
            """echo 'int main{};' | """
            """'%s' -c -arch ppc -x c -o /dev/null /dev/null 2>/dev/null"""
            %(_config_vars['CC'].replace("'", "'\"'\"'"),))
        if status:
            # The compile failed for some reason.  Because of differences
            # across Xcode and compiler versions, there is no reliable way
            # to be sure why it failed.  Assume here it was due to lack of
            # PPC support and remove the related '-arch' flags from each
            # config variables not explicitly overridden by an environment
            # variable.  If the error was for some other reason, we hope the
            # failure will show up again when trying to compile an extension
            # module.
            for cv in _UNIVERSAL_CONFIG_VARS:
                if cv in _config_vars and cv not in os.environ:
                    flags = _config_vars[cv]
                    flags = re.sub(r'-arch\s+ppc\w*\s', ' ', flags)
                    _save_modified_value(_config_vars, cv, flags)

    return _config_vars


def _override_all_archs(_config_vars):
    """Allow override of all archs with ARCHFLAGS env var"""
    # NOTE: This name was introduced by Apple in OSX 10.5 and
    # is used by several scripting languages distributed with
    # that OS release.
    if 'ARCHFLAGS' in os.environ:
        arch = os.environ['ARCHFLAGS']
        for cv in _UNIVERSAL_CONFIG_VARS:
            if cv in _config_vars and '-arch' in _config_vars[cv]:
                flags = _config_vars[cv]
                flags = re.sub(r'-arch\s+\w+\s', ' ', flags)
                flags = flags + ' ' + arch
                _save_modified_value(_config_vars, cv, flags)

    return _config_vars


def _check_for_unavailable_sdk(_config_vars):
    """Remove references to any SDKs not available"""
    # If we're on OSX 10.5 or later and the user tries to
    # compile an extension using an SDK that is not present
    # on the current machine it is better to not use an SDK
    # than to fail.  This is particularly important with
    # the standalone Command Line Tools alternative to a
    # full-blown Xcode install since the CLT packages do not
    # provide SDKs.  If the SDK is not present, it is assumed
    # that the header files and dev libs have been installed
    # to /usr and /System/Library by either a standalone CLT
    # package or the CLT component within Xcode.
    cflags = _config_vars.get('CFLAGS', '')
    m = re.search(r'-isysroot\s*(\S+)', cflags)
    if m is not None:
        sdk = m.group(1)
        if not os.path.exists(sdk):
            for cv in _UNIVERSAL_CONFIG_VARS:
                # Do not alter a config var explicitly overridden by env var
                if cv in _config_vars and cv not in os.environ:
                    flags = _config_vars[cv]
                    flags = re.sub(r'-isysroot\s*\S+(?:\s|$)', ' ', flags)
                    _save_modified_value(_config_vars, cv, flags)

    return _config_vars


def compiler_fixup(compiler_so, cc_args):
    """
    This function will strip '-isysroot PATH' and '-arch ARCH' from the
    compile flags if the user has specified one them in extra_compile_flags.

    This is needed because '-arch ARCH' adds another architecture to the
    build, without a way to remove an architecture. Furthermore GCC will
    barf if multiple '-isysroot' arguments are present.
    """
    stripArch = stripSysroot = False

    compiler_so = list(compiler_so)

    if not _supports_universal_builds():
        # OSX before 10.4.0, these don't support -arch and -isysroot at
        # all.
        stripArch = stripSysroot = True
    else:
        stripArch = '-arch' in cc_args
        stripSysroot = any(arg for arg in cc_args if arg.startswith('-isysroot'))

    if stripArch or 'ARCHFLAGS' in os.environ:
        while True:
            try:
                index = compiler_so.index('-arch')
                # Strip this argument and the next one:
                del compiler_so[index:index+2]
            except ValueError:
                break

    elif not _supports_arm64_builds():
        # Look for "-arch arm64" and drop that
        for idx in reversed(range(len(compiler_so))):
            if compiler_so[idx] == '-arch' and compiler_so[idx+1] == "arm64":
                del compiler_so[idx:idx+2]

    if 'ARCHFLAGS' in os.environ and not stripArch:
        # User specified different -arch flags in the environ,
        # see also distutils.sysconfig
        compiler_so = compiler_so + os.environ['ARCHFLAGS'].split()

    if stripSysroot:
        while True:
            indices = [i for i,x in enumerate(compiler_so) if x.startswith('-isysroot')]
            if not indices:
                break
            index = indices[0]
            if compiler_so[index] == '-isysroot':
                # Strip this argument and the next one:
                del compiler_so[index:index+2]
            else:
                # It's '-isysroot/some/path' in one arg
                del compiler_so[index:index+1]

    # Check if the SDK that is used during compilation actually exists,
    # the universal build requires the usage of a universal SDK and not all
    # users have that installed by default.
    sysroot = None
    argvar = cc_args
    indices = [i for i,x in enumerate(cc_args) if x.startswith('-isysroot')]
    if not indices:
        argvar = compiler_so
        indices = [i for i,x in enumerate(compiler_so) if x.startswith('-isysroot')]

    for idx in indices:
        if argvar[idx] == '-isysroot':
            sysroot = argvar[idx+1]
            break
        else:
            sysroot = argvar[idx][len('-isysroot'):]
            break

    if sysroot and not os.path.isdir(sysroot):
        sys.stderr.write(f"Compiling with an SDK that doesn't seem to exist: {sysroot}\n")
        sys.stderr.write("Please check your Xcode installation\n")
        sys.stderr.flush()

    return compiler_so


def customize_config_vars(_config_vars):
    """Customize Python build configuration variables.

    Called internally from sysconfig with a mutable mapping
    containing name/value pairs parsed from the configured
    makefile used to build this interpreter.  Returns
    the mapping updated as needed to reflect the environment
    in which the interpreter is running; in the case of
    a Python from a binary installer, the installed
    environment may be very different from the build
    environment, i.e. different OS levels, different
    built tools, different available CPU architectures.

    This customization is performed whenever
    distutils.sysconfig.get_config_vars() is first
    called.  It may be used in environments where no
    compilers are present, i.e. when installing pure
    Python dists.  Customization of compiler paths
    and detection of unavailable archs is deferred
    until the first extension module build is
    requested (in distutils.sysconfig.customize_compiler).

    Currently called from distutils.sysconfig
    """

    if not _supports_universal_builds():
        # On Mac OS X before 10.4, check if -arch and -isysroot
        # are in CFLAGS or LDFLAGS and remove them if they are.
        # This is needed when building extensions on a 10.3 system
        # using a universal build of python.
        _remove_universal_flags(_config_vars)

    # Allow user to override all archs with ARCHFLAGS env var
    _override_all_archs(_config_vars)

    # Remove references to sdks that are not found
    _check_for_unavailable_sdk(_config_vars)

    return _config_vars


def customize_compiler(_config_vars):
    """Customize compiler path and configuration variables.

    This customization is performed when the first
    extension module build is requested
    in distutils.sysconfig.customize_compiler.
    """

    # Find a compiler to use for extension module builds
    _find_appropriate_compiler(_config_vars)

    # Remove ppc arch flags if not supported here
    _remove_unsupported_archs(_config_vars)

    # Allow user to override all archs with ARCHFLAGS env var
    _override_all_archs(_config_vars)

    return _config_vars


def get_platform_osx(_config_vars, osname, release, machine):
    """Filter values for get_platform()"""
    # called from get_platform() in sysconfig and distutils.util
    #
    # For our purposes, we'll assume that the system version from
    # distutils' perspective is what MACOSX_DEPLOYMENT_TARGET is set
    # to. This makes the compatibility story a bit more sane because the
    # machine is going to compile and link as if it were
    # MACOSX_DEPLOYMENT_TARGET.

    macver = _config_vars.get('MACOSX_DEPLOYMENT_TARGET', '')
    if macver and '.' not in macver:
        # Ensure that the version includes at least a major
        # and minor version, even if MACOSX_DEPLOYMENT_TARGET
        # is set to a single-label version like "14".
        macver += '.0'
    macrelease = _get_system_version() or macver
    macver = macver or macrelease

    if macver:
        release = macver
        osname = "macosx"

        # Use the original CFLAGS value, if available, so that we
        # return the same machine type for the platform string.
        # Otherwise, distutils may consider this a cross-compiling
        # case and disallow installs.
        cflags = _config_vars.get(_INITPRE+'CFLAGS',
                                    _config_vars.get('CFLAGS', ''))
        if macrelease:
            try:
                macrelease = tuple(int(i) for i in macrelease.split('.')[0:2])
            except ValueError:
                macrelease = (10, 3)
        else:
            # assume no universal support
            macrelease = (10, 3)

        if (macrelease >= (10, 4)) and '-arch' in cflags.strip():
            # The universal build will build fat binaries, but not on
            # systems before 10.4

            machine = 'fat'

            archs = re.findall(r'-arch\s+(\S+)', cflags)
            archs = tuple(sorted(set(archs)))

            if len(archs) == 1:
                machine = archs[0]
            elif archs == ('arm64', 'x86_64'):
                machine = 'universal2'
            elif archs == ('i386', 'ppc'):
                machine = 'fat'
            elif archs == ('i386', 'x86_64'):
                machine = 'intel'
            elif archs == ('i386', 'ppc', 'x86_64'):
                machine = 'fat3'
            elif archs == ('ppc64', 'x86_64'):
                machine = 'fat64'
            elif archs == ('i386', 'ppc', 'ppc64', 'x86_64'):
                machine = 'universal'
            else:
                raise ValueError(
                   "Don't know machine value for archs=%r" % (archs,))

        elif machine == 'i386':
            # On OSX the machine type returned by uname is always the
            # 32-bit variant, even if the executable architecture is
            # the 64-bit variant
            if sys.maxsize >= 2**32:
                machine = 'x86_64'

        elif machine in ('PowerPC', 'Power_Macintosh'):
            # Pick a sane name for the PPC architecture.
            # See 'i386' case
            if sys.maxsize >= 2**32:
                machine = 'ppc64'
            else:
                machine = 'ppc'

    return (osname, release, machine)
