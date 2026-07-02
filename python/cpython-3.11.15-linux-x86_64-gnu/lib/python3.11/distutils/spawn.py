"""distutils.spawn

Provides the 'spawn()' function, a front-end to various platform-
specific functions for launching another program in a sub-process.
Also provides the 'find_executable()' to search the path for a given
executable name.
"""

import sys
import os
import subprocess

from distutils.errors import DistutilsPlatformError, DistutilsExecError
from distutils.debug import DEBUG
from distutils import log


if sys.platform == 'darwin':
    _cfg_target = None
    _cfg_target_split = None


def spawn(cmd, search_path=1, verbose=0, dry_run=0):
    """Run another program, specified as a command list 'cmd', in a new process.

    'cmd' is just the argument list for the new process, ie.
    cmd[0] is the program to run and cmd[1:] are the rest of its arguments.
    There is no way to run a program with a name different from that of its
    executable.

    If 'search_path' is true (the default), the system's executable
    search path will be used to find the program; otherwise, cmd[0]
    must be the exact path to the executable.  If 'dry_run' is true,
    the command will not actually be run.

    Raise DistutilsExecError if running the program fails in any way; just
    return on success.
    """
    # cmd is documented as a list, but just in case some code passes a tuple
    # in, protect our %-formatting code against horrible death
    cmd = list(cmd)

    log.info(' '.join(cmd))
    if dry_run:
        return

    if search_path:
        executable = find_executable(cmd[0])
        if executable is not None:
            cmd[0] = executable

    env = None
    if sys.platform == 'darwin':
        global _cfg_target, _cfg_target_split
        if _cfg_target is None:
            from distutils import sysconfig
            _cfg_target = sysconfig.get_config_var(
                                  'MACOSX_DEPLOYMENT_TARGET') or ''
            if _cfg_target:
                _cfg_target_split = [int(x) for x in _cfg_target.split('.')]
        if _cfg_target:
            # Ensure that the deployment target of the build process is not
            # less than 10.3 if the interpreter was built for 10.3 or later.
            # This ensures extension modules are built with correct
            # compatibility values, specifically LDSHARED which can use
            # '-undefined dynamic_lookup' which only works on >= 10.3.
            cur_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', _cfg_target)
            cur_target_split = [int(x) for x in cur_target.split('.')]
            if _cfg_target_split[:2] >= [10, 3] and cur_target_split[:2] < [10, 3]:
                my_msg = ('$MACOSX_DEPLOYMENT_TARGET mismatch: '
                          'now "%s" but "%s" during configure;'
                          'must use 10.3 or later'
                                % (cur_target, _cfg_target))
                raise DistutilsPlatformError(my_msg)
            env = dict(os.environ,
                       MACOSX_DEPLOYMENT_TARGET=cur_target)

    try:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        exitcode = proc.returncode
    except OSError as exc:
        if not DEBUG:
            cmd = cmd[0]
        raise DistutilsExecError(
            "command %r failed: %s" % (cmd, exc.args[-1])) from exc

    if exitcode:
        if not DEBUG:
            cmd = cmd[0]
        raise DistutilsExecError(
              "command %r failed with exit code %s" % (cmd, exitcode))


def find_executable(executable, path=None):
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    _, ext = os.path.splitext(executable)
    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'

    if os.path.isfile(executable):
        return executable

    if path is None:
        path = os.environ.get('PATH', None)
        if path is None:
            try:
                path = os.confstr("CS_PATH")
            except (AttributeError, ValueError):
                # os.confstr() or CS_PATH is not available
                path = os.defpath
        # bpo-35755: Don't use os.defpath if the PATH environment variable is
        # set to an empty string

    # PATH='' doesn't match, whereas PATH=':' looks in the current directory
    if not path:
        return None

    paths = path.split(os.pathsep)
    for p in paths:
        f = os.path.join(p, executable)
        if os.path.isfile(f):
            # the file exists, we have a shot at spawn working
            return f
    return None
