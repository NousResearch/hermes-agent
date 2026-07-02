# mypy: disable_error_code=call-overload
# pyright: reportCallIssue=false, reportArgumentType=false
# Can't disable on the exact line because distutils doesn't exists on Python 3.12
# and type-checkers aren't aware of distutils_hack,
# causing distutils.command.bdist.bdist.format_commands to be Any.

import sys

from distutils.command.bdist import bdist

if 'egg' not in bdist.format_commands:
    try:
        # format_commands is a dict in vendored distutils
        # It used to be a list in older (stdlib) distutils
        # We support both for backwards compatibility
        bdist.format_commands['egg'] = ('bdist_egg', "Python .egg file")
    except TypeError:
        bdist.format_command['egg'] = ('bdist_egg', "Python .egg file")
        bdist.format_commands.append('egg')

del bdist, sys
