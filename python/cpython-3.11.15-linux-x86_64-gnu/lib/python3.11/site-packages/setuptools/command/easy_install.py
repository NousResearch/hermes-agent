import os
import sys
import types

from setuptools import Command

from .. import _scripts, warnings


class easy_install(Command):
    """Stubbed command for temporary pbr compatibility."""


def __getattr__(name):
    attr = getattr(
        types.SimpleNamespace(
            ScriptWriter=_scripts.ScriptWriter,
            sys_executable=os.environ.get(
                "__PYVENV_LAUNCHER__", os.path.normpath(sys.executable)
            ),
        ),
        name,
    )
    warnings.SetuptoolsDeprecationWarning.emit(
        summary="easy_install module is deprecated",
        details="Avoid accessing attributes of setuptools.command.easy_install.",
        due_date=(2025, 10, 31),
        see_url="https://github.com/pypa/setuptools/issues/4976",
    )
    return attr
