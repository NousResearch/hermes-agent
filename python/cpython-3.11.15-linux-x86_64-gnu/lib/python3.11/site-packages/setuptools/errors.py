"""setuptools.errors

Provides exceptions used by setuptools modules.
"""

from __future__ import annotations

from distutils import errors as _distutils_errors

# Re-export errors from distutils to facilitate the migration to PEP632

ByteCompileError = _distutils_errors.DistutilsByteCompileError
CCompilerError = _distutils_errors.CCompilerError
ClassError = _distutils_errors.DistutilsClassError
CompileError = _distutils_errors.CompileError
ExecError = _distutils_errors.DistutilsExecError
FileError = _distutils_errors.DistutilsFileError
InternalError = _distutils_errors.DistutilsInternalError
LibError = _distutils_errors.LibError
LinkError = _distutils_errors.LinkError
ModuleError = _distutils_errors.DistutilsModuleError
OptionError = _distutils_errors.DistutilsOptionError
PlatformError = _distutils_errors.DistutilsPlatformError
PreprocessError = _distutils_errors.PreprocessError
SetupError = _distutils_errors.DistutilsSetupError
TemplateError = _distutils_errors.DistutilsTemplateError
UnknownFileError = _distutils_errors.UnknownFileError

# The root error class in the hierarchy
BaseError = _distutils_errors.DistutilsError


class InvalidConfigError(OptionError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for invalid configurations."""


class RemovedConfigError(OptionError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for configurations that were deprecated and removed."""


class RemovedCommandError(BaseError, RuntimeError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Error used for commands that have been removed in setuptools.

    Since ``setuptools`` is built on ``distutils``, simply removing a command
    from ``setuptools`` will make the behavior fall back to ``distutils``; this
    error is raised if a command exists in ``distutils`` but has been actively
    removed in ``setuptools``.
    """


class PackageDiscoveryError(BaseError, RuntimeError):  # type: ignore[valid-type, misc] # distutils imports are `Any` on python 3.12+
    """Impossible to perform automatic discovery of packages and/or modules.

    The current project layout or given discovery options can lead to problems when
    scanning the project directory.

    Setuptools might also refuse to complete auto-discovery if an error prone condition
    is detected (e.g. when a project is organised as a flat-layout but contains
    multiple directories that can be taken as top-level packages inside a single
    distribution [*]_). In these situations the users are encouraged to be explicit
    about which packages to include or to make the discovery parameters more specific.

    .. [*] Since multi-package distributions are uncommon it is very likely that the
       developers did not intend for all the directories to be packaged, and are just
       leaving auxiliary code in the repository top-level, such as maintenance-related
       scripts.
    """
