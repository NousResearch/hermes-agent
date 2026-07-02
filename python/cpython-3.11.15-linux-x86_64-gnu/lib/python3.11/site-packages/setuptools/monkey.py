"""
Monkey patching of distutils.
"""

from __future__ import annotations

import inspect
import platform
import sys
import types
from typing import TypeVar, cast, overload

import distutils.filelist

_T = TypeVar("_T")
_UnpatchT = TypeVar("_UnpatchT", type, types.FunctionType)


__all__: list[str] = []
"""
Everything is private. Contact the project team
if you think you need this functionality.
"""


def _get_mro(cls):
    """
    Returns the bases classes for cls sorted by the MRO.

    Works around an issue on Jython where inspect.getmro will not return all
    base classes if multiple classes share the same name. Instead, this
    function will return a tuple containing the class itself, and the contents
    of cls.__bases__. See https://github.com/pypa/setuptools/issues/1024.
    """
    if platform.python_implementation() == "Jython":
        return (cls,) + cls.__bases__
    return inspect.getmro(cls)


@overload
def get_unpatched(item: _UnpatchT) -> _UnpatchT: ...
@overload
def get_unpatched(item: object) -> None: ...
def get_unpatched(
    item: type | types.FunctionType | object,
) -> type | types.FunctionType | None:
    if isinstance(item, type):
        return get_unpatched_class(item)
    if isinstance(item, types.FunctionType):
        return get_unpatched_function(item)
    return None


def get_unpatched_class(cls: type[_T]) -> type[_T]:
    """Protect against re-patching the distutils if reloaded

    Also ensures that no other distutils extension monkeypatched the distutils
    first.
    """
    external_bases = (
        cast(type[_T], cls)
        for cls in _get_mro(cls)
        if not cls.__module__.startswith('setuptools')
    )
    base = next(external_bases)
    if not base.__module__.startswith('distutils'):
        msg = f"distutils has already been patched by {cls!r}"
        raise AssertionError(msg)
    return base


def patch_all() -> None:
    import setuptools

    # we can't patch distutils.cmd, alas
    distutils.core.Command = setuptools.Command  # type: ignore[misc,assignment] # monkeypatching

    _patch_distribution_metadata()

    # Install Distribution throughout the distutils
    for module in distutils.dist, distutils.core, distutils.cmd:
        module.Distribution = setuptools.dist.Distribution

    # Install the patched Extension
    distutils.core.Extension = setuptools.extension.Extension  # type: ignore[misc,assignment] # monkeypatching
    distutils.extension.Extension = setuptools.extension.Extension  # type: ignore[misc,assignment] # monkeypatching
    if 'distutils.command.build_ext' in sys.modules:
        sys.modules[
            'distutils.command.build_ext'
        ].Extension = setuptools.extension.Extension


def _patch_distribution_metadata():
    from . import _core_metadata

    """Patch write_pkg_file and read_pkg_file for higher metadata standards"""
    for attr in (
        'write_pkg_info',
        'write_pkg_file',
        'read_pkg_file',
        'get_metadata_version',
        'get_fullname',
    ):
        new_val = getattr(_core_metadata, attr)
        setattr(distutils.dist.DistributionMetadata, attr, new_val)


def patch_func(replacement, target_mod, func_name) -> None:
    """
    Patch func_name in target_mod with replacement

    Important - original must be resolved by name to avoid
    patching an already patched function.
    """
    original = getattr(target_mod, func_name)

    # set the 'unpatched' attribute on the replacement to
    # point to the original.
    vars(replacement).setdefault('unpatched', original)

    # replace the function in the original module
    setattr(target_mod, func_name, replacement)


def get_unpatched_function(candidate):
    return candidate.unpatched
