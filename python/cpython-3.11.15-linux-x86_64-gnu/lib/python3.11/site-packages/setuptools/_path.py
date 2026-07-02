from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING, TypeVar, Union

from more_itertools import unique_everseen

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

StrPath: TypeAlias = Union[str, os.PathLike[str]]  #  Same as _typeshed.StrPath
StrPathT = TypeVar("StrPathT", bound=Union[str, os.PathLike[str]])


def ensure_directory(path):
    """Ensure that the parent directory of `path` exists"""
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def same_path(p1: StrPath, p2: StrPath) -> bool:
    """Differs from os.path.samefile because it does not require paths to exist.
    Purely string based (no comparison between i-nodes).
    >>> same_path("a/b", "./a/b")
    True
    >>> same_path("a/b", "a/./b")
    True
    >>> same_path("a/b", "././a/b")
    True
    >>> same_path("a/b", "./a/b/c/..")
    True
    >>> same_path("a/b", "../a/b/c")
    False
    >>> same_path("a", "a/b")
    False
    """
    return normpath(p1) == normpath(p2)


def _cygwin_patch(filename: StrPath):  # pragma: nocover
    """
    Contrary to POSIX 2008, on Cygwin, getcwd (3) contains
    symlink components. Using
    os.path.abspath() works around this limitation. A fix in os.getcwd()
    would probably better, in Cygwin even more so, except
    that this seems to be by design...
    """
    return os.path.abspath(filename) if sys.platform == 'cygwin' else filename


def normpath(filename: StrPath) -> str:
    """Normalize a file/dir name for comparison purposes."""
    return os.path.normcase(os.path.realpath(os.path.normpath(_cygwin_patch(filename))))


@contextlib.contextmanager
def paths_on_pythonpath(paths):
    """
    Add the indicated paths to the head of the PYTHONPATH environment
    variable so that subprocesses will also see the packages at
    these paths.

    Do this in a context that restores the value on exit.

    >>> getfixture('monkeypatch').setenv('PYTHONPATH', 'anything')
    >>> with paths_on_pythonpath(['foo', 'bar']):
    ...     assert 'foo' in os.environ['PYTHONPATH']
    ...     assert 'anything' in os.environ['PYTHONPATH']
    >>> os.environ['PYTHONPATH']
    'anything'

    >>> getfixture('monkeypatch').delenv('PYTHONPATH')
    >>> with paths_on_pythonpath(['foo', 'bar']):
    ...     assert 'foo' in os.environ['PYTHONPATH']
    >>> os.environ.get('PYTHONPATH')
    """
    nothing = object()
    orig_pythonpath = os.environ.get('PYTHONPATH', nothing)
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    try:
        prefix = os.pathsep.join(unique_everseen(paths))
        to_join = filter(None, [prefix, current_pythonpath])
        new_path = os.pathsep.join(to_join)
        if new_path:
            os.environ['PYTHONPATH'] = new_path
        yield
    finally:
        if orig_pythonpath is nothing:
            os.environ.pop('PYTHONPATH', None)
        else:
            os.environ['PYTHONPATH'] = orig_pythonpath
