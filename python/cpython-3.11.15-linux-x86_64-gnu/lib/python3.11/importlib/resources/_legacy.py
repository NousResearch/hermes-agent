import functools
import os
import pathlib
import types
import warnings

from typing import Union, Iterable, ContextManager, BinaryIO, TextIO, Any

from . import _common

Package = Union[types.ModuleType, str]
Resource = str


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated. Use files() instead. "
            "Refer to https://importlib-resources.readthedocs.io"
            "/en/latest/using.html#migrating-from-legacy for migration advice.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


def normalize_path(path):
    # type: (Any) -> str
    """Normalize a path by ensuring it is a string.

    If the resulting string contains path separators, an exception is raised.
    """
    str_path = str(path)
    parent, file_name = os.path.split(str_path)
    if parent:
        raise ValueError(f'{path!r} must be only a file name')
    return file_name


@deprecated
def open_binary(package: Package, resource: Resource) -> BinaryIO:
    """Return a file-like object opened for binary reading of the resource."""
    return (_common.files(package) / normalize_path(resource)).open('rb')


@deprecated
def read_binary(package: Package, resource: Resource) -> bytes:
    """Return the binary contents of the resource."""
    return (_common.files(package) / normalize_path(resource)).read_bytes()


@deprecated
def open_text(
    package: Package,
    resource: Resource,
    encoding: str = 'utf-8',
    errors: str = 'strict',
) -> TextIO:
    """Return a file-like object opened for text reading of the resource."""
    return (_common.files(package) / normalize_path(resource)).open(
        'r', encoding=encoding, errors=errors
    )


@deprecated
def read_text(
    package: Package,
    resource: Resource,
    encoding: str = 'utf-8',
    errors: str = 'strict',
) -> str:
    """Return the decoded string of the resource.

    The decoding-related arguments have the same semantics as those of
    bytes.decode().
    """
    with open_text(package, resource, encoding, errors) as fp:
        return fp.read()


@deprecated
def contents(package: Package) -> Iterable[str]:
    """Return an iterable of entries in `package`.

    Note that not all entries are resources.  Specifically, directories are
    not considered resources.  Use `is_resource()` on each entry returned here
    to check if it is a resource or not.
    """
    return [path.name for path in _common.files(package).iterdir()]


@deprecated
def is_resource(package: Package, name: str) -> bool:
    """True if `name` is a resource inside `package`.

    Directories are *not* resources.
    """
    resource = normalize_path(name)
    return any(
        traversable.name == resource and traversable.is_file()
        for traversable in _common.files(package).iterdir()
    )


@deprecated
def path(
    package: Package,
    resource: Resource,
) -> ContextManager[pathlib.Path]:
    """A context manager providing a file path object to the resource.

    If the resource does not already exist on its own on the file system,
    a temporary file will be created. If the file was created, the file
    will be deleted upon exiting the context manager (no exception is
    raised if the file was deleted prior to the context manager
    exiting).
    """
    return _common.as_file(_common.files(package) / normalize_path(resource))
