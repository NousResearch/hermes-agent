import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib

from typing import Union, Optional
from .abc import ResourceReader, Traversable

from ._adapters import wrap_spec

Package = Union[types.ModuleType, str]


def files(package):
    # type: (Package) -> Traversable
    """
    Get a Traversable resource from a package
    """
    return from_package(get_package(package))


def get_resource_reader(package):
    # type: (types.ModuleType) -> Optional[ResourceReader]
    """
    Return the package's loader if it's a ResourceReader.
    """
    # We can't use
    # a issubclass() check here because apparently abc.'s __subclasscheck__()
    # hook wants to create a weak reference to the object, but
    # zipimport.zipimporter does not support weak references, resulting in a
    # TypeError.  That seems terrible.
    spec = package.__spec__
    reader = getattr(spec.loader, 'get_resource_reader', None)  # type: ignore
    if reader is None:
        return None
    return reader(spec.name)  # type: ignore


def resolve(cand):
    # type: (Package) -> types.ModuleType
    return cand if isinstance(cand, types.ModuleType) else importlib.import_module(cand)


def get_package(package):
    # type: (Package) -> types.ModuleType
    """Take a package name or module object and return the module.

    Raise an exception if the resolved module is not a package.
    """
    resolved = resolve(package)
    if wrap_spec(resolved).submodule_search_locations is None:
        raise TypeError(f'{package!r} is not a package')
    return resolved


def from_package(package):
    """
    Return a Traversable object for the given package.

    """
    spec = wrap_spec(package)
    reader = spec.loader.get_resource_reader(spec.name)
    return reader.files()


@contextlib.contextmanager
def _tempfile(reader, suffix='',
              # gh-93353: Keep a reference to call os.remove() in late Python
              # finalization.
              *, _os_remove=os.remove):
    # Not using tempfile.NamedTemporaryFile as it leads to deeper 'try'
    # blocks due to the need to close the temporary file to work on Windows
    # properly.
    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    try:
        try:
            os.write(fd, reader())
        finally:
            os.close(fd)
        del reader
        yield pathlib.Path(raw_path)
    finally:
        try:
            _os_remove(raw_path)
        except FileNotFoundError:
            pass


@functools.singledispatch
def as_file(path):
    """
    Given a Traversable object, return that object as a
    path on the local file system in a context manager.
    """
    return _tempfile(path.read_bytes, suffix=path.name)


@as_file.register(pathlib.Path)
@contextlib.contextmanager
def _(path):
    """
    Degenerate behavior for pathlib.Path objects.
    """
    yield path
