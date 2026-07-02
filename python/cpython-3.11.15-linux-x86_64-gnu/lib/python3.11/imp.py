"""This module provides the components needed to build your own __import__
function.  Undocumented functions are obsolete.

In most cases it is preferred you consider using the importlib module's
functionality over this module.

"""
# (Probably) need to stay in _imp
from _imp import (lock_held, acquire_lock, release_lock,
                  get_frozen_object, is_frozen_package,
                  init_frozen, is_builtin, is_frozen,
                  _fix_co_filename, _frozen_module_names)
try:
    from _imp import create_dynamic
except ImportError:
    # Platform doesn't support dynamic loading.
    create_dynamic = None

from importlib._bootstrap import _ERR_MSG, _exec, _load, _builtin_from_name
from importlib._bootstrap_external import SourcelessFileLoader

from importlib import machinery
from importlib import util
import importlib
import os
import sys
import tokenize
import types
import warnings

warnings.warn("the imp module is deprecated in favour of importlib and slated "
              "for removal in Python 3.12; "
              "see the module's documentation for alternative uses",
              DeprecationWarning, stacklevel=2)

# DEPRECATED
SEARCH_ERROR = 0
PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
PY_RESOURCE = 4
PKG_DIRECTORY = 5
C_BUILTIN = 6
PY_FROZEN = 7
PY_CODERESOURCE = 8
IMP_HOOK = 9


def new_module(name):
    """**DEPRECATED**

    Create a new module.

    The module is not entered into sys.modules.

    """
    return types.ModuleType(name)


def get_magic():
    """**DEPRECATED**

    Return the magic number for .pyc files.
    """
    return util.MAGIC_NUMBER


def get_tag():
    """Return the magic tag for .pyc files."""
    return sys.implementation.cache_tag


def cache_from_source(path, debug_override=None):
    """**DEPRECATED**

    Given the path to a .py file, return the path to its .pyc file.

    The .py file does not need to exist; this simply returns the path to the
    .pyc file calculated as if the .py file were imported.

    If debug_override is not None, then it must be a boolean and is used in
    place of sys.flags.optimize.

    If sys.implementation.cache_tag is None then NotImplementedError is raised.

    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return util.cache_from_source(path, debug_override)


def source_from_cache(path):
    """**DEPRECATED**

    Given the path to a .pyc. file, return the path to its .py file.

    The .pyc file does not need to exist; this simply returns the path to
    the .py file calculated to correspond to the .pyc file.  If path does
    not conform to PEP 3147 format, ValueError will be raised. If
    sys.implementation.cache_tag is None then NotImplementedError is raised.

    """
    return util.source_from_cache(path)


def get_suffixes():
    """**DEPRECATED**"""
    extensions = [(s, 'rb', C_EXTENSION) for s in machinery.EXTENSION_SUFFIXES]
    source = [(s, 'r', PY_SOURCE) for s in machinery.SOURCE_SUFFIXES]
    bytecode = [(s, 'rb', PY_COMPILED) for s in machinery.BYTECODE_SUFFIXES]

    return extensions + source + bytecode


class NullImporter:

    """**DEPRECATED**

    Null import object.

    """

    def __init__(self, path):
        if path == '':
            raise ImportError('empty pathname', path='')
        elif os.path.isdir(path):
            raise ImportError('existing directory', path=path)

    def find_module(self, fullname):
        """Always returns None."""
        return None


class _HackedGetData:

    """Compatibility support for 'file' arguments of various load_*()
    functions."""

    def __init__(self, fullname, path, file=None):
        super().__init__(fullname, path)
        self.file = file

    def get_data(self, path):
        """Gross hack to contort loader to deal w/ load_*()'s bad API."""
        if self.file and path == self.path:
            # The contract of get_data() requires us to return bytes. Reopen the
            # file in binary mode if needed.
            if not self.file.closed:
                file = self.file
                if 'b' not in file.mode:
                    file.close()
            if self.file.closed:
                self.file = file = open(self.path, 'rb')

            with file:
                return file.read()
        else:
            return super().get_data(path)


class _LoadSourceCompatibility(_HackedGetData, machinery.SourceFileLoader):

    """Compatibility support for implementing load_source()."""


def load_source(name, pathname, file=None):
    loader = _LoadSourceCompatibility(name, pathname, file)
    spec = util.spec_from_file_location(name, pathname, loader=loader)
    if name in sys.modules:
        module = _exec(spec, sys.modules[name])
    else:
        module = _load(spec)
    # To allow reloading to potentially work, use a non-hacked loader which
    # won't rely on a now-closed file object.
    module.__loader__ = machinery.SourceFileLoader(name, pathname)
    module.__spec__.loader = module.__loader__
    return module


class _LoadCompiledCompatibility(_HackedGetData, SourcelessFileLoader):

    """Compatibility support for implementing load_compiled()."""


def load_compiled(name, pathname, file=None):
    """**DEPRECATED**"""
    loader = _LoadCompiledCompatibility(name, pathname, file)
    spec = util.spec_from_file_location(name, pathname, loader=loader)
    if name in sys.modules:
        module = _exec(spec, sys.modules[name])
    else:
        module = _load(spec)
    # To allow reloading to potentially work, use a non-hacked loader which
    # won't rely on a now-closed file object.
    module.__loader__ = SourcelessFileLoader(name, pathname)
    module.__spec__.loader = module.__loader__
    return module


def load_package(name, path):
    """**DEPRECATED**"""
    if os.path.isdir(path):
        extensions = (machinery.SOURCE_SUFFIXES[:] +
                      machinery.BYTECODE_SUFFIXES[:])
        for extension in extensions:
            init_path = os.path.join(path, '__init__' + extension)
            if os.path.exists(init_path):
                path = init_path
                break
        else:
            raise ValueError('{!r} is not a package'.format(path))
    spec = util.spec_from_file_location(name, path,
                                        submodule_search_locations=[])
    if name in sys.modules:
        return _exec(spec, sys.modules[name])
    else:
        return _load(spec)


def load_module(name, file, filename, details):
    """**DEPRECATED**

    Load a module, given information returned by find_module().

    The module name must include the full package name, if any.

    """
    suffix, mode, type_ = details
    if mode and (not mode.startswith('r') or '+' in mode):
        raise ValueError('invalid file open mode {!r}'.format(mode))
    elif file is None and type_ in {PY_SOURCE, PY_COMPILED}:
        msg = 'file object required for import (type code {})'.format(type_)
        raise ValueError(msg)
    elif type_ == PY_SOURCE:
        return load_source(name, filename, file)
    elif type_ == PY_COMPILED:
        return load_compiled(name, filename, file)
    elif type_ == C_EXTENSION and load_dynamic is not None:
        if file is None:
            with open(filename, 'rb') as opened_file:
                return load_dynamic(name, filename, opened_file)
        else:
            return load_dynamic(name, filename, file)
    elif type_ == PKG_DIRECTORY:
        return load_package(name, filename)
    elif type_ == C_BUILTIN:
        return init_builtin(name)
    elif type_ == PY_FROZEN:
        return init_frozen(name)
    else:
        msg =  "Don't know how to import {} (type code {})".format(name, type_)
        raise ImportError(msg, name=name)


def find_module(name, path=None):
    """**DEPRECATED**

    Search for a module.

    If path is omitted or None, search for a built-in, frozen or special
    module and continue search in sys.path. The module name cannot
    contain '.'; to search for a submodule of a package, pass the
    submodule name and the package's __path__.

    """
    if not isinstance(name, str):
        raise TypeError("'name' must be a str, not {}".format(type(name)))
    elif not isinstance(path, (type(None), list)):
        # Backwards-compatibility
        raise RuntimeError("'path' must be None or a list, "
                           "not {}".format(type(path)))

    if path is None:
        if is_builtin(name):
            return None, None, ('', '', C_BUILTIN)
        elif is_frozen(name):
            return None, None, ('', '', PY_FROZEN)
        else:
            path = sys.path

    for entry in path:
        package_directory = os.path.join(entry, name)
        for suffix in ['.py', machinery.BYTECODE_SUFFIXES[0]]:
            package_file_name = '__init__' + suffix
            file_path = os.path.join(package_directory, package_file_name)
            if os.path.isfile(file_path):
                return None, package_directory, ('', '', PKG_DIRECTORY)
        for suffix, mode, type_ in get_suffixes():
            file_name = name + suffix
            file_path = os.path.join(entry, file_name)
            if os.path.isfile(file_path):
                break
        else:
            continue
        break  # Break out of outer loop when breaking out of inner loop.
    else:
        raise ImportError(_ERR_MSG.format(name), name=name)

    encoding = None
    if 'b' not in mode:
        with open(file_path, 'rb') as file:
            encoding = tokenize.detect_encoding(file.readline)[0]
    file = open(file_path, mode, encoding=encoding)
    return file, file_path, (suffix, mode, type_)


def reload(module):
    """**DEPRECATED**

    Reload the module and return it.

    The module must have been successfully imported before.

    """
    return importlib.reload(module)


def init_builtin(name):
    """**DEPRECATED**

    Load and return a built-in module by name, or None is such module doesn't
    exist
    """
    try:
        return _builtin_from_name(name)
    except ImportError:
        return None


if create_dynamic:
    def load_dynamic(name, path, file=None):
        """**DEPRECATED**

        Load an extension module.
        """
        import importlib.machinery
        loader = importlib.machinery.ExtensionFileLoader(name, path)

        # Issue #24748: Skip the sys.modules check in _load_module_shim;
        # always load new extension
        spec = importlib.machinery.ModuleSpec(
            name=name, loader=loader, origin=path)
        return _load(spec)

else:
    load_dynamic = None
