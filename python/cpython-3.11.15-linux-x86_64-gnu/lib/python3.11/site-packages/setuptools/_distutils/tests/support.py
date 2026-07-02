"""Support code for distutils test cases."""

import itertools
import os
import pathlib
import shutil
import sys
import sysconfig
import tempfile
from distutils.core import Distribution

import pytest
from more_itertools import always_iterable


@pytest.mark.usefixtures('distutils_managed_tempdir')
class TempdirManager:
    """
    Mix-in class that handles temporary directories for test cases.
    """

    def mkdtemp(self):
        """Create a temporary directory that will be cleaned up.

        Returns the path of the directory.
        """
        d = tempfile.mkdtemp()
        self.tempdirs.append(d)
        return d

    def write_file(self, path, content='xxx'):
        """Writes a file in the given path.

        path can be a string or a sequence.
        """
        pathlib.Path(*always_iterable(path)).write_text(content, encoding='utf-8')

    def create_dist(self, pkg_name='foo', **kw):
        """Will generate a test environment.

        This function creates:
         - a Distribution instance using keywords
         - a temporary directory with a package structure

        It returns the package directory and the distribution
        instance.
        """
        tmp_dir = self.mkdtemp()
        pkg_dir = os.path.join(tmp_dir, pkg_name)
        os.mkdir(pkg_dir)
        dist = Distribution(attrs=kw)

        return pkg_dir, dist


class DummyCommand:
    """Class to store options for retrieval via set_undefined_options()."""

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def ensure_finalized(self):
        pass


def copy_xxmodule_c(directory):
    """Helper for tests that need the xxmodule.c source file.

    Example use:

        def test_compile(self):
            copy_xxmodule_c(self.tmpdir)
            self.assertIn('xxmodule.c', os.listdir(self.tmpdir))

    If the source file can be found, it will be copied to *directory*.  If not,
    the test will be skipped.  Errors during copy are not caught.
    """
    shutil.copy(_get_xxmodule_path(), os.path.join(directory, 'xxmodule.c'))


def _get_xxmodule_path():
    source_name = 'xxmodule.c' if sys.version_info > (3, 9) else 'xxmodule-3.8.c'
    return os.path.join(os.path.dirname(__file__), source_name)


def fixup_build_ext(cmd):
    """Function needed to make build_ext tests pass.

    When Python was built with --enable-shared on Unix, -L. is not enough to
    find libpython<blah>.so, because regrtest runs in a tempdir, not in the
    source directory where the .so lives.

    When Python was built with in debug mode on Windows, build_ext commands
    need their debug attribute set, and it is not done automatically for
    some reason.

    This function handles both of these things.  Example use:

        cmd = build_ext(dist)
        support.fixup_build_ext(cmd)
        cmd.ensure_finalized()

    Unlike most other Unix platforms, Mac OS X embeds absolute paths
    to shared libraries into executables, so the fixup is not needed there.
    """
    if os.name == 'nt':
        cmd.debug = sys.executable.endswith('_d.exe')
    elif sysconfig.get_config_var('Py_ENABLE_SHARED'):
        # To further add to the shared builds fun on Unix, we can't just add
        # library_dirs to the Extension() instance because that doesn't get
        # plumbed through to the final compiler command.
        runshared = sysconfig.get_config_var('RUNSHARED')
        if runshared is None:
            cmd.library_dirs = ['.']
        else:
            if sys.platform == 'darwin':
                cmd.library_dirs = []
            else:
                name, equals, value = runshared.partition('=')
                cmd.library_dirs = [d for d in value.split(os.pathsep) if d]


def combine_markers(cls):
    """
    pytest will honor markers as found on the class, but when
    markers are on multiple subclasses, only one appears. Use
    this decorator to combine those markers.
    """
    cls.pytestmark = [
        mark
        for base in itertools.chain([cls], cls.__bases__)
        for mark in getattr(base, 'pytestmark', [])
    ]
    return cls
