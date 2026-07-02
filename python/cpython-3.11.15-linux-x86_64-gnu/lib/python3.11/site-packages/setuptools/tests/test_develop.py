"""develop tests"""

import os
import platform
import subprocess
import sys

import pytest

from setuptools._path import paths_on_pythonpath

from . import contexts, namespaces

SETUP_PY = """\
from setuptools import setup

setup(name='foo',
    packages=['foo'],
)
"""

INIT_PY = """print "foo"
"""


@pytest.fixture
def temp_user(monkeypatch):
    with contexts.tempdir() as user_base:
        with contexts.tempdir() as user_site:
            monkeypatch.setattr('site.USER_BASE', user_base)
            monkeypatch.setattr('site.USER_SITE', user_site)
            yield


@pytest.fixture
def test_env(tmpdir, temp_user):
    target = tmpdir
    foo = target.mkdir('foo')
    setup = target / 'setup.py'
    if setup.isfile():
        raise ValueError(dir(target))
    with setup.open('w') as f:
        f.write(SETUP_PY)
    init = foo / '__init__.py'
    with init.open('w') as f:
        f.write(INIT_PY)
    with target.as_cwd():
        yield target


class TestNamespaces:
    @staticmethod
    def install_develop(src_dir, target):
        develop_cmd = [
            sys.executable,
            'setup.py',
            'develop',
            '--install-dir',
            str(target),
        ]
        with src_dir.as_cwd():
            with paths_on_pythonpath([str(target)]):
                subprocess.check_call(develop_cmd)

    @pytest.mark.xfail(reason="pkg_resources has been removed")
    @pytest.mark.skipif(
        bool(os.environ.get("APPVEYOR")),
        reason="https://github.com/pypa/setuptools/issues/851",
    )
    @pytest.mark.skipif(
        platform.python_implementation() == 'PyPy',
        reason="https://github.com/pypa/setuptools/issues/1202",
    )
    @pytest.mark.uses_network
    def test_namespace_package_importable(self, tmpdir):
        """
        Installing two packages sharing the same namespace, one installed
        naturally using pip or `--single-version-externally-managed`
        and the other installed using `develop` should leave the namespace
        in tact and both packages reachable by import.
        """
        pkg_A = namespaces.build_namespace_package(tmpdir, 'myns.pkgA')
        pkg_B = namespaces.build_namespace_package(tmpdir, 'myns.pkgB')
        target = tmpdir / 'packages'
        # use pip to install to the target directory
        install_cmd = [
            sys.executable,
            '-m',
            'pip',
            'install',
            str(pkg_A),
            '-t',
            str(target),
        ]
        subprocess.check_call(install_cmd)
        self.install_develop(pkg_B, target)
        namespaces.make_site_dir(target)
        try_import = [
            sys.executable,
            '-c',
            'import myns.pkgA; import myns.pkgB',
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(try_import)

        # additionally ensure that pkg_resources import works
        pkg_resources_imp = [
            sys.executable,
            '-c',
            'import pkg_resources',
        ]
        with paths_on_pythonpath([str(target)]):
            subprocess.check_call(pkg_resources_imp)
