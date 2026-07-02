"""Tests for automatic package discovery"""

import os
import shutil
import tempfile

import pytest

from setuptools import find_namespace_packages, find_packages
from setuptools.discovery import FlatLayoutPackageFinder

from .compat.py39 import os_helper


class TestFindPackages:
    def setup_method(self, method):
        self.dist_dir = tempfile.mkdtemp()
        self._make_pkg_structure()

    def teardown_method(self, method):
        shutil.rmtree(self.dist_dir)

    def _make_pkg_structure(self):
        """Make basic package structure.

        dist/
            docs/
                conf.py
            pkg/
                __pycache__/
                nspkg/
                    mod.py
                subpkg/
                    assets/
                        asset
                    __init__.py
            setup.py

        """
        self.docs_dir = self._mkdir('docs', self.dist_dir)
        self._touch('conf.py', self.docs_dir)
        self.pkg_dir = self._mkdir('pkg', self.dist_dir)
        self._mkdir('__pycache__', self.pkg_dir)
        self.ns_pkg_dir = self._mkdir('nspkg', self.pkg_dir)
        self._touch('mod.py', self.ns_pkg_dir)
        self.sub_pkg_dir = self._mkdir('subpkg', self.pkg_dir)
        self.asset_dir = self._mkdir('assets', self.sub_pkg_dir)
        self._touch('asset', self.asset_dir)
        self._touch('__init__.py', self.sub_pkg_dir)
        self._touch('setup.py', self.dist_dir)

    def _mkdir(self, path, parent_dir=None):
        if parent_dir:
            path = os.path.join(parent_dir, path)
        os.mkdir(path)
        return path

    def _touch(self, path, dir_=None):
        if dir_:
            path = os.path.join(dir_, path)
        open(path, 'wb').close()
        return path

    def test_regular_package(self):
        self._touch('__init__.py', self.pkg_dir)
        packages = find_packages(self.dist_dir)
        assert packages == ['pkg', 'pkg.subpkg']

    def test_exclude(self):
        self._touch('__init__.py', self.pkg_dir)
        packages = find_packages(self.dist_dir, exclude=('pkg.*',))
        assert packages == ['pkg']

    def test_exclude_recursive(self):
        """
        Excluding a parent package should not exclude child packages as well.
        """
        self._touch('__init__.py', self.pkg_dir)
        self._touch('__init__.py', self.sub_pkg_dir)
        packages = find_packages(self.dist_dir, exclude=('pkg',))
        assert packages == ['pkg.subpkg']

    def test_include_excludes_other(self):
        """
        If include is specified, other packages should be excluded.
        """
        self._touch('__init__.py', self.pkg_dir)
        alt_dir = self._mkdir('other_pkg', self.dist_dir)
        self._touch('__init__.py', alt_dir)
        packages = find_packages(self.dist_dir, include=['other_pkg'])
        assert packages == ['other_pkg']

    def test_dir_with_dot_is_skipped(self):
        shutil.rmtree(os.path.join(self.dist_dir, 'pkg/subpkg/assets'))
        data_dir = self._mkdir('some.data', self.pkg_dir)
        self._touch('__init__.py', data_dir)
        self._touch('file.dat', data_dir)
        packages = find_packages(self.dist_dir)
        assert 'pkg.some.data' not in packages

    def test_dir_with_packages_in_subdir_is_excluded(self):
        """
        Ensure that a package in a non-package such as build/pkg/__init__.py
        is excluded.
        """
        build_dir = self._mkdir('build', self.dist_dir)
        build_pkg_dir = self._mkdir('pkg', build_dir)
        self._touch('__init__.py', build_pkg_dir)
        packages = find_packages(self.dist_dir)
        assert 'build.pkg' not in packages

    @pytest.mark.skipif(not os_helper.can_symlink(), reason='Symlink support required')
    def test_symlinked_packages_are_included(self):
        """
        A symbolically-linked directory should be treated like any other
        directory when matched as a package.

        Create a link from lpkg -> pkg.
        """
        self._touch('__init__.py', self.pkg_dir)
        linked_pkg = os.path.join(self.dist_dir, 'lpkg')
        os.symlink('pkg', linked_pkg)
        assert os.path.isdir(linked_pkg)
        packages = find_packages(self.dist_dir)
        assert 'lpkg' in packages

    def _assert_packages(self, actual, expected):
        assert set(actual) == set(expected)

    def test_pep420_ns_package(self):
        packages = find_namespace_packages(
            self.dist_dir, include=['pkg*'], exclude=['pkg.subpkg.assets']
        )
        self._assert_packages(packages, ['pkg', 'pkg.nspkg', 'pkg.subpkg'])

    def test_pep420_ns_package_no_includes(self):
        packages = find_namespace_packages(self.dist_dir, exclude=['pkg.subpkg.assets'])
        self._assert_packages(packages, ['docs', 'pkg', 'pkg.nspkg', 'pkg.subpkg'])

    def test_pep420_ns_package_no_includes_or_excludes(self):
        packages = find_namespace_packages(self.dist_dir)
        expected = ['docs', 'pkg', 'pkg.nspkg', 'pkg.subpkg', 'pkg.subpkg.assets']
        self._assert_packages(packages, expected)

    def test_regular_package_with_nested_pep420_ns_packages(self):
        self._touch('__init__.py', self.pkg_dir)
        packages = find_namespace_packages(
            self.dist_dir, exclude=['docs', 'pkg.subpkg.assets']
        )
        self._assert_packages(packages, ['pkg', 'pkg.nspkg', 'pkg.subpkg'])

    def test_pep420_ns_package_no_non_package_dirs(self):
        shutil.rmtree(self.docs_dir)
        shutil.rmtree(os.path.join(self.dist_dir, 'pkg/subpkg/assets'))
        packages = find_namespace_packages(self.dist_dir)
        self._assert_packages(packages, ['pkg', 'pkg.nspkg', 'pkg.subpkg'])


class TestFlatLayoutPackageFinder:
    EXAMPLES = {
        "hidden-folders": (
            [".pkg/__init__.py", "pkg/__init__.py", "pkg/nested/file.txt"],
            ["pkg", "pkg.nested"],
        ),
        "private-packages": (
            ["_pkg/__init__.py", "pkg/_private/__init__.py"],
            ["pkg", "pkg._private"],
        ),
        "invalid-name": (
            ["invalid-pkg/__init__.py", "other.pkg/__init__.py", "yet,another/file.py"],
            [],
        ),
        "docs": (["pkg/__init__.py", "docs/conf.py", "docs/readme.rst"], ["pkg"]),
        "tests": (
            ["pkg/__init__.py", "tests/test_pkg.py", "tests/__init__.py"],
            ["pkg"],
        ),
        "examples": (
            [
                "pkg/__init__.py",
                "examples/__init__.py",
                "examples/file.py",
                "example/other_file.py",
                # Sub-packages should always be fine
                "pkg/example/__init__.py",
                "pkg/examples/__init__.py",
            ],
            ["pkg", "pkg.examples", "pkg.example"],
        ),
        "tool-specific": (
            [
                "htmlcov/index.html",
                "pkg/__init__.py",
                "tasks/__init__.py",
                "tasks/subpackage/__init__.py",
                "fabfile/__init__.py",
                "fabfile/subpackage/__init__.py",
                # Sub-packages should always be fine
                "pkg/tasks/__init__.py",
                "pkg/fabfile/__init__.py",
            ],
            ["pkg", "pkg.tasks", "pkg.fabfile"],
        ),
    }

    @pytest.mark.parametrize("example", EXAMPLES.keys())
    def test_unwanted_directories_not_included(self, tmp_path, example):
        files, expected_packages = self.EXAMPLES[example]
        ensure_files(tmp_path, files)
        found_packages = FlatLayoutPackageFinder.find(str(tmp_path))
        assert set(found_packages) == set(expected_packages)


def ensure_files(root_path, files):
    for file in files:
        path = root_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
