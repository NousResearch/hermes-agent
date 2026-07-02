"""Tests for distutils.command.build_py."""

import os
import sys
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support

import jaraco.path
import pytest


@support.combine_markers
class TestBuildPy(support.TempdirManager):
    def test_package_data(self):
        sources = self.mkdtemp()
        jaraco.path.build(
            {
                '__init__.py': "# Pretend this is a package.",
                'README.txt': 'Info about this package',
            },
            sources,
        )

        destination = self.mkdtemp()

        dist = Distribution({"packages": ["pkg"], "package_dir": {"pkg": sources}})
        # script_name need not exist, it just need to be initialized
        dist.script_name = os.path.join(sources, "setup.py")
        dist.command_obj["build"] = support.DummyCommand(
            force=False, build_lib=destination
        )
        dist.packages = ["pkg"]
        dist.package_data = {"pkg": ["README.txt"]}
        dist.package_dir = {"pkg": sources}

        cmd = build_py(dist)
        cmd.compile = True
        cmd.ensure_finalized()
        assert cmd.package_data == dist.package_data

        cmd.run()

        # This makes sure the list of outputs includes byte-compiled
        # files for Python modules but not for package data files
        # (there shouldn't *be* byte-code files for those!).
        assert len(cmd.get_outputs()) == 3
        pkgdest = os.path.join(destination, "pkg")
        files = os.listdir(pkgdest)
        pycache_dir = os.path.join(pkgdest, "__pycache__")
        assert "__init__.py" in files
        assert "README.txt" in files
        if sys.dont_write_bytecode:
            assert not os.path.exists(pycache_dir)
        else:
            pyc_files = os.listdir(pycache_dir)
            assert f"__init__.{sys.implementation.cache_tag}.pyc" in pyc_files

    def test_empty_package_dir(self):
        # See bugs #1668596/#1720897
        sources = self.mkdtemp()
        jaraco.path.build({'__init__.py': '', 'doc': {'testfile': ''}}, sources)

        os.chdir(sources)
        dist = Distribution({
            "packages": ["pkg"],
            "package_dir": {"pkg": ""},
            "package_data": {"pkg": ["doc/*"]},
        })
        # script_name need not exist, it just need to be initialized
        dist.script_name = os.path.join(sources, "setup.py")
        dist.script_args = ["build"]
        dist.parse_command_line()

        try:
            dist.run_commands()
        except DistutilsFileError:
            self.fail("failed package_data test when package_dir is ''")

    @pytest.mark.skipif('sys.dont_write_bytecode')
    def test_byte_compile(self):
        project_dir, dist = self.create_dist(py_modules=['boiledeggs'])
        os.chdir(project_dir)
        self.write_file('boiledeggs.py', 'import antigravity')
        cmd = build_py(dist)
        cmd.compile = True
        cmd.build_lib = 'here'
        cmd.finalize_options()
        cmd.run()

        found = os.listdir(cmd.build_lib)
        assert sorted(found) == ['__pycache__', 'boiledeggs.py']
        found = os.listdir(os.path.join(cmd.build_lib, '__pycache__'))
        assert found == [f'boiledeggs.{sys.implementation.cache_tag}.pyc']

    @pytest.mark.skipif('sys.dont_write_bytecode')
    def test_byte_compile_optimized(self):
        project_dir, dist = self.create_dist(py_modules=['boiledeggs'])
        os.chdir(project_dir)
        self.write_file('boiledeggs.py', 'import antigravity')
        cmd = build_py(dist)
        cmd.compile = False
        cmd.optimize = 1
        cmd.build_lib = 'here'
        cmd.finalize_options()
        cmd.run()

        found = os.listdir(cmd.build_lib)
        assert sorted(found) == ['__pycache__', 'boiledeggs.py']
        found = os.listdir(os.path.join(cmd.build_lib, '__pycache__'))
        expect = f'boiledeggs.{sys.implementation.cache_tag}.opt-1.pyc'
        assert sorted(found) == [expect]

    def test_dir_in_package_data(self):
        """
        A directory in package_data should not be added to the filelist.
        """
        # See bug 19286
        sources = self.mkdtemp()
        jaraco.path.build(
            {
                'pkg': {
                    '__init__.py': '',
                    'doc': {
                        'testfile': '',
                        # create a directory that could be incorrectly detected as a file
                        'otherdir': {},
                    },
                }
            },
            sources,
        )

        os.chdir(sources)
        dist = Distribution({"packages": ["pkg"], "package_data": {"pkg": ["doc/*"]}})
        # script_name need not exist, it just need to be initialized
        dist.script_name = os.path.join(sources, "setup.py")
        dist.script_args = ["build"]
        dist.parse_command_line()

        try:
            dist.run_commands()
        except DistutilsFileError:
            self.fail("failed package_data when data dir includes a dir")

    def test_dont_write_bytecode(self, caplog):
        # makes sure byte_compile is not used
        dist = self.create_dist()[1]
        cmd = build_py(dist)
        cmd.compile = True
        cmd.optimize = 1

        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            cmd.byte_compile([])
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode

        assert 'byte-compiling is disabled' in caplog.records[0].message

    def test_namespace_package_does_not_warn(self, caplog):
        """
        Originally distutils implementation did not account for PEP 420
        and included warns for package directories that did not contain
        ``__init__.py`` files.
        After the acceptance of PEP 420, these warnings don't make more sense
        so we want to ensure there are not displayed to not confuse the users.
        """
        # Create a fake project structure with a package namespace:
        tmp = self.mkdtemp()
        jaraco.path.build({'ns': {'pkg': {'module.py': ''}}}, tmp)
        os.chdir(tmp)

        # Configure the package:
        attrs = {
            "name": "ns.pkg",
            "packages": ["ns", "ns.pkg"],
            "script_name": "setup.py",
        }
        dist = Distribution(attrs)

        # Run code paths that would trigger the trap:
        cmd = dist.get_command_obj("build_py")
        cmd.finalize_options()
        modules = cmd.find_all_modules()
        assert len(modules) == 1
        module_path = modules[0][-1]
        assert module_path.replace(os.sep, "/") == "ns/pkg/module.py"

        cmd.run()

        assert not any(
            "package init file" in msg and "not found" in msg for msg in caplog.messages
        )
