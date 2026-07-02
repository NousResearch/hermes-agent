"""Tests for distutils.command.install."""

import logging
import os
import pathlib
import site
import sys
from distutils import sysconfig
from distutils.command import install as install_module
from distutils.command.build_ext import build_ext
from distutils.command.install import INSTALL_SCHEMES, install
from distutils.core import Distribution
from distutils.errors import DistutilsOptionError
from distutils.extension import Extension
from distutils.tests import missing_compiler_executable, support
from distutils.util import is_mingw

import pytest


def _make_ext_name(modname):
    return modname + sysconfig.get_config_var('EXT_SUFFIX')


@support.combine_markers
@pytest.mark.usefixtures('save_env')
class TestInstall(
    support.TempdirManager,
):
    @pytest.mark.xfail(
        'platform.system() == "Windows" and sys.version_info > (3, 11)',
        reason="pypa/distutils#148",
    )
    def test_home_installation_scheme(self):
        # This ensure two things:
        # - that --home generates the desired set of directory names
        # - test --home is supported on all platforms
        builddir = self.mkdtemp()
        destination = os.path.join(builddir, "installation")

        dist = Distribution({"name": "foopkg"})
        # script_name need not exist, it just need to be initialized
        dist.script_name = os.path.join(builddir, "setup.py")
        dist.command_obj["build"] = support.DummyCommand(
            build_base=builddir,
            build_lib=os.path.join(builddir, "lib"),
        )

        cmd = install(dist)
        cmd.home = destination
        cmd.ensure_finalized()

        assert cmd.install_base == destination
        assert cmd.install_platbase == destination

        def check_path(got, expected):
            got = os.path.normpath(got)
            expected = os.path.normpath(expected)
            assert got == expected

        impl_name = sys.implementation.name.replace("cpython", "python")
        libdir = os.path.join(destination, "lib", impl_name)
        check_path(cmd.install_lib, libdir)
        _platlibdir = getattr(sys, "platlibdir", "lib")
        platlibdir = os.path.join(destination, _platlibdir, impl_name)
        check_path(cmd.install_platlib, platlibdir)
        check_path(cmd.install_purelib, libdir)
        check_path(
            cmd.install_headers,
            os.path.join(destination, "include", impl_name, "foopkg"),
        )
        check_path(cmd.install_scripts, os.path.join(destination, "bin"))
        check_path(cmd.install_data, destination)

    def test_user_site(self, monkeypatch):
        # test install with --user
        # preparing the environment for the test
        self.tmpdir = self.mkdtemp()
        orig_site = site.USER_SITE
        orig_base = site.USER_BASE
        monkeypatch.setattr(site, 'USER_BASE', os.path.join(self.tmpdir, 'B'))
        monkeypatch.setattr(site, 'USER_SITE', os.path.join(self.tmpdir, 'S'))
        monkeypatch.setattr(install_module, 'USER_BASE', site.USER_BASE)
        monkeypatch.setattr(install_module, 'USER_SITE', site.USER_SITE)

        def _expanduser(path):
            if path.startswith('~'):
                return os.path.normpath(self.tmpdir + path[1:])
            return path

        monkeypatch.setattr(os.path, 'expanduser', _expanduser)

        for key in ('nt_user', 'posix_user'):
            assert key in INSTALL_SCHEMES

        dist = Distribution({'name': 'xx'})
        cmd = install(dist)

        # making sure the user option is there
        options = [name for name, short, label in cmd.user_options]
        assert 'user' in options

        # setting a value
        cmd.user = True

        # user base and site shouldn't be created yet
        assert not os.path.exists(site.USER_BASE)
        assert not os.path.exists(site.USER_SITE)

        # let's run finalize
        cmd.ensure_finalized()

        # now they should
        assert os.path.exists(site.USER_BASE)
        assert os.path.exists(site.USER_SITE)

        assert 'userbase' in cmd.config_vars
        assert 'usersite' in cmd.config_vars

        actual_headers = os.path.relpath(cmd.install_headers, site.USER_BASE)
        if os.name == 'nt' and not is_mingw():
            site_path = os.path.relpath(os.path.dirname(orig_site), orig_base)
            include = os.path.join(site_path, 'Include')
        else:
            include = sysconfig.get_python_inc(0, '')
        expect_headers = os.path.join(include, 'xx')

        assert os.path.normcase(actual_headers) == os.path.normcase(expect_headers)

    def test_handle_extra_path(self):
        dist = Distribution({'name': 'xx', 'extra_path': 'path,dirs'})
        cmd = install(dist)

        # two elements
        cmd.handle_extra_path()
        assert cmd.extra_path == ['path', 'dirs']
        assert cmd.extra_dirs == 'dirs'
        assert cmd.path_file == 'path'

        # one element
        cmd.extra_path = ['path']
        cmd.handle_extra_path()
        assert cmd.extra_path == ['path']
        assert cmd.extra_dirs == 'path'
        assert cmd.path_file == 'path'

        # none
        dist.extra_path = cmd.extra_path = None
        cmd.handle_extra_path()
        assert cmd.extra_path is None
        assert cmd.extra_dirs == ''
        assert cmd.path_file is None

        # three elements (no way !)
        cmd.extra_path = 'path,dirs,again'
        with pytest.raises(DistutilsOptionError):
            cmd.handle_extra_path()

    def test_finalize_options(self):
        dist = Distribution({'name': 'xx'})
        cmd = install(dist)

        # must supply either prefix/exec-prefix/home or
        # install-base/install-platbase -- not both
        cmd.prefix = 'prefix'
        cmd.install_base = 'base'
        with pytest.raises(DistutilsOptionError):
            cmd.finalize_options()

        # must supply either home or prefix/exec-prefix -- not both
        cmd.install_base = None
        cmd.home = 'home'
        with pytest.raises(DistutilsOptionError):
            cmd.finalize_options()

        # can't combine user with prefix/exec_prefix/home or
        # install_(plat)base
        cmd.prefix = None
        cmd.user = 'user'
        with pytest.raises(DistutilsOptionError):
            cmd.finalize_options()

    def test_record(self):
        install_dir = self.mkdtemp()
        project_dir, dist = self.create_dist(py_modules=['hello'], scripts=['sayhi'])
        os.chdir(project_dir)
        self.write_file('hello.py', "def main(): print('o hai')")
        self.write_file('sayhi', 'from hello import main; main()')

        cmd = install(dist)
        dist.command_obj['install'] = cmd
        cmd.root = install_dir
        cmd.record = os.path.join(project_dir, 'filelist')
        cmd.ensure_finalized()
        cmd.run()

        content = pathlib.Path(cmd.record).read_text(encoding='utf-8')

        found = [pathlib.Path(line).name for line in content.splitlines()]
        expected = [
            'hello.py',
            f'hello.{sys.implementation.cache_tag}.pyc',
            'sayhi',
            'UNKNOWN-0.0.0-py{}.{}.egg-info'.format(*sys.version_info[:2]),
        ]
        assert found == expected

    def test_record_extensions(self):
        cmd = missing_compiler_executable()
        if cmd is not None:
            pytest.skip(f'The {cmd!r} command is not found')
        install_dir = self.mkdtemp()
        project_dir, dist = self.create_dist(
            ext_modules=[Extension('xx', ['xxmodule.c'])]
        )
        os.chdir(project_dir)
        support.copy_xxmodule_c(project_dir)

        buildextcmd = build_ext(dist)
        support.fixup_build_ext(buildextcmd)
        buildextcmd.ensure_finalized()

        cmd = install(dist)
        dist.command_obj['install'] = cmd
        dist.command_obj['build_ext'] = buildextcmd
        cmd.root = install_dir
        cmd.record = os.path.join(project_dir, 'filelist')
        cmd.ensure_finalized()
        cmd.run()

        content = pathlib.Path(cmd.record).read_text(encoding='utf-8')

        found = [pathlib.Path(line).name for line in content.splitlines()]
        expected = [
            _make_ext_name('xx'),
            'UNKNOWN-0.0.0-py{}.{}.egg-info'.format(*sys.version_info[:2]),
        ]
        assert found == expected

    def test_debug_mode(self, caplog, monkeypatch):
        # this covers the code called when DEBUG is set
        monkeypatch.setattr(install_module, 'DEBUG', True)
        caplog.set_level(logging.DEBUG)
        self.test_record()
        assert any(rec for rec in caplog.records if rec.levelno == logging.DEBUG)
