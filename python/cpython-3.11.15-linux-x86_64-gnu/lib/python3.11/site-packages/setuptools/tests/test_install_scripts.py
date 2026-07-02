"""install_scripts tests"""

import sys

import pytest

from setuptools.command.install_scripts import install_scripts
from setuptools.dist import Distribution

from . import contexts


class TestInstallScripts:
    settings = dict(
        name='foo',
        entry_points={'console_scripts': ['foo=foo:foo']},
        version='0.0',
    )
    unix_exe = '/usr/dummy-test-path/local/bin/python'
    unix_spaces_exe = '/usr/bin/env dummy-test-python'
    win32_exe = 'C:\\Dummy Test Path\\Program Files\\Python 3.6\\python.exe'

    def _run_install_scripts(self, install_dir, executable=None):
        dist = Distribution(self.settings)
        dist.script_name = 'setup.py'
        cmd = install_scripts(dist)
        cmd.install_dir = install_dir
        if executable is not None:
            bs = cmd.get_finalized_command('build_scripts')
            bs.executable = executable
        cmd.ensure_finalized()
        with contexts.quiet():
            cmd.run()

    @pytest.mark.skipif(sys.platform == 'win32', reason='non-Windows only')
    def test_sys_executable_escaping_unix(self, tmpdir, monkeypatch):
        """
        Ensure that shebang is not quoted on Unix when getting the Python exe
        from sys.executable.
        """
        expected = f'#!{self.unix_exe}\n'
        monkeypatch.setattr('sys.executable', self.unix_exe)
        with tmpdir.as_cwd():
            self._run_install_scripts(str(tmpdir))
            with open(str(tmpdir.join('foo')), 'r', encoding="utf-8") as f:
                actual = f.readline()
        assert actual == expected

    @pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
    def test_sys_executable_escaping_win32(self, tmpdir, monkeypatch):
        """
        Ensure that shebang is quoted on Windows when getting the Python exe
        from sys.executable and it contains a space.
        """
        expected = f'#!"{self.win32_exe}"\n'
        monkeypatch.setattr('sys.executable', self.win32_exe)
        with tmpdir.as_cwd():
            self._run_install_scripts(str(tmpdir))
            with open(str(tmpdir.join('foo-script.py')), 'r', encoding="utf-8") as f:
                actual = f.readline()
        assert actual == expected

    @pytest.mark.skipif(sys.platform == 'win32', reason='non-Windows only')
    def test_executable_with_spaces_escaping_unix(self, tmpdir):
        """
        Ensure that shebang on Unix is not quoted, even when
        a value with spaces
        is specified using --executable.
        """
        expected = f'#!{self.unix_spaces_exe}\n'
        with tmpdir.as_cwd():
            self._run_install_scripts(str(tmpdir), self.unix_spaces_exe)
            with open(str(tmpdir.join('foo')), 'r', encoding="utf-8") as f:
                actual = f.readline()
        assert actual == expected

    @pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
    def test_executable_arg_escaping_win32(self, tmpdir):
        """
        Ensure that shebang on Windows is quoted when
        getting a path with spaces
        from --executable, that is itself properly quoted.
        """
        expected = f'#!"{self.win32_exe}"\n'
        with tmpdir.as_cwd():
            self._run_install_scripts(str(tmpdir), '"' + self.win32_exe + '"')
            with open(str(tmpdir.join('foo-script.py')), 'r', encoding="utf-8") as f:
                actual = f.readline()
        assert actual == expected
