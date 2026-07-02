"""Tests for distutils.command.bdist_dumb."""

import os
import sys
import zipfile
from distutils.command.bdist_dumb import bdist_dumb
from distutils.core import Distribution
from distutils.tests import support

import pytest

SETUP_PY = """\
from distutils.core import setup
import foo

setup(name='foo', version='0.1', py_modules=['foo'],
      url='xxx', author='xxx', author_email='xxx')

"""


@support.combine_markers
@pytest.mark.usefixtures('save_env')
@pytest.mark.usefixtures('save_argv')
@pytest.mark.usefixtures('save_cwd')
class TestBuildDumb(
    support.TempdirManager,
):
    @pytest.mark.usefixtures('needs_zlib')
    def test_simple_built(self):
        # let's create a simple package
        tmp_dir = self.mkdtemp()
        pkg_dir = os.path.join(tmp_dir, 'foo')
        os.mkdir(pkg_dir)
        self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
        self.write_file((pkg_dir, 'foo.py'), '#')
        self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
        self.write_file((pkg_dir, 'README'), '')

        dist = Distribution({
            'name': 'foo',
            'version': '0.1',
            'py_modules': ['foo'],
            'url': 'xxx',
            'author': 'xxx',
            'author_email': 'xxx',
        })
        dist.script_name = 'setup.py'
        os.chdir(pkg_dir)

        sys.argv = ['setup.py']
        cmd = bdist_dumb(dist)

        # so the output is the same no matter
        # what is the platform
        cmd.format = 'zip'

        cmd.ensure_finalized()
        cmd.run()

        # see what we have
        dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
        base = f"{dist.get_fullname()}.{cmd.plat_name}.zip"

        assert dist_created == [base]

        # now let's check what we have in the zip file
        fp = zipfile.ZipFile(os.path.join('dist', base))
        try:
            contents = fp.namelist()
        finally:
            fp.close()

        contents = sorted(filter(None, map(os.path.basename, contents)))
        wanted = ['foo-0.1-py{}.{}.egg-info'.format(*sys.version_info[:2]), 'foo.py']
        if not sys.dont_write_bytecode:
            wanted.append(f'foo.{sys.implementation.cache_tag}.pyc')
        assert contents == sorted(wanted)
