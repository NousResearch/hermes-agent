import configparser
import contextlib
import inspect
import re
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from packaging.requirements import InvalidRequirement

from setuptools.config.setupcfg import ConfigHandler, Target, read_configuration
from setuptools.dist import Distribution, _Distribution
from setuptools.warnings import SetuptoolsDeprecationWarning

from ..textwrap import DALS

from distutils.errors import DistutilsFileError, DistutilsOptionError

IS_PYPY = '__pypy__' in sys.builtin_module_names


class ErrConfigHandler(ConfigHandler[Target]):
    """Erroneous handler. Fails to implement required methods."""

    section_prefix = "**err**"


def make_package_dir(name, base_dir, ns=False):
    dir_package = base_dir
    for dir_name in name.split('/'):
        dir_package = dir_package.mkdir(dir_name)
    init_file = None
    if not ns:
        init_file = dir_package.join('__init__.py')
        init_file.write('')
    return dir_package, init_file


def fake_env(
    tmpdir, setup_cfg, setup_py=None, encoding='ascii', package_path='fake_package'
):
    if setup_py is None:
        setup_py = 'from setuptools import setup\nsetup()\n'

    tmpdir.join('setup.py').write(setup_py)
    config = tmpdir.join('setup.cfg')
    config.write(setup_cfg.encode(encoding), mode='wb')

    package_dir, init_file = make_package_dir(package_path, tmpdir)

    init_file.write(
        'VERSION = (1, 2, 3)\n'
        '\n'
        'VERSION_MAJOR = 1'
        '\n'
        'def get_version():\n'
        '    return [3, 4, 5, "dev"]\n'
        '\n'
    )

    return package_dir, config


@contextlib.contextmanager
def get_dist(tmpdir, kwargs_initial=None, parse=True):
    kwargs_initial = kwargs_initial or {}

    with tmpdir.as_cwd():
        dist = Distribution(kwargs_initial)
        dist.script_name = 'setup.py'
        parse and dist.parse_config_files()

        yield dist


def test_parsers_implemented():
    with pytest.raises(NotImplementedError):
        handler = ErrConfigHandler(None, {}, False, Mock())
        handler.parsers


class TestConfigurationReader:
    def test_basic(self, tmpdir):
        _, config = fake_env(
            tmpdir,
            '[metadata]\n'
            'version = 10.1.1\n'
            'keywords = one, two\n'
            '\n'
            '[options]\n'
            'scripts = bin/a.py, bin/b.py\n',
        )
        config_dict = read_configuration(str(config))
        assert config_dict['metadata']['version'] == '10.1.1'
        assert config_dict['metadata']['keywords'] == ['one', 'two']
        assert config_dict['options']['scripts'] == ['bin/a.py', 'bin/b.py']

    def test_no_config(self, tmpdir):
        with pytest.raises(DistutilsFileError):
            read_configuration(str(tmpdir.join('setup.cfg')))

    def test_ignore_errors(self, tmpdir):
        _, config = fake_env(
            tmpdir,
            '[metadata]\nversion = attr: none.VERSION\nkeywords = one, two\n',
        )
        with pytest.raises(ImportError):
            read_configuration(str(config))

        config_dict = read_configuration(str(config), ignore_option_errors=True)

        assert config_dict['metadata']['keywords'] == ['one', 'two']
        assert 'version' not in config_dict['metadata']

        config.remove()


class TestMetadata:
    def test_basic(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'version = 10.1.1\n'
            'description = Some description\n'
            'long_description_content_type = text/something\n'
            'long_description = file: README\n'
            'name = fake_name\n'
            'keywords = one, two\n'
            'provides = package, package.sub\n'
            'license = otherlic\n'
            'download_url = http://test.test.com/test/\n'
            'maintainer_email = test@test.com\n',
        )

        tmpdir.join('README').write('readme contents\nline2')

        meta_initial = {
            # This will be used so `otherlic` won't replace it.
            'license': 'BSD 3-Clause License',
        }

        with get_dist(tmpdir, meta_initial) as dist:
            metadata = dist.metadata

            assert metadata.version == '10.1.1'
            assert metadata.description == 'Some description'
            assert metadata.long_description_content_type == 'text/something'
            assert metadata.long_description == 'readme contents\nline2'
            assert metadata.provides == ['package', 'package.sub']
            assert metadata.license == 'BSD 3-Clause License'
            assert metadata.name == 'fake_name'
            assert metadata.keywords == ['one', 'two']
            assert metadata.download_url == 'http://test.test.com/test/'
            assert metadata.maintainer_email == 'test@test.com'

    def test_license_cfg(self, tmpdir):
        fake_env(
            tmpdir,
            DALS(
                """
            [metadata]
            name=foo
            version=0.0.1
            license=Apache 2.0
            """
            ),
        )

        with get_dist(tmpdir) as dist:
            metadata = dist.metadata

            assert metadata.name == "foo"
            assert metadata.version == "0.0.1"
            assert metadata.license == "Apache 2.0"

    def test_file_mixed(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\nlong_description = file: README.rst, CHANGES.rst\n\n',
        )

        tmpdir.join('README.rst').write('readme contents\nline2')
        tmpdir.join('CHANGES.rst').write('changelog contents\nand stuff')

        with get_dist(tmpdir) as dist:
            assert dist.metadata.long_description == (
                'readme contents\nline2\nchangelog contents\nand stuff'
            )

    def test_file_sandboxed(self, tmpdir):
        tmpdir.ensure("README")
        project = tmpdir.join('depth1', 'depth2')
        project.ensure(dir=True)
        fake_env(project, '[metadata]\nlong_description = file: ../../README\n')

        with get_dist(project, parse=False) as dist:
            with pytest.raises(DistutilsOptionError):
                dist.parse_config_files()  # file: out of sandbox

    def test_aliases(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'author_email = test@test.com\n'
            'home_page = http://test.test.com/test/\n'
            'summary = Short summary\n'
            'platform = a, b\n'
            'classifier =\n'
            '  Framework :: Django\n'
            '  Programming Language :: Python :: 3.5\n',
        )

        with get_dist(tmpdir) as dist:
            metadata = dist.metadata
            assert metadata.author_email == 'test@test.com'
            assert metadata.url == 'http://test.test.com/test/'
            assert metadata.description == 'Short summary'
            assert metadata.platforms == ['a', 'b']
            assert metadata.classifiers == [
                'Framework :: Django',
                'Programming Language :: Python :: 3.5',
            ]

    def test_multiline(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'name = fake_name\n'
            'keywords =\n'
            '  one\n'
            '  two\n'
            'classifiers =\n'
            '  Framework :: Django\n'
            '  Programming Language :: Python :: 3.5\n',
        )
        with get_dist(tmpdir) as dist:
            metadata = dist.metadata
            assert metadata.keywords == ['one', 'two']
            assert metadata.classifiers == [
                'Framework :: Django',
                'Programming Language :: Python :: 3.5',
            ]

    def test_dict(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'project_urls =\n'
            '  Link One = https://example.com/one/\n'
            '  Link Two = https://example.com/two/\n',
        )
        with get_dist(tmpdir) as dist:
            metadata = dist.metadata
            assert metadata.project_urls == {
                'Link One': 'https://example.com/one/',
                'Link Two': 'https://example.com/two/',
            }

    def test_version(self, tmpdir):
        package_dir, config = fake_env(
            tmpdir, '[metadata]\nversion = attr: fake_package.VERSION\n'
        )

        sub_a = package_dir.mkdir('subpkg_a')
        sub_a.join('__init__.py').write('')
        sub_a.join('mod.py').write('VERSION = (2016, 11, 26)')

        sub_b = package_dir.mkdir('subpkg_b')
        sub_b.join('__init__.py').write('')
        sub_b.join('mod.py').write(
            'import third_party_module\nVERSION = (2016, 11, 26)'
        )

        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1.2.3'

        config.write('[metadata]\nversion = attr: fake_package.get_version\n')
        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '3.4.5.dev'

        config.write('[metadata]\nversion = attr: fake_package.VERSION_MAJOR\n')
        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1'

        config.write('[metadata]\nversion = attr: fake_package.subpkg_a.mod.VERSION\n')
        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '2016.11.26'

        config.write('[metadata]\nversion = attr: fake_package.subpkg_b.mod.VERSION\n')
        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '2016.11.26'

    def test_version_file(self, tmpdir):
        fake_env(tmpdir, '[metadata]\nversion = file: fake_package/version.txt\n')
        tmpdir.join('fake_package', 'version.txt').write('1.2.3\n')

        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1.2.3'

        tmpdir.join('fake_package', 'version.txt').write('1.2.3\n4.5.6\n')
        with pytest.raises(DistutilsOptionError):
            with get_dist(tmpdir) as dist:
                dist.metadata.version

    def test_version_with_package_dir_simple(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'version = attr: fake_package_simple.VERSION\n'
            '[options]\n'
            'package_dir =\n'
            '    = src\n',
            package_path='src/fake_package_simple',
        )

        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1.2.3'

    def test_version_with_package_dir_rename(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'version = attr: fake_package_rename.VERSION\n'
            '[options]\n'
            'package_dir =\n'
            '    fake_package_rename = fake_dir\n',
            package_path='fake_dir',
        )

        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1.2.3'

    def test_version_with_package_dir_complex(self, tmpdir):
        fake_env(
            tmpdir,
            '[metadata]\n'
            'version = attr: fake_package_complex.VERSION\n'
            '[options]\n'
            'package_dir =\n'
            '    fake_package_complex = src/fake_dir\n',
            package_path='src/fake_dir',
        )

        with get_dist(tmpdir) as dist:
            assert dist.metadata.version == '1.2.3'

    def test_unknown_meta_item(self, tmpdir):
        fake_env(tmpdir, '[metadata]\nname = fake_name\nunknown = some\n')
        with get_dist(tmpdir, parse=False) as dist:
            dist.parse_config_files()  # Skip unknown.

    def test_usupported_section(self, tmpdir):
        fake_env(tmpdir, '[metadata.some]\nkey = val\n')
        with get_dist(tmpdir, parse=False) as dist:
            with pytest.raises(DistutilsOptionError):
                dist.parse_config_files()

    def test_classifiers(self, tmpdir):
        expected = set([
            'Framework :: Django',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
        ])

        # From file.
        _, config = fake_env(tmpdir, '[metadata]\nclassifiers = file: classifiers\n')

        tmpdir.join('classifiers').write(
            'Framework :: Django\n'
            'Programming Language :: Python :: 3\n'
            'Programming Language :: Python :: 3.5\n'
        )

        with get_dist(tmpdir) as dist:
            assert set(dist.metadata.classifiers) == expected

        # From list notation
        config.write(
            '[metadata]\n'
            'classifiers =\n'
            '    Framework :: Django\n'
            '    Programming Language :: Python :: 3\n'
            '    Programming Language :: Python :: 3.5\n'
        )
        with get_dist(tmpdir) as dist:
            assert set(dist.metadata.classifiers) == expected

    def test_interpolation(self, tmpdir):
        fake_env(tmpdir, '[metadata]\ndescription = %(message)s\n')
        with pytest.raises(configparser.InterpolationMissingOptionError):
            with get_dist(tmpdir):
                pass

    def test_non_ascii_1(self, tmpdir):
        fake_env(tmpdir, '[metadata]\ndescription = éàïôñ\n', encoding='utf-8')
        with get_dist(tmpdir):
            pass

    def test_non_ascii_3(self, tmpdir):
        fake_env(tmpdir, '\n# -*- coding: invalid\n')
        with get_dist(tmpdir):
            pass

    def test_non_ascii_4(self, tmpdir):
        fake_env(
            tmpdir,
            '# -*- coding: utf-8\n[metadata]\ndescription = éàïôñ\n',
            encoding='utf-8',
        )
        with get_dist(tmpdir) as dist:
            assert dist.metadata.description == 'éàïôñ'

    def test_not_utf8(self, tmpdir):
        """
        Config files encoded not in UTF-8 will fail
        """
        fake_env(
            tmpdir,
            '# vim: set fileencoding=iso-8859-15 :\n[metadata]\ndescription = éàïôñ\n',
            encoding='iso-8859-15',
        )
        with pytest.raises(UnicodeDecodeError):
            with get_dist(tmpdir):
                pass

    @pytest.mark.parametrize(
        ("error_msg", "config", "invalid"),
        [
            (
                "Invalid dash-separated key 'author-email' in 'metadata' (setup.cfg)",
                DALS(
                    """
                    [metadata]
                    author-email = test@test.com
                    maintainer_email = foo@foo.com
                    """
                ),
                {"author-email": "test@test.com"},
            ),
            (
                "Invalid uppercase key 'Name' in 'metadata' (setup.cfg)",
                DALS(
                    """
                    [metadata]
                    Name = foo
                    description = Some description
                    """
                ),
                {"Name": "foo"},
            ),
        ],
    )
    def test_invalid_options_previously_deprecated(
        self, tmpdir, error_msg, config, invalid
    ):
        # This test and related methods can be removed when no longer needed.
        # Deprecation postponed due to push-back from the community in
        # https://github.com/pypa/setuptools/issues/4910
        fake_env(tmpdir, config)
        with pytest.warns(SetuptoolsDeprecationWarning, match=re.escape(error_msg)):
            dist = get_dist(tmpdir).__enter__()

        tmpdir.join('setup.cfg').remove()

        for field, value in invalid.items():
            attr = field.replace("-", "_").lower()
            assert getattr(dist.metadata, attr) == value


class TestOptions:
    def test_basic(self, tmpdir):
        fake_env(
            tmpdir,
            '[options]\n'
            'zip_safe = True\n'
            'include_package_data = yes\n'
            'package_dir = b=c, =src\n'
            'packages = pack_a, pack_b.subpack\n'
            'namespace_packages = pack1, pack2\n'
            'scripts = bin/one.py, bin/two.py\n'
            'eager_resources = bin/one.py, bin/two.py\n'
            'install_requires = docutils>=0.3; pack ==1.1, ==1.3; hey\n'
            'setup_requires = docutils>=0.3; spack ==1.1, ==1.3; there\n'
            'dependency_links = http://some.com/here/1, '
            'http://some.com/there/2\n'
            'python_requires = >=1.0, !=2.8\n'
            'py_modules = module1, module2\n',
        )
        deprec = pytest.warns(SetuptoolsDeprecationWarning, match="namespace_packages")
        with deprec, get_dist(tmpdir) as dist:
            assert dist.zip_safe
            assert dist.include_package_data
            assert dist.package_dir == {'': 'src', 'b': 'c'}
            assert dist.packages == ['pack_a', 'pack_b.subpack']
            assert dist.namespace_packages == ['pack1', 'pack2']
            assert dist.scripts == ['bin/one.py', 'bin/two.py']
            assert dist.dependency_links == ([
                'http://some.com/here/1',
                'http://some.com/there/2',
            ])
            assert dist.install_requires == ([
                'docutils>=0.3',
                'pack==1.1,==1.3',
                'hey',
            ])
            assert dist.setup_requires == ([
                'docutils>=0.3',
                'spack ==1.1, ==1.3',
                'there',
            ])
            assert dist.python_requires == '>=1.0, !=2.8'
            assert dist.py_modules == ['module1', 'module2']

    def test_multiline(self, tmpdir):
        fake_env(
            tmpdir,
            '[options]\n'
            'package_dir = \n'
            '  b=c\n'
            '  =src\n'
            'packages = \n'
            '  pack_a\n'
            '  pack_b.subpack\n'
            'namespace_packages = \n'
            '  pack1\n'
            '  pack2\n'
            'scripts = \n'
            '  bin/one.py\n'
            '  bin/two.py\n'
            'eager_resources = \n'
            '  bin/one.py\n'
            '  bin/two.py\n'
            'install_requires = \n'
            '  docutils>=0.3\n'
            '  pack ==1.1, ==1.3\n'
            '  hey\n'
            'setup_requires = \n'
            '  docutils>=0.3\n'
            '  spack ==1.1, ==1.3\n'
            '  there\n'
            'dependency_links = \n'
            '  http://some.com/here/1\n'
            '  http://some.com/there/2\n',
        )
        deprec = pytest.warns(SetuptoolsDeprecationWarning, match="namespace_packages")
        with deprec, get_dist(tmpdir) as dist:
            assert dist.package_dir == {'': 'src', 'b': 'c'}
            assert dist.packages == ['pack_a', 'pack_b.subpack']
            assert dist.namespace_packages == ['pack1', 'pack2']
            assert dist.scripts == ['bin/one.py', 'bin/two.py']
            assert dist.dependency_links == ([
                'http://some.com/here/1',
                'http://some.com/there/2',
            ])
            assert dist.install_requires == ([
                'docutils>=0.3',
                'pack==1.1,==1.3',
                'hey',
            ])
            assert dist.setup_requires == ([
                'docutils>=0.3',
                'spack ==1.1, ==1.3',
                'there',
            ])

    def test_package_dir_fail(self, tmpdir):
        fake_env(tmpdir, '[options]\npackage_dir = a b\n')
        with get_dist(tmpdir, parse=False) as dist:
            with pytest.raises(DistutilsOptionError):
                dist.parse_config_files()

    def test_package_data(self, tmpdir):
        fake_env(
            tmpdir,
            '[options.package_data]\n'
            '* = *.txt, *.rst\n'
            'hello = *.msg\n'
            '\n'
            '[options.exclude_package_data]\n'
            '* = fake1.txt, fake2.txt\n'
            'hello = *.dat\n',
        )

        with get_dist(tmpdir) as dist:
            assert dist.package_data == {
                '': ['*.txt', '*.rst'],
                'hello': ['*.msg'],
            }
            assert dist.exclude_package_data == {
                '': ['fake1.txt', 'fake2.txt'],
                'hello': ['*.dat'],
            }

    def test_packages(self, tmpdir):
        fake_env(tmpdir, '[options]\npackages = find:\n')

        with get_dist(tmpdir) as dist:
            assert dist.packages == ['fake_package']

    def test_find_directive(self, tmpdir):
        dir_package, config = fake_env(tmpdir, '[options]\npackages = find:\n')

        make_package_dir('sub_one', dir_package)
        make_package_dir('sub_two', dir_package)

        with get_dist(tmpdir) as dist:
            assert set(dist.packages) == set([
                'fake_package',
                'fake_package.sub_two',
                'fake_package.sub_one',
            ])

        config.write(
            '[options]\n'
            'packages = find:\n'
            '\n'
            '[options.packages.find]\n'
            'where = .\n'
            'include =\n'
            '    fake_package.sub_one\n'
            '    two\n'
        )
        with get_dist(tmpdir) as dist:
            assert dist.packages == ['fake_package.sub_one']

        config.write(
            '[options]\n'
            'packages = find:\n'
            '\n'
            '[options.packages.find]\n'
            'exclude =\n'
            '    fake_package.sub_one\n'
        )
        with get_dist(tmpdir) as dist:
            assert set(dist.packages) == set(['fake_package', 'fake_package.sub_two'])

    def test_find_namespace_directive(self, tmpdir):
        dir_package, config = fake_env(
            tmpdir, '[options]\npackages = find_namespace:\n'
        )

        make_package_dir('sub_one', dir_package)
        make_package_dir('sub_two', dir_package, ns=True)

        with get_dist(tmpdir) as dist:
            assert set(dist.packages) == {
                'fake_package',
                'fake_package.sub_two',
                'fake_package.sub_one',
            }

        config.write(
            '[options]\n'
            'packages = find_namespace:\n'
            '\n'
            '[options.packages.find]\n'
            'where = .\n'
            'include =\n'
            '    fake_package.sub_one\n'
            '    two\n'
        )
        with get_dist(tmpdir) as dist:
            assert dist.packages == ['fake_package.sub_one']

        config.write(
            '[options]\n'
            'packages = find_namespace:\n'
            '\n'
            '[options.packages.find]\n'
            'exclude =\n'
            '    fake_package.sub_one\n'
        )
        with get_dist(tmpdir) as dist:
            assert set(dist.packages) == {'fake_package', 'fake_package.sub_two'}

    def test_extras_require(self, tmpdir):
        fake_env(
            tmpdir,
            '[options.extras_require]\n'
            'pdf = ReportLab>=1.2; RXP\n'
            'rest = \n'
            '  docutils>=0.3\n'
            '  pack ==1.1, ==1.3\n',
        )

        with get_dist(tmpdir) as dist:
            assert dist.extras_require == {
                'pdf': ['ReportLab>=1.2', 'RXP'],
                'rest': ['docutils>=0.3', 'pack==1.1,==1.3'],
            }
            assert set(dist.metadata.provides_extras) == {'pdf', 'rest'}

    @pytest.mark.parametrize(
        "config",
        [
            "[options.extras_require]\nfoo = bar;python_version<'3'",
            "[options.extras_require]\nfoo = bar;os_name=='linux'",
            "[options.extras_require]\nfoo = bar;python_version<'3'\n",
            "[options.extras_require]\nfoo = bar;os_name=='linux'\n",
            "[options]\ninstall_requires = bar;python_version<'3'",
            "[options]\ninstall_requires = bar;os_name=='linux'",
            "[options]\ninstall_requires = bar;python_version<'3'\n",
            "[options]\ninstall_requires = bar;os_name=='linux'\n",
        ],
    )
    @pytest.mark.xfail(IS_PYPY, reason="Exceptions missing on PyPy")
    # TODO: investigate PyPy problem
    def test_raises_accidental_env_marker_misconfig(self, config, tmpdir):
        fake_env(tmpdir, config)
        match = (
            r"One of the parsed requirements in `(install_requires|extras_require.+)` "
            "looks like a valid environment marker.*"
        )
        with pytest.raises(InvalidRequirement, match=match):
            with get_dist(tmpdir) as _:
                pass

    @pytest.mark.parametrize(
        "config",
        [
            "[options.extras_require]\nfoo = bar;python_version<3",
            "[options.extras_require]\nfoo = bar;python_version<3\n",
            "[options]\ninstall_requires = bar;python_version<3",
            "[options]\ninstall_requires = bar;python_version<3\n",
        ],
    )
    @pytest.mark.xfail(IS_PYPY, reason="Warnings missing on PyPy (minor issue)")
    # TODO: investigate PyPy problem
    def test_warn_accidental_env_marker_misconfig(self, config, tmpdir):
        fake_env(tmpdir, config)
        match = (
            r"One of the parsed requirements in `(install_requires|extras_require.+)` "
            "looks like a valid environment marker.*"
        )
        with pytest.warns(SetuptoolsDeprecationWarning, match=match):
            with get_dist(tmpdir) as _:
                pass

    @pytest.mark.parametrize(
        "config",
        [
            "[options.extras_require]\nfoo =\n    bar;python_version<'3'",
            "[options.extras_require]\nfoo = bar;baz\nboo = xxx;yyy",
            "[options.extras_require]\nfoo =\n    bar;python_version<'3'\n",
            "[options.extras_require]\nfoo = bar;baz\nboo = xxx;yyy\n",
            "[options.extras_require]\nfoo =\n    bar\n    python_version<3\n",
            "[options]\ninstall_requires =\n    bar;python_version<'3'",
            "[options]\ninstall_requires = bar;baz\nboo = xxx;yyy",
            "[options]\ninstall_requires =\n    bar;python_version<'3'\n",
            "[options]\ninstall_requires = bar;baz\nboo = xxx;yyy\n",
            "[options]\ninstall_requires =\n    bar\n    python_version<3\n",
        ],
    )
    @pytest.mark.filterwarnings("error::setuptools.SetuptoolsDeprecationWarning")
    def test_nowarn_accidental_env_marker_misconfig(self, config, tmpdir, recwarn):
        fake_env(tmpdir, config)
        num_warnings = len(recwarn)
        with get_dist(tmpdir) as _:
            pass
        # The examples are valid, no warnings shown
        assert len(recwarn) == num_warnings

    def test_dash_preserved_extras_require(self, tmpdir):
        fake_env(tmpdir, '[options.extras_require]\nfoo-a = foo\nfoo_b = test\n')

        with get_dist(tmpdir) as dist:
            assert dist.extras_require == {'foo-a': ['foo'], 'foo_b': ['test']}

    def test_entry_points(self, tmpdir):
        _, config = fake_env(
            tmpdir,
            '[options.entry_points]\n'
            'group1 = point1 = pack.module:func, '
            '.point2 = pack.module2:func_rest [rest]\n'
            'group2 = point3 = pack.module:func2\n',
        )

        with get_dist(tmpdir) as dist:
            assert dist.entry_points == {
                'group1': [
                    'point1 = pack.module:func',
                    '.point2 = pack.module2:func_rest [rest]',
                ],
                'group2': ['point3 = pack.module:func2'],
            }

        expected = (
            '[blogtool.parsers]\n'
            '.rst = some.nested.module:SomeClass.some_classmethod[reST]\n'
        )

        tmpdir.join('entry_points').write(expected)

        # From file.
        config.write('[options]\nentry_points = file: entry_points\n')

        with get_dist(tmpdir) as dist:
            assert dist.entry_points == expected

    def test_case_sensitive_entry_points(self, tmpdir):
        fake_env(
            tmpdir,
            '[options.entry_points]\n'
            'GROUP1 = point1 = pack.module:func, '
            '.point2 = pack.module2:func_rest [rest]\n'
            'group2 = point3 = pack.module:func2\n',
        )

        with get_dist(tmpdir) as dist:
            assert dist.entry_points == {
                'GROUP1': [
                    'point1 = pack.module:func',
                    '.point2 = pack.module2:func_rest [rest]',
                ],
                'group2': ['point3 = pack.module:func2'],
            }

    def test_data_files(self, tmpdir):
        fake_env(
            tmpdir,
            '[options.data_files]\n'
            'cfg =\n'
            '      a/b.conf\n'
            '      c/d.conf\n'
            'data = e/f.dat, g/h.dat\n',
        )

        with get_dist(tmpdir) as dist:
            expected = [
                ('cfg', ['a/b.conf', 'c/d.conf']),
                ('data', ['e/f.dat', 'g/h.dat']),
            ]
            assert sorted(dist.data_files) == sorted(expected)

    def test_data_files_globby(self, tmpdir):
        fake_env(
            tmpdir,
            '[options.data_files]\n'
            'cfg =\n'
            '      a/b.conf\n'
            '      c/d.conf\n'
            'data = *.dat\n'
            'icons = \n'
            '      *.ico\n'
            'audio = \n'
            '      *.wav\n'
            '      sounds.db\n',
        )

        # Create dummy files for glob()'s sake:
        tmpdir.join('a.dat').write('')
        tmpdir.join('b.dat').write('')
        tmpdir.join('c.dat').write('')
        tmpdir.join('a.ico').write('')
        tmpdir.join('b.ico').write('')
        tmpdir.join('c.ico').write('')
        tmpdir.join('beep.wav').write('')
        tmpdir.join('boop.wav').write('')
        tmpdir.join('sounds.db').write('')

        with get_dist(tmpdir) as dist:
            expected = [
                ('cfg', ['a/b.conf', 'c/d.conf']),
                ('data', ['a.dat', 'b.dat', 'c.dat']),
                ('icons', ['a.ico', 'b.ico', 'c.ico']),
                ('audio', ['beep.wav', 'boop.wav', 'sounds.db']),
            ]
            assert sorted(dist.data_files) == sorted(expected)

    def test_python_requires_simple(self, tmpdir):
        fake_env(
            tmpdir,
            DALS(
                """
            [options]
            python_requires=>=2.7
            """
            ),
        )
        with get_dist(tmpdir) as dist:
            dist.parse_config_files()

    def test_python_requires_compound(self, tmpdir):
        fake_env(
            tmpdir,
            DALS(
                """
            [options]
            python_requires=>=2.7,!=3.0.*
            """
            ),
        )
        with get_dist(tmpdir) as dist:
            dist.parse_config_files()

    def test_python_requires_invalid(self, tmpdir):
        fake_env(
            tmpdir,
            DALS(
                """
            [options]
            python_requires=invalid
            """
            ),
        )
        with pytest.raises(Exception):
            with get_dist(tmpdir) as dist:
                dist.parse_config_files()

    def test_cmdclass(self, tmpdir):
        module_path = Path(tmpdir, "src/custom_build.py")  # auto discovery for src
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(
            "from distutils.core import Command\nclass CustomCmd(Command): pass\n",
            encoding="utf-8",
        )

        setup_cfg = """
            [options]
            cmdclass =
                customcmd = custom_build.CustomCmd
        """
        fake_env(tmpdir, inspect.cleandoc(setup_cfg))

        with get_dist(tmpdir) as dist:
            cmdclass = dist.cmdclass['customcmd']
            assert cmdclass.__name__ == "CustomCmd"
            assert cmdclass.__module__ == "custom_build"
            assert module_path.samefile(inspect.getfile(cmdclass))

    def test_requirements_file(self, tmpdir):
        fake_env(
            tmpdir,
            DALS(
                """
            [options]
            install_requires = file:requirements.txt
            [options.extras_require]
            colors = file:requirements-extra.txt
            """
            ),
        )

        tmpdir.join('requirements.txt').write('\ndocutils>=0.3\n\n')
        tmpdir.join('requirements-extra.txt').write('colorama')

        with get_dist(tmpdir) as dist:
            assert dist.install_requires == ['docutils>=0.3']
            assert dist.extras_require == {'colors': ['colorama']}


saved_dist_init = _Distribution.__init__


class TestExternalSetters:
    # During creation of the setuptools Distribution() object, we call
    # the init of the parent distutils Distribution object via
    # _Distribution.__init__ ().
    #
    # It's possible distutils calls out to various keyword
    # implementations (i.e. distutils.setup_keywords entry points)
    # that may set a range of variables.
    #
    # This wraps distutil's Distribution.__init__ and simulates
    # pbr or something else setting these values.
    def _fake_distribution_init(self, dist, attrs):
        saved_dist_init(dist, attrs)
        # see self._DISTUTILS_UNSUPPORTED_METADATA
        dist.metadata.long_description_content_type = 'text/something'
        # Test overwrite setup() args
        dist.metadata.project_urls = {
            'Link One': 'https://example.com/one/',
            'Link Two': 'https://example.com/two/',
        }

    @patch.object(_Distribution, '__init__', autospec=True)
    def test_external_setters(self, mock_parent_init, tmpdir):
        mock_parent_init.side_effect = self._fake_distribution_init

        dist = Distribution(attrs={'project_urls': {'will_be': 'ignored'}})

        assert dist.metadata.long_description_content_type == 'text/something'
        assert dist.metadata.project_urls == {
            'Link One': 'https://example.com/one/',
            'Link Two': 'https://example.com/two/',
        }
