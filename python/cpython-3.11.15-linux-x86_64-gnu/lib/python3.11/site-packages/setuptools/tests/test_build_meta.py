import contextlib
import importlib
import os
import re
import shutil
import signal
import sys
import tarfile
import warnings
from concurrent import futures
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import pytest
from jaraco import path
from packaging.requirements import Requirement

from setuptools.warnings import SetuptoolsDeprecationWarning

from .textwrap import DALS

SETUP_SCRIPT_STUB = "__import__('setuptools').setup()"


TIMEOUT = int(os.getenv("TIMEOUT_BACKEND_TEST", "180"))  # in seconds
IS_PYPY = '__pypy__' in sys.builtin_module_names


pytestmark = pytest.mark.skipif(
    sys.platform == "win32" and IS_PYPY,
    reason="The combination of PyPy + Windows + pytest-xdist + ProcessPoolExecutor "
    "is flaky and problematic",
)


class BuildBackendBase:
    def __init__(self, cwd='.', env=None, backend_name='setuptools.build_meta') -> None:
        self.cwd = cwd
        self.env = env or {}
        self.backend_name = backend_name


class BuildBackend(BuildBackendBase):
    """PEP 517 Build Backend"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = futures.ProcessPoolExecutor(max_workers=1)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Handles arbitrary function invocations on the build backend."""

        def method(*args, **kw):
            root = os.path.abspath(self.cwd)
            caller = BuildBackendCaller(root, self.env, self.backend_name)
            pid = None
            try:
                pid = self.pool.submit(os.getpid).result(TIMEOUT)
                return self.pool.submit(caller, name, *args, **kw).result(TIMEOUT)
            except futures.TimeoutError:
                self.pool.shutdown(wait=False)  # doesn't stop already running processes
                self._kill(pid)
                pytest.xfail(f"Backend did not respond before timeout ({TIMEOUT} s)")
            except (futures.process.BrokenProcessPool, MemoryError, OSError):
                if IS_PYPY:
                    pytest.xfail("PyPy frequently fails tests with ProcessPoolExector")
                raise

        return method

    def _kill(self, pid):
        if pid is None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)


class BuildBackendCaller(BuildBackendBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        (self.backend_name, _, self.backend_obj) = self.backend_name.partition(':')

    def __call__(self, name, *args, **kw) -> Any:
        """Handles arbitrary function invocations on the build backend."""
        os.chdir(self.cwd)
        os.environ.update(self.env)
        mod = importlib.import_module(self.backend_name)

        if self.backend_obj:
            backend = getattr(mod, self.backend_obj)
        else:
            backend = mod

        return getattr(backend, name)(*args, **kw)


defns = [
    {  # simple setup.py script
        'setup.py': DALS(
            """
            __import__('setuptools').setup(
                name='foo',
                version='0.0.0',
                py_modules=['hello'],
                setup_requires=['six'],
            )
            """
        ),
        'hello.py': DALS(
            """
            def run():
                print('hello')
            """
        ),
    },
    {  # setup.py that relies on __name__
        'setup.py': DALS(
            """
            assert __name__ == '__main__'
            __import__('setuptools').setup(
                name='foo',
                version='0.0.0',
                py_modules=['hello'],
                setup_requires=['six'],
            )
            """
        ),
        'hello.py': DALS(
            """
            def run():
                print('hello')
            """
        ),
    },
    {  # setup.py script that runs arbitrary code
        'setup.py': DALS(
            """
            variable = True
            def function():
                return variable
            assert variable
            __import__('setuptools').setup(
                name='foo',
                version='0.0.0',
                py_modules=['hello'],
                setup_requires=['six'],
            )
            """
        ),
        'hello.py': DALS(
            """
            def run():
                print('hello')
            """
        ),
    },
    {  # setup.py script that constructs temp files to be included in the distribution
        'setup.py': DALS(
            """
            # Some packages construct files on the fly, include them in the package,
            # and immediately remove them after `setup()` (e.g. pybind11==2.9.1).
            # Therefore, we cannot use `distutils.core.run_setup(..., stop_after=...)`
            # to obtain a distribution object first, and then run the distutils
            # commands later, because these files will be removed in the meantime.

            with open('world.py', 'w', encoding="utf-8") as f:
                f.write('x = 42')

            try:
                __import__('setuptools').setup(
                    name='foo',
                    version='0.0.0',
                    py_modules=['world'],
                    setup_requires=['six'],
                )
            finally:
                # Some packages will clean temporary files
                __import__('os').unlink('world.py')
            """
        ),
    },
    {  # setup.cfg only
        'setup.cfg': DALS(
            """
        [metadata]
        name = foo
        version = 0.0.0

        [options]
        py_modules=hello
        setup_requires=six
        """
        ),
        'hello.py': DALS(
            """
        def run():
            print('hello')
        """
        ),
    },
    {  # setup.cfg and setup.py
        'setup.cfg': DALS(
            """
        [metadata]
        name = foo
        version = 0.0.0

        [options]
        py_modules=hello
        setup_requires=six
        """
        ),
        'setup.py': "__import__('setuptools').setup()",
        'hello.py': DALS(
            """
        def run():
            print('hello')
        """
        ),
    },
]


class TestBuildMetaBackend:
    backend_name = 'setuptools.build_meta'

    def get_build_backend(self):
        return BuildBackend(backend_name=self.backend_name)

    @pytest.fixture(params=defns)
    def build_backend(self, tmpdir, request):
        path.build(request.param, prefix=str(tmpdir))
        with tmpdir.as_cwd():
            yield self.get_build_backend()

    def test_get_requires_for_build_wheel(self, build_backend):
        actual = build_backend.get_requires_for_build_wheel()
        expected = ['six']
        assert sorted(actual) == sorted(expected)

    def test_get_requires_for_build_sdist(self, build_backend):
        actual = build_backend.get_requires_for_build_sdist()
        expected = ['six']
        assert sorted(actual) == sorted(expected)

    def test_build_wheel(self, build_backend):
        dist_dir = os.path.abspath('pip-wheel')
        os.makedirs(dist_dir)
        wheel_name = build_backend.build_wheel(dist_dir)

        wheel_file = os.path.join(dist_dir, wheel_name)
        assert os.path.isfile(wheel_file)

        # Temporary files should be removed
        assert not os.path.isfile('world.py')

        with ZipFile(wheel_file) as zipfile:
            wheel_contents = set(zipfile.namelist())

        # Each one of the examples have a single module
        # that should be included in the distribution
        python_scripts = (f for f in wheel_contents if f.endswith('.py'))
        modules = [f for f in python_scripts if not f.endswith('setup.py')]
        assert len(modules) == 1

    @pytest.mark.parametrize('build_type', ('wheel', 'sdist'))
    def test_build_with_existing_file_present(self, build_type, tmpdir_cwd):
        # Building a sdist/wheel should still succeed if there's
        # already a sdist/wheel in the destination directory.
        files = {
            'setup.py': "from setuptools import setup\nsetup()",
            'VERSION': "0.0.1",
            'setup.cfg': DALS(
                """
                [metadata]
                name = foo
                version = file: VERSION
                """
            ),
            'pyproject.toml': DALS(
                """
                [build-system]
                requires = ["setuptools", "wheel"]
                build-backend = "setuptools.build_meta"
                """
            ),
        }

        path.build(files)

        dist_dir = os.path.abspath('preexisting-' + build_type)

        build_backend = self.get_build_backend()
        build_method = getattr(build_backend, 'build_' + build_type)

        # Build a first sdist/wheel.
        # Note: this also check the destination directory is
        # successfully created if it does not exist already.
        first_result = build_method(dist_dir)

        # Change version.
        with open("VERSION", "wt", encoding="utf-8") as version_file:
            version_file.write("0.0.2")

        # Build a *second* sdist/wheel.
        second_result = build_method(dist_dir)

        assert os.path.isfile(os.path.join(dist_dir, first_result))
        assert first_result != second_result

        # And if rebuilding the exact same sdist/wheel?
        open(os.path.join(dist_dir, second_result), 'wb').close()
        third_result = build_method(dist_dir)
        assert third_result == second_result
        assert os.path.getsize(os.path.join(dist_dir, third_result)) > 0

    @pytest.mark.parametrize("setup_script", [None, SETUP_SCRIPT_STUB])
    def test_build_with_pyproject_config(self, tmpdir, setup_script):
        files = {
            'pyproject.toml': DALS(
                """
                [build-system]
                requires = ["setuptools", "wheel"]
                build-backend = "setuptools.build_meta"

                [project]
                name = "foo"
                license = {text = "MIT"}
                description = "This is a Python package"
                dynamic = ["version", "readme"]
                classifiers = [
                    "Development Status :: 5 - Production/Stable",
                    "Intended Audience :: Developers"
                ]
                urls = {Homepage = "http://github.com"}
                dependencies = [
                    "appdirs",
                ]

                [project.optional-dependencies]
                all = [
                    "tomli>=1",
                    "pyscaffold>=4,<5",
                    'importlib; python_version == "2.6"',
                ]

                [project.scripts]
                foo = "foo.cli:main"

                [tool.setuptools]
                zip-safe = false
                package-dir = {"" = "src"}
                packages = {find = {where = ["src"]}}
                license-files = ["LICENSE*"]

                [tool.setuptools.dynamic]
                version = {attr = "foo.__version__"}
                readme = {file = "README.rst"}

                [tool.distutils.sdist]
                formats = "gztar"
                """
            ),
            "MANIFEST.in": DALS(
                """
                global-include *.py *.txt
                global-exclude *.py[cod]
                """
            ),
            "README.rst": "This is a ``README``",
            "LICENSE.txt": "---- placeholder MIT license ----",
            "src": {
                "foo": {
                    "__init__.py": "__version__ = '0.1'",
                    "__init__.pyi": "__version__: str",
                    "cli.py": "def main(): print('hello world')",
                    "data.txt": "def main(): print('hello world')",
                    "py.typed": "",
                }
            },
        }
        if setup_script:
            files["setup.py"] = setup_script

        build_backend = self.get_build_backend()
        with tmpdir.as_cwd():
            path.build(files)
            msgs = [
                "'tool.setuptools.license-files' is deprecated in favor of 'project.license-files'",
                "`project.license` as a TOML table is deprecated",
            ]
            with warnings.catch_warnings():
                for msg in msgs:
                    warnings.filterwarnings("ignore", msg, SetuptoolsDeprecationWarning)
                sdist_path = build_backend.build_sdist("temp")
                wheel_file = build_backend.build_wheel("temp")

        with tarfile.open(os.path.join(tmpdir, "temp", sdist_path)) as tar:
            sdist_contents = set(tar.getnames())

        with ZipFile(os.path.join(tmpdir, "temp", wheel_file)) as zipfile:
            wheel_contents = set(zipfile.namelist())
            metadata = str(zipfile.read("foo-0.1.dist-info/METADATA"), "utf-8")
            license = str(
                zipfile.read("foo-0.1.dist-info/licenses/LICENSE.txt"), "utf-8"
            )
            epoints = str(zipfile.read("foo-0.1.dist-info/entry_points.txt"), "utf-8")

        assert sdist_contents - {"foo-0.1/setup.py"} == {
            'foo-0.1',
            'foo-0.1/LICENSE.txt',
            'foo-0.1/MANIFEST.in',
            'foo-0.1/PKG-INFO',
            'foo-0.1/README.rst',
            'foo-0.1/pyproject.toml',
            'foo-0.1/setup.cfg',
            'foo-0.1/src',
            'foo-0.1/src/foo',
            'foo-0.1/src/foo/__init__.py',
            'foo-0.1/src/foo/__init__.pyi',
            'foo-0.1/src/foo/cli.py',
            'foo-0.1/src/foo/data.txt',
            'foo-0.1/src/foo/py.typed',
            'foo-0.1/src/foo.egg-info',
            'foo-0.1/src/foo.egg-info/PKG-INFO',
            'foo-0.1/src/foo.egg-info/SOURCES.txt',
            'foo-0.1/src/foo.egg-info/dependency_links.txt',
            'foo-0.1/src/foo.egg-info/entry_points.txt',
            'foo-0.1/src/foo.egg-info/requires.txt',
            'foo-0.1/src/foo.egg-info/top_level.txt',
            'foo-0.1/src/foo.egg-info/not-zip-safe',
        }
        assert wheel_contents == {
            "foo/__init__.py",
            "foo/__init__.pyi",  # include type information by default
            "foo/cli.py",
            "foo/data.txt",  # include_package_data defaults to True
            "foo/py.typed",  # include type information by default
            "foo-0.1.dist-info/licenses/LICENSE.txt",
            "foo-0.1.dist-info/METADATA",
            "foo-0.1.dist-info/WHEEL",
            "foo-0.1.dist-info/entry_points.txt",
            "foo-0.1.dist-info/top_level.txt",
            "foo-0.1.dist-info/RECORD",
        }
        assert license == "---- placeholder MIT license ----"

        for line in (
            "Summary: This is a Python package",
            "License: MIT",
            "License-File: LICENSE.txt",
            "Classifier: Intended Audience :: Developers",
            "Requires-Dist: appdirs",
            "Requires-Dist: " + str(Requirement('tomli>=1 ; extra == "all"')),
            "Requires-Dist: "
            + str(Requirement('importlib; python_version=="2.6" and extra =="all"')),
        ):
            assert line in metadata, (line, metadata)

        assert metadata.strip().endswith("This is a ``README``")
        assert epoints.strip() == "[console_scripts]\nfoo = foo.cli:main"

    def test_static_metadata_in_pyproject_config(self, tmpdir):
        # Make sure static metadata in pyproject.toml is not overwritten by setup.py
        # as required by PEP 621
        files = {
            'pyproject.toml': DALS(
                """
                [build-system]
                requires = ["setuptools", "wheel"]
                build-backend = "setuptools.build_meta"

                [project]
                name = "foo"
                description = "This is a Python package"
                version = "42"
                dependencies = ["six"]
                """
            ),
            'hello.py': DALS(
                """
                def run():
                    print('hello')
                """
            ),
            'setup.py': DALS(
                """
                __import__('setuptools').setup(
                    name='bar',
                    version='13',
                )
                """
            ),
        }
        build_backend = self.get_build_backend()
        with tmpdir.as_cwd():
            path.build(files)
            sdist_path = build_backend.build_sdist("temp")
            wheel_file = build_backend.build_wheel("temp")

        assert (tmpdir / "temp/foo-42.tar.gz").exists()
        assert (tmpdir / "temp/foo-42-py3-none-any.whl").exists()
        assert not (tmpdir / "temp/bar-13.tar.gz").exists()
        assert not (tmpdir / "temp/bar-42.tar.gz").exists()
        assert not (tmpdir / "temp/foo-13.tar.gz").exists()
        assert not (tmpdir / "temp/bar-13-py3-none-any.whl").exists()
        assert not (tmpdir / "temp/bar-42-py3-none-any.whl").exists()
        assert not (tmpdir / "temp/foo-13-py3-none-any.whl").exists()

        with tarfile.open(os.path.join(tmpdir, "temp", sdist_path)) as tar:
            pkg_info = str(tar.extractfile('foo-42/PKG-INFO').read(), "utf-8")
            members = tar.getnames()
            assert "bar-13/PKG-INFO" not in members

        with ZipFile(os.path.join(tmpdir, "temp", wheel_file)) as zipfile:
            metadata = str(zipfile.read("foo-42.dist-info/METADATA"), "utf-8")
            members = zipfile.namelist()
            assert "bar-13.dist-info/METADATA" not in members

        for file in pkg_info, metadata:
            for line in ("Name: foo", "Version: 42"):
                assert line in file
            for line in ("Name: bar", "Version: 13"):
                assert line not in file

    def test_build_sdist(self, build_backend):
        dist_dir = os.path.abspath('pip-sdist')
        os.makedirs(dist_dir)
        sdist_name = build_backend.build_sdist(dist_dir)

        assert os.path.isfile(os.path.join(dist_dir, sdist_name))

    def test_prepare_metadata_for_build_wheel(self, build_backend):
        dist_dir = os.path.abspath('pip-dist-info')
        os.makedirs(dist_dir)

        dist_info = build_backend.prepare_metadata_for_build_wheel(dist_dir)

        assert os.path.isfile(os.path.join(dist_dir, dist_info, 'METADATA'))

    def test_prepare_metadata_inplace(self, build_backend):
        """
        Some users might pass metadata_directory pre-populated with `.tox` or `.venv`.
        See issue #3523.
        """
        for pre_existing in [
            ".tox/python/lib/python3.10/site-packages/attrs-22.1.0.dist-info",
            ".tox/python/lib/python3.10/site-packages/autocommand-2.2.1.dist-info",
            ".nox/python/lib/python3.10/site-packages/build-0.8.0.dist-info",
            ".venv/python3.10/site-packages/click-8.1.3.dist-info",
            "venv/python3.10/site-packages/distlib-0.3.5.dist-info",
            "env/python3.10/site-packages/docutils-0.19.dist-info",
        ]:
            os.makedirs(pre_existing, exist_ok=True)
        dist_info = build_backend.prepare_metadata_for_build_wheel(".")
        assert os.path.isfile(os.path.join(dist_info, 'METADATA'))

    def test_build_sdist_explicit_dist(self, build_backend):
        # explicitly specifying the dist folder should work
        # the folder sdist_directory and the ``--dist-dir`` can be the same
        dist_dir = os.path.abspath('dist')
        sdist_name = build_backend.build_sdist(dist_dir)
        assert os.path.isfile(os.path.join(dist_dir, sdist_name))

    def test_build_sdist_version_change(self, build_backend):
        sdist_into_directory = os.path.abspath("out_sdist")
        os.makedirs(sdist_into_directory)

        sdist_name = build_backend.build_sdist(sdist_into_directory)
        assert os.path.isfile(os.path.join(sdist_into_directory, sdist_name))

        # if the setup.py changes subsequent call of the build meta
        # should still succeed, given the
        # sdist_directory the frontend specifies is empty
        setup_loc = os.path.abspath("setup.py")
        if not os.path.exists(setup_loc):
            setup_loc = os.path.abspath("setup.cfg")

        with open(setup_loc, 'rt', encoding="utf-8") as file_handler:
            content = file_handler.read()
        with open(setup_loc, 'wt', encoding="utf-8") as file_handler:
            file_handler.write(content.replace("version='0.0.0'", "version='0.0.1'"))

        shutil.rmtree(sdist_into_directory)
        os.makedirs(sdist_into_directory)

        sdist_name = build_backend.build_sdist("out_sdist")
        assert os.path.isfile(os.path.join(os.path.abspath("out_sdist"), sdist_name))

    def test_build_sdist_pyproject_toml_exists(self, tmpdir_cwd):
        files = {
            'setup.py': DALS(
                """
                __import__('setuptools').setup(
                    name='foo',
                    version='0.0.0',
                    py_modules=['hello']
                )"""
            ),
            'hello.py': '',
            'pyproject.toml': DALS(
                """
                [build-system]
                requires = ["setuptools", "wheel"]
                build-backend = "setuptools.build_meta"
                """
            ),
        }
        path.build(files)
        build_backend = self.get_build_backend()
        targz_path = build_backend.build_sdist("temp")
        with tarfile.open(os.path.join("temp", targz_path)) as tar:
            assert any('pyproject.toml' in name for name in tar.getnames())

    def test_build_sdist_setup_py_exists(self, tmpdir_cwd):
        # If build_sdist is called from a script other than setup.py,
        # ensure setup.py is included
        path.build(defns[0])

        build_backend = self.get_build_backend()
        targz_path = build_backend.build_sdist("temp")
        with tarfile.open(os.path.join("temp", targz_path)) as tar:
            assert any('setup.py' in name for name in tar.getnames())

    def test_build_sdist_setup_py_manifest_excluded(self, tmpdir_cwd):
        # Ensure that MANIFEST.in can exclude setup.py
        files = {
            'setup.py': DALS(
                """
        __import__('setuptools').setup(
            name='foo',
            version='0.0.0',
            py_modules=['hello']
        )"""
            ),
            'hello.py': '',
            'MANIFEST.in': DALS(
                """
        exclude setup.py
        """
            ),
        }

        path.build(files)

        build_backend = self.get_build_backend()
        targz_path = build_backend.build_sdist("temp")
        with tarfile.open(os.path.join("temp", targz_path)) as tar:
            assert not any('setup.py' in name for name in tar.getnames())

    def test_build_sdist_builds_targz_even_if_zip_indicated(self, tmpdir_cwd):
        files = {
            'setup.py': DALS(
                """
                __import__('setuptools').setup(
                    name='foo',
                    version='0.0.0',
                    py_modules=['hello']
                )"""
            ),
            'hello.py': '',
            'setup.cfg': DALS(
                """
                [sdist]
                formats=zip
                """
            ),
        }

        path.build(files)

        build_backend = self.get_build_backend()
        build_backend.build_sdist("temp")

    _relative_path_import_files = {
        'setup.py': DALS(
            """
            __import__('setuptools').setup(
                name='foo',
                version=__import__('hello').__version__,
                py_modules=['hello']
            )"""
        ),
        'hello.py': '__version__ = "0.0.0"',
        'setup.cfg': DALS(
            """
            [sdist]
            formats=zip
            """
        ),
    }

    def test_build_sdist_relative_path_import(self, tmpdir_cwd):
        path.build(self._relative_path_import_files)
        build_backend = self.get_build_backend()
        with pytest.raises(ImportError, match="^No module named 'hello'$"):
            build_backend.build_sdist("temp")

    _simple_pyproject_example = {
        "pyproject.toml": DALS(
            """
            [project]
            name = "proj"
            version = "42"
            """
        ),
        "src": {"proj": {"__init__.py": ""}},
    }

    def _assert_link_tree(self, parent_dir):
        """All files in the directory should be either links or hard links"""
        files = list(Path(parent_dir).glob("**/*"))
        assert files  # Should not be empty
        for file in files:
            assert file.is_symlink() or os.stat(file).st_nlink > 0

    def test_editable_without_config_settings(self, tmpdir_cwd):
        """
        Sanity check to ensure tests with --mode=strict are different from the ones
        without --mode.

        --mode=strict should create a local directory with a package tree.
        The directory should not get created otherwise.
        """
        path.build(self._simple_pyproject_example)
        build_backend = self.get_build_backend()
        assert not Path("build").exists()
        build_backend.build_editable("temp")
        assert not Path("build").exists()

    def test_build_wheel_inplace(self, tmpdir_cwd):
        config_settings = {"--build-option": ["build_ext", "--inplace"]}
        path.build(self._simple_pyproject_example)
        build_backend = self.get_build_backend()
        assert not Path("build").exists()
        Path("build").mkdir()
        build_backend.prepare_metadata_for_build_wheel("build", config_settings)
        build_backend.build_wheel("build", config_settings)
        assert Path("build/proj-42-py3-none-any.whl").exists()

    @pytest.mark.parametrize("config_settings", [{"editable-mode": "strict"}])
    def test_editable_with_config_settings(self, tmpdir_cwd, config_settings):
        path.build({**self._simple_pyproject_example, '_meta': {}})
        assert not Path("build").exists()
        build_backend = self.get_build_backend()
        build_backend.prepare_metadata_for_build_editable("_meta", config_settings)
        build_backend.build_editable("temp", config_settings, "_meta")
        self._assert_link_tree(next(Path("build").glob("__editable__.*")))

    @pytest.mark.parametrize(
        ("setup_literal", "requirements"),
        [
            ("'foo'", ['foo']),
            ("['foo']", ['foo']),
            (r"'foo\n'", ['foo']),
            (r"'foo\n\n'", ['foo']),
            ("['foo', 'bar']", ['foo', 'bar']),
            (r"'# Has a comment line\nfoo'", ['foo']),
            (r"'foo # Has an inline comment'", ['foo']),
            (r"'foo \\\n >=3.0'", ['foo>=3.0']),
            (r"'foo\nbar'", ['foo', 'bar']),
            (r"'foo\nbar\n'", ['foo', 'bar']),
            (r"['foo\n', 'bar\n']", ['foo', 'bar']),
        ],
    )
    @pytest.mark.parametrize('use_wheel', [True, False])
    def test_setup_requires(self, setup_literal, requirements, use_wheel, tmpdir_cwd):
        files = {
            'setup.py': DALS(
                """
                from setuptools import setup

                setup(
                    name="qux",
                    version="0.0.0",
                    py_modules=["hello"],
                    setup_requires={setup_literal},
                )
            """
            ).format(setup_literal=setup_literal),
            'hello.py': DALS(
                """
            def run():
                print('hello')
            """
            ),
        }

        path.build(files)

        build_backend = self.get_build_backend()

        if use_wheel:
            get_requires = build_backend.get_requires_for_build_wheel
        else:
            get_requires = build_backend.get_requires_for_build_sdist

        # Ensure that the build requirements are properly parsed
        expected = sorted(requirements)
        actual = get_requires()

        assert expected == sorted(actual)

    def test_setup_requires_with_auto_discovery(self, tmpdir_cwd):
        # Make sure patches introduced to retrieve setup_requires don't accidentally
        # activate auto-discovery and cause problems due to the incomplete set of
        # attributes passed to MinimalDistribution
        files = {
            'pyproject.toml': DALS(
                """
                [project]
                name = "proj"
                version = "42"
            """
            ),
            "setup.py": DALS(
                """
                __import__('setuptools').setup(
                    setup_requires=["foo"],
                    py_modules = ["hello", "world"]
                )
            """
            ),
            'hello.py': "'hello'",
            'world.py': "'world'",
        }
        path.build(files)
        build_backend = self.get_build_backend()
        setup_requires = build_backend.get_requires_for_build_wheel()
        assert setup_requires == ["foo"]

    def test_dont_install_setup_requires(self, tmpdir_cwd):
        files = {
            'setup.py': DALS(
                """
                        from setuptools import setup

                        setup(
                            name="qux",
                            version="0.0.0",
                            py_modules=["hello"],
                            setup_requires=["does-not-exist >99"],
                        )
                    """
            ),
            'hello.py': DALS(
                """
                    def run():
                        print('hello')
                    """
            ),
        }

        path.build(files)

        build_backend = self.get_build_backend()

        dist_dir = os.path.abspath('pip-dist-info')
        os.makedirs(dist_dir)

        # does-not-exist can't be satisfied, so if it attempts to install
        # setup_requires, it will fail.
        build_backend.prepare_metadata_for_build_wheel(dist_dir)

    _sys_argv_0_passthrough = {
        'setup.py': DALS(
            """
            import os
            import sys

            __import__('setuptools').setup(
                name='foo',
                version='0.0.0',
            )

            sys_argv = os.path.abspath(sys.argv[0])
            file_path = os.path.abspath('setup.py')
            assert sys_argv == file_path
            """
        )
    }

    def test_sys_argv_passthrough(self, tmpdir_cwd):
        path.build(self._sys_argv_0_passthrough)
        build_backend = self.get_build_backend()
        with pytest.raises(AssertionError):
            build_backend.build_sdist("temp")

    _setup_py_file_abspath = {
        'setup.py': DALS(
            """
            import os
            assert os.path.isabs(__file__)
            __import__('setuptools').setup(
                name='foo',
                version='0.0.0',
                py_modules=['hello'],
                setup_requires=['six'],
            )
            """
        )
    }

    def test_setup_py_file_abspath(self, tmpdir_cwd):
        path.build(self._setup_py_file_abspath)
        build_backend = self.get_build_backend()
        build_backend.build_sdist("temp")

    @pytest.mark.parametrize('build_hook', ('build_sdist', 'build_wheel'))
    def test_build_with_empty_setuppy(self, build_backend, build_hook):
        files = {'setup.py': ''}
        path.build(files)

        msg = re.escape('No distribution was found.')
        with pytest.raises(ValueError, match=msg):
            getattr(build_backend, build_hook)("temp")


class TestBuildMetaLegacyBackend(TestBuildMetaBackend):
    backend_name = 'setuptools.build_meta:__legacy__'

    # build_meta_legacy-specific tests
    def test_build_sdist_relative_path_import(self, tmpdir_cwd):
        # This must fail in build_meta, but must pass in build_meta_legacy
        path.build(self._relative_path_import_files)

        build_backend = self.get_build_backend()
        build_backend.build_sdist("temp")

    def test_sys_argv_passthrough(self, tmpdir_cwd):
        path.build(self._sys_argv_0_passthrough)

        build_backend = self.get_build_backend()
        build_backend.build_sdist("temp")


@pytest.mark.filterwarnings("ignore::setuptools.SetuptoolsDeprecationWarning")
def test_sys_exit_0_in_setuppy(monkeypatch, tmp_path):
    """Setuptools should be resilient to setup.py with ``sys.exit(0)`` (#3973)."""
    monkeypatch.chdir(tmp_path)
    setuppy = """
        import sys, setuptools
        setuptools.setup(name='foo', version='0.0.0')
        sys.exit(0)
        """
    (tmp_path / "setup.py").write_text(DALS(setuppy), encoding="utf-8")
    backend = BuildBackend(backend_name="setuptools.build_meta")
    assert backend.get_requires_for_build_wheel() == []


def test_system_exit_in_setuppy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setuppy = "import sys; sys.exit('some error')"
    (tmp_path / "setup.py").write_text(setuppy, encoding="utf-8")
    with pytest.raises(SystemExit, match="some error"):
        backend = BuildBackend(backend_name="setuptools.build_meta")
        backend.get_requires_for_build_wheel()
