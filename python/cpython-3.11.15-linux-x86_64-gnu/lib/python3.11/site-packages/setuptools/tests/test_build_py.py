import os
import shutil
import stat
import warnings
from pathlib import Path
from unittest.mock import Mock

import jaraco.path
import pytest

from setuptools import SetuptoolsDeprecationWarning
from setuptools.dist import Distribution

from .textwrap import DALS


def test_directories_in_package_data_glob(tmpdir_cwd):
    """
    Directories matching the glob in package_data should
    not be included in the package data.

    Regression test for #261.
    """
    dist = Distribution(
        dict(
            script_name='setup.py',
            script_args=['build_py'],
            packages=[''],
            package_data={'': ['path/*']},
        )
    )
    os.makedirs('path/subpath')
    dist.parse_command_line()
    dist.run_commands()


def test_recursive_in_package_data_glob(tmpdir_cwd):
    """
    Files matching recursive globs (**) in package_data should
    be included in the package data.

    #1806
    """
    dist = Distribution(
        dict(
            script_name='setup.py',
            script_args=['build_py'],
            packages=[''],
            package_data={'': ['path/**/data']},
        )
    )
    os.makedirs('path/subpath/subsubpath')
    open('path/subpath/subsubpath/data', 'wb').close()

    dist.parse_command_line()
    dist.run_commands()

    assert stat.S_ISREG(os.stat('build/lib/path/subpath/subsubpath/data').st_mode), (
        "File is not included"
    )


def test_read_only(tmpdir_cwd):
    """
    Ensure read-only flag is not preserved in copy
    for package modules and package data, as that
    causes problems with deleting read-only files on
    Windows.

    #1451
    """
    dist = Distribution(
        dict(
            script_name='setup.py',
            script_args=['build_py'],
            packages=['pkg'],
            package_data={'pkg': ['data.dat']},
        )
    )
    os.makedirs('pkg')
    open('pkg/__init__.py', 'wb').close()
    open('pkg/data.dat', 'wb').close()
    os.chmod('pkg/__init__.py', stat.S_IREAD)
    os.chmod('pkg/data.dat', stat.S_IREAD)
    dist.parse_command_line()
    dist.run_commands()
    shutil.rmtree('build')


@pytest.mark.xfail(
    'platform.system() == "Windows"',
    reason="On Windows, files do not have executable bits",
    raises=AssertionError,
    strict=True,
)
def test_executable_data(tmpdir_cwd):
    """
    Ensure executable bit is preserved in copy for
    package data, as users rely on it for scripts.

    #2041
    """
    dist = Distribution(
        dict(
            script_name='setup.py',
            script_args=['build_py'],
            packages=['pkg'],
            package_data={'pkg': ['run-me']},
        )
    )
    os.makedirs('pkg')
    open('pkg/__init__.py', 'wb').close()
    open('pkg/run-me', 'wb').close()
    os.chmod('pkg/run-me', 0o700)

    dist.parse_command_line()
    dist.run_commands()

    assert os.stat('build/lib/pkg/run-me').st_mode & stat.S_IEXEC, (
        "Script is not executable"
    )


EXAMPLE_WITH_MANIFEST = {
    "setup.cfg": DALS(
        """
        [metadata]
        name = mypkg
        version = 42

        [options]
        include_package_data = True
        packages = find:

        [options.packages.find]
        exclude = *.tests*
        """
    ),
    "mypkg": {
        "__init__.py": "",
        "resource_file.txt": "",
        "tests": {
            "__init__.py": "",
            "test_mypkg.py": "",
            "test_file.txt": "",
        },
    },
    "MANIFEST.in": DALS(
        """
        global-include *.py *.txt
        global-exclude *.py[cod]
        prune dist
        prune build
        prune *.egg-info
        """
    ),
}


def test_excluded_subpackages(tmpdir_cwd):
    jaraco.path.build(EXAMPLE_WITH_MANIFEST)
    dist = Distribution({"script_name": "%PEP 517%"})
    dist.parse_config_files()

    build_py = dist.get_command_obj("build_py")

    msg = r"Python recognizes 'mypkg\.tests' as an importable package"
    with pytest.warns(SetuptoolsDeprecationWarning, match=msg):  # noqa: PT031
        # TODO: To fix #3260 we need some transition period to deprecate the
        # existing behavior of `include_package_data`. After the transition, we
        # should remove the warning and fix the behavior.

        if os.getenv("SETUPTOOLS_USE_DISTUTILS") == "stdlib":
            # pytest.warns reset the warning filter temporarily
            # https://github.com/pytest-dev/pytest/issues/4011#issuecomment-423494810
            warnings.filterwarnings(
                "ignore",
                "'encoding' argument not specified",
                module="distutils.text_file",
                # This warning is already fixed in pypa/distutils but not in stdlib
            )

        build_py.finalize_options()
        build_py.run()

    build_dir = Path(dist.get_command_obj("build_py").build_lib)
    assert (build_dir / "mypkg/__init__.py").exists()
    assert (build_dir / "mypkg/resource_file.txt").exists()

    # Setuptools is configured to ignore `mypkg.tests`, therefore the following
    # files/dirs should not be included in the distribution.
    for f in [
        "mypkg/tests/__init__.py",
        "mypkg/tests/test_mypkg.py",
        "mypkg/tests/test_file.txt",
        "mypkg/tests",
    ]:
        with pytest.raises(AssertionError):
            # TODO: Enforce the following assertion once #3260 is fixed
            # (remove context manager and the following xfail).
            assert not (build_dir / f).exists()

    pytest.xfail("#3260")


@pytest.mark.filterwarnings("ignore::setuptools.SetuptoolsDeprecationWarning")
def test_existing_egg_info(tmpdir_cwd, monkeypatch):
    """When provided with the ``existing_egg_info_dir`` attribute, build_py should not
    attempt to run egg_info again.
    """
    # == Pre-condition ==
    # Generate an egg-info dir
    jaraco.path.build(EXAMPLE_WITH_MANIFEST)
    dist = Distribution({"script_name": "%PEP 517%"})
    dist.parse_config_files()
    assert dist.include_package_data

    egg_info = dist.get_command_obj("egg_info")
    dist.run_command("egg_info")
    egg_info_dir = next(Path(egg_info.egg_base).glob("*.egg-info"))
    assert egg_info_dir.is_dir()

    # == Setup ==
    build_py = dist.get_command_obj("build_py")
    build_py.finalize_options()
    egg_info = dist.get_command_obj("egg_info")
    egg_info_run = Mock(side_effect=egg_info.run)
    monkeypatch.setattr(egg_info, "run", egg_info_run)

    # == Remove caches ==
    # egg_info is called when build_py looks for data_files, which gets cached.
    # We need to ensure it is not cached yet, otherwise it may impact on the tests
    build_py.__dict__.pop('data_files', None)
    dist.reinitialize_command(egg_info)

    # == Sanity check ==
    # Ensure that if existing_egg_info is not given, build_py attempts to run egg_info
    build_py.existing_egg_info_dir = None
    build_py.run()
    egg_info_run.assert_called()

    # == Remove caches ==
    egg_info_run.reset_mock()
    build_py.__dict__.pop('data_files', None)
    dist.reinitialize_command(egg_info)

    # == Actual test ==
    # Ensure that if existing_egg_info_dir is given, egg_info doesn't run
    build_py.existing_egg_info_dir = egg_info_dir
    build_py.run()
    egg_info_run.assert_not_called()
    assert build_py.data_files

    # Make sure the list of outputs is actually OK
    outputs = map(lambda x: x.replace(os.sep, "/"), build_py.get_outputs())
    assert outputs
    example = str(Path(build_py.build_lib, "mypkg/__init__.py")).replace(os.sep, "/")
    assert example in outputs


EXAMPLE_ARBITRARY_MAPPING = {
    "pyproject.toml": DALS(
        """
        [project]
        name = "mypkg"
        version = "42"

        [tool.setuptools]
        packages = ["mypkg", "mypkg.sub1", "mypkg.sub2", "mypkg.sub2.nested"]

        [tool.setuptools.package-dir]
        "" = "src"
        "mypkg.sub2" = "src/mypkg/_sub2"
        "mypkg.sub2.nested" = "other"
        """
    ),
    "src": {
        "mypkg": {
            "__init__.py": "",
            "resource_file.txt": "",
            "sub1": {
                "__init__.py": "",
                "mod1.py": "",
            },
            "_sub2": {
                "mod2.py": "",
            },
        },
    },
    "other": {
        "__init__.py": "",
        "mod3.py": "",
    },
    "MANIFEST.in": DALS(
        """
        global-include *.py *.txt
        global-exclude *.py[cod]
        """
    ),
}


def test_get_outputs(tmpdir_cwd):
    jaraco.path.build(EXAMPLE_ARBITRARY_MAPPING)
    dist = Distribution({"script_name": "%test%"})
    dist.parse_config_files()

    build_py = dist.get_command_obj("build_py")
    build_py.editable_mode = True
    build_py.ensure_finalized()
    build_lib = build_py.build_lib.replace(os.sep, "/")
    outputs = {x.replace(os.sep, "/") for x in build_py.get_outputs()}
    assert outputs == {
        f"{build_lib}/mypkg/__init__.py",
        f"{build_lib}/mypkg/resource_file.txt",
        f"{build_lib}/mypkg/sub1/__init__.py",
        f"{build_lib}/mypkg/sub1/mod1.py",
        f"{build_lib}/mypkg/sub2/mod2.py",
        f"{build_lib}/mypkg/sub2/nested/__init__.py",
        f"{build_lib}/mypkg/sub2/nested/mod3.py",
    }
    mapping = {
        k.replace(os.sep, "/"): v.replace(os.sep, "/")
        for k, v in build_py.get_output_mapping().items()
    }
    assert mapping == {
        f"{build_lib}/mypkg/__init__.py": "src/mypkg/__init__.py",
        f"{build_lib}/mypkg/resource_file.txt": "src/mypkg/resource_file.txt",
        f"{build_lib}/mypkg/sub1/__init__.py": "src/mypkg/sub1/__init__.py",
        f"{build_lib}/mypkg/sub1/mod1.py": "src/mypkg/sub1/mod1.py",
        f"{build_lib}/mypkg/sub2/mod2.py": "src/mypkg/_sub2/mod2.py",
        f"{build_lib}/mypkg/sub2/nested/__init__.py": "other/__init__.py",
        f"{build_lib}/mypkg/sub2/nested/mod3.py": "other/mod3.py",
    }


class TestTypeInfoFiles:
    PYPROJECTS = {
        "default_pyproject": DALS(
            """
            [project]
            name = "foo"
            version = "1"
            """
        ),
        "dont_include_package_data": DALS(
            """
            [project]
            name = "foo"
            version = "1"

            [tool.setuptools]
            include-package-data = false
            """
        ),
        "exclude_type_info": DALS(
            """
            [project]
            name = "foo"
            version = "1"

            [tool.setuptools]
            include-package-data = false

            [tool.setuptools.exclude-package-data]
            "*" = ["py.typed", "*.pyi"]
            """
        ),
    }

    EXAMPLES = {
        "simple_namespace": {
            "directory_structure": {
                "foo": {
                    "bar.pyi": "",
                    "py.typed": "",
                    "__init__.py": "",
                }
            },
            "expected_type_files": {"foo/bar.pyi", "foo/py.typed"},
        },
        "nested_inside_namespace": {
            "directory_structure": {
                "foo": {
                    "bar": {
                        "py.typed": "",
                        "mod.pyi": "",
                    }
                }
            },
            "expected_type_files": {"foo/bar/mod.pyi", "foo/bar/py.typed"},
        },
        "namespace_nested_inside_regular": {
            "directory_structure": {
                "foo": {
                    "namespace": {
                        "foo.pyi": "",
                    },
                    "__init__.pyi": "",
                    "py.typed": "",
                }
            },
            "expected_type_files": {
                "foo/namespace/foo.pyi",
                "foo/__init__.pyi",
                "foo/py.typed",
            },
        },
    }

    @pytest.mark.parametrize(
        "pyproject",
        [
            "default_pyproject",
            pytest.param(
                "dont_include_package_data",
                marks=pytest.mark.xfail(reason="pypa/setuptools#4350"),
            ),
        ],
    )
    @pytest.mark.parametrize("example", EXAMPLES.keys())
    def test_type_files_included_by_default(self, tmpdir_cwd, pyproject, example):
        structure = {
            **self.EXAMPLES[example]["directory_structure"],
            "pyproject.toml": self.PYPROJECTS[pyproject],
        }
        expected_type_files = self.EXAMPLES[example]["expected_type_files"]
        jaraco.path.build(structure)

        build_py = get_finalized_build_py()
        outputs = get_outputs(build_py)
        assert expected_type_files <= outputs

    @pytest.mark.parametrize("pyproject", ["exclude_type_info"])
    @pytest.mark.parametrize("example", EXAMPLES.keys())
    def test_type_files_can_be_excluded(self, tmpdir_cwd, pyproject, example):
        structure = {
            **self.EXAMPLES[example]["directory_structure"],
            "pyproject.toml": self.PYPROJECTS[pyproject],
        }
        expected_type_files = self.EXAMPLES[example]["expected_type_files"]
        jaraco.path.build(structure)

        build_py = get_finalized_build_py()
        outputs = get_outputs(build_py)
        assert expected_type_files.isdisjoint(outputs)

    def test_stub_only_package(self, tmpdir_cwd):
        structure = {
            "pyproject.toml": DALS(
                """
                [project]
                name = "foo-stubs"
                version = "1"
                """
            ),
            "foo-stubs": {"__init__.pyi": "", "bar.pyi": ""},
        }
        expected_type_files = {"foo-stubs/__init__.pyi", "foo-stubs/bar.pyi"}
        jaraco.path.build(structure)

        build_py = get_finalized_build_py()
        outputs = get_outputs(build_py)
        assert expected_type_files <= outputs


def get_finalized_build_py(script_name="%build_py-test%"):
    dist = Distribution({"script_name": script_name})
    dist.parse_config_files()
    build_py = dist.get_command_obj("build_py")
    build_py.finalize_options()
    return build_py


def get_outputs(build_py):
    build_dir = Path(build_py.build_lib)
    return {
        os.path.relpath(x, build_dir).replace(os.sep, "/")
        for x in build_py.get_outputs()
    }
