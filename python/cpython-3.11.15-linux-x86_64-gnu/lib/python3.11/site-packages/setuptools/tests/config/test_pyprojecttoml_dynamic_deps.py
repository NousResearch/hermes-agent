from inspect import cleandoc

import pytest
from jaraco import path

from setuptools.config.pyprojecttoml import apply_configuration
from setuptools.dist import Distribution
from setuptools.warnings import SetuptoolsWarning


def test_dynamic_dependencies(tmp_path):
    files = {
        "requirements.txt": "six\n  # comment\n",
        "pyproject.toml": cleandoc(
            """
            [project]
            name = "myproj"
            version = "1.0"
            dynamic = ["dependencies"]

            [build-system]
            requires = ["setuptools", "wheel"]
            build-backend = "setuptools.build_meta"

            [tool.setuptools.dynamic.dependencies]
            file = ["requirements.txt"]
            """
        ),
    }
    path.build(files, prefix=tmp_path)
    dist = Distribution()
    dist = apply_configuration(dist, tmp_path / "pyproject.toml")
    assert dist.install_requires == ["six"]


def test_dynamic_optional_dependencies(tmp_path):
    files = {
        "requirements-docs.txt": "sphinx\n  # comment\n",
        "pyproject.toml": cleandoc(
            """
            [project]
            name = "myproj"
            version = "1.0"
            dynamic = ["optional-dependencies"]

            [tool.setuptools.dynamic.optional-dependencies.docs]
            file = ["requirements-docs.txt"]

            [build-system]
            requires = ["setuptools", "wheel"]
            build-backend = "setuptools.build_meta"
            """
        ),
    }
    path.build(files, prefix=tmp_path)
    dist = Distribution()
    dist = apply_configuration(dist, tmp_path / "pyproject.toml")
    assert dist.extras_require == {"docs": ["sphinx"]}


def test_mixed_dynamic_optional_dependencies(tmp_path):
    """
    Test that if PEP 621 was loosened to allow mixing of dynamic and static
    configurations in the case of fields containing sub-fields (groups),
    things would work out.
    """
    files = {
        "requirements-images.txt": "pillow~=42.0\n  # comment\n",
        "pyproject.toml": cleandoc(
            """
            [project]
            name = "myproj"
            version = "1.0"
            dynamic = ["optional-dependencies"]

            [project.optional-dependencies]
            docs = ["sphinx"]

            [tool.setuptools.dynamic.optional-dependencies.images]
            file = ["requirements-images.txt"]
            """
        ),
    }

    path.build(files, prefix=tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    with pytest.raises(ValueError, match="project.optional-dependencies"):
        apply_configuration(Distribution(), pyproject)


def test_mixed_extras_require_optional_dependencies(tmp_path):
    files = {
        "pyproject.toml": cleandoc(
            """
            [project]
            name = "myproj"
            version = "1.0"
            optional-dependencies.docs = ["sphinx"]
            """
        ),
    }

    path.build(files, prefix=tmp_path)
    pyproject = tmp_path / "pyproject.toml"

    dist = Distribution({"extras_require": {"hello": ["world"]}})

    with pytest.warns(SetuptoolsWarning, match=".extras_require. overwritten"):
        dist = apply_configuration(dist, pyproject)

    assert dist.extras_require == {"docs": ["sphinx"]}
