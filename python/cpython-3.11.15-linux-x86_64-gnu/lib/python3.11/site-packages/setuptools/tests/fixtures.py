import contextlib
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import jaraco.path
import path
import pytest

from setuptools._normalization import safer_name

from . import contexts, environment
from .textwrap import DALS


@pytest.fixture
def user_override(monkeypatch):
    """
    Override site.USER_BASE and site.USER_SITE with temporary directories in
    a context.
    """
    with contexts.tempdir() as user_base:
        monkeypatch.setattr('site.USER_BASE', user_base)
        with contexts.tempdir() as user_site:
            monkeypatch.setattr('site.USER_SITE', user_site)
            with contexts.save_user_site_setting():
                yield


@pytest.fixture
def tmpdir_cwd(tmpdir):
    with tmpdir.as_cwd() as orig:
        yield orig


@pytest.fixture(autouse=True, scope="session")
def workaround_xdist_376(request):
    """
    Workaround pytest-dev/pytest-xdist#376

    ``pytest-xdist`` tends to inject '' into ``sys.path``,
    which may break certain isolation expectations.
    Remove the entry so the import
    machinery behaves the same irrespective of xdist.
    """
    if not request.config.pluginmanager.has_plugin('xdist'):
        return

    with contextlib.suppress(ValueError):
        sys.path.remove('')


@pytest.fixture
def sample_project(tmp_path):
    """
    Clone the 'sampleproject' and return a path to it.
    """
    cmd = ['git', 'clone', 'https://github.com/pypa/sampleproject']
    try:
        subprocess.check_call(cmd, cwd=str(tmp_path))
    except Exception:
        pytest.skip("Unable to clone sampleproject")
    return tmp_path / 'sampleproject'


@pytest.fixture
def sample_project_cwd(sample_project):
    with path.Path(sample_project):
        yield


# sdist and wheel artifacts should be stable across a round of tests
# so we can build them once per session and use the files as "readonly"

# In the case of setuptools, building the wheel without sdist may cause
# it to contain the `build` directory, and therefore create situations with
# `setuptools/build/lib/build/lib/...`. To avoid that, build both artifacts at once.


def _build_distributions(tmp_path_factory, request):
    with contexts.session_locked_tmp_dir(
        request, tmp_path_factory, "dist_build"
    ) as tmp:  # pragma: no cover
        sdist = next(tmp.glob("*.tar.gz"), None)
        wheel = next(tmp.glob("*.whl"), None)
        if sdist and wheel:
            return (sdist, wheel)

        # Sanity check: should not create recursive setuptools/build/lib/build/lib/...
        assert not Path(request.config.rootdir, "build/lib/build").exists()

        subprocess.check_output([
            sys.executable,
            "-m",
            "build",
            "--outdir",
            str(tmp),
            str(request.config.rootdir),
        ])

        # Sanity check: should not create recursive setuptools/build/lib/build/lib/...
        assert not Path(request.config.rootdir, "build/lib/build").exists()

        return next(tmp.glob("*.tar.gz")), next(tmp.glob("*.whl"))


@pytest.fixture(scope="session")
def setuptools_sdist(tmp_path_factory, request):
    prebuilt = os.getenv("PRE_BUILT_SETUPTOOLS_SDIST")
    if prebuilt and os.path.exists(prebuilt):  # pragma: no cover
        return Path(prebuilt).resolve()

    sdist, _ = _build_distributions(tmp_path_factory, request)
    return sdist


@pytest.fixture(scope="session")
def setuptools_wheel(tmp_path_factory, request):
    prebuilt = os.getenv("PRE_BUILT_SETUPTOOLS_WHEEL")
    if prebuilt and os.path.exists(prebuilt):  # pragma: no cover
        return Path(prebuilt).resolve()

    _, wheel = _build_distributions(tmp_path_factory, request)
    return wheel


@pytest.fixture
def venv(tmp_path, setuptools_wheel):
    """Virtual env with the version of setuptools under test installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'venv')
    env.create_opts = ['--no-setuptools', '--wheel=bundle']
    # TODO: Use `--no-wheel` when setuptools implements its own bdist_wheel
    env.req = str(setuptools_wheel)
    # In some environments (eg. downstream distro packaging),
    # where tox isn't used to run tests and PYTHONPATH is set to point to
    # a specific setuptools codebase, PYTHONPATH will leak into the spawned
    # processes.
    # env.create() should install the just created setuptools
    # wheel, but it doesn't if it finds another existing matching setuptools
    # installation present on PYTHONPATH:
    # `setuptools is already installed with the same version as the provided
    # wheel. Use --force-reinstall to force an installation of the wheel.`
    # This prevents leaking PYTHONPATH to the created environment.
    with contexts.environment(PYTHONPATH=None):
        return env.create()


@pytest.fixture
def venv_without_setuptools(tmp_path):
    """Virtual env without any version of setuptools installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'venv_without_setuptools')
    env.create_opts = ['--no-setuptools', '--no-wheel']
    env.ensure_env()
    return env


@pytest.fixture
def bare_venv(tmp_path):
    """Virtual env without any common packages installed"""
    env = environment.VirtualEnv()
    env.root = path.Path(tmp_path / 'bare_venv')
    env.create_opts = ['--no-setuptools', '--no-pip', '--no-wheel', '--no-seed']
    env.ensure_env()
    return env


def make_sdist(dist_path, files):
    """
    Create a simple sdist tarball at dist_path, containing the files
    listed in ``files`` as ``(filename, content)`` tuples.
    """

    # Distributions with only one file don't play well with pip.
    assert len(files) > 1
    with tarfile.open(dist_path, 'w:gz') as dist:
        for filename, content in files:
            file_bytes = io.BytesIO(content.encode('utf-8'))
            file_info = tarfile.TarInfo(name=filename)
            file_info.size = len(file_bytes.getvalue())
            file_info.mtime = int(time.time())
            dist.addfile(file_info, fileobj=file_bytes)


def make_trivial_sdist(dist_path, distname, version, setuptools_wheel=None):
    """
    Create a simple sdist tarball at dist_path, containing just a simple
    setup.py.

    If ``setuptools_wheel`` is passed, a ``pyproject.toml`` file will also
    be generated and the passed value will be used as location for
    setuptools (as build dependency).
    """
    files = [
        (
            'setup.py',
            DALS(
                f"""\
                 import setuptools
                 setuptools.setup(
                     name={distname!r},
                     version={version!r}
                 )
                 """
            ),
        ),
        ('setup.cfg', ''),
    ]

    if setuptools_wheel:
        files.append((
            "pyproject.toml",
            DALS(
                f"""\
                [build-system]
                requires = ["setuptools @ {setuptools_wheel.as_uri()}"]
                build-backend = "setuptools.build_meta"
                """
            ),
        ))

    make_sdist(dist_path, files)


def make_nspkg_sdist(dist_path, distname, version):
    """
    Make an sdist tarball with distname and version which also contains one
    package with the same name as distname.  The top-level package is
    designated a namespace package).
    """
    # Assert that the distname contains at least one period
    assert '.' in distname

    parts = distname.split('.')
    nspackage = parts[0]

    packages = ['.'.join(parts[:idx]) for idx in range(1, len(parts) + 1)]

    setup_py = DALS(
        f"""\
        import setuptools
        setuptools.setup(
            name={distname!r},
            version={version!r},
            packages={packages!r},
            namespace_packages=[{nspackage!r}]
        )
    """
    )

    init = "__import__('pkg_resources').declare_namespace(__name__)"

    files = [('setup.py', setup_py), (os.path.join(nspackage, '__init__.py'), init)]
    for package in packages[1:]:
        filename = os.path.join(*(package.split('.') + ['__init__.py']))
        files.append((filename, ''))

    make_sdist(dist_path, files)


def make_python_requires_sdist(dist_path, distname, version, python_requires):
    make_sdist(
        dist_path,
        [
            (
                'setup.py',
                DALS(
                    """\
                import setuptools
                setuptools.setup(
                  name={name!r},
                  version={version!r},
                  python_requires={python_requires!r},
                )
                """
                ).format(
                    name=distname, version=version, python_requires=python_requires
                ),
            ),
            ('setup.cfg', ''),
        ],
    )


def create_setup_requires_package(
    path,
    distname='foobar',
    version='0.1',
    make_package=make_trivial_sdist,
    setup_py_template=None,
    setup_attrs=None,
    use_setup_cfg=(),
):
    """Creates a source tree under path for a trivial test package that has a
    single requirement in setup_requires--a tarball for that requirement is
    also created and added to the dependency_links argument.

    ``distname`` and ``version`` refer to the name/version of the package that
    the test package requires via ``setup_requires``.  The name of the test
    package itself is just 'test_pkg'.
    """

    normalized_distname = safer_name(distname)
    test_setup_attrs = {
        'name': 'test_pkg',
        'version': '0.0',
        'setup_requires': [f'{normalized_distname}=={version}'],
        'dependency_links': [os.path.abspath(path)],
    }
    if setup_attrs:
        test_setup_attrs.update(setup_attrs)

    test_pkg = os.path.join(path, 'test_pkg')
    os.mkdir(test_pkg)

    # setup.cfg
    if use_setup_cfg:
        options = []
        metadata = []
        for name in use_setup_cfg:
            value = test_setup_attrs.pop(name)
            if name in 'name version'.split():
                section = metadata
            else:
                section = options
            if isinstance(value, (tuple, list)):
                value = ';'.join(value)
            section.append(f'{name}: {value}')
        test_setup_cfg_contents = DALS(
            """
            [metadata]
            {metadata}
            [options]
            {options}
            """
        ).format(
            options='\n'.join(options),
            metadata='\n'.join(metadata),
        )
    else:
        test_setup_cfg_contents = ''
    with open(os.path.join(test_pkg, 'setup.cfg'), 'w', encoding="utf-8") as f:
        f.write(test_setup_cfg_contents)

    # setup.py
    if setup_py_template is None:
        setup_py_template = DALS(
            """\
            import setuptools
            setuptools.setup(**%r)
        """
        )
    with open(os.path.join(test_pkg, 'setup.py'), 'w', encoding="utf-8") as f:
        f.write(setup_py_template % test_setup_attrs)

    foobar_path = os.path.join(path, f'{normalized_distname}-{version}.tar.gz')
    make_package(foobar_path, distname, version)

    return test_pkg


@pytest.fixture
def pbr_package(tmp_path, monkeypatch, venv):
    files = {
        "pyproject.toml": DALS(
            """
            [build-system]
            requires = ["setuptools"]
            build-backend = "setuptools.build_meta"
            """
        ),
        "setup.py": DALS(
            """
            __import__('setuptools').setup(
                pbr=True,
                setup_requires=["pbr"],
            )
            """
        ),
        "setup.cfg": DALS(
            """
            [metadata]
            name = mypkg

            [files]
            packages =
                mypkg
            """
        ),
        "mypkg": {
            "__init__.py": "",
            "hello.py": "print('Hello world!')",
        },
        "other": {"test.txt": "Another file in here."},
    }
    venv.run(["python", "-m", "pip", "install", "pbr"])
    prefix = tmp_path / 'mypkg'
    prefix.mkdir()
    jaraco.path.build(files, prefix=prefix)
    monkeypatch.setenv('PBR_VERSION', "0.42")
    return prefix
