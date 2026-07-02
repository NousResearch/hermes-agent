from __future__ import annotations

import os
import sys
from importlib.util import cache_from_source as _compiled_file_name

import pytest
from jaraco import path

from setuptools.command.build_ext import build_ext, get_abi3_suffix
from setuptools.dist import Distribution
from setuptools.errors import CompileError
from setuptools.extension import Extension

from . import environment
from .textwrap import DALS

import distutils.command.build_ext as orig
from distutils.sysconfig import get_config_var

IS_PYPY = '__pypy__' in sys.builtin_module_names


class TestBuildExt:
    def test_get_ext_filename(self):
        """
        Setuptools needs to give back the same
        result as distutils, even if the fullname
        is not in ext_map.
        """
        dist = Distribution()
        cmd = build_ext(dist)
        cmd.ext_map['foo/bar'] = ''
        res = cmd.get_ext_filename('foo')
        wanted = orig.build_ext.get_ext_filename(cmd, 'foo')
        assert res == wanted

    def test_abi3_filename(self):
        """
        Filename needs to be loadable by several versions
        of Python 3 if 'is_abi3' is truthy on Extension()
        """
        print(get_abi3_suffix())

        extension = Extension('spam.eggs', ['eggs.c'], py_limited_api=True)
        dist = Distribution(dict(ext_modules=[extension]))
        cmd = build_ext(dist)
        cmd.finalize_options()
        assert 'spam.eggs' in cmd.ext_map
        res = cmd.get_ext_filename('spam.eggs')

        if not get_abi3_suffix():
            assert res.endswith(get_config_var('EXT_SUFFIX'))
        elif sys.platform == 'win32':
            assert res.endswith('eggs.pyd')
        else:
            assert 'abi3' in res

    def test_ext_suffix_override(self):
        """
        SETUPTOOLS_EXT_SUFFIX variable always overrides
        default extension options.
        """
        dist = Distribution()
        cmd = build_ext(dist)
        cmd.ext_map['for_abi3'] = ext = Extension(
            'for_abi3',
            ['s.c'],
            # Override shouldn't affect abi3 modules
            py_limited_api=True,
        )
        # Mock value needed to pass tests
        ext._links_to_dynamic = False

        if not IS_PYPY:
            expect = cmd.get_ext_filename('for_abi3')
        else:
            # PyPy builds do not use ABI3 tag, so they will
            # also get the overridden suffix.
            expect = 'for_abi3.test-suffix'

        try:
            os.environ['SETUPTOOLS_EXT_SUFFIX'] = '.test-suffix'
            res = cmd.get_ext_filename('normal')
            assert 'normal.test-suffix' == res
            res = cmd.get_ext_filename('for_abi3')
            assert expect == res
        finally:
            del os.environ['SETUPTOOLS_EXT_SUFFIX']

    def dist_with_example(self):
        files = {
            "src": {"mypkg": {"subpkg": {"ext2.c": ""}}},
            "c-extensions": {"ext1": {"main.c": ""}},
        }

        ext1 = Extension("mypkg.ext1", ["c-extensions/ext1/main.c"])
        ext2 = Extension("mypkg.subpkg.ext2", ["src/mypkg/subpkg/ext2.c"])
        ext3 = Extension("ext3", ["c-extension/ext3.c"])

        path.build(files)
        return Distribution({
            "script_name": "%test%",
            "ext_modules": [ext1, ext2, ext3],
            "package_dir": {"": "src"},
        })

    def test_get_outputs(self, tmpdir_cwd, monkeypatch):
        monkeypatch.setenv('SETUPTOOLS_EXT_SUFFIX', '.mp3')  # make test OS-independent
        monkeypatch.setattr('setuptools.command.build_ext.use_stubs', False)
        dist = self.dist_with_example()

        # Regular build: get_outputs not empty, but get_output_mappings is empty
        build_ext = dist.get_command_obj("build_ext")
        build_ext.editable_mode = False
        build_ext.ensure_finalized()
        build_lib = build_ext.build_lib.replace(os.sep, "/")
        outputs = [x.replace(os.sep, "/") for x in build_ext.get_outputs()]
        assert outputs == [
            f"{build_lib}/ext3.mp3",
            f"{build_lib}/mypkg/ext1.mp3",
            f"{build_lib}/mypkg/subpkg/ext2.mp3",
        ]
        assert build_ext.get_output_mapping() == {}

        # Editable build: get_output_mappings should contain everything in get_outputs
        dist.reinitialize_command("build_ext")
        build_ext.editable_mode = True
        build_ext.ensure_finalized()
        mapping = {
            k.replace(os.sep, "/"): v.replace(os.sep, "/")
            for k, v in build_ext.get_output_mapping().items()
        }
        assert mapping == {
            f"{build_lib}/ext3.mp3": "src/ext3.mp3",
            f"{build_lib}/mypkg/ext1.mp3": "src/mypkg/ext1.mp3",
            f"{build_lib}/mypkg/subpkg/ext2.mp3": "src/mypkg/subpkg/ext2.mp3",
        }

    def test_get_output_mapping_with_stub(self, tmpdir_cwd, monkeypatch):
        monkeypatch.setenv('SETUPTOOLS_EXT_SUFFIX', '.mp3')  # make test OS-independent
        monkeypatch.setattr('setuptools.command.build_ext.use_stubs', True)
        dist = self.dist_with_example()

        # Editable build should create compiled stubs (.pyc files only, no .py)
        build_ext = dist.get_command_obj("build_ext")
        build_ext.editable_mode = True
        build_ext.ensure_finalized()
        for ext in build_ext.extensions:
            monkeypatch.setattr(ext, "_needs_stub", True)

        build_lib = build_ext.build_lib.replace(os.sep, "/")
        mapping = {
            k.replace(os.sep, "/"): v.replace(os.sep, "/")
            for k, v in build_ext.get_output_mapping().items()
        }

        def C(file):
            """Make it possible to do comparisons and tests in a OS-independent way"""
            return _compiled_file_name(file).replace(os.sep, "/")

        assert mapping == {
            C(f"{build_lib}/ext3.py"): C("src/ext3.py"),
            f"{build_lib}/ext3.mp3": "src/ext3.mp3",
            C(f"{build_lib}/mypkg/ext1.py"): C("src/mypkg/ext1.py"),
            f"{build_lib}/mypkg/ext1.mp3": "src/mypkg/ext1.mp3",
            C(f"{build_lib}/mypkg/subpkg/ext2.py"): C("src/mypkg/subpkg/ext2.py"),
            f"{build_lib}/mypkg/subpkg/ext2.mp3": "src/mypkg/subpkg/ext2.mp3",
        }

        # Ensure only the compiled stubs are present not the raw .py stub
        assert f"{build_lib}/mypkg/ext1.py" not in mapping
        assert f"{build_lib}/mypkg/subpkg/ext2.py" not in mapping

        # Visualize what the cached stub files look like
        example_stub = C(f"{build_lib}/mypkg/ext1.py")
        assert example_stub in mapping
        assert example_stub.startswith(f"{build_lib}/mypkg/__pycache__/ext1")
        assert example_stub.endswith(".pyc")


class TestBuildExtInplace:
    def get_build_ext_cmd(self, optional: bool, **opts) -> build_ext:
        files: dict[str, str | dict[str, dict[str, str]]] = {
            "eggs.c": "#include missingheader.h\n",
            ".build": {"lib": {}, "tmp": {}},
        }
        path.build(files)
        extension = Extension('spam.eggs', ['eggs.c'], optional=optional)
        dist = Distribution(dict(ext_modules=[extension]))
        dist.script_name = 'setup.py'
        cmd = build_ext(dist)
        vars(cmd).update(build_lib=".build/lib", build_temp=".build/tmp", **opts)
        cmd.ensure_finalized()
        return cmd

    def get_log_messages(self, caplog, capsys):
        """
        Historically, distutils "logged" by printing to sys.std*.
        Later versions adopted the logging framework. Grab
        messages regardless of how they were captured.
        """
        std = capsys.readouterr()
        return std.out.splitlines() + std.err.splitlines() + caplog.messages

    def test_optional(self, tmpdir_cwd, caplog, capsys):
        """
        If optional extensions fail to build, setuptools should show the error
        in the logs but not fail to build
        """
        cmd = self.get_build_ext_cmd(optional=True, inplace=True)
        cmd.run()
        assert any(
            'build_ext: building extension "spam.eggs" failed'
            for msg in self.get_log_messages(caplog, capsys)
        )
        # No compile error exception should be raised

    def test_non_optional(self, tmpdir_cwd):
        # Non-optional extensions should raise an exception
        cmd = self.get_build_ext_cmd(optional=False, inplace=True)
        with pytest.raises(CompileError):
            cmd.run()


def test_build_ext_config_handling(tmpdir_cwd):
    files = {
        'setup.py': DALS(
            """
            from setuptools import Extension, setup
            setup(
                name='foo',
                version='0.0.0',
                ext_modules=[Extension('foo', ['foo.c'])],
            )
            """
        ),
        'foo.c': DALS(
            """
            #include "Python.h"

            #if PY_MAJOR_VERSION >= 3

            static struct PyModuleDef moduledef = {
                    PyModuleDef_HEAD_INIT,
                    "foo",
                    NULL,
                    0,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL
            };

            #define INITERROR return NULL

            PyMODINIT_FUNC PyInit_foo(void)

            #else

            #define INITERROR return

            void initfoo(void)

            #endif
            {
            #if PY_MAJOR_VERSION >= 3
                PyObject *module = PyModule_Create(&moduledef);
            #else
                PyObject *module = Py_InitModule("extension", NULL);
            #endif
                if (module == NULL)
                    INITERROR;
            #if PY_MAJOR_VERSION >= 3
                return module;
            #endif
            }
            """
        ),
        'setup.cfg': DALS(
            """
            [build]
            build_base = foo_build
            """
        ),
    }
    path.build(files)
    code, (stdout, stderr) = environment.run_setup_py(
        cmd=['build'],
        data_stream=(0, 2),
    )
    assert code == 0, f'\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}'
