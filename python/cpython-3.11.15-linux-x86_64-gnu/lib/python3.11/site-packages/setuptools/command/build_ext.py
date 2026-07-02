from __future__ import annotations

import itertools
import operator
import os
import sys
import textwrap
from collections.abc import Iterator
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import cache_from_source as _compiled_file_name
from pathlib import Path
from typing import TYPE_CHECKING

from setuptools.dist import Distribution
from setuptools.errors import BaseError
from setuptools.extension import Extension, Library

from distutils import log
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var

if TYPE_CHECKING:
    # Cython not installed on CI tests, causing _build_ext to be `Any`
    from distutils.command.build_ext import build_ext as _build_ext
else:
    try:
        # Attempt to use Cython for building extensions, if available
        from Cython.Distutils.build_ext import build_ext as _build_ext

        # Additionally, assert that the compiler module will load
        # also. Ref #1229.
        __import__('Cython.Compiler.Main')
    except ImportError:
        from distutils.command.build_ext import build_ext as _build_ext

# make sure _config_vars is initialized
get_config_var("LDSHARED")
# Not publicly exposed in typeshed distutils stubs, but this is done on purpose
# See https://github.com/pypa/setuptools/pull/4228#issuecomment-1959856400
from distutils.sysconfig import _config_vars as _CONFIG_VARS  # noqa: E402


def _customize_compiler_for_shlib(compiler):
    if sys.platform == "darwin":
        # building .dylib requires additional compiler flags on OSX; here we
        # temporarily substitute the pyconfig.h variables so that distutils'
        # 'customize_compiler' uses them before we build the shared libraries.
        tmp = _CONFIG_VARS.copy()
        try:
            # XXX Help!  I don't have any idea whether these are right...
            _CONFIG_VARS['LDSHARED'] = (
                "gcc -Wl,-x -dynamiclib -undefined dynamic_lookup"
            )
            _CONFIG_VARS['CCSHARED'] = " -dynamiclib"
            _CONFIG_VARS['SO'] = ".dylib"
            customize_compiler(compiler)
        finally:
            _CONFIG_VARS.clear()
            _CONFIG_VARS.update(tmp)
    else:
        customize_compiler(compiler)


have_rtld = False
use_stubs = False
libtype = 'shared'

if sys.platform == "darwin":
    use_stubs = True
elif os.name != 'nt':
    try:
        import dl  # type: ignore[import-not-found] # https://github.com/python/mypy/issues/13002

        use_stubs = have_rtld = hasattr(dl, 'RTLD_NOW')
    except ImportError:
        pass


def get_abi3_suffix():
    """Return the file extension for an abi3-compliant Extension()"""
    for suffix in EXTENSION_SUFFIXES:
        if '.abi3' in suffix:  # Unix
            return suffix
        elif suffix == '.pyd':  # Windows
            return suffix
    return None


class build_ext(_build_ext):
    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    editable_mode = False
    inplace = False

    def run(self) -> None:
        """Build extensions in build directory, then copy if --inplace"""
        old_inplace, self.inplace = self.inplace, False
        _build_ext.run(self)
        self.inplace = old_inplace
        if old_inplace:
            self.copy_extensions_to_source()

    def _get_inplace_equivalent(self, build_py, ext: Extension) -> tuple[str, str]:
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)
        modpath = fullname.split('.')
        package = '.'.join(modpath[:-1])
        package_dir = build_py.get_package_dir(package)
        inplace_file = os.path.join(package_dir, os.path.basename(filename))
        regular_file = os.path.join(self.build_lib, filename)
        return (inplace_file, regular_file)

    def copy_extensions_to_source(self) -> None:
        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if os.path.exists(regular_file) or not ext.optional:
                self.copy_file(regular_file, inplace_file, level=self.verbose)

            if ext._needs_stub:
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)
                # Always compile stub and remove the original (leave the cache behind)
                # (this behaviour was observed in previous iterations of the code)

    def _get_equivalent_stub(self, ext: Extension, output_file: str) -> str:
        dir_ = os.path.dirname(output_file)
        _, _, name = ext.name.rpartition(".")
        return f"{os.path.join(dir_, name)}.py"

    def _get_output_mapping(self) -> Iterator[tuple[str, str]]:
        if not self.inplace:
            return

        build_py = self.get_finalized_command('build_py')
        opt = self.get_finalized_command('install_lib').optimize or ""

        for ext in self.extensions:
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)
            yield (regular_file, inplace_file)

            if ext._needs_stub:
                # This version of `build_ext` always builds artifacts in another dir,
                # when "inplace=True" is given it just copies them back.
                # This is done in the `copy_extensions_to_source` function, which
                # always compile stub files via `_compile_and_remove_stub`.
                # At the end of the process, a `.pyc` stub file is created without the
                # corresponding `.py`.

                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                regular_stub = self._get_equivalent_stub(ext, regular_file)
                inplace_cache = _compiled_file_name(inplace_stub, optimization=opt)
                output_cache = _compiled_file_name(regular_stub, optimization=opt)
                yield (output_cache, inplace_cache)

    def get_ext_filename(self, fullname: str) -> str:
        so_ext = os.getenv('SETUPTOOLS_EXT_SUFFIX')
        if so_ext:
            filename = os.path.join(*fullname.split('.')) + so_ext
        else:
            filename = _build_ext.get_ext_filename(self, fullname)
            ext_suffix = get_config_var('EXT_SUFFIX')
            if not isinstance(ext_suffix, str):
                raise OSError(
                    "Configuration variable EXT_SUFFIX not found for this platform "
                    "and environment variable SETUPTOOLS_EXT_SUFFIX is missing"
                )
            so_ext = ext_suffix

        if fullname in self.ext_map:
            ext = self.ext_map[fullname]
            abi3_suffix = get_abi3_suffix()
            if ext.py_limited_api and abi3_suffix:  # Use abi3
                filename = filename[: -len(so_ext)] + abi3_suffix
            if isinstance(ext, Library):
                fn, ext = os.path.splitext(filename)
                return self.shlib_compiler.library_filename(fn, libtype)
            elif use_stubs and ext._links_to_dynamic:
                d, fn = os.path.split(filename)
                return os.path.join(d, 'dl-' + fn)
        return filename

    def initialize_options(self):
        _build_ext.initialize_options(self)
        self.shlib_compiler = None
        self.shlibs = []
        self.ext_map = {}
        self.editable_mode = False

    def finalize_options(self) -> None:
        _build_ext.finalize_options(self)
        self.extensions = self.extensions or []
        self.check_extensions_list(self.extensions)
        self.shlibs = [ext for ext in self.extensions if isinstance(ext, Library)]
        if self.shlibs:
            self.setup_shlib_compiler()
        for ext in self.extensions:
            ext._full_name = self.get_ext_fullname(ext.name)
        for ext in self.extensions:
            fullname = ext._full_name
            self.ext_map[fullname] = ext

            # distutils 3.1 will also ask for module names
            # XXX what to do with conflicts?
            self.ext_map[fullname.split('.')[-1]] = ext

            ltd = self.shlibs and self.links_to_dynamic(ext) or False
            ns = ltd and use_stubs and not isinstance(ext, Library)
            ext._links_to_dynamic = ltd
            ext._needs_stub = ns
            filename = ext._file_name = self.get_ext_filename(fullname)
            libdir = os.path.dirname(os.path.join(self.build_lib, filename))
            if ltd and libdir not in ext.library_dirs:
                ext.library_dirs.append(libdir)
            if ltd and use_stubs and os.curdir not in ext.runtime_library_dirs:
                ext.runtime_library_dirs.append(os.curdir)

        if self.editable_mode:
            self.inplace = True

    def setup_shlib_compiler(self) -> None:
        compiler = self.shlib_compiler = new_compiler(
            compiler=self.compiler, force=self.force
        )
        _customize_compiler_for_shlib(compiler)

        if self.include_dirs is not None:
            compiler.set_include_dirs(self.include_dirs)
        if self.define is not None:
            # 'define' option is a list of (name,value) tuples
            for name, value in self.define:
                compiler.define_macro(name, value)
        if self.undef is not None:
            for macro in self.undef:
                compiler.undefine_macro(macro)
        if self.libraries is not None:
            compiler.set_libraries(self.libraries)
        if self.library_dirs is not None:
            compiler.set_library_dirs(self.library_dirs)
        if self.rpath is not None:
            compiler.set_runtime_library_dirs(self.rpath)
        if self.link_objects is not None:
            compiler.set_link_objects(self.link_objects)

        # hack so distutils' build_extension() builds a library instead
        compiler.link_shared_object = link_shared_object.__get__(compiler)  # type: ignore[method-assign]

    def get_export_symbols(self, ext):
        if isinstance(ext, Library):
            return ext.export_symbols
        return _build_ext.get_export_symbols(self, ext)

    def build_extension(self, ext) -> None:
        ext._convert_pyx_sources_to_lang()
        _compiler = self.compiler
        try:
            if isinstance(ext, Library):
                self.compiler = self.shlib_compiler
            _build_ext.build_extension(self, ext)
            if ext._needs_stub:
                build_lib = self.get_finalized_command('build_py').build_lib
                self.write_stub(build_lib, ext)
        finally:
            self.compiler = _compiler

    def links_to_dynamic(self, ext):
        """Return true if 'ext' links to a dynamic lib in the same package"""
        # XXX this should check to ensure the lib is actually being built
        # XXX as dynamic, and not just using a locally-found version or a
        # XXX static-compiled version
        libnames = dict.fromkeys([lib._full_name for lib in self.shlibs])
        pkg = '.'.join(ext._full_name.split('.')[:-1] + [''])
        return any(pkg + libname in libnames for libname in ext.libraries)

    def get_source_files(self) -> list[str]:
        return [*_build_ext.get_source_files(self), *self._get_internal_depends()]

    def _get_internal_depends(self) -> Iterator[str]:
        """Yield ``ext.depends`` that are contained by the project directory"""
        project_root = Path(self.distribution.src_root or os.curdir).resolve()
        depends = (dep for ext in self.extensions for dep in ext.depends)

        def skip(orig_path: str, reason: str) -> None:
            log.info(
                "dependency %s won't be automatically "
                "included in the manifest: the path %s",
                orig_path,
                reason,
            )

        for dep in depends:
            path = Path(dep)

            if path.is_absolute():
                skip(dep, "must be relative")
                continue

            if ".." in path.parts:
                skip(dep, "can't have `..` segments")
                continue

            try:
                resolved = (project_root / path).resolve(strict=True)
            except OSError:
                skip(dep, "doesn't exist")
                continue

            try:
                resolved.relative_to(project_root)
            except ValueError:
                skip(dep, "must be inside the project root")
                continue

            yield path.as_posix()

    def get_outputs(self) -> list[str]:
        if self.inplace:
            return list(self.get_output_mapping().keys())
        return sorted(_build_ext.get_outputs(self) + self.__get_stubs_outputs())

    def get_output_mapping(self) -> dict[str, str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        mapping = self._get_output_mapping()
        return dict(sorted(mapping, key=operator.itemgetter(0)))

    def __get_stubs_outputs(self):
        # assemble the base name for each extension that needs a stub
        ns_ext_bases = (
            os.path.join(self.build_lib, *ext._full_name.split('.'))
            for ext in self.extensions
            if ext._needs_stub
        )
        # pair each base with the extension
        pairs = itertools.product(ns_ext_bases, self.__get_output_extensions())
        return list(base + fnext for base, fnext in pairs)

    def __get_output_extensions(self):
        yield '.py'
        yield '.pyc'
        if self.get_finalized_command('build_py').optimize:
            yield '.pyo'

    def write_stub(self, output_dir, ext, compile=False) -> None:
        stub_file = os.path.join(output_dir, *ext._full_name.split('.')) + '.py'
        self._write_stub_file(stub_file, ext, compile)

    def _write_stub_file(self, stub_file: str, ext: Extension, compile=False):
        log.info("writing stub loader for %s to %s", ext._full_name, stub_file)
        if compile and os.path.exists(stub_file):
            raise BaseError(stub_file + " already exists! Please delete.")
        with open(stub_file, 'w', encoding="utf-8") as f:
            content = (
                textwrap
                .dedent(f"""
                def __bootstrap__():
                   global __bootstrap__, __file__, __loader__
                   import sys, os, importlib.resources as irs, importlib.util
                #rtld   import dl
                   with irs.files(__name__).joinpath(
                     {os.path.basename(ext._file_name)!r}) as __file__:
                      del __bootstrap__
                      if '__loader__' in globals():
                          del __loader__
                #rtld      old_flags = sys.getdlopenflags()
                      old_dir = os.getcwd()
                      try:
                        os.chdir(os.path.dirname(__file__))
                #rtld        sys.setdlopenflags(dl.RTLD_NOW)
                        spec = importlib.util.spec_from_file_location(
                                   __name__, __file__)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                      finally:
                #rtld        sys.setdlopenflags(old_flags)
                        os.chdir(old_dir)
                __bootstrap__()
                """)
                .lstrip()
                .replace('#rtld', '#rtld' * (not have_rtld))
            )
            f.write(content)
        if compile:
            self._compile_and_remove_stub(stub_file)

    def _compile_and_remove_stub(self, stub_file: str):
        from distutils.util import byte_compile

        byte_compile([stub_file], optimize=0, force=True)
        optimize = self.get_finalized_command('install_lib').optimize
        if optimize > 0:
            byte_compile(
                [stub_file],
                optimize=optimize,
                force=True,
            )
        if os.path.exists(stub_file):
            os.unlink(stub_file)


if use_stubs or os.name == 'nt':
    # Build shared libraries
    #
    def link_shared_object(
        self,
        objects,
        output_libname,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug: bool = False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ) -> None:
        self.link(
            self.SHARED_LIBRARY,
            objects,
            output_libname,
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            export_symbols,
            debug,
            extra_preargs,
            extra_postargs,
            build_temp,
            target_lang,
        )

else:
    # Build static libraries everywhere else
    libtype = 'static'

    def link_shared_object(
        self,
        objects,
        output_libname,
        output_dir=None,
        libraries=None,
        library_dirs=None,
        runtime_library_dirs=None,
        export_symbols=None,
        debug: bool = False,
        extra_preargs=None,
        extra_postargs=None,
        build_temp=None,
        target_lang=None,
    ) -> None:
        # XXX we need to either disallow these attrs on Library instances,
        # or warn/abort here if set, or something...
        # libraries=None, library_dirs=None, runtime_library_dirs=None,
        # export_symbols=None, extra_preargs=None, extra_postargs=None,
        # build_temp=None

        assert output_dir is None  # distutils build_ext doesn't pass this
        output_dir, filename = os.path.split(output_libname)
        basename, _ext = os.path.splitext(filename)
        if self.library_filename("x").startswith('lib'):
            # strip 'lib' prefix; this is kludgy if some platform uses
            # a different prefix
            basename = basename[3:]

        self.create_static_lib(objects, basename, output_dir, debug, target_lang)
