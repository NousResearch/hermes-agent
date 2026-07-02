"""setuptools.command.bdist_egg

Build .egg distributions"""

from __future__ import annotations

import marshal
import os
import re
import sys
import textwrap
from collections.abc import Iterator
from sysconfig import get_path, get_platform, get_python_version
from types import CodeType
from typing import TYPE_CHECKING, AnyStr, Literal

from setuptools import Command
from setuptools.extension import Library

from .._path import StrPath, StrPathT, ensure_directory

from distutils import log
from distutils.dir_util import mkpath, remove_tree

if TYPE_CHECKING:
    from _typeshed import GenericPath
    from typing_extensions import TypeAlias

# Same as zipfile._ZipFileMode from typeshed
_ZipFileMode: TypeAlias = Literal["r", "w", "x", "a"]


def _get_purelib():
    return get_path("purelib")


def strip_module(filename):
    if '.' in filename:
        filename = os.path.splitext(filename)[0]
    filename = filename.removesuffix('module')
    return filename


def sorted_walk(
    dir: GenericPath[AnyStr],
) -> Iterator[tuple[AnyStr, list[AnyStr], list[AnyStr]]]:
    """Do os.walk in a reproducible way,
    independent of indeterministic filesystem readdir order
    """
    for base, dirs, files in os.walk(dir):
        dirs.sort()
        files.sort()
        yield base, dirs, files


def write_stub(resource, pyfile) -> None:
    _stub_template = textwrap.dedent(
        """
        def __bootstrap__():
            global __bootstrap__, __loader__, __file__
            import sys, importlib.resources as irs, importlib.util
            with irs.as_file(irs.files(__name__).joinpath(%r)) as __file__:
                __loader__ = None; del __bootstrap__, __loader__
                spec = importlib.util.spec_from_file_location(__name__,__file__)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
        __bootstrap__()
        """
    ).lstrip()
    with open(pyfile, 'w', encoding="utf-8") as f:
        f.write(_stub_template % resource)


class bdist_egg(Command):
    description = 'create an "egg" distribution'

    user_options = [
        ('bdist-dir=', 'b', "temporary directory for creating the distribution"),
        (
            'plat-name=',
            'p',
            "platform name to embed in generated filenames "
            "(by default uses `sysconfig.get_platform()`)",
        ),
        ('exclude-source-files', None, "remove all .py files from the generated egg"),
        (
            'keep-temp',
            'k',
            "keep the pseudo-installation tree around after "
            "creating the distribution archive",
        ),
        ('dist-dir=', 'd', "directory to put final built distributions in"),
        ('skip-build', None, "skip rebuilding everything (for testing/debugging)"),
    ]

    boolean_options = ['keep-temp', 'skip-build', 'exclude-source-files']

    def initialize_options(self):
        self.bdist_dir = None
        self.plat_name = None
        self.keep_temp = False
        self.dist_dir = None
        self.skip_build = False
        self.egg_output = None
        self.exclude_source_files = None

    def finalize_options(self) -> None:
        ei_cmd = self.ei_cmd = self.get_finalized_command("egg_info")
        self.egg_info = ei_cmd.egg_info

        if self.bdist_dir is None:
            bdist_base = self.get_finalized_command('bdist').bdist_base
            self.bdist_dir = os.path.join(bdist_base, 'egg')

        if self.plat_name is None:
            self.plat_name = get_platform()

        self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))

        if self.egg_output is None:
            # Compute filename of the output egg
            basename = ei_cmd._get_egg_basename(
                py_version=get_python_version(),
                platform=self.distribution.has_ext_modules() and self.plat_name,
            )

            self.egg_output = os.path.join(self.dist_dir, basename + '.egg')

    def do_install_data(self) -> None:
        # Hack for packages that install data to install's --install-lib
        self.get_finalized_command('install').install_lib = self.bdist_dir

        site_packages = os.path.normcase(os.path.realpath(_get_purelib()))
        old, self.distribution.data_files = self.distribution.data_files, []

        for item in old:
            if isinstance(item, tuple) and len(item) == 2:
                if os.path.isabs(item[0]):
                    realpath = os.path.realpath(item[0])
                    normalized = os.path.normcase(realpath)
                    if normalized == site_packages or normalized.startswith(
                        site_packages + os.sep
                    ):
                        item = realpath[len(site_packages) + 1 :], item[1]
                        # XXX else: raise ???
            self.distribution.data_files.append(item)

        try:
            log.info("installing package data to %s", self.bdist_dir)
            self.call_command('install_data', force=False, root=None)
        finally:
            self.distribution.data_files = old

    def get_outputs(self):
        return [self.egg_output]

    def call_command(self, cmdname, **kw):
        """Invoke reinitialized command `cmdname` with keyword args"""
        for dirname in INSTALL_DIRECTORY_ATTRS:
            kw.setdefault(dirname, self.bdist_dir)
        kw.setdefault('skip_build', self.skip_build)
        cmd = self.reinitialize_command(cmdname, **kw)
        self.run_command(cmdname)
        return cmd

    def run(self) -> None:  # noqa: C901  # is too complex (14)  # FIXME
        # Generate metadata first
        self.run_command("egg_info")
        # We run install_lib before install_data, because some data hacks
        # pull their data path from the install_lib command.
        log.info("installing library code to %s", self.bdist_dir)
        instcmd = self.get_finalized_command('install')
        old_root = instcmd.root
        instcmd.root = None
        if self.distribution.has_c_libraries() and not self.skip_build:
            self.run_command('build_clib')
        cmd = self.call_command('install_lib', warn_dir=False)
        instcmd.root = old_root

        all_outputs, ext_outputs = self.get_ext_outputs()
        self.stubs = []
        to_compile = []
        for p, ext_name in enumerate(ext_outputs):
            filename, _ext = os.path.splitext(ext_name)
            pyfile = os.path.join(self.bdist_dir, strip_module(filename) + '.py')
            self.stubs.append(pyfile)
            log.info("creating stub loader for %s", ext_name)
            write_stub(os.path.basename(ext_name), pyfile)
            to_compile.append(pyfile)
            ext_outputs[p] = ext_name.replace(os.sep, '/')

        if to_compile:
            cmd.byte_compile(to_compile)
        if self.distribution.data_files:
            self.do_install_data()

        # Make the EGG-INFO directory
        archive_root = self.bdist_dir
        egg_info = os.path.join(archive_root, 'EGG-INFO')
        self.mkpath(egg_info)
        if self.distribution.scripts:
            script_dir = os.path.join(egg_info, 'scripts')
            log.info("installing scripts to %s", script_dir)
            self.call_command('install_scripts', install_dir=script_dir, no_ep=True)

        self.copy_metadata_to(egg_info)
        native_libs = os.path.join(egg_info, "native_libs.txt")
        if all_outputs:
            log.info("writing %s", native_libs)
            ensure_directory(native_libs)
            with open(native_libs, 'wt', encoding="utf-8") as libs_file:
                libs_file.write('\n'.join(all_outputs))
                libs_file.write('\n')
        elif os.path.isfile(native_libs):
            log.info("removing %s", native_libs)
            os.unlink(native_libs)

        write_safety_flag(os.path.join(archive_root, 'EGG-INFO'), self.zip_safe())

        if os.path.exists(os.path.join(self.egg_info, 'depends.txt')):
            log.warn(
                "WARNING: 'depends.txt' will not be used by setuptools 0.6!\n"
                "Use the install_requires/extras_require setup() args instead."
            )

        if self.exclude_source_files:
            self.zap_pyfiles()

        # Make the archive
        make_zipfile(
            self.egg_output,
            archive_root,
            verbose=self.verbose,
            mode=self.gen_header(),
        )
        if not self.keep_temp:
            remove_tree(self.bdist_dir)

        # Add to 'Distribution.dist_files' so that the "upload" command works
        getattr(self.distribution, 'dist_files', []).append((
            'bdist_egg',
            get_python_version(),
            self.egg_output,
        ))

    def zap_pyfiles(self) -> None:
        log.info("Removing .py files from temporary directory")
        for base, dirs, files in walk_egg(self.bdist_dir):
            for name in files:
                path = os.path.join(base, name)

                if name.endswith('.py'):
                    log.debug("Deleting %s", path)
                    os.unlink(path)

                if base.endswith('__pycache__'):
                    path_old = path

                    pattern = r'(?P<name>.+)\.(?P<magic>[^.]+)\.pyc'
                    m = re.match(pattern, name)
                    # We shouldn't find any non-pyc files in __pycache__
                    assert m is not None
                    path_new = os.path.join(base, os.pardir, m.group('name') + '.pyc')
                    log.info(f"Renaming file from [{path_old}] to [{path_new}]")
                    try:
                        os.remove(path_new)
                    except OSError:
                        pass
                    os.rename(path_old, path_new)

    def zip_safe(self):
        safe = getattr(self.distribution, 'zip_safe', None)
        if safe is not None:
            return safe
        log.warn("zip_safe flag not set; analyzing archive contents...")
        return analyze_egg(self.bdist_dir, self.stubs)

    def gen_header(self) -> Literal["w"]:
        return 'w'

    def copy_metadata_to(self, target_dir) -> None:
        "Copy metadata (egg info) to the target_dir"
        # normalize the path (so that a forward-slash in egg_info will
        # match using startswith below)
        norm_egg_info = os.path.normpath(self.egg_info)
        prefix = os.path.join(norm_egg_info, '')
        for path in self.ei_cmd.filelist.files:
            if path.startswith(prefix):
                target = os.path.join(target_dir, path[len(prefix) :])
                ensure_directory(target)
                self.copy_file(path, target)

    def get_ext_outputs(self):
        """Get a list of relative paths to C extensions in the output distro"""

        all_outputs = []
        ext_outputs = []

        paths = {self.bdist_dir: ''}
        for base, dirs, files in sorted_walk(self.bdist_dir):
            all_outputs.extend(
                paths[base] + filename
                for filename in files
                if os.path.splitext(filename)[1].lower() in NATIVE_EXTENSIONS
            )
            for filename in dirs:
                paths[os.path.join(base, filename)] = paths[base] + filename + '/'

        if self.distribution.has_ext_modules():
            build_cmd = self.get_finalized_command('build_ext')
            for ext in build_cmd.extensions:
                if isinstance(ext, Library):
                    continue
                fullname = build_cmd.get_ext_fullname(ext.name)
                filename = build_cmd.get_ext_filename(fullname)
                if not os.path.basename(filename).startswith('dl-'):
                    if os.path.exists(os.path.join(self.bdist_dir, filename)):
                        ext_outputs.append(filename)

        return all_outputs, ext_outputs


NATIVE_EXTENSIONS: dict[str, None] = dict.fromkeys('.dll .so .dylib .pyd'.split())


def walk_egg(egg_dir: StrPath) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk an unpacked egg's contents, skipping the metadata directory"""
    walker = sorted_walk(egg_dir)
    base, dirs, files = next(walker)
    if 'EGG-INFO' in dirs:
        dirs.remove('EGG-INFO')
    yield base, dirs, files
    yield from walker


def analyze_egg(egg_dir, stubs):
    # check for existing flag in EGG-INFO
    for flag, fn in safety_flags.items():
        if os.path.exists(os.path.join(egg_dir, 'EGG-INFO', fn)):
            return flag
    if not can_scan():
        return False
    safe = True
    for base, dirs, files in walk_egg(egg_dir):
        for name in files:
            if name.endswith(('.py', '.pyw')):
                continue
            elif name.endswith(('.pyc', '.pyo')):
                # always scan, even if we already know we're not safe
                safe = scan_module(egg_dir, base, name, stubs) and safe
    return safe


def write_safety_flag(egg_dir, safe) -> None:
    # Write or remove zip safety flag file(s)
    for flag, fn in safety_flags.items():
        fn = os.path.join(egg_dir, fn)
        if os.path.exists(fn):
            if safe is None or bool(safe) != flag:
                os.unlink(fn)
        elif safe is not None and bool(safe) == flag:
            with open(fn, 'wt', encoding="utf-8") as f:
                f.write('\n')


safety_flags = {
    True: 'zip-safe',
    False: 'not-zip-safe',
}


def scan_module(egg_dir, base, name, stubs):
    """Check whether module possibly uses unsafe-for-zipfile stuff"""

    filename = os.path.join(base, name)
    if filename[:-1] in stubs:
        return True  # Extension module
    pkg = base[len(egg_dir) + 1 :].replace(os.sep, '.')
    module = pkg + (pkg and '.' or '') + os.path.splitext(name)[0]
    skip = 16  # skip magic & reserved? & date & file size
    f = open(filename, 'rb')
    f.read(skip)
    code = marshal.load(f)
    f.close()
    safe = True
    symbols = dict.fromkeys(iter_symbols(code))
    for bad in ['__file__', '__path__']:
        if bad in symbols:
            log.warn("%s: module references %s", module, bad)
            safe = False
    if 'inspect' in symbols:
        for bad in [
            'getsource',
            'getabsfile',
            'getfile',
            'getsourcefile',
            'getsourcelines',
            'findsource',
            'getcomments',
            'getframeinfo',
            'getinnerframes',
            'getouterframes',
            'stack',
            'trace',
        ]:
            if bad in symbols:
                log.warn("%s: module MAY be using inspect.%s", module, bad)
                safe = False
    return safe


def iter_symbols(code: CodeType) -> Iterator[str]:
    """Yield names and strings used by `code` and its nested code objects"""
    yield from code.co_names
    for const in code.co_consts:
        if isinstance(const, str):
            yield const
        elif isinstance(const, CodeType):
            yield from iter_symbols(const)


def can_scan() -> bool:
    if not sys.platform.startswith('java') and sys.platform != 'cli':
        # CPython, PyPy, etc.
        return True
    log.warn("Unable to analyze compiled code on this platform.")
    log.warn(
        "Please ask the author to include a 'zip_safe'"
        " setting (either True or False) in the package's setup.py"
    )
    return False


# Attribute names of options for commands that might need to be convinced to
# install to the egg build directory

INSTALL_DIRECTORY_ATTRS = ['install_lib', 'install_dir', 'install_data', 'install_base']


def make_zipfile(
    zip_filename: StrPathT,
    base_dir,
    verbose: bool = False,
    compress=True,
    mode: _ZipFileMode = 'w',
) -> StrPathT:
    """Create a zip file from all the files under 'base_dir'.  The output
    zip file will be named 'base_dir' + ".zip".  Uses either the "zipfile"
    Python module (if available) or the InfoZIP "zip" utility (if installed
    and found on the default search path).  If neither tool is available,
    raises DistutilsExecError.  Returns the name of the output zip file.
    """
    import zipfile

    mkpath(os.path.dirname(zip_filename))  # type: ignore[arg-type] # python/mypy#18075
    log.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)

    def visit(z, dirname, names):
        for name in names:
            path = os.path.normpath(os.path.join(dirname, name))
            if os.path.isfile(path):
                p = path[len(base_dir) + 1 :]
                z.write(path, p)
                log.debug("adding '%s'", p)

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    z = zipfile.ZipFile(zip_filename, mode, compression=compression)
    for dirname, dirs, files in sorted_walk(base_dir):
        visit(z, dirname, files)
    z.close()
    return zip_filename
