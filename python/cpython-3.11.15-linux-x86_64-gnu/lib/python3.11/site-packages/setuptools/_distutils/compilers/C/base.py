"""distutils.ccompiler

Contains Compiler, an abstract base class that defines the interface
for the Distutils compiler abstraction model."""

from __future__ import annotations

import os
import pathlib
import re
import sys
import warnings
from collections.abc import Callable, Iterable, MutableSequence, Sequence
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    TypeVar,
    Union,
    overload,
)

from more_itertools import always_iterable

from ..._log import log
from ..._modified import newer_group
from ...dir_util import mkpath
from ...errors import (
    DistutilsModuleError,
    DistutilsPlatformError,
)
from ...file_util import move_file
from ...spawn import spawn
from ...util import execute, is_mingw, split_quoted
from .errors import (
    CompileError,
    LinkError,
    UnknownFileType,
)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeVarTuple, Unpack

    _Ts = TypeVarTuple("_Ts")

_Macro: TypeAlias = Union[tuple[str], tuple[str, Union[str, None]]]
_StrPathT = TypeVar("_StrPathT", bound="str | os.PathLike[str]")
_BytesPathT = TypeVar("_BytesPathT", bound="bytes | os.PathLike[bytes]")


class Compiler:
    """Abstract base class to define the interface that must be implemented
    by real compiler classes.  Also has some utility methods used by
    several compiler classes.

    The basic idea behind a compiler abstraction class is that each
    instance can be used for all the compile/link steps in building a
    single project.  Thus, attributes common to all of those compile and
    link steps -- include directories, macros to define, libraries to link
    against, etc. -- are attributes of the compiler instance.  To allow for
    variability in how individual files are treated, most of those
    attributes may be varied on a per-compilation or per-link basis.
    """

    # 'compiler_type' is a class attribute that identifies this class.  It
    # keeps code that wants to know what kind of compiler it's dealing with
    # from having to import all possible compiler classes just to do an
    # 'isinstance'.  In concrete CCompiler subclasses, 'compiler_type'
    # should really, really be one of the keys of the 'compiler_class'
    # dictionary (see below -- used by the 'new_compiler()' factory
    # function) -- authors of new compiler interface classes are
    # responsible for updating 'compiler_class'!
    compiler_type: ClassVar[str] = None  # type: ignore[assignment]

    # XXX things not handled by this compiler abstraction model:
    #   * client can't provide additional options for a compiler,
    #     e.g. warning, optimization, debugging flags.  Perhaps this
    #     should be the domain of concrete compiler abstraction classes
    #     (UnixCCompiler, MSVCCompiler, etc.) -- or perhaps the base
    #     class should have methods for the common ones.
    #   * can't completely override the include or library searchg
    #     path, ie. no "cc -I -Idir1 -Idir2" or "cc -L -Ldir1 -Ldir2".
    #     I'm not sure how widely supported this is even by Unix
    #     compilers, much less on other platforms.  And I'm even less
    #     sure how useful it is; maybe for cross-compiling, but
    #     support for that is a ways off.  (And anyways, cross
    #     compilers probably have a dedicated binary with the
    #     right paths compiled in.  I hope.)
    #   * can't do really freaky things with the library list/library
    #     dirs, e.g. "-Ldir1 -lfoo -Ldir2 -lfoo" to link against
    #     different versions of libfoo.a in different locations.  I
    #     think this is useless without the ability to null out the
    #     library search path anyways.

    executables: ClassVar[dict]

    # Subclasses that rely on the standard filename generation methods
    # implemented below should override these; see the comment near
    # those methods ('object_filenames()' et. al.) for details:
    src_extensions: ClassVar[list[str] | None] = None
    obj_extension: ClassVar[str | None] = None
    static_lib_extension: ClassVar[str | None] = None
    shared_lib_extension: ClassVar[str | None] = None
    static_lib_format: ClassVar[str | None] = None  # format string
    shared_lib_format: ClassVar[str | None] = None  # prob. same as static_lib_format
    exe_extension: ClassVar[str | None] = None

    # Default language settings. language_map is used to detect a source
    # file or Extension target language, checking source filenames.
    # language_order is used to detect the language precedence, when deciding
    # what language to use when mixing source types. For example, if some
    # extension has two files with ".c" extension, and one with ".cpp", it
    # is still linked as c++.
    language_map: ClassVar[dict[str, str]] = {
        ".c": "c",
        ".cc": "c++",
        ".cpp": "c++",
        ".cxx": "c++",
        ".m": "objc",
    }
    language_order: ClassVar[list[str]] = ["c++", "objc", "c"]

    include_dirs: list[str] = []
    """
    include dirs specific to this compiler class
    """

    library_dirs: list[str] = []
    """
    library dirs specific to this compiler class
    """

    def __init__(self, verbose: bool = False, force: bool = False) -> None:
        self.force = force
        self.verbose = verbose

        # 'output_dir': a common output directory for object, library,
        # shared object, and shared library files
        self.output_dir: str | None = None

        # 'macros': a list of macro definitions (or undefinitions).  A
        # macro definition is a 2-tuple (name, value), where the value is
        # either a string or None (no explicit value).  A macro
        # undefinition is a 1-tuple (name,).
        self.macros: list[_Macro] = []

        # 'include_dirs': a list of directories to search for include files
        self.include_dirs = []

        # 'libraries': a list of libraries to include in any link
        # (library names, not filenames: eg. "foo" not "libfoo.a")
        self.libraries: list[str] = []

        # 'library_dirs': a list of directories to search for libraries
        self.library_dirs = []

        # 'runtime_library_dirs': a list of directories to search for
        # shared libraries/objects at runtime
        self.runtime_library_dirs: list[str] = []

        # 'objects': a list of object files (or similar, such as explicitly
        # named library files) to include on any link
        self.objects: list[str] = []

        for key in self.executables.keys():
            self.set_executable(key, self.executables[key])

    def set_executables(self, **kwargs: str) -> None:
        """Define the executables (and options for them) that will be run
        to perform the various stages of compilation.  The exact set of
        executables that may be specified here depends on the compiler
        class (via the 'executables' class attribute), but most will have:
          compiler      the C/C++ compiler
          linker_so     linker used to create shared objects and libraries
          linker_exe    linker used to create binary executables
          archiver      static library creator

        On platforms with a command-line (Unix, DOS/Windows), each of these
        is a string that will be split into executable name and (optional)
        list of arguments.  (Splitting the string is done similarly to how
        Unix shells operate: words are delimited by spaces, but quotes and
        backslashes can override this.  See
        'distutils.util.split_quoted()'.)
        """

        # Note that some CCompiler implementation classes will define class
        # attributes 'cpp', 'cc', etc. with hard-coded executable names;
        # this is appropriate when a compiler class is for exactly one
        # compiler/OS combination (eg. MSVCCompiler).  Other compiler
        # classes (UnixCCompiler, in particular) are driven by information
        # discovered at run-time, since there are many different ways to do
        # basically the same things with Unix C compilers.

        for key in kwargs:
            if key not in self.executables:
                raise ValueError(
                    f"unknown executable '{key}' for class {self.__class__.__name__}"
                )
            self.set_executable(key, kwargs[key])

    def set_executable(self, key, value):
        if isinstance(value, str):
            setattr(self, key, split_quoted(value))
        else:
            setattr(self, key, value)

    def _find_macro(self, name):
        i = 0
        for defn in self.macros:
            if defn[0] == name:
                return i
            i += 1
        return None

    def _check_macro_definitions(self, definitions):
        """Ensure that every element of 'definitions' is valid."""
        for defn in definitions:
            self._check_macro_definition(*defn)

    def _check_macro_definition(self, defn):
        """
        Raise a TypeError if defn is not valid.

        A valid definition is either a (name, value) 2-tuple or a (name,) tuple.
        """
        if not isinstance(defn, tuple) or not self._is_valid_macro(*defn):
            raise TypeError(
                f"invalid macro definition '{defn}': "
                "must be tuple (string,), (string, string), or (string, None)"
            )

    @staticmethod
    def _is_valid_macro(name, value=None):
        """
        A valid macro is a ``name : str`` and a ``value : str | None``.

        >>> Compiler._is_valid_macro('foo', None)
        True
        """
        return isinstance(name, str) and isinstance(value, (str, type(None)))

    # -- Bookkeeping methods -------------------------------------------

    def define_macro(self, name: str, value: str | None = None) -> None:
        """Define a preprocessor macro for all compilations driven by this
        compiler object.  The optional parameter 'value' should be a
        string; if it is not supplied, then the macro will be defined
        without an explicit value and the exact outcome depends on the
        compiler used (XXX true? does ANSI say anything about this?)
        """
        # Delete from the list of macro definitions/undefinitions if
        # already there (so that this one will take precedence).
        i = self._find_macro(name)
        if i is not None:
            del self.macros[i]

        self.macros.append((name, value))

    def undefine_macro(self, name: str) -> None:
        """Undefine a preprocessor macro for all compilations driven by
        this compiler object.  If the same macro is defined by
        'define_macro()' and undefined by 'undefine_macro()' the last call
        takes precedence (including multiple redefinitions or
        undefinitions).  If the macro is redefined/undefined on a
        per-compilation basis (ie. in the call to 'compile()'), then that
        takes precedence.
        """
        # Delete from the list of macro definitions/undefinitions if
        # already there (so that this one will take precedence).
        i = self._find_macro(name)
        if i is not None:
            del self.macros[i]

        undefn = (name,)
        self.macros.append(undefn)

    def add_include_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        header files.  The compiler is instructed to search directories in
        the order in which they are supplied by successive calls to
        'add_include_dir()'.
        """
        self.include_dirs.append(dir)

    def set_include_dirs(self, dirs: list[str]) -> None:
        """Set the list of directories that will be searched to 'dirs' (a
        list of strings).  Overrides any preceding calls to
        'add_include_dir()'; subsequence calls to 'add_include_dir()' add
        to the list passed to 'set_include_dirs()'.  This does not affect
        any list of standard include directories that the compiler may
        search by default.
        """
        self.include_dirs = dirs[:]

    def add_library(self, libname: str) -> None:
        """Add 'libname' to the list of libraries that will be included in
        all links driven by this compiler object.  Note that 'libname'
        should *not* be the name of a file containing a library, but the
        name of the library itself: the actual filename will be inferred by
        the linker, the compiler, or the compiler class (depending on the
        platform).

        The linker will be instructed to link against libraries in the
        order they were supplied to 'add_library()' and/or
        'set_libraries()'.  It is perfectly valid to duplicate library
        names; the linker will be instructed to link against libraries as
        many times as they are mentioned.
        """
        self.libraries.append(libname)

    def set_libraries(self, libnames: list[str]) -> None:
        """Set the list of libraries to be included in all links driven by
        this compiler object to 'libnames' (a list of strings).  This does
        not affect any standard system libraries that the linker may
        include by default.
        """
        self.libraries = libnames[:]

    def add_library_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        libraries specified to 'add_library()' and 'set_libraries()'.  The
        linker will be instructed to search for libraries in the order they
        are supplied to 'add_library_dir()' and/or 'set_library_dirs()'.
        """
        self.library_dirs.append(dir)

    def set_library_dirs(self, dirs: list[str]) -> None:
        """Set the list of library search directories to 'dirs' (a list of
        strings).  This does not affect any standard library search path
        that the linker may search by default.
        """
        self.library_dirs = dirs[:]

    def add_runtime_library_dir(self, dir: str) -> None:
        """Add 'dir' to the list of directories that will be searched for
        shared libraries at runtime.
        """
        self.runtime_library_dirs.append(dir)

    def set_runtime_library_dirs(self, dirs: list[str]) -> None:
        """Set the list of directories to search for shared libraries at
        runtime to 'dirs' (a list of strings).  This does not affect any
        standard search path that the runtime linker may search by
        default.
        """
        self.runtime_library_dirs = dirs[:]

    def add_link_object(self, object: str) -> None:
        """Add 'object' to the list of object files (or analogues, such as
        explicitly named library files or the output of "resource
        compilers") to be included in every link driven by this compiler
        object.
        """
        self.objects.append(object)

    def set_link_objects(self, objects: list[str]) -> None:
        """Set the list of object files (or analogues) to be included in
        every link to 'objects'.  This does not affect any standard object
        files that the linker may include by default (such as system
        libraries).
        """
        self.objects = objects[:]

    # -- Private utility methods --------------------------------------
    # (here for the convenience of subclasses)

    # Helper method to prep compiler in subclass compile() methods

    def _setup_compile(
        self,
        outdir: str | None,
        macros: list[_Macro] | None,
        incdirs: list[str] | tuple[str, ...] | None,
        sources,
        depends,
        extra,
    ):
        """Process arguments and decide which source files to compile."""
        outdir, macros, incdirs = self._fix_compile_args(outdir, macros, incdirs)

        if extra is None:
            extra = []

        # Get the list of expected output (object) files
        objects = self.object_filenames(sources, strip_dir=False, output_dir=outdir)
        assert len(objects) == len(sources)

        pp_opts = gen_preprocess_options(macros, incdirs)

        build = {}
        for i in range(len(sources)):
            src = sources[i]
            obj = objects[i]
            ext = os.path.splitext(src)[1]
            self.mkpath(os.path.dirname(obj))
            build[obj] = (src, ext)

        return macros, objects, extra, pp_opts, build

    def _get_cc_args(self, pp_opts, debug, before):
        # works for unixccompiler, cygwinccompiler
        cc_args = pp_opts + ['-c']
        if debug:
            cc_args[:0] = ['-g']
        if before:
            cc_args[:0] = before
        return cc_args

    def _fix_compile_args(
        self,
        output_dir: str | None,
        macros: list[_Macro] | None,
        include_dirs: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, list[_Macro], list[str]]:
        """Typecheck and fix-up some of the arguments to the 'compile()'
        method, and return fixed-up values.  Specifically: if 'output_dir'
        is None, replaces it with 'self.output_dir'; ensures that 'macros'
        is a list, and augments it with 'self.macros'; ensures that
        'include_dirs' is a list, and augments it with 'self.include_dirs'.
        Guarantees that the returned values are of the correct type,
        i.e. for 'output_dir' either string or None, and for 'macros' and
        'include_dirs' either list or None.
        """
        if output_dir is None:
            output_dir = self.output_dir
        elif not isinstance(output_dir, str):
            raise TypeError("'output_dir' must be a string or None")

        if macros is None:
            macros = list(self.macros)
        elif isinstance(macros, list):
            macros = macros + (self.macros or [])
        else:
            raise TypeError("'macros' (if supplied) must be a list of tuples")

        if include_dirs is None:
            include_dirs = list(self.include_dirs)
        elif isinstance(include_dirs, (list, tuple)):
            include_dirs = list(include_dirs) + (self.include_dirs or [])
        else:
            raise TypeError("'include_dirs' (if supplied) must be a list of strings")

        # add include dirs for class
        include_dirs += self.__class__.include_dirs

        return output_dir, macros, include_dirs

    def _prep_compile(self, sources, output_dir, depends=None):
        """Decide which source files must be recompiled.

        Determine the list of object files corresponding to 'sources',
        and figure out which ones really need to be recompiled.
        Return a list of all object files and a dictionary telling
        which source files can be skipped.
        """
        # Get the list of expected output (object) files
        objects = self.object_filenames(sources, output_dir=output_dir)
        assert len(objects) == len(sources)

        # Return an empty dict for the "which source files can be skipped"
        # return value to preserve API compatibility.
        return objects, {}

    def _fix_object_args(
        self, objects: list[str] | tuple[str, ...], output_dir: str | None
    ) -> tuple[list[str], str]:
        """Typecheck and fix up some arguments supplied to various methods.
        Specifically: ensure that 'objects' is a list; if output_dir is
        None, replace with self.output_dir.  Return fixed versions of
        'objects' and 'output_dir'.
        """
        if not isinstance(objects, (list, tuple)):
            raise TypeError("'objects' must be a list or tuple of strings")
        objects = list(objects)

        if output_dir is None:
            output_dir = self.output_dir
        elif not isinstance(output_dir, str):
            raise TypeError("'output_dir' must be a string or None")

        return (objects, output_dir)

    def _fix_lib_args(
        self,
        libraries: list[str] | tuple[str, ...] | None,
        library_dirs: list[str] | tuple[str, ...] | None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Typecheck and fix up some of the arguments supplied to the
        'link_*' methods.  Specifically: ensure that all arguments are
        lists, and augment them with their permanent versions
        (eg. 'self.libraries' augments 'libraries').  Return a tuple with
        fixed versions of all arguments.
        """
        if libraries is None:
            libraries = list(self.libraries)
        elif isinstance(libraries, (list, tuple)):
            libraries = list(libraries) + (self.libraries or [])
        else:
            raise TypeError("'libraries' (if supplied) must be a list of strings")

        if library_dirs is None:
            library_dirs = list(self.library_dirs)
        elif isinstance(library_dirs, (list, tuple)):
            library_dirs = list(library_dirs) + (self.library_dirs or [])
        else:
            raise TypeError("'library_dirs' (if supplied) must be a list of strings")

        # add library dirs for class
        library_dirs += self.__class__.library_dirs

        if runtime_library_dirs is None:
            runtime_library_dirs = list(self.runtime_library_dirs)
        elif isinstance(runtime_library_dirs, (list, tuple)):
            runtime_library_dirs = list(runtime_library_dirs) + (
                self.runtime_library_dirs or []
            )
        else:
            raise TypeError(
                "'runtime_library_dirs' (if supplied) must be a list of strings"
            )

        return (libraries, library_dirs, runtime_library_dirs)

    def _need_link(self, objects, output_file):
        """Return true if we need to relink the files listed in 'objects'
        to recreate 'output_file'.
        """
        if self.force:
            return True
        newer = newer_group(objects, output_file)
        return newer

    def detect_language(self, sources: str | list[str]) -> str | None:
        """Detect the language of a given file, or list of files. Uses
        language_map, and language_order to do the job.
        """
        if not isinstance(sources, list):
            sources = [sources]
        lang = None
        index = len(self.language_order)
        for source in sources:
            base, ext = os.path.splitext(source)
            extlang = self.language_map.get(ext)
            try:
                extindex = self.language_order.index(extlang)
                if extindex < index:
                    lang = extlang
                    index = extindex
            except ValueError:
                pass
        return lang

    # -- Worker methods ------------------------------------------------
    # (must be implemented by subclasses)

    def preprocess(
        self,
        source: str | os.PathLike[str],
        output_file: str | os.PathLike[str] | None = None,
        macros: list[_Macro] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        extra_preargs: list[str] | None = None,
        extra_postargs: Iterable[str] | None = None,
    ):
        """Preprocess a single C/C++ source file, named in 'source'.
        Output will be written to file named 'output_file', or stdout if
        'output_file' not supplied.  'macros' is a list of macro
        definitions as for 'compile()', which will augment the macros set
        with 'define_macro()' and 'undefine_macro()'.  'include_dirs' is a
        list of directory names that will be added to the default list.

        Raises PreprocessError on failure.
        """
        pass

    def compile(
        self,
        sources: Sequence[str | os.PathLike[str]],
        output_dir: str | None = None,
        macros: list[_Macro] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        depends: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        """Compile one or more source files.

        'sources' must be a list of filenames, most likely C/C++
        files, but in reality anything that can be handled by a
        particular compiler and compiler class (eg. MSVCCompiler can
        handle resource files in 'sources').  Return a list of object
        filenames, one per source filename in 'sources'.  Depending on
        the implementation, not all source files will necessarily be
        compiled, but all corresponding object filenames will be
        returned.

        If 'output_dir' is given, object files will be put under it, while
        retaining their original path component.  That is, "foo/bar.c"
        normally compiles to "foo/bar.o" (for a Unix implementation); if
        'output_dir' is "build", then it would compile to
        "build/foo/bar.o".

        'macros', if given, must be a list of macro definitions.  A macro
        definition is either a (name, value) 2-tuple or a (name,) 1-tuple.
        The former defines a macro; if the value is None, the macro is
        defined without an explicit value.  The 1-tuple case undefines a
        macro.  Later definitions/redefinitions/ undefinitions take
        precedence.

        'include_dirs', if given, must be a list of strings, the
        directories to add to the default include file search path for this
        compilation only.

        'debug' is a boolean; if true, the compiler will be instructed to
        output debug symbols in (or alongside) the object file(s).

        'extra_preargs' and 'extra_postargs' are implementation- dependent.
        On platforms that have the notion of a command-line (e.g. Unix,
        DOS/Windows), they are most likely lists of strings: extra
        command-line arguments to prepend/append to the compiler command
        line.  On other platforms, consult the implementation class
        documentation.  In any event, they are intended as an escape hatch
        for those occasions when the abstract compiler framework doesn't
        cut the mustard.

        'depends', if given, is a list of filenames that all targets
        depend on.  If a source file is older than any file in
        depends, then the source file will be recompiled.  This
        supports dependency tracking, but only at a coarse
        granularity.

        Raises CompileError on failure.
        """
        # A concrete compiler class can either override this method
        # entirely or implement _compile().
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        for obj in objects:
            try:
                src, ext = build[obj]
            except KeyError:
                continue
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        # Return *all* object filenames, not just the ones we just built.
        return objects

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to product 'obj'."""
        # A concrete compiler class that does not override compile()
        # should implement _compile().
        pass

    def create_static_lib(
        self,
        objects: list[str] | tuple[str, ...],
        output_libname: str,
        output_dir: str | None = None,
        debug: bool = False,
        target_lang: str | None = None,
    ) -> None:
        """Link a bunch of stuff together to create a static library file.
        The "bunch of stuff" consists of the list of object files supplied
        as 'objects', the extra object files supplied to
        'add_link_object()' and/or 'set_link_objects()', the libraries
        supplied to 'add_library()' and/or 'set_libraries()', and the
        libraries supplied as 'libraries' (if any).

        'output_libname' should be a library name, not a filename; the
        filename will be inferred from the library name.  'output_dir' is
        the directory where the library file will be put.

        'debug' is a boolean; if true, debugging information will be
        included in the library (note that on most platforms, it is the
        compile step where this matters: the 'debug' flag is included here
        just for consistency).

        'target_lang' is the target language for which the given objects
        are being compiled. This allows specific linkage time treatment of
        certain languages.

        Raises LibError on failure.
        """
        pass

    # values for target_desc parameter in link()
    SHARED_OBJECT = "shared_object"
    SHARED_LIBRARY = "shared_library"
    EXECUTABLE = "executable"

    def link(
        self,
        target_desc: str,
        objects: list[str] | tuple[str, ...],
        output_filename: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        """Link a bunch of stuff together to create an executable or
        shared library file.

        The "bunch of stuff" consists of the list of object files supplied
        as 'objects'.  'output_filename' should be a filename.  If
        'output_dir' is supplied, 'output_filename' is relative to it
        (i.e. 'output_filename' can provide directory components if
        needed).

        'libraries' is a list of libraries to link against.  These are
        library names, not filenames, since they're translated into
        filenames in a platform-specific way (eg. "foo" becomes "libfoo.a"
        on Unix and "foo.lib" on DOS/Windows).  However, they can include a
        directory component, which means the linker will look in that
        specific directory rather than searching all the normal locations.

        'library_dirs', if supplied, should be a list of directories to
        search for libraries that were specified as bare library names
        (ie. no directory component).  These are on top of the system
        default and those supplied to 'add_library_dir()' and/or
        'set_library_dirs()'.  'runtime_library_dirs' is a list of
        directories that will be embedded into the shared library and used
        to search for other shared libraries that *it* depends on at
        run-time.  (This may only be relevant on Unix.)

        'export_symbols' is a list of symbols that the shared library will
        export.  (This appears to be relevant only on Windows.)

        'debug' is as for 'compile()' and 'create_static_lib()', with the
        slight distinction that it actually matters on most platforms (as
        opposed to 'create_static_lib()', which includes a 'debug' flag
        mostly for form's sake).

        'extra_preargs' and 'extra_postargs' are as for 'compile()' (except
        of course that they supply command-line arguments for the
        particular linker being used).

        'target_lang' is the target language for which the given objects
        are being compiled. This allows specific linkage time treatment of
        certain languages.

        Raises LinkError on failure.
        """
        raise NotImplementedError

    # Old 'link_*()' methods, rewritten to use the new 'link()' method.

    def link_shared_lib(
        self,
        objects: list[str] | tuple[str, ...],
        output_libname: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.SHARED_LIBRARY,
            objects,
            self.library_filename(output_libname, lib_type='shared'),
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

    def link_shared_object(
        self,
        objects: list[str] | tuple[str, ...],
        output_filename: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        export_symbols: Iterable[str] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        build_temp: str | os.PathLike[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.SHARED_OBJECT,
            objects,
            output_filename,
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

    def link_executable(
        self,
        objects: list[str] | tuple[str, ...],
        output_progname: str,
        output_dir: str | None = None,
        libraries: list[str] | tuple[str, ...] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
        runtime_library_dirs: list[str] | tuple[str, ...] | None = None,
        debug: bool = False,
        extra_preargs: list[str] | None = None,
        extra_postargs: list[str] | None = None,
        target_lang: str | None = None,
    ):
        self.link(
            Compiler.EXECUTABLE,
            objects,
            self.executable_filename(output_progname),
            output_dir,
            libraries,
            library_dirs,
            runtime_library_dirs,
            None,
            debug,
            extra_preargs,
            extra_postargs,
            None,
            target_lang,
        )

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function; there is
    # no appropriate default implementation so subclasses should
    # implement all of these.

    def library_dir_option(self, dir: str) -> str:
        """Return the compiler option to add 'dir' to the list of
        directories searched for libraries.
        """
        raise NotImplementedError

    def runtime_library_dir_option(self, dir: str) -> str:
        """Return the compiler option to add 'dir' to the list of
        directories searched for runtime libraries.
        """
        raise NotImplementedError

    def library_option(self, lib: str) -> str:
        """Return the compiler option to add 'lib' to the list of libraries
        linked into the shared library or executable.
        """
        raise NotImplementedError

    def has_function(  # noqa: C901
        self,
        funcname: str,
        includes: Iterable[str] | None = None,
        include_dirs: list[str] | tuple[str, ...] | None = None,
        libraries: list[str] | None = None,
        library_dirs: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """Return a boolean indicating whether funcname is provided as
        a symbol on the current platform.  The optional arguments can
        be used to augment the compilation environment.

        The libraries argument is a list of flags to be passed to the
        linker to make additional symbol definitions available for
        linking.

        The includes and include_dirs arguments are deprecated.
        Usually, supplying include files with function declarations
        will cause function detection to fail even in cases where the
        symbol is available for linking.

        """
        # this can't be included at module scope because it tries to
        # import math which might not be available at that point - maybe
        # the necessary logic should just be inlined?
        import tempfile

        if includes is None:
            includes = []
        else:
            warnings.warn("includes is deprecated", DeprecationWarning)
        if include_dirs is None:
            include_dirs = []
        else:
            warnings.warn("include_dirs is deprecated", DeprecationWarning)
        if libraries is None:
            libraries = []
        if library_dirs is None:
            library_dirs = []
        fd, fname = tempfile.mkstemp(".c", funcname, text=True)
        with os.fdopen(fd, "w", encoding='utf-8') as f:
            for incl in includes:
                f.write(f"""#include "{incl}"\n""")
            if not includes:
                # Use "char func(void);" as the prototype to follow
                # what autoconf does.  This prototype does not match
                # any well-known function the compiler might recognize
                # as a builtin, so this ends up as a true link test.
                # Without a fake prototype, the test would need to
                # know the exact argument types, and the has_function
                # interface does not provide that level of information.
                f.write(
                    f"""\
#ifdef __cplusplus
extern "C"
#endif
char {funcname}(void);
"""
                )
            f.write(
                f"""\
int main (int argc, char **argv) {{
    {funcname}();
    return 0;
}}
"""
            )

        try:
            objects = self.compile([fname], include_dirs=include_dirs)
        except CompileError:
            return False
        finally:
            os.remove(fname)

        try:
            self.link_executable(
                objects, "a.out", libraries=libraries, library_dirs=library_dirs
            )
        except (LinkError, TypeError):
            return False
        else:
            os.remove(
                self.executable_filename("a.out", output_dir=self.output_dir or '')
            )
        finally:
            for fn in objects:
                os.remove(fn)
        return True

    def find_library_file(
        self, dirs: Iterable[str], lib: str, debug: bool = False
    ) -> str | None:
        """Search the specified list of directories for a static or shared
        library file 'lib' and return the full path to that file.  If
        'debug' true, look for a debugging version (if that makes sense on
        the current platform).  Return None if 'lib' wasn't found in any of
        the specified directories.
        """
        raise NotImplementedError

    # -- Filename generation methods -----------------------------------

    # The default implementation of the filename generating methods are
    # prejudiced towards the Unix/DOS/Windows view of the world:
    #   * object files are named by replacing the source file extension
    #     (eg. .c/.cpp -> .o/.obj)
    #   * library files (shared or static) are named by plugging the
    #     library name and extension into a format string, eg.
    #     "lib%s.%s" % (lib_name, ".a") for Unix static libraries
    #   * executables are named by appending an extension (possibly
    #     empty) to the program name: eg. progname + ".exe" for
    #     Windows
    #
    # To reduce redundant code, these methods expect to find
    # several attributes in the current object (presumably defined
    # as class attributes):
    #   * src_extensions -
    #     list of C/C++ source file extensions, eg. ['.c', '.cpp']
    #   * obj_extension -
    #     object file extension, eg. '.o' or '.obj'
    #   * static_lib_extension -
    #     extension for static library files, eg. '.a' or '.lib'
    #   * shared_lib_extension -
    #     extension for shared library/object files, eg. '.so', '.dll'
    #   * static_lib_format -
    #     format string for generating static library filenames,
    #     eg. 'lib%s.%s' or '%s.%s'
    #   * shared_lib_format
    #     format string for generating shared library filenames
    #     (probably same as static_lib_format, since the extension
    #     is one of the intended parameters to the format string)
    #   * exe_extension -
    #     extension for executable files, eg. '' or '.exe'

    def object_filenames(
        self,
        source_filenames: Iterable[str | os.PathLike[str]],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] | None = '',
    ) -> list[str]:
        if output_dir is None:
            output_dir = ''
        return list(
            self._make_out_path(output_dir, strip_dir, src_name)
            for src_name in source_filenames
        )

    @property
    def out_extensions(self):
        return dict.fromkeys(self.src_extensions, self.obj_extension)

    def _make_out_path(self, output_dir, strip_dir, src_name):
        return self._make_out_path_exts(
            output_dir, strip_dir, src_name, self.out_extensions
        )

    @classmethod
    def _make_out_path_exts(cls, output_dir, strip_dir, src_name, extensions):
        r"""
        >>> exts = {'.c': '.o'}
        >>> Compiler._make_out_path_exts('.', False, '/foo/bar.c', exts).replace('\\', '/')
        './foo/bar.o'
        >>> Compiler._make_out_path_exts('.', True, '/foo/bar.c', exts).replace('\\', '/')
        './bar.o'
        """
        src = pathlib.PurePath(src_name)
        # Ensure base is relative to honor output_dir (python/cpython#37775).
        base = cls._make_relative(src)
        try:
            new_ext = extensions[src.suffix]
        except LookupError:
            raise UnknownFileType(f"unknown file type '{src.suffix}' (from '{src}')")
        if strip_dir:
            base = pathlib.PurePath(base.name)
        return os.path.join(output_dir, base.with_suffix(new_ext))

    @staticmethod
    def _make_relative(base: pathlib.Path):
        return base.relative_to(base.anchor)

    @overload
    def shared_object_filename(
        self,
        basename: str,
        strip_dir: Literal[False] = False,
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    @overload
    def shared_object_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: Literal[True],
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    def shared_object_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = '',
    ) -> str:
        assert output_dir is not None
        if strip_dir:
            basename = os.path.basename(basename)
        return os.path.join(output_dir, basename + self.shared_lib_extension)

    @overload
    def executable_filename(
        self,
        basename: str,
        strip_dir: Literal[False] = False,
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    @overload
    def executable_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: Literal[True],
        output_dir: str | os.PathLike[str] = "",
    ) -> str: ...
    def executable_filename(
        self,
        basename: str | os.PathLike[str],
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = '',
    ) -> str:
        assert output_dir is not None
        if strip_dir:
            basename = os.path.basename(basename)
        return os.path.join(output_dir, basename + (self.exe_extension or ''))

    def library_filename(
        self,
        libname: str,
        lib_type: str = "static",
        strip_dir: bool = False,
        output_dir: str | os.PathLike[str] = "",  # or 'shared'
    ):
        assert output_dir is not None
        expected = '"static", "shared", "dylib", "xcode_stub"'
        if lib_type not in eval(expected):
            raise ValueError(f"'lib_type' must be {expected}")
        fmt = getattr(self, lib_type + "_lib_format")
        ext = getattr(self, lib_type + "_lib_extension")

        dir, base = os.path.split(libname)
        filename = fmt % (base, ext)
        if strip_dir:
            dir = ''

        return os.path.join(output_dir, dir, filename)

    # -- Utility methods -----------------------------------------------

    def announce(self, msg: object, level: int = 1) -> None:
        log.debug(msg)

    def debug_print(self, msg: object) -> None:
        from distutils.debug import DEBUG

        if DEBUG:
            print(msg)

    def warn(self, msg: object) -> None:
        sys.stderr.write(f"warning: {msg}\n")

    def execute(
        self,
        func: Callable[[Unpack[_Ts]], object],
        args: tuple[Unpack[_Ts]],
        msg: object = None,
        level: int = 1,
    ) -> None:
        execute(func, args, msg)

    def spawn(
        self, cmd: MutableSequence[bytes | str | os.PathLike[str]], **kwargs
    ) -> None:
        spawn(cmd, **kwargs)

    @overload
    def move_file(
        self, src: str | os.PathLike[str], dst: _StrPathT
    ) -> _StrPathT | str: ...
    @overload
    def move_file(
        self, src: bytes | os.PathLike[bytes], dst: _BytesPathT
    ) -> _BytesPathT | bytes: ...
    def move_file(
        self,
        src: str | os.PathLike[str] | bytes | os.PathLike[bytes],
        dst: str | os.PathLike[str] | bytes | os.PathLike[bytes],
    ) -> str | os.PathLike[str] | bytes | os.PathLike[bytes]:
        return move_file(src, dst)

    def mkpath(self, name, mode=0o777):
        mkpath(name, mode)


# Map a sys.platform/os.name ('posix', 'nt') to the default compiler
# type for that platform. Keys are interpreted as re match
# patterns. Order is important; platform mappings are preferred over
# OS names.
_default_compilers = (
    # Platform string mappings
    # on a cygwin built python we can use gcc like an ordinary UNIXish
    # compiler
    ('cygwin.*', 'unix'),
    ('zos', 'zos'),
    # OS name mappings
    ('posix', 'unix'),
    ('nt', 'msvc'),
)


def get_default_compiler(osname: str | None = None, platform: str | None = None) -> str:
    """Determine the default compiler to use for the given platform.

    osname should be one of the standard Python OS names (i.e. the
    ones returned by os.name) and platform the common value
    returned by sys.platform for the platform in question.

    The default values are os.name and sys.platform in case the
    parameters are not given.
    """
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    # Mingw is a special case where sys.platform is 'win32' but we
    # want to use the 'mingw32' compiler, so check it first
    if is_mingw():
        return 'mingw32'
    for pattern, compiler in _default_compilers:
        if (
            re.match(pattern, platform) is not None
            or re.match(pattern, osname) is not None
        ):
            return compiler
    # Default to Unix compiler
    return 'unix'


# Map compiler types to (module_name, class_name) pairs -- ie. where to
# find the code that implements an interface to this compiler.  (The module
# is assumed to be in the 'distutils' package.)
compiler_class = {
    'unix': ('unixccompiler', 'UnixCCompiler', "standard UNIX-style compiler"),
    'msvc': ('_msvccompiler', 'MSVCCompiler', "Microsoft Visual C++"),
    'cygwin': (
        'cygwinccompiler',
        'CygwinCCompiler',
        "Cygwin port of GNU C Compiler for Win32",
    ),
    'mingw32': (
        'cygwinccompiler',
        'Mingw32CCompiler',
        "Mingw32 port of GNU C Compiler for Win32",
    ),
    'bcpp': ('bcppcompiler', 'BCPPCompiler', "Borland C++ Compiler"),
    'zos': ('zosccompiler', 'zOSCCompiler', 'IBM XL C/C++ Compilers'),
}


def show_compilers() -> None:
    """Print list of available compilers (used by the "--help-compiler"
    options to "build", "build_ext", "build_clib").
    """
    # XXX this "knows" that the compiler option it's describing is
    # "--compiler", which just happens to be the case for the three
    # commands that use it.
    from distutils.fancy_getopt import FancyGetopt

    compilers = sorted(
        ("compiler=" + compiler, None, compiler_class[compiler][2])
        for compiler in compiler_class.keys()
    )
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help("List of available compilers:")


def new_compiler(
    plat: str | None = None,
    compiler: str | None = None,
    verbose: bool = False,
    force: bool = False,
) -> Compiler:
    """Generate an instance of some CCompiler subclass for the supplied
    platform/compiler combination.  'plat' defaults to 'os.name'
    (eg. 'posix', 'nt'), and 'compiler' defaults to the default compiler
    for that platform.  Currently only 'posix' and 'nt' are supported, and
    the default compilers are "traditional Unix interface" (UnixCCompiler
    class) and Visual C++ (MSVCCompiler class).  Note that it's perfectly
    possible to ask for a Unix compiler object under Windows, and a
    Microsoft compiler object under Unix -- if you supply a value for
    'compiler', 'plat' is ignored.
    """
    if plat is None:
        plat = os.name

    try:
        if compiler is None:
            compiler = get_default_compiler(plat)

        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = f"don't know how to compile C/C++ code on platform '{plat}'"
        if compiler is not None:
            msg = msg + f" with '{compiler}' compiler"
        raise DistutilsPlatformError(msg)

    try:
        module_name = "distutils." + module_name
        __import__(module_name)
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except ImportError:
        raise DistutilsModuleError(
            f"can't compile C/C++ code: unable to load module '{module_name}'"
        )
    except KeyError:
        raise DistutilsModuleError(
            f"can't compile C/C++ code: unable to find class '{class_name}' "
            f"in module '{module_name}'"
        )

    # XXX The None is necessary to preserve backwards compatibility
    # with classes that expect verbose to be the first positional
    # argument.
    return klass(None, force=force)


def gen_preprocess_options(
    macros: Iterable[_Macro], include_dirs: Iterable[str]
) -> list[str]:
    """Generate C pre-processor options (-D, -U, -I) as used by at least
    two types of compilers: the typical Unix compiler and Visual C++.
    'macros' is the usual thing, a list of 1- or 2-tuples, where (name,)
    means undefine (-U) macro 'name', and (name,value) means define (-D)
    macro 'name' to 'value'.  'include_dirs' is just a list of directory
    names to be added to the header file search path (-I).  Returns a list
    of command-line options suitable for either Unix compilers or Visual
    C++.
    """
    # XXX it would be nice (mainly aesthetic, and so we don't generate
    # stupid-looking command lines) to go over 'macros' and eliminate
    # redundant definitions/undefinitions (ie. ensure that only the
    # latest mention of a particular macro winds up on the command
    # line).  I don't think it's essential, though, since most (all?)
    # Unix C compilers only pay attention to the latest -D or -U
    # mention of a macro on their command line.  Similar situation for
    # 'include_dirs'.  I'm punting on both for now.  Anyways, weeding out
    # redundancies like this should probably be the province of
    # CCompiler, since the data structures used are inherited from it
    # and therefore common to all CCompiler classes.
    pp_opts = []
    for macro in macros:
        if not (isinstance(macro, tuple) and 1 <= len(macro) <= 2):
            raise TypeError(
                f"bad macro definition '{macro}': "
                "each element of 'macros' list must be a 1- or 2-tuple"
            )

        if len(macro) == 1:  # undefine this macro
            pp_opts.append(f"-U{macro[0]}")
        elif len(macro) == 2:
            if macro[1] is None:  # define with no explicit value
                pp_opts.append(f"-D{macro[0]}")
            else:
                # XXX *don't* need to be clever about quoting the
                # macro value here, because we're going to avoid the
                # shell at all costs when we spawn the command!
                pp_opts.append("-D{}={}".format(*macro))

    pp_opts.extend(f"-I{dir}" for dir in include_dirs)
    return pp_opts


def gen_lib_options(
    compiler: Compiler,
    library_dirs: Iterable[str],
    runtime_library_dirs: Iterable[str],
    libraries: Iterable[str],
) -> list[str]:
    """Generate linker options for searching library directories and
    linking with specific libraries.  'libraries' and 'library_dirs' are,
    respectively, lists of library names (not filenames!) and search
    directories.  Returns a list of command-line options suitable for use
    with some compiler (depending on the two format strings passed in).
    """
    lib_opts = [compiler.library_dir_option(dir) for dir in library_dirs]

    for dir in runtime_library_dirs:
        lib_opts.extend(always_iterable(compiler.runtime_library_dir_option(dir)))

    # XXX it's important that we *not* remove redundant library mentions!
    # sometimes you really do have to say "-lfoo -lbar -lfoo" in order to
    # resolve all symbols.  I just hope we never have to say "-lfoo obj.o
    # -lbar" to get things to work -- that's certainly a possibility, but a
    # pretty nasty way to arrange your C code.

    for lib in libraries:
        (lib_dir, lib_name) = os.path.split(lib)
        if lib_dir:
            lib_file = compiler.find_library_file([lib_dir], lib_name)
            if lib_file:
                lib_opts.append(lib_file)
            else:
                compiler.warn(
                    f"no library file corresponding to '{lib}' found (skipping)"
                )
        else:
            lib_opts.append(compiler.library_option(lib))
    return lib_opts
