"""Filename globbing utility."""

import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys

__all__ = ["glob", "iglob", "escape"]

def glob(pathname, *, root_dir=None, dir_fd=None, recursive=False,
        include_hidden=False):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. Unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns by default.

    If `include_hidden` is true, the patterns '*', '?', '**'  will match hidden
    directories.

    If `recursive` is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive,
                      include_hidden=include_hidden))

def iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False,
          include_hidden=False):
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    sys.audit("glob.glob", pathname, recursive)
    sys.audit("glob.glob/2", pathname, recursive, root_dir, dir_fd)
    if root_dir is not None:
        root_dir = os.fspath(root_dir)
    else:
        root_dir = pathname[:0]
    it = _iglob(pathname, root_dir, dir_fd, recursive, False,
                include_hidden=include_hidden)
    if not pathname or recursive and _isrecursive(pathname[:2]):
        try:
            s = next(it)  # skip empty string
            if s:
                it = itertools.chain((s,), it)
        except StopIteration:
            pass
    return it

def _iglob(pathname, root_dir, dir_fd, recursive, dironly,
           include_hidden=False):
    dirname, basename = os.path.split(pathname)
    if not has_magic(pathname):
        assert not dironly
        if basename:
            if _lexists(_join(root_dir, pathname), dir_fd):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if _isdir(_join(root_dir, dirname), dir_fd):
                yield pathname
        return
    if not dirname:
        if recursive and _isrecursive(basename):
            yield from _glob2(root_dir, basename, dir_fd, dironly,
                             include_hidden=include_hidden)
        else:
            yield from _glob1(root_dir, basename, dir_fd, dironly,
                              include_hidden=include_hidden)
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, root_dir, dir_fd, recursive, True,
                      include_hidden=include_hidden)
    else:
        dirs = [dirname]
    if has_magic(basename):
        if recursive and _isrecursive(basename):
            glob_in_dir = _glob2
        else:
            glob_in_dir = _glob1
    else:
        glob_in_dir = _glob0
    for dirname in dirs:
        for name in glob_in_dir(_join(root_dir, dirname), basename, dir_fd, dironly,
                               include_hidden=include_hidden):
            yield os.path.join(dirname, name)

# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames.  _glob1 accepts a pattern while _glob0
# takes a literal basename (so it only has to check for its existence).

def _glob1(dirname, pattern, dir_fd, dironly, include_hidden=False):
    names = _listdir(dirname, dir_fd, dironly)
    if include_hidden or not _ishidden(pattern):
        names = (x for x in names if include_hidden or not _ishidden(x))
    return fnmatch.filter(names, pattern)

def _glob0(dirname, basename, dir_fd, dironly, include_hidden=False):
    if basename:
        if _lexists(_join(dirname, basename), dir_fd):
            return [basename]
    else:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if _isdir(dirname, dir_fd):
            return [basename]
    return []

# Following functions are not public but can be used by third-party code.

def glob0(dirname, pattern):
    return _glob0(dirname, pattern, None, False)

def glob1(dirname, pattern):
    return _glob1(dirname, pattern, None, False)

# This helper function recursively yields relative pathnames inside a literal
# directory.

def _glob2(dirname, pattern, dir_fd, dironly, include_hidden=False):
    assert _isrecursive(pattern)
    if not dirname or _isdir(dirname, dir_fd):
        yield pattern[:0]
    yield from _rlistdir(dirname, dir_fd, dironly,
                         include_hidden=include_hidden)

# If dironly is false, yields all file names inside a directory.
# If dironly is true, yields only directory names.
def _iterdir(dirname, dir_fd, dironly):
    try:
        fd = None
        fsencode = None
        if dir_fd is not None:
            if dirname:
                fd = arg = os.open(dirname, _dir_open_flags, dir_fd=dir_fd)
            else:
                arg = dir_fd
            if isinstance(dirname, bytes):
                fsencode = os.fsencode
        elif dirname:
            arg = dirname
        elif isinstance(dirname, bytes):
            arg = bytes(os.curdir, 'ASCII')
        else:
            arg = os.curdir
        try:
            with os.scandir(arg) as it:
                for entry in it:
                    try:
                        if not dironly or entry.is_dir():
                            if fsencode is not None:
                                yield fsencode(entry.name)
                            else:
                                yield entry.name
                    except OSError:
                        pass
        finally:
            if fd is not None:
                os.close(fd)
    except OSError:
        return

def _listdir(dirname, dir_fd, dironly):
    with contextlib.closing(_iterdir(dirname, dir_fd, dironly)) as it:
        return list(it)

# Recursively yields relative pathnames inside a literal directory.
def _rlistdir(dirname, dir_fd, dironly, include_hidden=False):
    names = _listdir(dirname, dir_fd, dironly)
    for x in names:
        if include_hidden or not _ishidden(x):
            yield x
            path = _join(dirname, x) if dirname else x
            for y in _rlistdir(path, dir_fd, dironly,
                               include_hidden=include_hidden):
                yield _join(x, y)


def _lexists(pathname, dir_fd):
    # Same as os.path.lexists(), but with dir_fd
    if dir_fd is None:
        return os.path.lexists(pathname)
    try:
        os.lstat(pathname, dir_fd=dir_fd)
    except (OSError, ValueError):
        return False
    else:
        return True

def _isdir(pathname, dir_fd):
    # Same as os.path.isdir(), but with dir_fd
    if dir_fd is None:
        return os.path.isdir(pathname)
    try:
        st = os.stat(pathname, dir_fd=dir_fd)
    except (OSError, ValueError):
        return False
    else:
        return stat.S_ISDIR(st.st_mode)

def _join(dirname, basename):
    # It is common if dirname or basename is empty
    if not dirname or not basename:
        return dirname or basename
    return os.path.join(dirname, basename)

magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')

def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None

def _ishidden(path):
    return path[0] in ('.', b'.'[0])

def _isrecursive(pattern):
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'

def escape(pathname):
    """Escape all special characters.
    """
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(br'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname


_dir_open_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
