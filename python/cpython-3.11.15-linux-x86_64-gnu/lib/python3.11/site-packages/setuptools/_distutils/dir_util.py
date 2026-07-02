"""distutils.dir_util

Utility functions for manipulating directories and directory trees."""

import functools
import itertools
import os
import pathlib

from . import file_util
from ._log import log
from .errors import DistutilsFileError, DistutilsInternalError


class SkipRepeatAbsolutePaths(set):
    """
    Cache for mkpath.

    In addition to cheapening redundant calls, eliminates redundant
    "creating /foo/bar/baz" messages in dry-run mode.
    """

    def __init__(self):
        SkipRepeatAbsolutePaths.instance = self

    @classmethod
    def clear(cls):
        super(cls, cls.instance).clear()

    def wrap(self, func):
        @functools.wraps(func)
        def wrapper(path, *args, **kwargs):
            if path.absolute() in self:
                return
            result = func(path, *args, **kwargs)
            self.add(path.absolute())
            return result

        return wrapper


# Python 3.8 compatibility
wrapper = SkipRepeatAbsolutePaths().wrap


@functools.singledispatch
@wrapper
def mkpath(name: pathlib.Path, mode=0o777, verbose=True) -> None:
    """Create a directory and any missing ancestor directories.

    If the directory already exists (or if 'name' is the empty string, which
    means the current directory, which of course exists), then do nothing.
    Raise DistutilsFileError if unable to create some directory along the way
    (eg. some sub-path exists, but is a file rather than a directory).
    If 'verbose' is true, log the directory created.
    """
    if verbose and not name.is_dir():
        log.info("creating %s", name)

    try:
        name.mkdir(mode=mode, parents=True, exist_ok=True)
    except OSError as exc:
        raise DistutilsFileError(f"could not create '{name}': {exc.args[-1]}")


@mkpath.register
def _(name: str, *args, **kwargs):
    return mkpath(pathlib.Path(name), *args, **kwargs)


@mkpath.register
def _(name: None, *args, **kwargs):
    """
    Detect a common bug -- name is None.
    """
    raise DistutilsInternalError(f"mkpath: 'name' must be a string (got {name!r})")


def create_tree(base_dir, files, mode=0o777, verbose=True):
    """Create all the empty directories under 'base_dir' needed to put 'files'
    there.

    'base_dir' is just the name of a directory which doesn't necessarily
    exist yet; 'files' is a list of filenames to be interpreted relative to
    'base_dir'.  'base_dir' + the directory portion of every file in 'files'
    will be created if it doesn't already exist.  'mode' and 'verbose'
    flags are as for 'mkpath()'.
    """
    # First get the list of directories to create
    need_dir = set(os.path.join(base_dir, os.path.dirname(file)) for file in files)

    # Now create them
    for dir in sorted(need_dir):
        mkpath(dir, mode, verbose=verbose)


def copy_tree(
    src,
    dst,
    preserve_mode=True,
    preserve_times=True,
    preserve_symlinks=False,
    update=False,
    verbose=True,
):
    """Copy an entire directory tree 'src' to a new location 'dst'.

    Both 'src' and 'dst' must be directory names.  If 'src' is not a
    directory, raise DistutilsFileError.  If 'dst' does not exist, it is
    created with 'mkpath()'.  The end result of the copy is that every
    file in 'src' is copied to 'dst', and directories under 'src' are
    recursively copied to 'dst'.  Return the list of files that were
    copied or might have been copied, using their output name.  The
    return value is unaffected by 'update': it is simply
    the list of all files under 'src', with the names changed to be
    under 'dst'.

    'preserve_mode' and 'preserve_times' are the same as for
    'copy_file'; note that they only apply to regular files, not to
    directories.  If 'preserve_symlinks' is true, symlinks will be
    copied as symlinks (on platforms that support them!); otherwise
    (the default), the destination of the symlink will be copied.
    'update' and 'verbose' are the same as for 'copy_file'.
    """
    if not os.path.isdir(src):
        raise DistutilsFileError(f"cannot copy tree '{src}': not a directory")
    try:
        names = os.listdir(src)
    except OSError as e:
        raise DistutilsFileError(f"error listing files in '{src}': {e.strerror}")

    mkpath(dst, verbose=verbose)

    copy_one = functools.partial(
        _copy_one,
        src=src,
        dst=dst,
        preserve_symlinks=preserve_symlinks,
        verbose=verbose,
        preserve_mode=preserve_mode,
        preserve_times=preserve_times,
        update=update,
    )
    return list(itertools.chain.from_iterable(map(copy_one, names)))


def _copy_one(
    name,
    *,
    src,
    dst,
    preserve_symlinks,
    verbose,
    preserve_mode,
    preserve_times,
    update,
):
    src_name = os.path.join(src, name)
    dst_name = os.path.join(dst, name)

    if name.startswith('.nfs'):
        # skip NFS rename files
        return

    if preserve_symlinks and os.path.islink(src_name):
        link_dest = os.readlink(src_name)
        if verbose >= 1:
            log.info("linking %s -> %s", dst_name, link_dest)
        os.symlink(link_dest, dst_name)
        yield dst_name

    elif os.path.isdir(src_name):
        yield from copy_tree(
            src_name,
            dst_name,
            preserve_mode,
            preserve_times,
            preserve_symlinks,
            update,
            verbose=verbose,
        )
    else:
        file_util.copy_file(
            src_name,
            dst_name,
            preserve_mode,
            preserve_times,
            update,
            verbose=verbose,
        )
        yield dst_name


def _build_cmdtuple(path, cmdtuples):
    """Helper for remove_tree()."""
    for f in os.listdir(path):
        real_f = os.path.join(path, f)
        if os.path.isdir(real_f) and not os.path.islink(real_f):
            _build_cmdtuple(real_f, cmdtuples)
        else:
            cmdtuples.append((os.remove, real_f))
    cmdtuples.append((os.rmdir, path))


def remove_tree(directory, verbose=True):
    """Recursively remove an entire directory tree.

    Any errors are ignored (apart from being reported to stdout if 'verbose'
    is true).
    """
    if verbose >= 1:
        log.info("removing '%s' (and everything under it)", directory)
    cmdtuples = []
    _build_cmdtuple(directory, cmdtuples)
    for cmd in cmdtuples:
        try:
            cmd[0](cmd[1])
            # Clear the cache
            SkipRepeatAbsolutePaths.clear()
        except OSError as exc:
            log.warning("error removing %s: %s", directory, exc)


def ensure_relative(path):
    """Take the full path 'path', and make it a relative path.

    This is useful to make 'path' the second argument to os.path.join().
    """
    drive, path = os.path.splitdrive(path)
    if path[0:1] == os.sep:
        path = drive + path[1:]
    return path
