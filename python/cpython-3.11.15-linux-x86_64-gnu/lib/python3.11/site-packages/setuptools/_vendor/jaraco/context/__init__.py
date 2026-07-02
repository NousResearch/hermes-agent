from __future__ import annotations

import contextlib
import errno
import functools
import operator
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator

if sys.version_info < (3, 12):
    from backports import tarfile
else:
    import tarfile


@contextlib.contextmanager
def pushd(dir: str | os.PathLike) -> Iterator[str | os.PathLike]:
    """
    >>> tmp_path = getfixture('tmp_path')
    >>> with pushd(tmp_path):
    ...     assert os.getcwd() == os.fspath(tmp_path)
    >>> assert os.getcwd() != os.fspath(tmp_path)
    """

    orig = os.getcwd()
    os.chdir(dir)
    try:
        yield dir
    finally:
        os.chdir(orig)


@contextlib.contextmanager
def tarball(
    url, target_dir: str | os.PathLike | None = None
) -> Iterator[str | os.PathLike]:
    """
    Get a URL to a tarball, download, extract, yield, then clean up.

    Assumes everything in the tarball is prefixed with a common
    directory. That common path is stripped and the contents
    are extracted to ``target_dir``, similar to passing
    ``-C {target} --strip-components 1`` to the ``tar`` command.

    Uses the streaming protocol to extract the contents from a
    stream in a single pass without loading the whole file into
    memory.

    >>> import urllib.request
    >>> url = getfixture('tarfile_served')
    >>> target = getfixture('tmp_path') / 'out'
    >>> tb = tarball(url, target_dir=target)
    >>> import pathlib
    >>> with tb as extracted:
    ...     contents = pathlib.Path(extracted, 'contents.txt').read_text(encoding='utf-8')
    >>> assert not os.path.exists(extracted)

    If the target is not specified, contents are extracted to a
    directory relative to the current working directory named after
    the name of the file as extracted from the URL.

    >>> target = getfixture('tmp_path')
    >>> with pushd(target), tarball(url):
    ...     target.joinpath('served').is_dir()
    True
    """
    if target_dir is None:
        target_dir = os.path.basename(url).replace('.tar.gz', '').replace('.tgz', '')
    os.mkdir(target_dir)
    try:
        req = urllib.request.urlopen(url)
        with tarfile.open(fileobj=req, mode='r|*') as tf:
            tf.extractall(path=target_dir, filter=_default_filter)
        yield target_dir
    finally:
        shutil.rmtree(target_dir)


def _compose_tarfile_filters(*filters):
    def compose_two(f1, f2):
        return lambda member, path: f1(f2(member, path), path)

    return functools.reduce(compose_two, filters, lambda member, path: member)


def strip_first_component(
    member: tarfile.TarInfo,
    path,
) -> tarfile.TarInfo:
    _, member.name = member.name.split('/', 1)
    return member


_default_filter = _compose_tarfile_filters(tarfile.data_filter, strip_first_component)


def _compose(*cmgrs):
    """
    Compose any number of dependent context managers into a single one.

    The last, innermost context manager may take arbitrary arguments, but
    each successive context manager should accept the result from the
    previous as a single parameter.

    Like :func:`jaraco.functools.compose`, behavior works from right to
    left, so the context manager should be indicated from outermost to
    innermost.

    Example, to create a context manager to change to a temporary
    directory:

    >>> temp_dir_as_cwd = _compose(pushd, temp_dir)
    >>> with temp_dir_as_cwd() as dir:
    ...     assert os.path.samefile(os.getcwd(), dir)
    """

    def compose_two(inner, outer):
        def composed(*args, **kwargs):
            with inner(*args, **kwargs) as saved, outer(saved) as res:
                yield res

        return contextlib.contextmanager(composed)

    return functools.reduce(compose_two, reversed(cmgrs))


tarball_cwd = _compose(pushd, tarball)
"""
A tarball context with the current working directory pointing to the contents.
"""


def remove_readonly(func, path, exc_info):
    """
    Add support for removing read-only files on Windows.
    """
    _, exc, _ = exc_info
    if func in (os.rmdir, os.remove, os.unlink) and exc.errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # retry
        func(path)
    else:
        raise


def robust_remover():
    return (
        functools.partial(shutil.rmtree, onerror=remove_readonly)
        if platform.system() == 'Windows'
        else shutil.rmtree
    )


@contextlib.contextmanager
def temp_dir(remover=shutil.rmtree):
    """
    Create a temporary directory context. Pass a custom remover
    to override the removal behavior.

    >>> import pathlib
    >>> with temp_dir() as the_dir:
    ...     assert os.path.isdir(the_dir)
    >>> assert not os.path.exists(the_dir)
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        remover(temp_dir)


robust_temp_dir = functools.partial(temp_dir, remover=robust_remover())


@contextlib.contextmanager
def repo_context(
    url, branch: str | None = None, quiet: bool = True, dest_ctx=robust_temp_dir
):
    """
    Check out the repo indicated by url.

    If dest_ctx is supplied, it should be a context manager
    to yield the target directory for the check out.

    >>> getfixture('ensure_git')
    >>> getfixture('needs_internet')
    >>> repo = repo_context('https://github.com/jaraco/jaraco.context')
    >>> with repo as dest:
    ...     listing = os.listdir(dest)
    >>> 'README.rst' in listing
    True
    """
    exe = 'git' if 'git' in url else 'hg'
    with dest_ctx() as repo_dir:
        cmd = [exe, 'clone', url, repo_dir]
        cmd.extend(['--branch', branch] * bool(branch))
        stream = subprocess.DEVNULL if quiet else None
        subprocess.check_call(cmd, stdout=stream, stderr=stream)
        yield repo_dir


class ExceptionTrap:
    """
    A context manager that will catch certain exceptions and provide an
    indication they occurred.

    >>> with ExceptionTrap() as trap:
    ...     raise Exception()
    >>> bool(trap)
    True

    >>> with ExceptionTrap() as trap:
    ...     pass
    >>> bool(trap)
    False

    >>> with ExceptionTrap(ValueError) as trap:
    ...     raise ValueError("1 + 1 is not 3")
    >>> bool(trap)
    True
    >>> trap.value
    ValueError('1 + 1 is not 3')
    >>> trap.tb
    <traceback object at ...>

    >>> with ExceptionTrap(ValueError) as trap:
    ...     raise Exception()
    Traceback (most recent call last):
    ...
    Exception

    >>> bool(trap)
    False
    """

    exc_info = None, None, None

    def __init__(self, exceptions=(Exception,)):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    @property
    def type(self):
        return self.exc_info[0]

    @property
    def value(self):
        return self.exc_info[1]

    @property
    def tb(self):
        return self.exc_info[2]

    def __exit__(self, *exc_info):
        type = exc_info[0]
        matches = type and issubclass(type, self.exceptions)
        if matches:
            self.exc_info = exc_info
        return matches

    def __bool__(self):
        return bool(self.type)

    def raises(self, func, *, _test=bool):
        """
        Wrap func and replace the result with the truth
        value of the trap (True if an exception occurred).

        First, give the decorator an alias to support Python 3.8
        Syntax.

        >>> raises = ExceptionTrap(ValueError).raises

        Now decorate a function that always fails.

        >>> @raises
        ... def fail():
        ...     raise ValueError('failed')
        >>> fail()
        True
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ExceptionTrap(self.exceptions) as trap:
                func(*args, **kwargs)
            return _test(trap)

        return wrapper

    def passes(self, func):
        """
        Wrap func and replace the result with the truth
        value of the trap (True if no exception).

        First, give the decorator an alias to support Python 3.8
        Syntax.

        >>> passes = ExceptionTrap(ValueError).passes

        Now decorate a function that always fails.

        >>> @passes
        ... def fail():
        ...     raise ValueError('failed')

        >>> fail()
        False
        """
        return self.raises(func, _test=operator.not_)


class suppress(contextlib.suppress, contextlib.ContextDecorator):
    """
    A version of contextlib.suppress with decorator support.

    >>> @suppress(KeyError)
    ... def key_error():
    ...     {}['']
    >>> key_error()
    """


class on_interrupt(contextlib.ContextDecorator):
    """
    Replace a KeyboardInterrupt with SystemExit(1).

    Useful in conjunction with console entry point functions.

    >>> def do_interrupt():
    ...     raise KeyboardInterrupt()
    >>> on_interrupt('error')(do_interrupt)()
    Traceback (most recent call last):
    ...
    SystemExit: 1
    >>> on_interrupt('error', code=255)(do_interrupt)()
    Traceback (most recent call last):
    ...
    SystemExit: 255
    >>> on_interrupt('suppress')(do_interrupt)()
    >>> with __import__('pytest').raises(KeyboardInterrupt):
    ...     on_interrupt('ignore')(do_interrupt)()
    """

    def __init__(self, action='error', /, code=1):
        self.action = action
        self.code = code

    def __enter__(self):
        return self

    def __exit__(self, exctype, excinst, exctb):
        if exctype is not KeyboardInterrupt or self.action == 'ignore':
            return
        elif self.action == 'error':
            raise SystemExit(self.code) from excinst
        return self.action == 'suppress'
