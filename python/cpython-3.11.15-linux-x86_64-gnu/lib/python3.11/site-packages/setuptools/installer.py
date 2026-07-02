from __future__ import annotations

import glob
import itertools
import os
import subprocess
import sys
import tempfile

import packaging.requirements
import packaging.utils

from . import _reqs
from ._importlib import metadata
from .warnings import SetuptoolsDeprecationWarning
from .wheel import Wheel

from distutils import log
from distutils.errors import DistutilsError


def _fixup_find_links(find_links):
    """Ensure find-links option end-up being a list of strings."""
    if isinstance(find_links, str):
        return find_links.split()
    assert isinstance(find_links, (tuple, list))
    return find_links


def fetch_build_egg(dist, req) -> metadata.Distribution | metadata.PathDistribution:
    """Fetch an egg needed for building.

    Use pip/wheel to fetch/build a wheel."""
    _DeprecatedInstaller.emit()
    _warn_wheel_not_available(dist)
    return _fetch_build_egg_no_warn(dist, req)


def _present(req):
    return any(_dist_matches_req(dist, req) for dist in metadata.distributions())


def _fetch_build_eggs(dist, requires: _reqs._StrOrIter) -> list[metadata.Distribution]:
    _DeprecatedInstaller.emit(stacklevel=3)
    _warn_wheel_not_available(dist)

    parsed_reqs = _reqs.parse(requires)

    missing_reqs = itertools.filterfalse(_present, parsed_reqs)

    needed_reqs = (
        req for req in missing_reqs if not req.marker or req.marker.evaluate()
    )
    resolved_dists = [_fetch_build_egg_no_warn(dist, req) for req in needed_reqs]
    for dist in resolved_dists:
        # dist.locate_file('') is the directory containing EGG-INFO, where the importabl
        # contents can be found.
        sys.path.insert(0, str(dist.locate_file('')))
    return resolved_dists


def _dist_matches_req(egg_dist, req):
    return (
        packaging.utils.canonicalize_name(egg_dist.name)
        == packaging.utils.canonicalize_name(req.name)
        and egg_dist.version in req.specifier
    )


def _fetch_build_egg_no_warn(dist, req):  # noqa: C901  # is too complex (16)  # FIXME
    # Ignore environment markers; if supplied, it is required.
    req = strip_marker(req)
    # Take easy_install options into account, but do not override relevant
    # pip environment variables (like PIP_INDEX_URL or PIP_QUIET); they'll
    # take precedence.
    opts = dist.get_option_dict('easy_install')
    if 'allow_hosts' in opts:
        raise DistutilsError(
            'the `allow-hosts` option is not supported '
            'when using pip to install requirements.'
        )
    quiet = 'PIP_QUIET' not in os.environ and 'PIP_VERBOSE' not in os.environ
    if 'PIP_INDEX_URL' in os.environ:
        index_url = None
    elif 'index_url' in opts:
        index_url = opts['index_url'][1]
    else:
        index_url = None
    find_links = (
        _fixup_find_links(opts['find_links'][1])[:] if 'find_links' in opts else []
    )
    if dist.dependency_links:
        find_links.extend(dist.dependency_links)
    eggs_dir = os.path.realpath(dist.get_egg_cache_dir())
    cached_dists = metadata.Distribution.discover(path=glob.glob(f'{eggs_dir}/*.egg'))
    for egg_dist in cached_dists:
        if _dist_matches_req(egg_dist, req):
            return egg_dist
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            '-m',
            'pip',
            '--disable-pip-version-check',
            'wheel',
            '--no-deps',
            '-w',
            tmpdir,
        ]
        if quiet:
            cmd.append('--quiet')
        if index_url is not None:
            cmd.extend(('--index-url', index_url))
        for link in find_links or []:
            cmd.extend(('--find-links', link))
        # If requirement is a PEP 508 direct URL, directly pass
        # the URL to pip, as `req @ url` does not work on the
        # command line.
        cmd.append(req.url or str(req))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise DistutilsError(str(e)) from e
        wheel = Wheel(glob.glob(os.path.join(tmpdir, '*.whl'))[0])
        dist_location = os.path.join(eggs_dir, wheel.egg_name())
        wheel.install_as_egg(dist_location)
        return metadata.Distribution.at(dist_location + '/EGG-INFO')


def strip_marker(req) -> packaging.requirements.Requirement:
    """
    Return a new requirement without the environment marker to avoid
    calling pip with something like `babel; extra == "i18n"`, which
    would always be ignored.
    """
    # create a copy to avoid mutating the input
    req = packaging.requirements.Requirement(str(req))
    req.marker = None
    return req


def _warn_wheel_not_available(dist):
    try:
        metadata.distribution('wheel')
    except metadata.PackageNotFoundError:
        dist.announce('WARNING: The wheel package is not available.', log.WARN)


class _DeprecatedInstaller(SetuptoolsDeprecationWarning):
    _SUMMARY = "setuptools.installer and fetch_build_eggs are deprecated."
    _DETAILS = """
    Requirements should be satisfied by a PEP 517 installer.
    If you are using pip, you can try `pip install --use-pep517`.
    """
    _DUE_DATE = 2025, 10, 31
