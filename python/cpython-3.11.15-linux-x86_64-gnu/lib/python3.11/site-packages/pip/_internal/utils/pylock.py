from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

from pip._vendor.packaging.pylock import (
    Package,
    PackageArchive,
    PackageDirectory,
    PackageSdist,
    PackageVcs,
    PackageWheel,
    Pylock,
    is_valid_pylock_path,
)
from pip._vendor.packaging.version import Version

from pip._internal.exceptions import InstallationError
from pip._internal.models.link import Link
from pip._internal.utils.compat import tomllib
from pip._internal.utils.urls import path_to_url, url_to_path

if TYPE_CHECKING:
    from pip._internal.network.session import PipSession
    from pip._internal.req.req_install import InstallRequirement


def _pylock_package_from_install_requirement(
    ireq: InstallRequirement, base_dir: Path
) -> Package:
    base_dir = base_dir.resolve()
    dist = ireq.get_dist()
    download_info = ireq.download_info
    assert download_info
    package_version = None
    package_vcs = None
    package_directory = None
    package_archive = None
    package_sdist = None
    package_wheels = None
    if ireq.is_direct:
        if download_info.vcs_info:
            package_vcs = PackageVcs(
                type=download_info.vcs_info.vcs,
                url=download_info.url,
                path=None,
                requested_revision=download_info.vcs_info.requested_revision,
                commit_id=download_info.vcs_info.commit_id,
                subdirectory=download_info.subdirectory,
            )
        elif download_info.dir_info:
            package_directory = PackageDirectory(
                path=(
                    Path(url_to_path(download_info.url))
                    .resolve()
                    .relative_to(base_dir)
                    .as_posix()
                ),
                editable=(
                    download_info.dir_info.editable
                    if download_info.dir_info.editable
                    else None
                ),
                subdirectory=download_info.subdirectory,
            )
        elif download_info.archive_info:
            if not download_info.archive_info.hashes:
                raise NotImplementedError()
            package_archive = PackageArchive(
                url=download_info.url,
                path=None,
                hashes=download_info.archive_info.hashes,
                subdirectory=download_info.subdirectory,
            )
        else:
            # should never happen
            raise NotImplementedError()
    else:
        package_version = dist.version
        if download_info.archive_info:
            if not download_info.archive_info.hashes:
                raise NotImplementedError()
            link = Link(download_info.url)
            if link.is_wheel:
                package_wheels = [
                    PackageWheel(
                        name=link.filename,
                        url=download_info.url,
                        hashes=download_info.archive_info.hashes,
                    )
                ]
            else:
                package_sdist = PackageSdist(
                    name=link.filename,
                    url=download_info.url,
                    hashes=download_info.archive_info.hashes,
                )
        else:
            # should never happen
            raise NotImplementedError()
    return Package(
        name=dist.canonical_name,
        version=package_version,
        vcs=package_vcs,
        directory=package_directory,
        archive=package_archive,
        sdist=package_sdist,
        wheels=package_wheels,
    )


def pylock_from_install_requirements(
    install_requirements: Iterable[InstallRequirement], base_dir: Path
) -> Pylock:
    return Pylock(
        lock_version=Version("1.0"),
        created_by="pip",
        packages=sorted(
            (
                _pylock_package_from_install_requirement(ireq, base_dir)
                for ireq in install_requirements
            ),
            key=lambda p: p.name,
        ),
    )


_SCHEME_RE = re.compile("^(http|https|file)://", re.IGNORECASE)


def _is_url(s: str) -> bool:
    return bool(_SCHEME_RE.match(s))


def is_valid_pylock_filename(filename: str) -> bool:
    if _is_url(filename):
        path = Path(urlsplit(filename).path.rpartition("/")[-1])
    else:
        path = Path(filename)
    return is_valid_pylock_path(path)


def _package_dist_url(
    pylock_path_or_url: str, path: str | None, url: str | None
) -> str:
    """Compute an url from a Pylock package path and url.

    Give priority to path over url. If path is relative,
    compute an url using the pylock file location as base.
    """
    if path is not None:
        if not os.path.isabs(path):
            # relative path, join to pylock location
            if _is_url(pylock_path_or_url):
                return urljoin(pylock_path_or_url, path)
            else:
                return path_to_url(
                    os.path.join(os.path.dirname(pylock_path_or_url), path)
                )
        else:
            # absolute path, reject if pylock comes from a URL
            if _is_url(pylock_path_or_url):
                raise InstallationError(
                    f"Absolute paths are not supported in pylock files obtained "
                    f"from a URL: {path!r} in {pylock_path_or_url!r}"
                )
            return path_to_url(path)
    else:
        assert url is not None  # guaranteed by packaging.pylock validation
        return url


def package_vcs_requirement_url(
    pylock_path_or_url: str, package_vcs: PackageVcs
) -> str:
    dist_url = _package_dist_url(pylock_path_or_url, package_vcs.path, package_vcs.url)
    url = f"{package_vcs.type}+{dist_url}@{package_vcs.commit_id}"
    if package_vcs.subdirectory:
        if "#" in url:
            raise InstallationError(
                f"Package URL {url!r} cannot contain fragments in combination "
                f"with subdirectory field (in {pylock_path_or_url!r})"
            )
        url += "#subdirectory=" + package_vcs.subdirectory
    return url


def package_archive_requirement_url(
    pylock_path_or_url: str, package_archive: PackageArchive
) -> str:
    url = _package_dist_url(
        pylock_path_or_url, package_archive.path, package_archive.url
    )
    if package_archive.subdirectory:
        if "#" in url:
            raise InstallationError(
                f"Package URL {url!r} cannot contain fragments in combination "
                f"with subdirectory field (in {pylock_path_or_url!r})"
            )
        url += "#subdirectory=" + package_archive.subdirectory
    return url


def package_directory_requirement_url(
    pylock_path_or_url: str, package_directory: PackageDirectory
) -> str:
    if _is_url(pylock_path_or_url) and not pylock_path_or_url.startswith("file://"):
        raise InstallationError(
            f"Directory entries are not supported in remote pylock.toml "
            f"{pylock_path_or_url!r}"
        )
    url = _package_dist_url(pylock_path_or_url, package_directory.path, None)
    assert url.startswith("file://")
    if not url.endswith("/"):
        url += "/"
    if package_directory.subdirectory:
        url += package_directory.subdirectory
        if not url.endswith("/"):
            url += "/"
    return url


def package_sdist_requirement_url(
    pylock_path_or_url: str, package_sdist: PackageSdist
) -> str:
    return _package_dist_url(pylock_path_or_url, package_sdist.path, package_sdist.url)


def package_wheel_requirement_url(
    pylock_path_or_url: str, package_wheel: PackageWheel
) -> str:
    return _package_dist_url(pylock_path_or_url, package_wheel.path, package_wheel.url)


def _get_pylock_path_or_url_content(path_or_url: str, session: PipSession) -> str:
    # TODO: refactor - this is similar to req_file.get_file_content
    scheme = urlsplit(path_or_url).scheme
    # Pip has special support for file:// URLs (LocalFSAdapter).
    if scheme in ["http", "https", "file"]:
        # Delay importing heavy network modules until absolutely necessary.
        from pip._internal.network.utils import raise_for_status

        resp = session.get(path_or_url)
        raise_for_status(resp)
        return resp.text

    # Assume this is a bare path.
    return Path(path_or_url).read_text(encoding="utf-8")


def select_from_pylock_path_or_url(
    pylock_path_or_url: str,
    session: PipSession,
) -> Iterator[
    tuple[
        Package,
        PackageVcs | PackageDirectory | PackageArchive | PackageWheel | PackageSdist,
    ]
]:
    try:
        pylock_content = _get_pylock_path_or_url_content(pylock_path_or_url, session)
    except Exception as exc:
        raise InstallationError(
            f"Error reading pylock file {pylock_path_or_url!r}: {exc}"
        ) from exc

    try:
        lock = Pylock.from_dict(tomllib.loads(pylock_content))
    except Exception as exc:
        raise InstallationError(
            f"Invalid pylock file {pylock_path_or_url!r}: {exc}"
        ) from exc

    try:
        yield from lock.select()
    except Exception as exc:
        raise InstallationError(
            f"Cannot select requirements from pylock file {pylock_path_or_url!r}: {exc}"
        ) from exc
