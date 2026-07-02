from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
)

from .markers import Marker
from .specifiers import SpecifierSet
from .utils import NormalizedName, is_normalized_name
from .version import Version

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from typing_extensions import Self

_logger = logging.getLogger(__name__)

__all__ = [
    "Package",
    "PackageArchive",
    "PackageDirectory",
    "PackageSdist",
    "PackageVcs",
    "PackageWheel",
    "Pylock",
    "PylockUnsupportedVersionError",
    "PylockValidationError",
    "is_valid_pylock_path",
]

_T = TypeVar("_T")
_T2 = TypeVar("_T2")


class _FromMappingProtocol(Protocol):  # pragma: no cover
    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self: ...


_FromMappingProtocolT = TypeVar("_FromMappingProtocolT", bound=_FromMappingProtocol)


_PYLOCK_FILE_NAME_RE = re.compile(r"^pylock\.([^.]+)\.toml$")


def is_valid_pylock_path(path: Path) -> bool:
    """Check if the given path is a valid pylock file path."""
    return path.name == "pylock.toml" or bool(_PYLOCK_FILE_NAME_RE.match(path.name))


def _toml_key(key: str) -> str:
    return key.replace("_", "-")


def _toml_value(key: str, value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, (Version, Marker, SpecifierSet)):
        return str(value)
    if isinstance(value, Sequence) and key == "environments":
        return [str(v) for v in value]
    return value


def _toml_dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
    return {
        _toml_key(key): _toml_value(key, value)
        for key, value in data
        if value is not None
    }


def _get(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T | None:
    """Get a value from the dictionary and verify it's the expected type."""
    if (value := d.get(key)) is None:
        return None
    if not isinstance(value, expected_type):
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} "
            f"(expected {expected_type.__name__})",
            context=key,
        )
    return value


def _get_required(d: Mapping[str, Any], expected_type: type[_T], key: str) -> _T:
    """Get a required value from the dictionary and verify it's the expected type."""
    if (value := _get(d, expected_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence(
    d: Mapping[str, Any], expected_item_type: type[_T], key: str
) -> Sequence[_T] | None:
    """Get a list value from the dictionary and verify it's the expected items type."""
    if (value := _get(d, Sequence, key)) is None:  # type: ignore[type-abstract]
        return None
    if isinstance(value, (str, bytes)):
        # special case: str and bytes are Sequences, but we want to reject it
        raise PylockValidationError(
            f"Unexpected type {type(value).__name__} (expected Sequence)",
            context=key,
        )
    for i, item in enumerate(value):
        if not isinstance(item, expected_item_type):
            raise PylockValidationError(
                f"Unexpected type {type(item).__name__} "
                f"(expected {expected_item_type.__name__})",
                context=f"{key}[{i}]",
            )
    return value


def _get_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2 | None:
    """Get a value from the dictionary, verify it's the expected type,
    and convert to the target type.

    This assumes the target_type constructor accepts the value.
    """
    if (value := _get(d, expected_type, key)) is None:
        return None
    try:
        return target_type(value)
    except Exception as e:
        raise PylockValidationError(e, context=key) from e


def _get_required_as(
    d: Mapping[str, Any],
    expected_type: type[_T],
    target_type: Callable[[_T], _T2],
    key: str,
) -> _T2:
    """Get a required value from the dict, verify it's the expected type,
    and convert to the target type."""
    if (value := _get_as(d, expected_type, target_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return value


def _get_sequence_as(
    d: Mapping[str, Any],
    expected_item_type: type[_T],
    target_item_type: Callable[[_T], _T2],
    key: str,
) -> list[_T2] | None:
    """Get list value from dictionary and verify expected items type."""
    if (value := _get_sequence(d, expected_item_type, key)) is None:
        return None
    result = []
    try:
        for item in value:
            typed_item = target_item_type(item)
            result.append(typed_item)
    except Exception as e:
        raise PylockValidationError(e, context=f"{key}[{len(result)}]") from e
    return result


def _get_object(
    d: Mapping[str, Any], target_type: type[_FromMappingProtocolT], key: str
) -> _FromMappingProtocolT | None:
    """Get a dictionary value from the dictionary and convert it to a dataclass."""
    if (value := _get(d, Mapping, key)) is None:  # type: ignore[type-abstract]
        return None
    try:
        return target_type._from_dict(value)
    except Exception as e:
        raise PylockValidationError(e, context=key) from e


def _get_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> list[_FromMappingProtocolT] | None:
    """Get a list value from the dictionary and convert its items to a dataclass."""
    if (value := _get_sequence(d, Mapping, key)) is None:  # type: ignore[type-abstract]
        return None
    result: list[_FromMappingProtocolT] = []
    try:
        for item in value:
            typed_item = target_item_type._from_dict(item)
            result.append(typed_item)
    except Exception as e:
        raise PylockValidationError(e, context=f"{key}[{len(result)}]") from e
    return result


def _get_required_sequence_of_objects(
    d: Mapping[str, Any], target_item_type: type[_FromMappingProtocolT], key: str
) -> Sequence[_FromMappingProtocolT]:
    """Get a required list value from the dictionary and convert its items to a
    dataclass."""
    if (result := _get_sequence_of_objects(d, target_item_type, key)) is None:
        raise _PylockRequiredKeyError(key)
    return result


def _validate_normalized_name(name: str) -> NormalizedName:
    """Validate that a string is a NormalizedName."""
    if not is_normalized_name(name):
        raise PylockValidationError(f"Name {name!r} is not normalized")
    return NormalizedName(name)


def _validate_path_url(path: str | None, url: str | None) -> None:
    if not path and not url:
        raise PylockValidationError("path or url must be provided")


def _validate_hashes(hashes: Mapping[str, Any]) -> Mapping[str, Any]:
    if not hashes:
        raise PylockValidationError("At least one hash must be provided")
    if not all(isinstance(hash_val, str) for hash_val in hashes.values()):
        raise PylockValidationError("Hash values must be strings")
    return hashes


class PylockValidationError(Exception):
    """Raised when when input data is not spec-compliant."""

    context: str | None = None
    message: str

    def __init__(
        self,
        cause: str | Exception,
        *,
        context: str | None = None,
    ) -> None:
        if isinstance(cause, PylockValidationError):
            if cause.context:
                self.context = (
                    f"{context}.{cause.context}" if context else cause.context
                )
            else:
                self.context = context
            self.message = cause.message
        else:
            self.context = context
            self.message = str(cause)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} in {self.context!r}"
        return self.message


class _PylockRequiredKeyError(PylockValidationError):
    def __init__(self, key: str) -> None:
        super().__init__("Missing required value", context=key)


class PylockUnsupportedVersionError(PylockValidationError):
    """Raised when encountering an unsupported `lock_version`."""


@dataclass(frozen=True, init=False)
class PackageVcs:
    type: str
    url: str | None = None
    path: str | None = None
    requested_revision: str | None = None
    commit_id: str  # type: ignore[misc]
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        type: str,
        url: str | None = None,
        path: str | None = None,
        requested_revision: str | None = None,
        commit_id: str,
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "requested_revision", requested_revision)
        object.__setattr__(self, "commit_id", commit_id)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_vcs = cls(
            type=_get_required(d, str, "type"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            requested_revision=_get(d, str, "requested-revision"),
            commit_id=_get_required(d, str, "commit-id"),
            subdirectory=_get(d, str, "subdirectory"),
        )
        _validate_path_url(package_vcs.path, package_vcs.url)
        return package_vcs


@dataclass(frozen=True, init=False)
class PackageDirectory:
    path: str
    editable: bool | None = None
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        path: str,
        editable: bool | None = None,
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "editable", editable)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls(
            path=_get_required(d, str, "path"),
            editable=_get(d, bool, "editable"),
            subdirectory=_get(d, str, "subdirectory"),
        )


@dataclass(frozen=True, init=False)
class PackageArchive:
    url: str | None = None
    path: str | None = None
    size: int | None = None
    upload_time: datetime | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]
    subdirectory: str | None = None

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        upload_time: datetime | None = None,
        hashes: Mapping[str, str],
        subdirectory: str | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "hashes", hashes)
        object.__setattr__(self, "subdirectory", subdirectory)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_archive = cls(
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            upload_time=_get(d, datetime, "upload-time"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
            subdirectory=_get(d, str, "subdirectory"),
        )
        _validate_path_url(package_archive.path, package_archive.url)
        return package_archive


@dataclass(frozen=True, init=False)
class PackageSdist:
    name: str | None = None
    upload_time: datetime | None = None
    url: str | None = None
    path: str | None = None
    size: int | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]

    def __init__(
        self,
        *,
        name: str | None = None,
        upload_time: datetime | None = None,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        hashes: Mapping[str, str],
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "hashes", hashes)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_sdist = cls(
            name=_get(d, str, "name"),
            upload_time=_get(d, datetime, "upload-time"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
        )
        _validate_path_url(package_sdist.path, package_sdist.url)
        return package_sdist


@dataclass(frozen=True, init=False)
class PackageWheel:
    name: str | None = None
    upload_time: datetime | None = None
    url: str | None = None
    path: str | None = None
    size: int | None = None
    hashes: Mapping[str, str]  # type: ignore[misc]

    def __init__(
        self,
        *,
        name: str | None = None,
        upload_time: datetime | None = None,
        url: str | None = None,
        path: str | None = None,
        size: int | None = None,
        hashes: Mapping[str, str],
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "upload_time", upload_time)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "hashes", hashes)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package_wheel = cls(
            name=_get(d, str, "name"),
            upload_time=_get(d, datetime, "upload-time"),
            url=_get(d, str, "url"),
            path=_get(d, str, "path"),
            size=_get(d, int, "size"),
            hashes=_get_required_as(d, Mapping, _validate_hashes, "hashes"),  # type: ignore[type-abstract]
        )
        _validate_path_url(package_wheel.path, package_wheel.url)
        return package_wheel


@dataclass(frozen=True, init=False)
class Package:
    name: NormalizedName
    version: Version | None = None
    marker: Marker | None = None
    requires_python: SpecifierSet | None = None
    dependencies: Sequence[Mapping[str, Any]] | None = None
    vcs: PackageVcs | None = None
    directory: PackageDirectory | None = None
    archive: PackageArchive | None = None
    index: str | None = None
    sdist: PackageSdist | None = None
    wheels: Sequence[PackageWheel] | None = None
    attestation_identities: Sequence[Mapping[str, Any]] | None = None
    tool: Mapping[str, Any] | None = None

    def __init__(
        self,
        *,
        name: NormalizedName,
        version: Version | None = None,
        marker: Marker | None = None,
        requires_python: SpecifierSet | None = None,
        dependencies: Sequence[Mapping[str, Any]] | None = None,
        vcs: PackageVcs | None = None,
        directory: PackageDirectory | None = None,
        archive: PackageArchive | None = None,
        index: str | None = None,
        sdist: PackageSdist | None = None,
        wheels: Sequence[PackageWheel] | None = None,
        attestation_identities: Sequence[Mapping[str, Any]] | None = None,
        tool: Mapping[str, Any] | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "marker", marker)
        object.__setattr__(self, "requires_python", requires_python)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "vcs", vcs)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "archive", archive)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "sdist", sdist)
        object.__setattr__(self, "wheels", wheels)
        object.__setattr__(self, "attestation_identities", attestation_identities)
        object.__setattr__(self, "tool", tool)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        package = cls(
            name=_get_required_as(d, str, _validate_normalized_name, "name"),
            version=_get_as(d, str, Version, "version"),
            requires_python=_get_as(d, str, SpecifierSet, "requires-python"),
            dependencies=_get_sequence(d, Mapping, "dependencies"),  # type: ignore[type-abstract]
            marker=_get_as(d, str, Marker, "marker"),
            vcs=_get_object(d, PackageVcs, "vcs"),
            directory=_get_object(d, PackageDirectory, "directory"),
            archive=_get_object(d, PackageArchive, "archive"),
            index=_get(d, str, "index"),
            sdist=_get_object(d, PackageSdist, "sdist"),
            wheels=_get_sequence_of_objects(d, PackageWheel, "wheels"),
            attestation_identities=_get_sequence(d, Mapping, "attestation-identities"),  # type: ignore[type-abstract]
            tool=_get(d, Mapping, "tool"),  # type: ignore[type-abstract]
        )
        distributions = bool(package.sdist) + len(package.wheels or [])
        direct_urls = (
            bool(package.vcs) + bool(package.directory) + bool(package.archive)
        )
        if distributions > 0 and direct_urls > 0:
            raise PylockValidationError(
                "None of vcs, directory, archive must be set if sdist or wheels are set"
            )
        if distributions == 0 and direct_urls != 1:
            raise PylockValidationError(
                "Exactly one of vcs, directory, archive must be set "
                "if sdist and wheels are not set"
            )
        try:
            for i, attestation_identity in enumerate(  # noqa: B007
                package.attestation_identities or []
            ):
                _get_required(attestation_identity, str, "kind")
        except Exception as e:
            raise PylockValidationError(
                e, context=f"attestation-identities[{i}]"
            ) from e
        return package

    @property
    def is_direct(self) -> bool:
        return not (self.sdist or self.wheels)


@dataclass(frozen=True, init=False)
class Pylock:
    """A class representing a pylock file."""

    lock_version: Version
    environments: Sequence[Marker] | None = None
    requires_python: SpecifierSet | None = None
    extras: Sequence[NormalizedName] | None = None
    dependency_groups: Sequence[str] | None = None
    default_groups: Sequence[str] | None = None
    created_by: str  # type: ignore[misc]
    packages: Sequence[Package]  # type: ignore[misc]
    tool: Mapping[str, Any] | None = None

    def __init__(
        self,
        *,
        lock_version: Version,
        environments: Sequence[Marker] | None = None,
        requires_python: SpecifierSet | None = None,
        extras: Sequence[NormalizedName] | None = None,
        dependency_groups: Sequence[str] | None = None,
        default_groups: Sequence[str] | None = None,
        created_by: str,
        packages: Sequence[Package],
        tool: Mapping[str, Any] | None = None,
    ) -> None:
        # In Python 3.10+ make dataclass kw_only=True and remove __init__
        object.__setattr__(self, "lock_version", lock_version)
        object.__setattr__(self, "environments", environments)
        object.__setattr__(self, "requires_python", requires_python)
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "dependency_groups", dependency_groups)
        object.__setattr__(self, "default_groups", default_groups)
        object.__setattr__(self, "created_by", created_by)
        object.__setattr__(self, "packages", packages)
        object.__setattr__(self, "tool", tool)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> Self:
        pylock = cls(
            lock_version=_get_required_as(d, str, Version, "lock-version"),
            environments=_get_sequence_as(d, str, Marker, "environments"),
            extras=_get_sequence_as(d, str, _validate_normalized_name, "extras"),
            dependency_groups=_get_sequence(d, str, "dependency-groups"),
            default_groups=_get_sequence(d, str, "default-groups"),
            created_by=_get_required(d, str, "created-by"),
            requires_python=_get_as(d, str, SpecifierSet, "requires-python"),
            packages=_get_required_sequence_of_objects(d, Package, "packages"),
            tool=_get(d, Mapping, "tool"),  # type: ignore[type-abstract]
        )
        if not Version("1") <= pylock.lock_version < Version("2"):
            raise PylockUnsupportedVersionError(
                f"pylock version {pylock.lock_version} is not supported"
            )
        if pylock.lock_version > Version("1.0"):
            _logger.warning(
                "pylock minor version %s is not supported", pylock.lock_version
            )
        return pylock

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], /) -> Self:
        """Create and validate a Pylock instance from a TOML dictionary.

        Raises :class:`PylockValidationError` if the input data is not
        spec-compliant.
        """
        return cls._from_dict(d)

    def to_dict(self) -> Mapping[str, Any]:
        """Convert the Pylock instance to a TOML dictionary."""
        return dataclasses.asdict(self, dict_factory=_toml_dict_factory)

    def validate(self) -> None:
        """Validate the Pylock instance against the specification.

        Raises :class:`PylockValidationError` otherwise."""
        self.from_dict(self.to_dict())
