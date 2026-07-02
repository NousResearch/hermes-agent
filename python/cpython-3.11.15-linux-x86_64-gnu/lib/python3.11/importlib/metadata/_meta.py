from typing import Any, Dict, Iterator, List, Protocol, TypeVar, Union


_T = TypeVar("_T")


class PackageMetadata(Protocol):
    def __len__(self) -> int:
        ...  # pragma: no cover

    def __contains__(self, item: str) -> bool:
        ...  # pragma: no cover

    def __getitem__(self, key: str) -> str:
        ...  # pragma: no cover

    def __iter__(self) -> Iterator[str]:
        ...  # pragma: no cover

    def get_all(self, name: str, failobj: _T = ...) -> Union[List[Any], _T]:
        """
        Return all values associated with a possibly multi-valued key.
        """

    @property
    def json(self) -> Dict[str, Union[str, List[str]]]:
        """
        A JSON-compatible form of the metadata.
        """


class SimplePath(Protocol):
    """
    A minimal subset of pathlib.Path required by PathDistribution.
    """

    def joinpath(self) -> 'SimplePath':
        ...  # pragma: no cover

    def __truediv__(self) -> 'SimplePath':
        ...  # pragma: no cover

    def parent(self) -> 'SimplePath':
        ...  # pragma: no cover

    def read_text(self) -> str:
        ...  # pragma: no cover
