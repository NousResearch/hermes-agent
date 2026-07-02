from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, TypeVar, Union, overload

import jaraco.text as text
from packaging.requirements import Requirement

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

_T = TypeVar("_T")
_StrOrIter: TypeAlias = Union[str, Iterable[str]]


parse_req: Callable[[str], Requirement] = lru_cache()(Requirement)
# Setuptools parses the same requirement many times
# (e.g. first for validation than for normalisation),
# so it might be worth to cache.


def parse_strings(strs: _StrOrIter) -> Iterator[str]:
    """
    Yield requirement strings for each specification in `strs`.

    `strs` must be a string, or a (possibly-nested) iterable thereof.
    """
    return text.join_continuation(map(text.drop_comment, text.yield_lines(strs)))


# These overloads are only needed because of a mypy false-positive, pyright gets it right
# https://github.com/python/mypy/issues/3737
@overload
def parse(strs: _StrOrIter) -> Iterator[Requirement]: ...
@overload
def parse(strs: _StrOrIter, parser: Callable[[str], _T]) -> Iterator[_T]: ...
def parse(strs: _StrOrIter, parser: Callable[[str], _T] = parse_req) -> Iterator[_T]:  # type: ignore[assignment]
    """
    Parse requirements.
    """
    return map(parser, parse_strings(strs))
