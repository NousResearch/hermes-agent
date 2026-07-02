"""Represents a wheel file and provides access to the various parts of the
name that have meaning.
"""

from __future__ import annotations

from collections.abc import Iterable

from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import (
    InvalidWheelFilename as _PackagingInvalidWheelFilename,
)
from pip._vendor.packaging.utils import parse_wheel_filename

from pip._internal.exceptions import InvalidWheelFilename


class Wheel:
    """A wheel file"""

    def __init__(self, filename: str) -> None:
        self.filename = filename

        try:
            wheel_info = parse_wheel_filename(filename)
        except _PackagingInvalidWheelFilename as e:
            raise InvalidWheelFilename(e.args[0]) from None

        self.name, _version, self.build_tag, self.file_tags = wheel_info
        self.version = str(_version)

    def get_formatted_file_tags(self) -> list[str]:
        """Return the wheel's tags as a sorted list of strings."""
        return sorted(str(tag) for tag in self.file_tags)

    def support_index_min(self, tags: list[Tag]) -> int:
        """Return the lowest index that one of the wheel's file_tag combinations
        achieves in the given list of supported tags.

        For example, if there are 8 supported tags and one of the file tags
        is first in the list, then return 0.

        :param tags: the PEP 425 tags to check the wheel against, in order
            with most preferred first.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        try:
            return next(i for i, t in enumerate(tags) if t in self.file_tags)
        except StopIteration:
            raise ValueError()

    def find_most_preferred_tag(
        self, tags: list[Tag], tag_to_priority: dict[Tag, int]
    ) -> int:
        """Return the priority of the most preferred tag that one of the wheel's file
        tag combinations achieves in the given list of supported tags using the given
        tag_to_priority mapping, where lower priorities are more-preferred.

        This is used in place of support_index_min in some cases in order to avoid
        an expensive linear scan of a large list of tags.

        :param tags: the PEP 425 tags to check the wheel against.
        :param tag_to_priority: a mapping from tag to priority of that tag, where
            lower is more preferred.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        return min(
            tag_to_priority[tag] for tag in self.file_tags if tag in tag_to_priority
        )

    def supported(self, tags: Iterable[Tag]) -> bool:
        """Return whether the wheel is compatible with one of the given tags.

        :param tags: the PEP 425 tags to check the wheel against.
        """
        return not self.file_tags.isdisjoint(tags)
