"""People message store + identity + People-file writer (Phase 1 of #12323)."""

from plugins.people.store import PeopleMessageStore
from plugins.people.identity import IdentityResolver
from plugins.people.writer import write_people_markdown

__all__ = [
    "PeopleMessageStore",
    "IdentityResolver",
    "write_people_markdown",
]
