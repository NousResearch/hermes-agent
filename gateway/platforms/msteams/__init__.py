"""Microsoft Teams gateway platform adapter."""

from .adapter import (
    MsTeamsAdapter,
    check_msteams_requirements,
    strip_bot_mention,
    _activities_url,
)

__all__ = [
    "MsTeamsAdapter",
    "check_msteams_requirements",
    "strip_bot_mention",
    "_activities_url",
]
