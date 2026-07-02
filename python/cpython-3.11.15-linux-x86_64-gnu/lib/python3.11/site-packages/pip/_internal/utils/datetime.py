"""For when pip wants to check the date or time."""

import datetime
import sys


def today_is_later_than(year: int, month: int, day: int) -> bool:
    today = datetime.date.today()
    given = datetime.date(year, month, day)

    return today > given


def parse_iso_datetime(isodate: str) -> datetime.datetime:
    """Convert an ISO format string to a datetime.

    Handles the format 2020-01-22T14:24:01Z (trailing Z)
    which is not supported by older versions of fromisoformat.
    """
    # Python 3.11+ supports Z suffix natively in fromisoformat
    if sys.version_info >= (3, 11):
        return datetime.datetime.fromisoformat(isodate)
    else:
        return datetime.datetime.fromisoformat(
            isodate.replace("Z", "+00:00")
            if isodate.endswith("Z") and ("T" in isodate or " " in isodate.strip())
            else isodate
        )
