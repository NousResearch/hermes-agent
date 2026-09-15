"""Shared print helpers for CLI list commands."""

from hermes_cli.colors import Colors, color


def print_truncated_footer(more: int) -> None:
    """Print a dim footer when ``--limit`` silently cut list rows (B7).

    ``more`` is the count of hidden rows (0 = nothing cut, prints nothing).
    Pass ``more=1`` when the cap lived in the query and only "at least one
    more" is known (``sessions list`` fetches ``limit + 1`` for exactly this
    probe); pass the exact count when the full list is already in hand
    (the ``pets.py`` precedent).
    """
    if more <= 0:
        return
    hidden = "and more" if more == 1 else f"and {more} more"
    print(color(f"  … {hidden} (use --limit 0 to see all)", Colors.DIM))
