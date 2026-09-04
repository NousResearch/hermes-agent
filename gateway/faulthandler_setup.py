"""Safe signal-triggered Python traceback dumps for the gateway."""

from __future__ import annotations

import faulthandler
from typing import IO, Any


def register_traceback_signal(signum: int, *, file: IO[Any]) -> None:
    """Register a diagnostic signal without replaying its fatal default action."""
    faulthandler.register(
        signum,
        file=file,
        all_threads=True,
        chain=False,
    )
