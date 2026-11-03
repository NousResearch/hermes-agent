"""Atomic, owner-only publication of backup archives."""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.backup_staging import staged_backup_file


@contextmanager
def _atomic_output_path(final_path: Path, publish_path: Optional[Callable[[], Optional[Path]]] = None):
    """Yield a hidden sibling path and publish it only after a clean close.

    ``publish_path`` picks the destination at publish time (default ``final_path``) so a caller
    can divert an incomplete archive elsewhere without ever touching ``final_path``; returning
    ``None`` discards the partial instead of publishing it.
    """
    with staged_backup_file(final_path.parent, "archive.partial") as partial_path:
        # Archives carry credentials: create the partial owner-only before any byte is written so
        # neither it nor the published file ever has umask-derived group/other bits. O_EXCL refuses
        # a file or symlink planted in the staging directory; the writer reopens and truncates
        # this inode, which keeps its mode.
        os.close(os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        yield partial_path
        destination = publish_path() if publish_path else final_path
        if destination is not None:
            os.replace(partial_path, destination)
