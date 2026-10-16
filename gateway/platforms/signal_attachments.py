"""Shared-path staging for Signal daemons running outside Hermes' filesystem."""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def staged_signal_attachments(paths, staging_dir):
    """Keep sidecar-readable copies alive through one RPC; never remove caller-owned files."""
    if not paths or not staging_dir:
        yield paths
        return
    directory = Path(staging_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    staged, owned = [], []
    try:
        for path in paths:
            source = Path(path).resolve()
            if source.is_relative_to(directory):
                staged.append(str(source))
                continue
            fd, name = tempfile.mkstemp(prefix="hermes-signal-", suffix=source.suffix[:16], dir=directory)
            owned.append(Path(name))
            with os.fdopen(fd, "wb") as target:
                # The daemon may use a different UID in its container.
                os.chmod(name, 0o644)
                with source.open("rb") as original:
                    shutil.copyfileobj(original, target)
            staged.append(name)
        yield staged
    finally:
        for path in owned:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Signal: failed to remove staged attachment %s: %s", path, exc)
