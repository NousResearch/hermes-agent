"""Bounded read-back after an external prompt editor exits."""
import time


_EDITOR_SAVE_POLL_SECONDS = 0.05
_EDITOR_SAVE_STABLE_SECONDS = 0.2
_EDITOR_SAVE_UNCHANGED_GRACE_SECONDS = 0.3
_EDITOR_SAVE_TIMEOUT_SECONDS = 2.0


def _read_editor_file_when_settled(path: str, initial: str) -> str:
    """Read an editor file after delayed or atomic saves have settled."""
    started_at = time.monotonic()
    latest = initial
    stable_since = started_at
    observed_change = False

    while time.monotonic() - started_at < _EDITOR_SAVE_TIMEOUT_SECONDS:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                current = fh.read()
        except OSError:
            # Atomic-save editors can briefly replace or rename the target.
            pass
        else:
            now = time.monotonic()
            if current != latest:
                latest = current
                stable_since = now
                observed_change = True
            elif observed_change:
                if now - stable_since >= _EDITOR_SAVE_STABLE_SECONDS:
                    return latest
            elif now - started_at >= _EDITOR_SAVE_UNCHANGED_GRACE_SECONDS:
                return latest

        time.sleep(_EDITOR_SAVE_POLL_SECONDS)

    return latest
