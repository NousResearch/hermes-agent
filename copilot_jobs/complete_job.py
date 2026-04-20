"""Update copilot job state after the copilot process exits.

Called by the shell wrapper that ``launcher.py`` spawns.  Runs outside
the original hermes process, so it must bootstrap its own DB connection.

Usage::

    python complete_job.py <session_id> <exit_code>
"""

import sys
from pathlib import Path

# Ensure the hermes-agent root is on sys.path so ``hermes_state`` resolves.
_AGENT_ROOT = str(Path(__file__).resolve().parent.parent)
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from hermes_state import SessionDB  # noqa: E402


def finish(session_id: str, exit_code: int) -> None:
    state = "done" if exit_code == 0 else "failed"
    db = SessionDB()
    try:
        db.finish_copilot_job(session_id, state=state, exit_code=exit_code)
    finally:
        db.close()


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <session_id> <exit_code>", file=sys.stderr)
        sys.exit(1)

    session_id = sys.argv[1]
    try:
        exit_code = int(sys.argv[2])
    except ValueError:
        print(f"Invalid exit_code: {sys.argv[2]}", file=sys.stderr)
        sys.exit(1)

    finish(session_id, exit_code)


if __name__ == "__main__":
    main()
