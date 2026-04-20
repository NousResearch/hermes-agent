"""State enums and dataclasses for copilot jobs."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class JobState(str, Enum):
    """Valid states for a copilot job.

    Simplified: copilot sessions are cloud-managed via --remote/--connect,
    so we only track whether we've launched and whether it finished.
    """
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class RepoEntry:
    """A repository entry discovered from the workspace filesystem."""
    slug: str
    path: str
    readme_summary: str = ""
    description: str = ""
    default_branch: str = "main"
