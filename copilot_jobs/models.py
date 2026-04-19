"""State enums and dataclasses for copilot jobs."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class JobState(str, Enum):
    """Valid states for a copilot job."""
    PENDING = "pending"
    RUNNING = "running"
    IDLE = "idle"
    CLOSED = "closed"
    FAILED = "failed"


class JobOwner(str, Enum):
    """Who currently owns a copilot job."""
    HERMES = "hermes"
    HUMAN = "human"


@dataclass
class RepoEntry:
    """A repository entry discovered from the workspace filesystem."""
    slug: str
    path: str
    readme_summary: str = ""
    description: str = ""
    default_branch: str = "main"
