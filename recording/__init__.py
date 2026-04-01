"""
Action recording and replay for Hermes Agent.

Records tool call sequences during agent sessions and replays them,
allowing users to capture workflows and schedule them as cron jobs.

Recordings are stored in ~/.hermes/recordings/{name}.yaml
"""

from recording.store import (
    create_recording,
    get_recording,
    list_recordings,
    delete_recording,
    add_step,
    RECORDINGS_DIR,
)
from recording.replay import replay_recording
from recording.capture import RecordingSession, get_active_session

__all__ = [
    "create_recording",
    "get_recording",
    "list_recordings",
    "delete_recording",
    "add_step",
    "replay_recording",
    "RecordingSession",
    "get_active_session",
    "RECORDINGS_DIR",
]
