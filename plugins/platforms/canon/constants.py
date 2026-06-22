"""Constants for the Canon platform plugin."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api-6m6mlelskq-uc.a.run.app"
DEFAULT_STREAM_URL = "https://canon-agent-stream-195218560334.us-central1.run.app"
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_SEEN_MESSAGE_IDS = 1024
MAX_MEDIA_BYTES = 10 * 1024 * 1024
RUNTIME_STATUS_INTERVAL_SECONDS = 30.0
RUNTIME_SIGNAL_POLL_SECONDS = 1.0
RUNTIME_INPUT_POLL_SECONDS = 1.0
RUNTIME_APPROVAL_POLL_SECONDS = 1.0
RUNTIME_HITL_MAX_TIMEOUT_SECONDS = 30 * 60
FINAL_MESSAGE_HANDOFF_SECONDS = 0.75
REGISTRATION_POLL_INTERVAL_SECONDS = 3.0
REGISTRATION_TIMEOUT_SECONDS = 5 * 60.0

AUDIO_EXTS = {".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm", ".flac"}
IMAGE_EXTS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTS = {".avi", ".mkv", ".mov", ".mp4", ".webm", ".3gp"}

TURN_COMPLETE_METADATA = {
    "turnSemantics": "turn_complete",
    "turnComplete": True,
}
CANON_AGENTS_JSON_BOOTSTRAP_ENV = "CANON_AGENTS_JSON_BOOTSTRAP"

CONTROL_METADATA_TYPES = {
    "approval_reply",
    "approval_outcome",
    "runtime_input_reply",
    "runtime_input_outcome",
}
