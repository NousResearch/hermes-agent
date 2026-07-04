"""Shared constants for the OpenViking memory plugin."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_OPENVIKING_SERVICE_ENDPOINT = "https://api.vikingdb.cn-beijing.volces.com/openviking"
_DEFAULT_AGENT = "hermes"
_AGENT_PROMPT_LABEL = "Hermes peer ID in OpenViking"
_OVCLI_CONFIG_ENV = "OPENVIKING_CLI_CONFIG_FILE"
_OVCLI_DEFAULT_RELATIVE_PATH = ".openviking/ovcli.conf"
_OVCLI_SAVED_PREFIX = "ovcli.conf."
_OPENVIKING_ENV_KEYS = (
    "OPENVIKING_ENDPOINT",
    "OPENVIKING_API_KEY",
    "OPENVIKING_ACCOUNT",
    "OPENVIKING_USER",
    "OPENVIKING_AGENT",
)
_TIMEOUT = 30.0
_SESSION_DRAIN_TIMEOUT = 10.0
_DEFERRED_COMMIT_TIMEOUT = (_TIMEOUT * 2) + 5.0
_REMOTE_RESOURCE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")
_SYNC_TRACE_ENV = "HERMES_OPENVIKING_SYNC_TRACE"
_DEFAULT_RECALL_LIMIT = 6
_DEFAULT_RECALL_SCORE_THRESHOLD = 0.15
_DEFAULT_RECALL_MAX_INJECTED_CHARS = 4000
_DEFAULT_RECALL_TIMEOUT_SECONDS = 4.0
_DEFAULT_RECALL_REQUEST_TIMEOUT_SECONDS = 3.0
_DEFAULT_RECALL_FULL_READ_LIMIT = 2
_RECALL_QUERY_MIN_CHARS = 5
_RECALL_MIN_TIMEOUT_SECONDS = 0.05
_READ_BATCH_LIMIT = 3
_READ_BATCH_FULL_LIMIT = 2500

# Maps the viking_remember `category` enum to a viking:// subdirectory.
# Keep in sync with REMEMBER_SCHEMA.parameters.properties.category.enum.
_CATEGORY_SUBDIR_MAP = {
    "preference": "preferences",
    "entity": "entities",
    "event": "events",
    "case": "cases",
    "pattern": "patterns",
}
_DEFAULT_MEMORY_SUBDIR = "preferences"

# Maps the built-in memory tool's `target` ("user" vs "memory") to a subdir
# for on_memory_write mirroring. User profile facts → preferences; agent
# notes / observations → patterns. Anything unknown falls back to the default.
_MEMORY_WRITE_TARGET_SUBDIR_MAP = {
    "user": "preferences",
    "memory": "patterns",
}
# OpenViking-generated markdown summaries. Non-.md sidecars such as
# .relations.json are rejected earlier by the exact memory-file check.
_GENERATED_MEMORY_SUMMARY_FILENAMES = {
    ".abstract.md",
    ".overview.md",
}
_LOCAL_OPENVIKING_HOSTS = {"localhost", "127.0.0.1", "::1"}
_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT = 60.0
_OPENVIKING_SERVER_LOG_RELATIVE_PATH = Path("logs") / "openviking-server.log"
_OPENVIKING_RESPONDED_FAILURE_PREFIX = "OpenViking server responded"
_SETUP_CANCELLED = object()
