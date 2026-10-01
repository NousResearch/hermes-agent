"""Responsibility common; employee contracts on native Hermes."""

from __future__ import annotations
import re
from hermes_constants import get_hermes_home

def get_responsibilities_root():
    return get_hermes_home() / "responsibilities"


MAX_RESPONSIBILITY_NAME_LENGTH = 64

MAX_RESPONSIBILITY_MD_CHARS = 12_000

MAX_RESPONSIBILITY_FRONTMATTER_BYTES = 64 * 1024

MAX_RESPONSIBILITY_DESCRIPTION_LENGTH = 1024

SCHEDULES_DIRNAME = "schedules"

MAX_SCHEDULE_DECLARATION_BYTES = 16 * 1024

WEBHOOKS_DIRNAME = "webhooks"

SCRIPTS_DIRNAME = "scripts"

MAX_WEBHOOK_DECLARATION_BYTES = 16 * 1024

MAX_WEBHOOK_ACK_BODY_BYTES = 8 * 1024

MAX_WEBHOOK_ACK_CONTENT_TYPE_LENGTH = 100

STATE_DIRNAME = "state"

ARCHIVE_DIRNAME = "archive"

REFERENCES_DIRNAME = "references"

CONTEXT_DIRNAME = "context"

MAX_STATE_ROOT_CHARS = 6_000

MAX_STATE_FILE_CHARS = 10_000

MAX_ARCHIVE_FILE_CHARS = 130_000

MAX_CONTEXT_FILE_CHARS = 10_000

MAX_STATE_FILES = 100

MAX_ARCHIVE_FILES = 500

MAX_PACKAGE_SCRIPT_BYTES = 128 * 1024

RESPONSIBILITY_MD_CHAR_CEILING = 130_000

RESPONSIBILITY_MD_BYTE_CEILING = 4 * RESPONSIBILITY_MD_CHAR_CEILING

STATE_ROOT_BYTE_CEILING = 128 * 1024

STATE_FILE_BYTE_CEILING = 4 * MAX_STATE_FILE_CHARS

ARCHIVE_FILE_BYTE_CEILING = 4 * MAX_ARCHIVE_FILE_CHARS

CONTEXT_FILE_BYTE_CEILING = 128 * 1024

_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_FRONTMATTER_END_RE = re.compile(rb"\n---[ \t]*\r?\n")

_ALLOWED_FRONTMATTER = frozenset({
    "name",
    "trigger",
    "description",
    "linked_schedules",
    "author",
    "version",
    "lifecycle",
})

_ALLOWED_DECLARATION_FIELDS = frozenset(
    {"schedule", "scope", "report", "repeat", "script", "timezone"}
)

_ALLOWED_WEBHOOK_FIELDS = frozenset(
    {"scope", "key", "report", "handshake", "ack", "verify"}
)

_VERIFY_SIGNED_INPUTS = frozenset({"body", "timestamp.body", "v0:timestamp:body"})

_VERIFY_HEADER_SOURCE_RE = re.compile(
    r"[!#$%&'*+^_`|~0-9A-Za-z-]+(\.[A-Za-z0-9_]+)?"
)

_HMAC_PREFIX_RE = re.compile(r"[ -~]{0,16}")

_ACK_MEDIA_TYPE_RE = re.compile(
    r"[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+"
)

_LIFECYCLE_VALUES = frozenset({"ongoing", "finite"})

_DONE_WHEN_HEADING_RE = re.compile(r"(?i)^#{1,6}\s+done when\s*$")

_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

def _has_done_when_heading(body: str) -> bool:
    """True when a Done-when heading exists outside fenced code blocks."""

    fence: str | None = None
    for line in body.splitlines():
        stripped = line.strip()
        match = _CODE_FENCE_RE.match(stripped)
        if match is not None:
            marker = match.group(1)
            if fence is None:
                fence = marker
                continue
            if (
                marker[0] == fence[0]
                and len(marker) >= len(fence)
                and stripped == marker
            ):
                fence = None
            continue
        if fence is None and _DONE_WHEN_HEADING_RE.match(line.rstrip()):
            return True
    return False

class ResponsibilityFilesystemError(ValueError):
    """A responsibility path or package is invalid or unsafe."""

_SCRIPT_SUFFIXES = (".sh", ".bash", ".py")

_STRAY_FILES_LIMIT = 20

_STRAY_SCAN_MAX_ENTRIES = 2000

_MARKDOWN_TREE_DIRNAMES = frozenset(
    {REFERENCES_DIRNAME, CONTEXT_DIRNAME, STATE_DIRNAME, ARCHIVE_DIRNAME}
)

def validate_responsibility_name(name: str) -> str | None:
    if not name:
        return "Responsibility name is required."
    if len(name) > MAX_RESPONSIBILITY_NAME_LENGTH:
        return (
            f"Responsibility name exceeds {MAX_RESPONSIBILITY_NAME_LENGTH} characters."
        )
    if not _VALID_NAME_RE.fullmatch(name):
        return (
            f"Invalid responsibility name '{name}'. Use lowercase letters, "
            "numbers, hyphens, dots, and underscores. Must start with a letter "
            "or digit."
        )
    return None

_ROSTER_DESCRIPTION_LENGTH = 120
