"""
_common.py — Shared logic for the desired-state skill scripts.

Single source of truth for:
- HERMES_HOME resolution (works inside the Hermes process or bare system Python)
- Store layout: <HERMES_HOME>/state/desired/<domain>/<slug>.md
- The GoalDoc model: a desired-state artifact = YAML-ish frontmatter + markdown body
- A small, stdlib-only frontmatter parser/serializer for the constrained goal schema
- Slugging, validation, and enumerations

Stdlib-only by design. Skill scripts may run outside the Hermes process
(system Python, nix, CI, Termux) where third-party deps such as PyYAML are
not importable, so the constrained goal frontmatter is parsed/serialized
here with the stdlib alone. Python 3.11+.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# HERMES_HOME + store layout
# ---------------------------------------------------------------------------
#
# Mirrors hermes_constants.get_hermes_home(); prefers the real module when it
# is importable so future enhancements (profile resolution, Docker detection)
# are picked up automatically, and falls back to the stdlib otherwise. Same
# contract as skills/productivity/google-workspace/scripts/_hermes_home.py.
try:
    from hermes_constants import get_hermes_home as get_hermes_home
except (ModuleNotFoundError, ImportError):

    def get_hermes_home() -> Path:
        """Return the Hermes home directory.

        Mirrors ``hermes_constants.get_hermes_home()``: ``HERMES_HOME`` wins,
        else the platform-native default — ``%LOCALAPPDATA%\\hermes`` on native
        Windows (matching ``_get_platform_default_hermes_home``), ``~/.hermes``
        on POSIX. A bare ``~/.hermes`` fallback would point Windows users at the
        wrong store when the module isn't importable.
        """
        val = os.environ.get("HERMES_HOME", "").strip()
        if val:
            return Path(val)
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
            return base / "hermes"
        return Path.home() / ".hermes"


def desired_root() -> Path:
    """Root of the desired-state store: <HERMES_HOME>/state/desired/."""
    return get_hermes_home() / "state" / "desired"


def display_root() -> str:
    """`~/`-shortened display string for the store root (user-facing messages)."""
    root = desired_root()
    try:
        return "~/" + str(root.relative_to(Path.home()))
    except ValueError:
        return str(root)


def goal_path(domain: str, slug: str, *, root: Path | None = None) -> Path:
    """Absolute path of a goal artifact for (domain, slug)."""
    base = root if root is not None else desired_root()
    return base / slugify(domain) / f"{slugify(slug)}.md"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

HORIZONS = ("short", "medium", "long")
STATUSES = ("active", "paused", "achieved", "dropped")
DIRECTIONS = ("increase", "decrease", "maintain")

# Frontmatter keys we treat as string lists (block or inline YAML flow).
LIST_KEYS = ("linked_projects", "linked_people", "linked_todos", "tags")
# Frontmatter keys whose bare scalar values are schema strings, not YAML bools/numbers.
STRING_KEYS = (
    "domain", "goal", "horizon", "status", "direction", "unit",
    "target_date", "start_date", "measurement_source", "created_at", "updated_at",
    # Also used by the local SKILL.md metadata guard.
    "name", "description", "version", "author", "license",
)


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Lowercase, hyphen-separated, filesystem-safe slug.

    "Savings Rate 2026!" -> "savings-rate-2026". Collapses runs of
    non-alphanumeric characters to a single hyphen and trims them.
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower())
    return s.strip("-") or "untitled"


# ---------------------------------------------------------------------------
# Time helpers (injectable clock keeps callers + tests deterministic)
# ---------------------------------------------------------------------------

def utcnow(now: datetime | None = None) -> datetime:
    """Return an aware UTC datetime; pass `now` to pin it (tests, replay)."""
    return now if now is not None else datetime.now(timezone.utc)


def iso_now(now: datetime | None = None) -> str:
    """UTC timestamp as `YYYY-MM-DDTHH:MM:SSZ` (second precision)."""
    return utcnow(now).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date(value: Any) -> date | None:
    """Best-effort parse of an ISO date/datetime string to a `date`.

    Accepts `YYYY-MM-DD` and full ISO timestamps (trailing `Z` allowed).
    Returns None for empty/unparseable input rather than raising, so a
    hand-edited artifact never crashes gap computation.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Frontmatter parse / serialize (stdlib-only, constrained schema)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)$")
_ESCAPE_RE = re.compile(r"\\(.)")  # unescape \\ and \" from double-quoted dumps


def _coerce_scalar(raw: str, *, key: str | None = None) -> Any:
    """Coerce a scalar frontmatter value string to a Python value."""
    v = raw.strip()
    if v in ("", "~", "null", "None"):
        return None
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return _unquote(v)
    if key in STRING_KEYS:
        return v
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _NUM_RE.match(v):
        return float(v) if ("." in v) else int(v)
    return v


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body). Raises ValueError if no valid block.

    A valid artifact starts at byte 0 with `---\\n` and closes with a line
    that is exactly `---` before the markdown body.
    """
    if not text.startswith("---"):
        raise ValueError("missing frontmatter: file must start with '---'")
    m = re.search(r"\n---[ \t]*\n", text[3:])
    if not m:
        raise ValueError("unterminated frontmatter: no closing '---'")
    fm = text[3 : m.start() + 3].lstrip("\n")
    body = text[m.end() + 3 :]
    return fm, body


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a goal artifact into (frontmatter dict, markdown body).

    Supports the constrained goal schema only: scalars, inline flow lists
    (`key: [a, b]`), and block lists (`key:` then indented `- item` lines).
    Bare numeric-looking values are coerced to int/float unless the key is a
    schema string field; known list keys always yield a list (possibly empty).
    """
    fm_text, body = _split_frontmatter(text)
    data: dict[str, Any] = {}
    lines = fm_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            # Possible block list: consume following indented "- " lines.
            items: list[str] = []
            j = i + 1
            while j < len(lines) and re.match(r"\s*-\s+", lines[j]):
                items.append(_unquote(re.sub(r"^\s*-\s+", "", lines[j]).strip()))
                j += 1
            if items:
                data[key] = items
                i = j
                continue
            data[key] = [] if key in LIST_KEYS else None
            i += 1
            continue
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            data[key] = [_unquote(p.strip()) for p in _split_inline_list(inner) if p.strip()] if inner else []
            i += 1
            continue
        data[key] = _coerce_scalar(rest, key=key)
        i += 1

    # Normalize declared shapes so downstream code never type-checks.
    for k in LIST_KEYS:
        if k in data and not isinstance(data[k], list):
            data[k] = [data[k]] if data[k] not in (None, "") else []
    return data, body


def _split_inline_list(inner: str) -> list[str]:
    """Split a constrained inline list while respecting quoted commas."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in inner:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if quote == '"' and ch == "\\":
            buf.append(ch)
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            buf.append(ch)
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def _unquote(v: str) -> str:
    """Strip surrounding quotes and unescape a dumped double-quoted value."""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        return _ESCAPE_RE.sub(r"\1", inner) if v[0] == '"' else inner
    return v


def _dump_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    # Quote when a bare scalar would be ambiguous to our parser or to a reader.
    if (
        text == ""
        or text[0] in "[]{}#&*!|>'\"%@`-"
        or ": " in text
        or text != text.strip()
        or text in ("~", "null", "None")
        or text.lower() in ("true", "false")
        or _NUM_RE.match(text) is not None
    ):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return '"' + escaped + '"'
    return text


def dump_frontmatter(data: dict[str, Any]) -> str:
    """Serialize a frontmatter dict back to text (block lists, insertion order).

    Deterministic and round-trip-stable with parse_frontmatter for the goal
    schema. Empty list keys are emitted as `key: []`.
    """
    out: list[str] = []
    for key, value in data.items():
        if isinstance(value, list):
            if not value:
                out.append(f"{key}: []")
            else:
                out.append(f"{key}:")
                out.extend(f"  - {_dump_scalar(v)}" for v in value)
        else:
            rendered = _dump_scalar(value)
            out.append(f"{key}: {rendered}" if rendered != "" else f"{key}:")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# GoalDoc model
# ---------------------------------------------------------------------------


@dataclass
class GoalDoc:
    """A single desired-state artifact.

    Frontmatter fields map to attributes; the markdown body carries context,
    rationale, constraints, and `- [ ]` milestone checkboxes.
    """

    domain: str
    goal: str
    horizon: str = "medium"
    status: str = "active"
    direction: str | None = None
    target_value: float | int | str | None = None
    current_value: float | int | str | None = None
    baseline_value: float | int | str | None = None
    unit: str | None = None
    target_date: str | None = None
    start_date: str | None = None
    measurement_source: str | None = None
    linked_projects: list[str] = field(default_factory=list)
    linked_people: list[str] = field(default_factory=list)
    linked_todos: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    body: str = ""
    # Non-persisted: where this doc was loaded from (set by the store).
    path: Path | None = field(default=None, compare=False)

    # Frontmatter key order for stable serialization.
    _FIELD_ORDER = (
        "domain", "goal", "horizon", "status", "direction",
        "target_value", "current_value", "baseline_value", "unit",
        "target_date", "start_date", "measurement_source",
        "linked_projects", "linked_people", "linked_todos", "tags",
        "created_at", "updated_at",
    )

    @property
    def slug(self) -> str:
        return slugify(self.path.stem) if self.path else slugify(self.goal)

    def to_text(self) -> str:
        """Render to `---frontmatter---\\n\\nbody` artifact text."""
        fm = {k: getattr(self, k) for k in self._FIELD_ORDER}
        body = self.body if self.body.endswith("\n") or self.body == "" else self.body + "\n"
        return f"---\n{dump_frontmatter(fm)}\n---\n\n{body}"

    @classmethod
    def from_text(cls, text: str, *, path: Path | None = None) -> GoalDoc:
        data, body = parse_frontmatter(text)
        known = {f for f in cls._FIELD_ORDER}
        kwargs = {k: v for k, v in data.items() if k in known}
        for lk in LIST_KEYS:
            kwargs.setdefault(lk, [])
        # Raise a clear ValueError (not the raw TypeError from a missing
        # positional arg) so a hand-edited artifact lacking a required field is
        # skipped by list_goals and reported readably by a direct get.
        missing = [f for f in ("domain", "goal") if not str(kwargs.get(f, "")).strip()]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        doc = cls(**kwargs)
        doc.body = body.lstrip("\n")
        doc.path = path
        return doc

    @classmethod
    def from_file(cls, path: Path) -> GoalDoc:
        return cls.from_text(Path(path).read_text(encoding="utf-8"), path=Path(path))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means valid."""
        problems: list[str] = []
        if not str(self.domain).strip():
            problems.append("domain is required")
        if not str(self.goal).strip():
            problems.append("goal is required")
        if self.horizon not in HORIZONS:
            problems.append(f"horizon must be one of {HORIZONS}, got {self.horizon!r}")
        if self.status not in STATUSES:
            problems.append(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.direction is not None and self.direction not in DIRECTIONS:
            problems.append(f"direction must be one of {DIRECTIONS} or empty, got {self.direction!r}")
        if self.target_date and parse_date(self.target_date) is None:
            problems.append(f"target_date is not a valid ISO date: {self.target_date!r}")
        if self.start_date and parse_date(self.start_date) is None:
            problems.append(f"start_date is not a valid ISO date: {self.start_date!r}")
        return problems

    def is_quantifiable(self) -> bool:
        return _as_number(self.target_value) is not None and _as_number(self.current_value) is not None

    def effective_direction(self) -> str:
        """Resolved goal direction.

        Uses the explicit ``direction`` if set, else infers from the *stable*
        reference — baseline first, then current. Inferring from the mutable
        current value alone flips a decrease goal to increase once tracking
        crosses the target, so ``ds_store.create_goal`` persists this at define
        time to lock it in.
        """
        if self.direction in DIRECTIONS:
            return self.direction
        target = _as_number(self.target_value)
        ref = _as_number(self.baseline_value)
        if ref is None:
            ref = _as_number(self.current_value)
        if target is not None and ref is not None and target < ref:
            return "decrease"
        return "increase"

    def lock_direction(self) -> None:
        """Freeze the inferred direction once a target and a reference (baseline
        or current) both exist, so it can't flip on later tracking. No-op if
        direction is already set or there is nothing to infer from. Called by
        ds_store on both create and update, so a target-only goal locks its
        direction the moment its first value is tracked."""
        has_reference = _as_number(self.baseline_value) is not None or _as_number(self.current_value) is not None
        if self.direction is None and _as_number(self.target_value) is not None and has_reference:
            self.direction = self.effective_direction()


def _as_number(value: Any) -> float | None:
    """Return `value` as float if numeric, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUM_RE.match(value.strip()):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Milestone helpers — edit the body's "- [ ] / - [x]" lines without a full rewrite
# ---------------------------------------------------------------------------

_MILESTONE_LINE = re.compile(r"^(\s*[-*]\s+\[)([ xX])(\]\s*)(.*)$")


def milestones_in(body: str) -> list[tuple[bool, str]]:
    """Return [(done, text), ...] for each checkbox line in the body, in order."""
    out: list[tuple[bool, str]] = []
    for line in (body or "").splitlines():
        m = _MILESTONE_LINE.match(line)
        if m:
            out.append((m.group(2) in "xX", m.group(4).strip()))
    return out


def add_milestone(body: str, text: str) -> str:
    """Append `- [ ] text` under a `## Milestones` heading, creating it if absent."""
    text = text.strip()
    if not text:
        raise ValueError("milestone text is empty")
    new_line = f"- [ ] {text}"
    lines = body.rstrip("\n").split("\n") if body.strip() else []
    heading = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "## milestones"), None)
    if heading is None:
        lines = lines + ([""] if lines else []) + ["## Milestones", new_line]
        return "\n".join(lines) + "\n"
    insert_at = heading + 1
    for j in range(heading + 1, len(lines)):
        if _MILESTONE_LINE.match(lines[j]):
            insert_at = j + 1
        elif lines[j].strip():
            break
    lines.insert(insert_at, new_line)
    return "\n".join(lines) + "\n"


def set_milestone(body: str, index: int, done: bool) -> str:
    """Check/uncheck the 1-based Nth milestone. Raises IndexError if out of range."""
    lines = body.split("\n")
    seen = 0
    for i, line in enumerate(lines):
        m = _MILESTONE_LINE.match(line)
        if not m:
            continue
        seen += 1
        if seen == index:
            mark = "x" if done else " "
            lines[i] = f"{m.group(1)}{mark}{m.group(3)}{m.group(4)}"
            return "\n".join(lines)
    raise IndexError(f"no milestone #{index} (goal has {seen})")
