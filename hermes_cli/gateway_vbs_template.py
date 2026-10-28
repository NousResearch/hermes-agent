"""Single source of truth for the Windows gateway launcher's VBScript shape.

The launcher is WRITTEN by :mod:`hermes_cli.gateway_windows` and VERIFIED by
:mod:`hermes_cli.gateway_runtime_service_identity`. Both read the shape from
here, so the launcher this repo emits and the launcher it accepts can never
drift apart again. They did once: a generator-only change shipped a launcher
the verifier rejected, and the Windows native-task service stopped verifying
at all (``service_identity_unverified`` on every discovery attempt) -- the
Windows task could no longer be adopted, started, or stopped.

Two shapes are recognized:

``legacy``
    The one-shot launcher: fire the gateway and forget it. ``sh.Run`` is
    asynchronous, so the VBS exits immediately and the gateway is orphaned
    (adopted by init) and outlives the launcher. Still ACCEPTED so an install
    that has not run ``hermes update`` yet keeps verifying.

``supervised``
    The bounded launcher: wait for the gateway, and while it asks to be
    retried (``GATEWAY_SERVICE_RESTART_EXIT_CODE``), try again -- at most
    ``MAX_ATTEMPTS`` times, with a delay that grows each round. The wait is
    what makes the exit code observable, so this shape is the one that
    actually caps a respawn storm: the launcher, not the gateway, owns the
    retry budget.

The shape is declared ONCE, as templates whose slots are ``{q:NAME}`` (a
VBScript double-quoted literal) and ``{i:NAME}`` (an integer). The renderer
substitutes values into them; the verifier compiles the very same templates
into regexes. A change to the shape therefore cannot leave the two halves
disagreeing -- which is precisely the bug this module exists to prevent.

The integer slots are matched as bounded integers rather than as fixed
literals: a launcher written by an older build (or by a newer one) must keep
verifying, and only the *structure* is load-bearing for identity. The
constants this build SHIPS are asserted against
``gateway.restart.GATEWAY_SERVICE_RESTART_EXIT_CODE`` by test, so the retry
code the launcher acts on is the code the gateway actually returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

# The gateway's "please retry me" exit code (gateway.restart.
# GATEWAY_SERVICE_RESTART_EXIT_CODE). Only this code is retried; a crash
# (any other code) is reported to the supervisor as-is.
RETRY_EXIT_CODE = 75

# Attempts the supervised launcher makes before it gives up, and the delay
# between them. The delay doubles each round so a stuck host backs off
# instead of hammering it.
MAX_ATTEMPTS = 3
BASE_DELAY_MS = 3000

# The environment variables the launcher may set. The verifier accepts any
# subset and any order (it needs HERMES_HOME and the supervised marker) but
# refuses an unknown key or a duplicate, so a foreign script cannot smuggle a
# variable past it.
ENV_KEYS = frozenset({
    "HERMES_HOME",
    "HERMES_SUPERVISED_CHILD",
    "HERMES_GATEWAY_DETACHED",
    "PYTHONIOENCODING",
    "VIRTUAL_ENV",
})

# The comment that opens every launcher script.
COMMENT_PREFIX = "' "

# Lines between the comment and the environment block.
_SET_LINES = (
    'Set sh = CreateObject("WScript.Shell")',
    'Set env = sh.Environment("PROCESS")',
)

# A VBScript double-quoted literal: an embedded quote is doubled.
_LITERAL_BODY = r'(?:[^"\r\n]|"")*'
# The captured form, for the one place the literal's CONTENT is needed.
_LITERAL_CAPTURE = r'([^"\r\n]|"")*'
_ENV_ITEM_RE = re.compile(rf'env\.Item\("([A-Z_]+)"\) = "({_LITERAL_CAPTURE})"')

# Slot syntax: {q:NAME} is a VBScript literal, {i:NAME} is an integer.
_SLOT_RE = re.compile(r"\{(q|i):(\w+)\}")

# Bounds for the integer slots. The structure is pinned; only these tuning
# values are free, so a launcher from a build with different backoff numbers
# still verifies while an absurd one does not.
_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "delay_ms": (0, 600_000),
    "retry_code": (0, 255),
    "max_attempts": (1, 1_000),
}


@dataclass(frozen=True)
class Shape:
    """One launcher's shape: the ``Dim`` line that identifies it and its tail."""

    name: str
    dim_line: str
    tail: tuple[str, ...]


# The one-shot launcher: fire the gateway asynchronously and exit. The gateway
# is orphaned and adopted by init.
LEGACY = Shape(
    name="legacy",
    dim_line="Dim sh, env, existing_pp",
    tail=(
        'existing_pp = env.Item("PYTHONPATH")',
        "If Len(existing_pp) > 0 Then",
        '  env.Item("PYTHONPATH") = "{q:pythonpath_with_existing}" & existing_pp',
        "Else",
        '  env.Item("PYTHONPATH") = "{q:pythonpath}"',
        "End If",
        "sh.CurrentDirectory = \"{q:cwd}\"",
        "sh.Run \"{q:command}\", 0, False",
    ),
)

# The bounded launcher: wait for the gateway, and retry only while the gateway
# asks to be retried. The attempt cap and the growing delay are the storm
# brake; the wait is what makes the exit code observable in the first place.
SUPERVISED = Shape(
    name="supervised",
    dim_line="Dim sh, env, existing_pp, rc, attempts, delay",
    tail=(
        'existing_pp = env.Item("PYTHONPATH")',
        "If Len(existing_pp) > 0 Then",
        '  env.Item("PYTHONPATH") = "{q:pythonpath_with_existing}" & existing_pp',
        "Else",
        '  env.Item("PYTHONPATH") = "{q:pythonpath}"',
        "End If",
        "sh.CurrentDirectory = \"{q:cwd}\"",
        "attempts = 0",
        "delay = {i:delay_ms}",
        "Do",
        "  rc = sh.Run(\"{q:command}\", 0, True)",
        "  If rc <> {i:retry_code} Then Exit Do",
        "  attempts = attempts + 1",
        "  If attempts >= {i:max_attempts} Then Exit Do",
        "  WScript.Sleep delay",
        "  delay = delay * 2",
        "Loop",
    ),
)

# Every shape the verifier accepts, keyed by the line that identifies it.
SHAPES: tuple[Shape, ...] = (LEGACY, SUPERVISED)


@dataclass(frozen=True)
class Parsed:
    """A launcher script that matched a known shape."""

    shape: str
    env: dict[str, str]
    home: str
    command: str
    values: dict[str, int]


def quote(value: str) -> str:
    """VBScript double-quoted literal (embedded quote doubled; newline refused)."""
    if "\r" in value or "\n" in value:
        raise ValueError(f"refusing to quote VBScript value containing newline: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def unquote(literal: str) -> str:
    """Inverse of :func:`quote`."""
    return literal.replace('""', '"')


def _render_line(template: str, values: dict[str, object]) -> str:
    def _sub(match: re.Match[str]) -> str:
        kind, name = match.group(1), match.group(2)
        if name not in values:
            raise KeyError(f"VBS template slot {name!r} has no value")
        value = values[name]
        if kind == "q":
            # The template carries the surrounding quotes, so the slot renders
            # only the CONTENT -- the same thing the verifier captures.
            return str(value).replace('"', '""')
        return str(value)

    return _SLOT_RE.sub(_sub, template)


def render(
    shape: Shape,
    *,
    description: str,
    env_items: Sequence[tuple[str, str]],
    values: dict[str, object],
) -> str:
    """Render ``shape`` into the launcher script text (CRLF terminated).

    ``env_items`` are the ``(key, value)`` pairs to export; ``values`` fill the
    template slots.
    """
    for key, _ in env_items:
        if key not in ENV_KEYS:
            raise ValueError(f"launcher env key {key!r} is not in the verified allowlist")
    lines: list[str] = [f"{COMMENT_PREFIX}{description}", "Option Explicit", shape.dim_line]
    lines.extend(_SET_LINES)
    lines.extend(f'env.Item("{key}") = {quote(value)}' for key, value in env_items)
    lines.extend(_render_line(template, values) for template in shape.tail)
    return "\r\n".join(lines) + "\r\n"


def _compile(template: str) -> re.Pattern[str]:
    """Compile one template line into a regex that captures its slots.

    The group name is prefixed with the slot kind (``q_``/``i_``) so the caller
    can tell a literal from an integer without re-deriving it from the
    template. A literal's group holds the CONTENT (the surrounding quotes are
    matched as literal text), exactly like :func:`unquote` expects.
    """
    parts: list[str] = []
    position = 0
    for match in _SLOT_RE.finditer(template):
        parts.append(re.escape(template[position:match.start()]))
        kind, name = match.group(1), match.group(2)
        if kind == "q":
            parts.append(f'(?P<q_{name}>{_LITERAL_CAPTURE})')
        else:
            parts.append(f"(?P<i_{name}>\\d{{1,9}})")
        position = match.end()
    parts.append(re.escape(template[position:]))
    return re.compile("".join(parts))


def parse(script: str) -> Optional[Parsed]:
    """Match ``script`` against a known shape; ``None`` when it is not ours.

    The whole script must be accounted for: a fixed head, an environment block
    of known keys, and the tail of exactly one shape. Anything left over --
    an extra line, a second ``sh.Run``, a trailing command -- is not ours.
    """
    lines = script.splitlines()
    if len(lines) < 6 or not lines[0].startswith(COMMENT_PREFIX):
        return None
    if lines[1] != "Option Explicit":
        return None
    shape = next((candidate for candidate in SHAPES if candidate.dim_line == lines[2]), None)
    if shape is None or tuple(lines[3:5]) != _SET_LINES:
        return None

    env: dict[str, str] = {}
    index = 5
    while index < len(lines):
        found = _ENV_ITEM_RE.fullmatch(lines[index])
        if found is None:
            break
        key = found.group(1)
        if key not in ENV_KEYS or key in env:
            return None
        env[key] = unquote(found.group(2))
        index += 1

    tail = lines[index:]
    if len(tail) != len(shape.tail):
        return None
    values: dict[str, int] = {}
    command = ""
    for actual, template in zip(tail, shape.tail):
        found = _compile(template).fullmatch(actual)
        if found is None:
            return None
        for name, raw in found.groupdict().items():
            if name == "q_command":
                command = unquote(raw)
            elif name.startswith("i_"):
                slot = name[2:]
                number = int(raw)
                low, high = _INT_BOUNDS[slot]
                if not low <= number <= high:
                    return None
                values[slot] = number
    return Parsed(
        shape=shape.name,
        env=env,
        home=env.get("HERMES_HOME", ""),
        command=command,
        values=values,
    )
