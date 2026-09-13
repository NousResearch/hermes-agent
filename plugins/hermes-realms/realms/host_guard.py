"""Refuse terminal commands that re-point a realm's tools at the host desktop.

A realm strips the host's display, bus and input handles from the environment
every child inherits. Nothing stops a command from *putting them back* — the
values are guessable (`wayland-0`, `/run/user/1000/…`) and one `env WAYLAND_DISPLAY=…`
prefix is enough for a GUI tool inside a private desktop to act on the user's
real one instead. That is not hypothetical: it is how a previous session
reached the user's live Hyprland while reporting it was working in a realm.

The environment sanitizing is still the mechanism; this is the guard that says
so out loud rather than letting the escape happen silently. It is an
agent-judgment guardrail, not a containment boundary — a realm has never
claimed to stop code that is trying to get out, and the VM kind exists exactly
because that claim needs a different implementation to be true.
"""

import re
import shlex


# Re-pointing any of these at a host value puts a realm's GUI tooling on the
# user's own desktop. Each is an endpoint a compositor or input stack reads
# directly, so a single assignment is sufficient to escape.
GUARDED_KEYS = (
    "WAYLAND_DISPLAY",
    "HYPRLAND_INSTANCE_SIGNATURE",
    "HYPRLAND_CMD",
    "DISPLAY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "AT_SPI_BUS_ADDRESS",
    "SWAYSOCK",
    "I3SOCK",
    "YDOTOOL_SOCKET",
    "CUA_INJECT_SOCKET",
    "CUA_DRIVER_SOCKET",
    "XAUTHORITY",
)

MESSAGE = (
    "Blocked: this command re-points {keys} at the host desktop from inside a "
    "realm. That is how an agent ends up driving the user's real session while "
    "reporting it is working privately. Ask the user, or run /realm off if they "
    "explicitly want host access; existing approvals still apply."
)

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
# shlex keeps shell operators attached to the word before them, so a realistic
# `export DISPLAY=:0; cmd` yields the token `DISPLAY=:0;` and the value would
# never match a host display without this.
_OPERATORS = re.compile(r"[;&|]+$")


def _split_operator(token):
    return _OPERATORS.sub("", token), bool(_OPERATORS.search(token))


def _is_host_value(key, value):
    """A guarded key is only an escape when it is pointed somewhere real.

    Clearing a variable is not an escape, and blocking that would break
    ordinary commands that normalize their environment.

    This deliberately does NOT exempt "the value already in this process's
    environment". ``pre_tool`` runs in the Hermes backend, which normally sits
    on the user's own desktop session — so that environment holds the HOST's
    handles, and the realm's private ones collide with them by string value
    anyway (both compositors are ``wayland-0``). Treating "same as mine" as
    safe therefore exempted precisely the escape this guard exists to stop.
    """
    if not value:
        return False
    if key == "XDG_RUNTIME_DIR":
        # A realm's runtime dir is a private subdirectory; the bare user
        # runtime directory is the host's.
        return re.fullmatch(r"/run/user/[0-9]+/?", value) is not None
    if key == "DISPLAY":
        return re.fullmatch(r":[0-9]+(\.[0-9]+)?", value) is not None
    if key in ("DBUS_SESSION_BUS_ADDRESS", "AT_SPI_BUS_ADDRESS"):
        return re.search(r"/run/user/[0-9]+/(bus|at-spi)", value) is not None
    return True


def host_escape(command):
    """Return a refusal message when *command* re-points a guarded key.

    ``None`` means the command is ordinary. Unparseable shell text is ordinary
    too: this guard reports a specific, recognised pattern rather than pretending
    to understand every possible line, and a scan that guessed would block real
    work.
    """
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError:
        return None
    found = []
    for index, token in enumerate(tokens):
        bare, _ = _split_operator(token)
        match = _ASSIGNMENT.match(bare)
        if match is None:
            continue
        key, value = match.groups()
        if key not in GUARDED_KEYS or not _is_host_value(key, value):
            continue
        previous, ended = _split_operator(tokens[index - 1]) if index else ("", True)
        # A bare `VAR=x` at the start of a command, an `export VAR=x`, and an
        # `env VAR=x cmd` prefix all place the value in a child's environment.
        # An assignment appearing as a normal argument does not. A preceding
        # operator (`cmd; VAR=x ...`) starts a new command, so the assignment
        # there is a prefix too.
        if index == 0 or ended \
                or previous in ("env", "export", "declare", "typeset", "setenv") \
                or _ASSIGNMENT.match(previous) or previous in (";", "&&", "||", "|", "&"):
            found.append(key)
    if not found:
        return None
    return MESSAGE.format(keys=", ".join(dict.fromkeys(found)))
