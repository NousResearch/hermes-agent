"""Gateway lifecycle guard for cron job creation (#30719).

An agent running inside a gateway can schedule a cron job that calls
``hermes gateway restart`` (or ``launchctl kickstart ai.hermes.gateway``
or ``systemctl restart hermes-gateway``).  When the cron fires, the
gateway dies, the supervisor (launchd KeepAlive / systemd Restart=)
revives it, auto-resume picks up the offending session, and the resumed
turn re-runs the same logic — a SIGTERM-respawn loop every ~10 seconds
until manually broken.

This module rejects cron job specs whose prompt or script contains a
direct shell-level gateway-lifecycle command.  It is enforced at
``cron.jobs.create_job`` so it fires on every job-creation path: the
``hermes cron create`` CLI subcommand AND the agent's ``cronjob`` model
tool (which calls ``create_job`` directly, bypassing the CLI layer).

The pattern is intentionally command-shaped: it anchors on a concrete
command identifier (``hermes gateway``, ``launchctl ... hermes-gateway``,
``systemctl ... hermes-gateway``, ``pkill`` against the gateway) so it
cannot fire on prose.  A cron ``prompt`` is fed to a future LLM, not a
shell, so an over-broad substring match on English ("Kong API gateway
autoscaling and restart behavior") would produce a high false-positive
rate without preventing the actual foot-gun, which requires a real
command shape.

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` already
blocks these commands at *execution* time when ``_HERMES_GATEWAY=1``, and
``hermes gateway stop|restart`` refuse to self-target from inside the
gateway.  Blocking at *creation* time as well means the agent gets an
immediate, informative rejection instead of scheduling a job that will
only fail (silently) when it fires.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle. Each branch
# is anchored on a concrete command identifier so a match can only fire on
# actual shell-command-shaped strings, not on prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: `hermes gateway restart|stop` — the canonical foot-gun.
    # `start` is intentionally excluded: starting a gateway from inside a
    # gateway is benign (a no-op or "already running" error), and a
    # legitimate cron job might start a sibling profile's gateway.
    r"(?:hermes\s+gateway\s+(?:restart|stop))"
    # Branch B: launchctl ops on a hermes-gateway label. macOS launchd
    # labels look like `ai.hermes.gateway` / `hermes-gateway`. Requiring the
    # gateway identifier prevents blocking unrelated hermes services (e.g.
    # `launchctl unload ai.hermes.update-checker.plist`).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    #   - LEADING \b on the command so `skill`/`skills` can't match the bare
    #     `kill` substring (the false-positive that blocked read-only commands
    #     mentioning skill paths: `skills-...safe-gateway...hermes-harness`).
    #   - the gap between kill and its target tokens is bounded to a single
    #     shell segment ([^\n;|&]*) so it can't greedily span across `;`/`&&`
    #     into unrelated `hermes`/`gateway` path tokens later on the line.
    r"|(?:\b(?:pkill|kill)\b[^\n;|&]*\bhermes\b[^\n;|&]*\bgateway)"
    r"|(?:\b(?:pkill|kill)\b[^\n;|&]*\bgateway\b[^\n;|&]*\bhermes)"
)


# A lifecycle command wrapped in an ssh invocation targets a *remote* host's
# gateway, so the local foot-gun rationale does not apply: the command runs
# under the remote sshd, the local gateway is never SIGTERMed, and no
# supervisor respawn loop can form on this machine.  Fleet maintenance
# (restarting a sibling machine's gateway over ssh) is a legitimate, common
# operation and must not be blocked.
#
# Loopback targets (``ssh localhost ...``) are still blocked: the *effect*
# (this host's gateway dying, on a schedule for the cron path) can still
# produce the #30719 respawn loop even though the ssh client itself would
# survive.  We cannot resolve arbitrary hostnames in a text guard, so an ssh
# to this machine's own LAN hostname is an accepted residual gap.
_SSH_COMMAND_RE = re.compile(r"(?i)(?:^|\s)(?:/\S*/)?(?:ssh|autossh)\s")
_LOOPBACK_HOST_RE = re.compile(
    r"(?i)(?:^|[\s@\[:])(?:localhost|(?:::ffff:)?127\.\d{1,3}\.\d{1,3}\.\d{1,3}|::1|0\.0\.0\.0)\b"
)

# Rough shell-segment separators.  This is a heuristic split (it does not
# honour quoting), which errs on the side of BLOCKING: a separator inside an
# ssh remote-command string starts a "new segment" that no longer contains
# ``ssh``, so such a match falls back to blocked rather than allowed.
_SEGMENT_SPLIT_RE = re.compile(r"(?:\|\||&&|;|\||&|\$\(|`)")


def _match_is_ssh_remote(text: str, match_start: int) -> bool:
    """Return True if the lifecycle match at *match_start* sits inside an
    ssh invocation targeting a non-loopback host."""
    line_start = text.rfind("\n", 0, match_start) + 1
    prefix = text[line_start:match_start]
    # The command context for the match is the last shell segment before it.
    segment = _SEGMENT_SPLIT_RE.split(prefix)[-1]
    if not _SSH_COMMAND_RE.search(segment):
        return False
    if _LOOPBACK_HOST_RE.search(segment):
        return False
    return True


# Text-only consumer commands: when a lifecycle phrase appears as a QUOTED
# ARGUMENT to one of these, it is DATA being printed/searched/read, not a
# gateway command being executed — so it cannot SIGTERM this process.
# Deliberately EXCLUDES shell interpreters (bash/sh/zsh/dash/eval/xargs/env
# etc.): `bash -c "hermes gateway restart"` re-executes the phrase and MUST
# stay blocked. Anchored at the start of the segment (optional leading path).
_TEXT_CONSUMER_RE = re.compile(
    r"(?i)(?:^|\s)(?:/\S*/)?(?:echo|printf|grep|egrep|fgrep|rg|cat|head|tail|"
    r"less|more|comm|diff|sed\s+-n|awk|jq|tee|column|sort|uniq|wc)\b"
)

# Shell quote chars that open a data region. A `'` or `"` region makes the
# OTHER quote char (and backticks) literal until it closes — so we track the
# active region with a left-to-right scan rather than naive per-char counting
# (which mis-reads a backtick nested inside a single-quoted string).
_OPENING_QUOTES = ("'", '"')


def _open_quote_at(s: str) -> Optional[str]:
    """Left-to-right scan of *s*; return the quote char still OPEN at the end
    of the string, or None if all quotes are balanced. Inside an active
    single/double quote region the other quote char is literal."""
    active: Optional[str] = None
    for ch in s:
        if active is None:
            if ch in _OPENING_QUOTES:
                active = ch
        elif ch == active:
            active = None
    return active


def _match_is_quoted_data(text: str, match_start: int, match_str: str) -> bool:
    """Return True if the Branch-A `hermes gateway restart|stop` match at
    *match_start* is a QUOTED DATA argument to a text-only consumer command
    (echo/grep/printf/…), rather than an executed gateway command.

    Two conditions BOTH required (fail-closed — any doubt → not-data → blocked):
      1. The match sits inside an open single/double quote region (a proper
         left-to-right scan, so a backtick or the other quote nested inside is
         treated as literal), and that region closes after the match.
      2. The enclosing shell segment's leading command is a text-only consumer
         and NOT a shell interpreter (bash -c "…" stays blocked).

    Only applied to Branch A. The launchctl/systemctl/pkill branches are not
    exempted here — their command identifiers are distinctive enough that a
    quoted-data occurrence is vanishingly rare and not worth the bypass risk.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_start)
    if line_end == -1:
        line_end = len(text)
    prefix = text[line_start:match_start]
    suffix = text[match_start + len(match_str):line_end]

    # Condition 1: a single/double quote region is OPEN at the match, and it
    # closes somewhere in the suffix (data is bounded, not a trailing dangle).
    open_q = _open_quote_at(prefix)
    if open_q is None or open_q not in suffix:
        return False

    # Condition 2: the segment's command is a text-only consumer, not an
    # interpreter. Split on shell separators OUTSIDE quotes isn't worth the
    # complexity here — the prefix up to the match is within one quoted arg, so
    # take the segment before the opening quote and check its leading command.
    seg_prefix = prefix[: prefix.rfind(open_q)]
    segment = _SEGMENT_SPLIT_RE.split(seg_prefix)[-1]
    return bool(_TEXT_CONSUMER_RE.search(segment))


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern.

    Matches that are ssh-wrapped to a remote (non-loopback) host are
    exempt — restarting a *different* machine's gateway is legitimate fleet
    maintenance and cannot SIGTERM-loop this process (see
    ``_match_is_ssh_remote``).
    """
    if not text:
        return False
    for match in _GATEWAY_LIFECYCLE_PATTERN.finditer(text):
        # ssh-wrapped remote lifecycle commands are legitimate fleet ops.
        if _match_is_ssh_remote(text, match.start()):
            continue
        # Branch A only (`hermes gateway restart|stop`): exempt when the phrase
        # is quoted DATA fed to a text-only consumer (echo/grep/printf/…), not
        # an executed command. Interpreter re-exec (bash -c "…") is NOT exempt.
        matched = match.group(0)
        if matched.lower().startswith("hermes") and _match_is_quoted_data(
            text, match.start(), matched
        ):
            continue
        return True
    return False


def _resolve_script_path(script_path: str) -> Path:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.
    """
    from hermes_constants import get_hermes_home

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        return raw
    return get_hermes_home() / "scripts" / raw


def _read_script_for_scanning(script_path: str) -> str:
    """Read a script file for lifecycle-pattern scanning.

    Decodes with ``errors="replace"`` so binary or non-UTF-8 content does not
    silently bypass the check — a plain text-mode read raises
    ``UnicodeDecodeError`` on such files, and swallowing that error would let
    an attacker hide the command in binary noise.  Returns an empty string
    only when the file cannot be read at all.
    """
    try:
        return _resolve_script_path(script_path).read_bytes().decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a
    gateway-lifecycle command pattern.

    ``prompt`` is scanned directly.  ``script``, when supplied, is read from
    disk and concatenated for the scan.  Both are considered together so a
    job cannot slip through by splitting the command across the prompt and
    the script.

    Callers should let the exception propagate when they want the create to
    fail with a ``ValueError``-shaped error (the agent's ``cronjob`` tool
    surfaces this as a tool error; the CLI prints it in red and exits 1).
    """
    combined = prompt or ""
    if script:
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if contains_gateway_lifecycle_command(combined):
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command "
            "(restart/stop/kill). This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
