"""Gateway lifecycle guard for cron job creation.

A cron job that restarts/stops the gateway from inside the gateway (``hermes gateway restart``,
``launchctl kickstart ai.hermes.gateway``, ``systemctl restart hermes-gateway``) kills the process,
the supervisor revives it, auto-resume re-runs the turn: a SIGTERM-respawn loop.
``cron.jobs.create_job`` rejects such specs on every creation path. Patterns are command-shaped —
anchored on concrete command identifiers — so they cannot fire on prose. Defence-in-depth layer.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shlex
import socket
import stat
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle; each branch is anchored on a
# concrete command identifier so it fires only on command-shaped strings, never prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: destructive `hermes gateway` ops. `start` is excluded: starting from inside a
    # gateway is benign and a job may legitimately start a sibling profile. The lookbehind keeps
    # `hermes` from being a path component or word tail (`/docs/hermes gateway restart-notes.md`)
    # while every real command position (text start, whitespace, `;`/`&`/`|`, `$(`, backtick,
    # U+FFFD) still matches.
    # See #77173.
    r"(?:(?<![/\w.\-])hermes\s+gateway\s+(?:restart|stop|uninstall)\b)"
    # Branch B: launchctl ops anchored on a hermes-gateway label so unrelated hermes services stay
    # unblocked. `submit`/`bootstrap` register a NEW keepalive job wrapping an arbitrary helper (a
    # laundered restart); neutral-label submissions are caught by
    # `contains_launchctl_submit_command`. `bootout`/`remove`/`disable` are the
    # modern/legacy/durable forms of `unload`.
    # `submit` and `bootstrap` are included alongside the direct verbs (kickstart/etc.): `launchctl submit
    # -l ai.hermes.gateway-<suffix> -- <helper-script>` (or `launchctl bootstrap gui/<uid> <plist>`) creates
    # a NEW keepalive job wrapping an arbitrary helper, which is how a blocked direct restart/kill gets
    # laundered into a persistent restart loop instead (#62891) — same foot-gun, indirect shape.
    # Neutral-label submissions that dodge this text anchor are caught separately by
    # `contains_launchctl_submit_command` (execution-aware, label-independent). `bootout`/`remove`/`disable`
    # sit alongside `unload`: Apple deprecated load/unload in favour of bootstrap/bootout, so `bootout` is
    # the modern spelling of an already-listed verb, `remove` is its legacy sibling, and `disable` is what
    # makes an unload durable across boots. Omitting them left the bypassable approval layer
    # (tools/approval.py, skipped on force=True) as the only cover, while this hard block — documented as
    # "force=True cannot help here" — let them through (#80260).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap|bootout|remove|disable)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill/kill of the gateway process, both token orders. Leading \b keeps "skill" from
    # matching as "kill".
    r"|(?:\bp?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:\bp?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)

# Every branch uses `[^\n]*` between verb and label so matches cannot span unrelated lines. A POSIX
# backslash-newline continuation is therefore collapsed to a space before matching (as the shell
# does) rather than loosening `[^\n]*`.
# Every branch above uses `[^\n]*` between its verb and the gateway identifier so the match can't span
# unrelated lines of a longer cron prompt/script, but that also means a real multi-line shell invocation
# split across continuation lines (e.g. `launchctl submit \` / `  -l ai.hermes.gateway-... \` / `  -- ...`,
# the exact reported shape in #62891) would otherwise slip past. Collapse continuations to a single space
# before matching, mirroring what the shell itself does, rather than loosening `[^\n]*` and risking false
# positives across genuinely separate lines.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")

# Python argv-list punctuation (`subprocess.run(["launchctl", "bootout", ...])`) separates exec'd
# words with brackets/commas. Stripped only for the token-join re-scan, never from raw text.
# See #68289.
_ARGV_LIST_PUNCTUATION = re.compile(r"[\[\],]+")

# Branch A2: `hermes -p <profile> gateway restart|stop` (also `--profile <name>` /
# `--profile=<name>`). The selector breaks Branch A's adjacency. A sibling-profile restart is a
# legitimate fleet operation, so the profile name is captured and blocked only when it equals the
# profile running the guard. `start` stays excluded as in Branch A.
# Unlike Branch A this form is NOT unconditionally self-targeting: issued from inside gateway `zeus`,
# `hermes -p venus gateway restart` operates on a sibling profile's gateway and is a legitimate fleet
# operation. The pattern captures the named profile so `contains_gateway_lifecycle_command` can block only
# the self-targeting shape (named profile == the profile running the guard). See #78028.
_PROFILE_FLAG_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    r"hermes\s+"
    # Any global flags before the profile selector (each may carry a value).
    r"(?:-{1,2}\S+(?:\s+\S+)?\s+)*"
    # The selector: exactly the shapes the CLI's `_apply_profile_override` accepts.
    r"(?:--profile=([^\s]+)|(?:-p|--profile)\s+([^\s]+))"
    # Any global flags between the selector and the subcommand.
    r"(?:\s+-{1,2}\S+(?:\s+\S+)?)*"
    r"\s+gateway\s+(?:restart|stop)"
)

# Branch B needs the label AFTER the verb in one `[^\n]*` span; a loop that builds the label in an
# EARLIER `;`-segment (`label=${item%%:*}; launchctl bootout "gui/$uid/$label"`) leaves only
# `$label` next to the verb. These verbs act on an EXISTING job, so the hermes-gateway label anchor
# stays correct, but the check is "verb anywhere AND label anywhere".
# No profile identity available: cannot prove self-targeting, so do not block — sibling restarts must stay
# allowed (#78028).
_LAUNCHCTL_LIFECYCLE_VERBS_RE = re.compile(
    r"(?i)\blaunchctl\s+(?:kickstart|unload|load|stop|restart|bootout|kill|disable|remove)\b"
)
_HERMES_GATEWAY_LABEL_RE = re.compile(r"(?i)\bhermes[.\-]?gateway\b")

_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_SHELL_COMMAND_FLAGS = {"-c", "--command"}
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")

# `ssh`/`slogin` run their trailing command on a DIFFERENT host, so a gateway
# lifecycle verb in that payload targets the REMOTE gateway — SIGTERM does not
# cross the SSH boundary back to the caller. `scp`/`sftp`/`rsync` are excluded:
# they move files, not run a login shell command, so no lifecycle payload rides
# along. See profiles/marin/workspace/reports/remote-gateway-guard-bug.md.
_SSH_EXECUTABLES = frozenset({"ssh", "slogin"})
# ssh options that consume the NEXT token as their value, so the value is never
# mistaken for the destination host. The space-separated form is what matters;
# attached forms (`-p2222`, `-oKey=Val`) are single option-shaped tokens already
# skipped as flags. Unknown flags are treated as valueless — a misparse yields a
# non-classifiable host and fails closed (stays blocked), never a false exempt.
_SSH_OPTIONS_WITH_VALUES = frozenset({
    "-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l",
    "-m", "-O", "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
})
# Shell-expansion markers in a destination token: the true host is only known at
# runtime, so the target is ambiguous and must fail closed (stay blocked).
_SSH_DEST_AMBIGUOUS_MARKERS = ("$", "`", "*", "?")
_MAX_SSH_CONFIG_BYTES = 256 * 1024
# Options that redirect the connection AWAY from the destination token, so the
# destination is NOT the effective host and cannot be classified from it: e.g.
# `ssh -o HostName=127.0.0.1 8.8.8.8 ...` connects to loopback while the token
# reads 8.8.8.8. Their presence anywhere in the invocation forces a fail-closed
# verdict (the payload is left scanned/blocked). `-o RemoteCommand`/forwarding
# options are NOT here: they don't change WHICH host the login lands on.
_SSH_HOST_OVERRIDE_O_KEYS = frozenset({"hostname", "proxycommand", "proxyjump"})
# `-o` keys whose value runs a command on the CALLING host (not the remote): a lifecycle payload
# here restarts the local gateway even though the connection is genuinely remote. Their presence
# forces a fail-closed verdict. (`Match exec` is a config-block keyword, not settable via `-o`.)
_SSH_LOCAL_EXEC_O_KEYS = frozenset({"localcommand", "permitlocalcommand", "knownhostscommand"})
# Union checked by the cluster decoder: any of these `-o` keys ⇒ do not mask (fail closed).
_SSH_FAILCLOSED_O_KEYS = _SSH_HOST_OVERRIDE_O_KEYS | _SSH_LOCAL_EXEC_O_KEYS
# Short-option letters (no dash) that consume an argument — derived from the option set above so the
# cluster parser knows where an argument begins. ssh clusters short flags (`-tqo HostName=...` ==
# `-t -q -o HostName=...`), and a value may be attached to its letter or supplied as the next token.
# This set matches OpenSSH's getopt arg-taking letters (verified against 10.3); a future ssh that
# adds/removes a VALUE-taking letter could misalign destination identification, but never grants a
# local-exec exemption — local-exec/redirect `-o` values stay visible to the regex regardless.
_SSH_ARG_OPTION_CHARS = frozenset(option[1] for option in _SSH_OPTIONS_WITH_VALUES)
# Argument-taking short options whose mere presence redirects the login host: `-J` (ProxyJump),
# `-F` (alternate/`none` config that can rewrite HostName). `-o` is handled separately by key.
_SSH_HOST_OVERRIDE_SHORT_CHARS = frozenset({"J", "F"})

# Directory names directly under `Library` that mark a FileProvider-backed subtree: `Mobile
# Documents` is iCloud Drive; `CloudStorage` hosts third-party providers (Dropbox, OneDrive, Google
# Drive, ...).
_CLOUD_PLACEHOLDER_MARKERS = frozenset({"Mobile Documents", "CloudStorage"})

# Executables whose arguments are DATA (search patterns, SQL, log filters) and cannot execute their
# argument text. Deliberately conservative: no `awk` (system()), no `sed` (`s///e`), no
# `echo`/`printf` (routinely piped into a shell), no `mysql` (`\\!` and `system` escapes).
_DATA_SINK_EXECUTABLES = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ag", "ack", "journalctl", "sqlite3", "psql"}
)
# Argument shapes that smuggle execution back INTO a data sink (command/process substitution, psql
# `\!`). Any hit disables masking for the whole segment — fail closed to the plain regex verdict.
_UNSAFE_DATA_ARG_MARKERS = ("`", "$(", "<(", ">(", "\\!")
# sqlite3 dot-commands (`.shell`, `.system`) also disable masking. Dot must be followed by a NAME
# character so relative paths (`.`, `./x`) stay paths.
_DOT_COMMAND_ARGUMENT = re.compile(r"^\.[A-Za-z]")
# A data sink piped into a shell/interpreter can feed matched lines to execution; never mask it.
_PIPE_TO_INTERPRETER = re.compile(
    r"\|\s*&?\s*(?:sudo\s+)?(?:sh|bash|dash|ksh|zsh|xargs|eval|source)\b"
)

# Bytes sniffed before reading a referenced file in full (see _BINARY_MAGICS).
_BINARY_SNIFF_BYTES = 4096

_ReadRemoteScriptFn = Callable[[str], Optional[str]]

# Wrappers that hand execution to their argument tail: the real command sits further right, so a
# first-token-only guard would let `sudo bash ~/restart.sh` / `sudo launchctl submit ...` walk past.
# A guard that reads only the first token sees `sudo`/`env`/`nohup` and never inspects what they run, so
# `sudo bash ~/restart.sh` walked past the same walk that stops `bash ~/restart.sh`, and `sudo launchctl
# submit ...` past the label-independent submit block (#62891). `_PIPE_TO_INTERPRETER` above already reads
# `sudo ` this way for the pipe case; this generalises that reading to the command position.
_TRANSPARENT_COMMAND_PREFIXES = frozenset({
    "sudo", "doas", "env", "nohup", "setsid", "nice", "ionice", "stdbuf",
    "timeout", "exec", "command", "builtin", "eatmydata",
    # Privilege and namespace wrappers: options, then the command they run.
    "pkexec", "su", "runuser", "setpriv", "systemd-run", "nsenter", "unshare",
})

# Wrapper options that consume the NEXT token, so a value is never mistaken for the command.
_TRANSPARENT_PREFIX_VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-U", "-C", "-p", "-r", "-t", "-T", "--user", "--group", "--prompt"},
    "doas": {"-u", "-C"},
    "env": {"-u", "--unset", "-S", "--split-string", "-C", "--chdir"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "--class", "--classdata"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "pkexec": {"--user"},
    "su": {"-s", "--shell", "-g", "--group", "-G", "--supp-group"},
    "runuser": {"-u", "--user", "-s", "--shell", "-g", "--group", "-G", "--supp-group"},
    "setpriv": {"--reuid", "--regid", "--groups", "--inh-caps", "--ambient-caps", "--bounding-set",
                "--selinux-label", "--apparmor-profile"},
    "systemd-run": {"-u", "--unit", "-p", "--property", "-E", "--setenv", "--slice",
                    "--description", "--uid", "--gid", "--on-calendar", "--service-type"},
    "nsenter": {"-t", "--target", "-S", "--setuid", "-G", "--setgid", "-r", "--root", "-w", "--wd"},
    "unshare": {"--map-user", "--map-group", "--setgroups", "-R", "--root", "-w", "--wd"},
}

# Wrapper options carrying a COMMAND STRING (shell source, re-scanned like `sh -c`); treating it as
# an opaque value would hide whatever it runs (`env -S 'bash ~/restart.sh'`).
_STRING_COMMAND_OPTIONS = {
    "env": ("-S", "--split-string"),
    "su": ("-c", "--command"),
    "runuser": ("-c", "--command"),
}

# Wrappers whose first non-option operand is a VALUE, not the command (`timeout 60 bash x.sh`).
_TRANSPARENT_PREFIX_OPERANDS = {"timeout": 1}

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Bound the walk: a pathological token run must not spin here.
_MAX_PREFIX_PEELS = 8

_BINARY_MAGICS = (
    b"\x7fELF",              # ELF — Linux/BSD executables and shared objects
    b"\xfe\xed\xfa\xce",     # Mach-O 32-bit
    b"\xfe\xed\xfa\xcf",     # Mach-O 64-bit
    b"\xce\xfa\xed\xfe",     # Mach-O 32-bit, byte-swapped
    b"\xcf\xfa\xed\xfe",     # Mach-O 64-bit, byte-swapped
    b"\xca\xfe\xba\xbe",     # Mach-O universal ("fat") binary
    b"MZ",                   # PE/COFF — Windows .exe/.dll
    b"!<arch>",              # static archive (.a)
    b"\x1f\x8b",             # gzip
    b"PK\x03\x04",           # zip (also .jar/.whl/.egg)
)


# --- profile identity -------------------------------------------------------------------------

def _current_profile_name() -> Optional[str]:
    """Profile running the guard: ``HERMES_PROFILE_NAME``/``HERMES_PROFILE`` env first, then
    ``hermes_cli.profiles.get_active_profile_name`` (from ``HERMES_HOME``); ``None`` if neither."""
    for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or None
    except Exception:
        return None


def _named_profile_is_current(named: str) -> bool:
    """True when *named* is the profile executing the guard. Without a profile identity
    self-targeting cannot be proven, so nothing is blocked (sibling restarts stay allowed)."""
    current = _current_profile_name()
    return bool(current) and named.strip().casefold() == current.strip().casefold()


# --- direct string scans ----------------------------------------------------------------------

def _contains_launchctl_gateway_lifecycle(normalized_text: str) -> bool:
    """Order-independent companion to Branch B — see the verbs regex comment."""
    return bool(_LAUNCHCTL_LIFECYCLE_VERBS_RE.search(normalized_text)) and bool(
        _HERMES_GATEWAY_LABEL_RE.search(normalized_text)
    )


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern.

    Passes, in order: raw-text regex (the only pass that fires on inputs shlex cannot tokenize, e.g.
    Python source); profile-flag form; the same regex on each shell-tokenized segment with quotes/
    escapes resolved (closes splice bypasses like ``kick"start"`` / ``kick\\start``);
    order-independent launchctl pass. Single choke point for every recursion level of
    ``_contains_unsafe_gateway_action``.

    That second pass exists because a real shell resolves quote-splicing (``kick"start"``) and
    backslash-escaping (``kick\\start``) into one literal word — ``kickstart`` — before the command ever
    runs. The raw text still has the quote or backslash sitting between the verb's two halves, so the first
    pass alone lets a spliced verb reach ``launchctl``/``systemctl`` untouched while still executing as the
    blocked lifecycle command (#80269, reported against #80260's bootout parity fix). Tokenizing closes that
    gap while keeping the same gateway-label anchoring (``_GATEWAY_LIFECYCLE_PATTERN`` still requires a
    ``hermes``/``gateway`` token) — this function is the single choke point
    ``_contains_unsafe_gateway_action`` calls at every recursion level, so referenced-script and ``sh -c``
    payload scanning inherit the fix automatically.
    """
    if not text:
        return False
    # Provably inert heredoc bodies (quoted delimiter, data-sink consumer like `cat > f <<'EOF'`)
    # are documentation, not commands. The stripper fails open on ANY ambiguity (unquoted delimiter,
    # shell consumer, unterminated body), so executable heredocs are still scanned.
    # Heredoc bodies that are provably inert data (quoted delimiter, data-sink consumer like `cat > file
    # <<'EOF'`) are masked before scanning (#88336): a runbook line "a human can run: hermes gateway
    # restart" inside such a body is documentation, not a command this shell will execute.
    from tools.shell_heredoc import strip_inert_heredoc_bodies

    text = strip_inert_heredoc_bodies(text)
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    if _GATEWAY_LIFECYCLE_PATTERN.search(normalized):
        return True
    # Profile-flag form: blocked only when the named profile IS the one running the guard.
    # Profile-flag form (#78028): `hermes -p <profile> gateway restart|stop` bypasses Branch A because the
    # selector sits between `hermes` and `gateway`. It is only the same foot-gun when the named profile IS
    # the profile running the guard — sibling-profile restarts are legitimate fleet operations and stay
    # allowed.
    profile_match = _PROFILE_FLAG_LIFECYCLE_PATTERN.search(normalized)
    if profile_match:
        named = profile_match.group(1) or profile_match.group(2)
        # Profile ids cannot contain quotes (`^[a-z0-9][a-z0-9_-]{0,63}$`), so a shell-quoted
        # `-p 'zeus'` compares equal to the bare name.
        if named and _named_profile_is_current(named.strip().strip("\"'")):
            return True
    # Token-aware pass. Tokens are also re-joined with Python argv-list punctuation stripped, since
    # `subprocess.run(["launchctl", "bootout", ...])` separates argv words with commas/brackets.
    # Token-aware second pass (#80269): re-run the pattern on shell-tokenized segments where quotes/escapes
    # are resolved, closing splice bypasses like `kick"start"`. Runs after the profile-flag check so both
    # passes apply independently.
    for segment in _iter_command_segments(normalized):
        joined = " ".join(segment)
        if joined and _GATEWAY_LIFECYCLE_PATTERN.search(joined):
            return True
        stripped = _ARGV_LIST_PUNCTUATION.sub(" ", joined)
        if stripped != joined and _GATEWAY_LIFECYCLE_PATTERN.search(stripped):
            return True
    # The label may be built in an earlier `;`-segment, so no pass above sees verb + label together.
    # Order-independent launchctl pass (#77083): a shell loop can build the gateway label from a variable
    # defined in an earlier `;`-separated segment (`label=${item%%:*}; launchctl bootout
    # "gui/$uid/$label"`), so neither the same-span regex nor same-segment tokenization sees verb and label
    # together. Check "verb anywhere AND label anywhere" instead.
    return _contains_launchctl_gateway_lifecycle(normalized)


# Whole-walk work limits. The per-file cap and depth bound above limit one read, not the walk: a
# command can reference arbitrarily many scripts, and the pure-Python shlex pass (quadratic on a
# giant token) once held the GIL for minutes. These caps bound one whole walk and are charged
# BEFORE any text reaches shlex. Exhaustion fails closed (an unscanned script could hide a
# lifecycle command) and is logged at WARNING so an operator can tell it from a real block. Sizes
# sit well above any legitimate wrapper graph; remote reads are a backend roundtrip each, so they
# get a far tighter cap.
# See #78398.
_MAX_LIFECYCLE_SCAN_BYTES = _MAX_REFERENCED_SCRIPT_BYTES  # 1 MiB across the walk
_MAX_LIFECYCLE_SCAN_LINES = 16384
_MAX_LIFECYCLE_SCAN_LINE_BYTES = 64 * 1024
_MAX_LIFECYCLE_SCAN_PATHS = 1024
_MAX_LIFECYCLE_SCAN_REMOTE_READS = 64


class _LifecycleScanBudget:
    """Shared work budget for one complete referenced-script walk."""

    __slots__ = ("bytes_remaining", "lines_remaining", "paths_remaining", "remote_reads_remaining")

    def __init__(self) -> None:
        # Read the module constants at construction so tests/operators can lower them at runtime.
        self.bytes_remaining = _MAX_LIFECYCLE_SCAN_BYTES
        self.lines_remaining = _MAX_LIFECYCLE_SCAN_LINES
        self.paths_remaining = _MAX_LIFECYCLE_SCAN_PATHS
        self.remote_reads_remaining = _MAX_LIFECYCLE_SCAN_REMOTE_READS

    def charge_text(self, text: str) -> bool:
        """Charge *text* before tokenization; False when it does not fit."""
        # UTF-8 is >= one byte per code point, so the char count is a free lower bound.
        if len(text) > self.bytes_remaining:
            return False
        encoded = len(text.encode("utf-8", errors="replace"))
        if encoded > self.bytes_remaining:
            return False
        lines = text.count("\n") + 1
        if lines > self.lines_remaining:
            return False
        # One huge token is the quadratic shlex case; bound the longest physical line (chars, a
        # lower bound on bytes — tight enough for a DoS bound without a per-line encode).
        longest = max((len(line) for line in text.split("\n")), default=0)
        if longest > _MAX_LIFECYCLE_SCAN_LINE_BYTES:
            return False
        self.bytes_remaining -= encoded
        self.lines_remaining -= lines
        return True

    def charge_path(self) -> bool:
        """Charge one unique referenced path before any local/remote read."""
        if self.paths_remaining <= 0:
            return False
        self.paths_remaining -= 1
        return True

    def charge_remote_read(self) -> bool:
        """Charge one remote-backend read (a network roundtrip each)."""
        if self.remote_reads_remaining <= 0:
            return False
        self.remote_reads_remaining -= 1
        return True


def _capped_read_limit(max_bytes: Optional[int]) -> int:
    """Per-read byte cap: never above the per-file cap, never negative. One definition so local and
    remote reads cannot diverge.

    See #76762, #77703.
    """
    if max_bytes is None:
        return _MAX_REFERENCED_SCRIPT_BYTES
    return min(_MAX_REFERENCED_SCRIPT_BYTES, max(0, int(max_bytes)))


def lifecycle_scan_root_within_budget(text: str) -> bool:
    """Whether *text* may safely enter an optional tokenizer pass (``tools/terminal_tool.py`` gates
    its launchctl pre-scan on this). A FRESH budget, independent of the full guard's walk: the
    pre-scan may pass while the walk later exhausts, still fail-closed — only the friendlier
    launchctl diagnostic is lost. ``False`` is not a verdict: callers must still run the full guard."""
    try:
        return _LifecycleScanBudget().charge_text(text)
    except Exception:
        return False


def _budget_exhausted(what: str, depth: int) -> bool:
    logger.warning(
        "lifecycle guard scan budget exhausted (%s at depth %d); "
        "failing closed — see _MAX_LIFECYCLE_SCAN_* in cron/lifecycle_guard.py",
        what, depth,
    )
    return True


# --- shell tokenization -----------------------------------------------------------------------

def _split_logical_lines(text: str) -> list[str]:
    """Split on newlines outside quotes (a quoted newline is data, not a separator); honors
    escapes."""
    lines: list[str] = []
    current: list[str] = []
    in_single = in_double = escape = False
    for ch in text:
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "\n" and not in_single and not in_double:
            lines.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        lines.append("".join(current))
    return lines


def _shlex_tokens(line: str) -> list[str]:
    """POSIX-tokenize one shell line, honoring quotes and `#` comments; raises ValueError."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _split_segments(tokens: list[str], *, keep_controls: bool = False) -> Iterator[list[str]]:
    """Yield non-empty runs of *tokens* between control-operator tokens; with *keep_controls* each
    control token is also yielded as its own segment so the line can be rebuilt in order."""
    segment: list[str] = []
    for token in tokens:
        if token and set(token) <= _CONTROL_CHARS:
            if segment:
                yield segment
                segment = []
            if keep_controls:
                yield [token]
            continue
        segment.append(token)
    if segment:
        yield segment


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments per logical line; a line shlex rejects (unbalanced
    quotes) falls back to per-physical-line tokenization."""
    for line in _split_logical_lines(command.replace("\\\n", "")):
        try:
            tokens = _shlex_tokens(line)
        except ValueError:
            for physical_line in line.splitlines():
                try:
                    yield from _split_segments(_shlex_tokens(physical_line))
                except ValueError:
                    continue
            continue
        yield from _split_segments(tokens)


def _executable_name(token: str) -> str:
    """Command name of an executable token. ``Path(token).name`` is "" for ``.``, ``..`` and ``/``;
    the POSIX dot-source builtin is spelled ``.``, so fall back to the raw token or ``.
    ./helper.sh`` would escape the sourced-script scan."""
    return Path(token).name or token


def _peel_transparent_prefixes(segment: list[str], index: int) -> int:
    """Index of the command a wrapper chain actually executes. Unchanged if not a wrapper; may be
    ``len(segment)`` when a wrapper has no operand — callers must bounds-check."""
    for _ in range(_MAX_PREFIX_PEELS):
        if index >= len(segment):
            return index
        name = _executable_name(segment[index])
        if name not in _TRANSPARENT_COMMAND_PREFIXES:
            return index
        value_options = _TRANSPARENT_PREFIX_VALUE_OPTIONS.get(name, frozenset())
        index += 1
        while index < len(segment):
            token = segment[index]
            if token == "--":
                # POSIX end-of-options: the command starts at the next token.
                index += 1
                break
            if token in value_options:
                index += 2
                continue
            if token.startswith("-") or _ENV_ASSIGNMENT.match(token):
                index += 1
                continue
            break
        for _ in range(_TRANSPARENT_PREFIX_OPERANDS.get(name, 0)):
            if index < len(segment) and not segment[index].startswith("-"):
                index += 1
    return index


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    return next((i for i, token in enumerate(segment) if not _ENV_ASSIGNMENT.match(token)), None)


def _executed_command_index(segment: list[str]) -> Optional[int]:
    """Index of the command a segment actually executes (env assignments and wrappers peeled)."""
    index = _command_token_index(segment)
    if index is None:
        return None
    index = _peel_transparent_prefixes(segment, index)
    return index if index < len(segment) else None


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: a NEW job's label is attacker-chosen, so a neutral name defeats any
    label-anchored regex. Both verbs register a persistent launchd job — never safe in the gateway.

    See #62891.
    """
    for segment in _iter_command_segments(command):
        index = _executed_command_index(segment)
        if index is not None and _executable_name(segment[index]) == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _mask_data_sink_arguments(text: str) -> str:
    """Replace data-sink executables' arguments with a neutral placeholder.

    The regex cannot tell an EXECUTED lifecycle command from the same characters as *data* (a grep
    pattern, a SQL literal): every argument of a ``_DATA_SINK_EXECUTABLES`` segment becomes ``arg``
    and a match that survives is a real command. Strictly fail-closed: masking is skipped when the
    line pipes into a shell/interpreter, any argument carries an execution marker, or the line
    cannot be tokenized. Masking can only ALLOW what the plain regex would block, never block more.
    """
    lines_out: list[str] = []
    changed = False
    for line in text.splitlines() or [text]:
        if _PIPE_TO_INTERPRETER.search(line):
            lines_out.append(line)
            continue
        try:
            tokens = _shlex_tokens(line)
        except ValueError:
            lines_out.append(line)
            continue

        rebuilt: list[str] = []
        for segment in _split_segments(tokens, keep_controls=True):
            index = _command_token_index(segment)
            if index is not None and Path(segment[index]).name in _DATA_SINK_EXECUTABLES:
                arguments = segment[index + 1 :]
                if not any(
                    _DOT_COMMAND_ARGUMENT.match(argument)
                    or any(marker in argument for marker in _UNSAFE_DATA_ARG_MARKERS)
                    for argument in arguments
                ):
                    changed = True
                    rebuilt.extend(segment[: index + 1])
                    rebuilt.extend("arg" for _ in arguments)
                    continue
            rebuilt.extend(segment)
        lines_out.append(" ".join(rebuilt))
    return "\n".join(lines_out) if changed else text


# --- remote-SSH maintenance boundary ----------------------------------------------------------

def _probe_local_source_ip(family: int, target: str) -> Optional[str]:
    """Source address the kernel would use to reach *target*, or ``None``.

    A connected UDP socket only consults the routing table to select a source address — no datagram
    is sent, so this never blocks or leaks traffic. Probing several representative destinations
    surfaces per-route source IPs (default route, a second NIC, a VPN/Tailscale interface), so a
    multi-homed host's OWN addresses are all recognized as local — without any DNS."""
    try:
        probe = socket.socket(family, socket.SOCK_DGRAM)
        try:
            probe.connect((target, 9))
            return probe.getsockname()[0]
        finally:
            probe.close()
    except Exception:
        return None


def _local_host_identities() -> frozenset[str]:
    """Casefolded names/IPs that mean "this machine". A destination in this set is a loopback back
    to the calling gateway, so its lifecycle payload keeps full protection.

    Everything here is derived WITHOUT network I/O: ``gethostname`` is a local syscall and the
    source-IP probes use connected UDP sockets, which only consult the routing table (no packet is
    sent, nothing blocks). The guard runs on the tool-executor thread; a DNS hang there would wedge
    every terminal command until the gateway restarts (#77780/#78256), so DNS is never used — an
    unresolvable name is treated as ambiguous and stays blocked instead.
    """
    ids = {"localhost", "localhost.localdomain", "127.0.0.1", "::1", "0.0.0.0"}
    try:
        hostname = socket.gethostname()
        if hostname:
            ids.add(hostname)
            ids.add(hostname.split(".", 1)[0])
    except Exception:
        pass
    # Primary source of truth: every address on every local interface (getifaddrs, a local syscall —
    # no DNS, no network). This catches a multi-homed host's secondary NIC / VPN / bridge IPs, not
    # just the default-route address, so `ssh <any-own-ip> …` is recognized as a self-restart.
    try:
        import psutil

        for interface_addrs in psutil.net_if_addrs().values():
            for addr in interface_addrs:
                candidate = (addr.address or "").split("%", 1)[0]  # strip IPv6 zone id
                try:
                    ipaddress.ip_address(candidate)
                except ValueError:
                    continue  # MAC address / non-IP family
                ids.add(candidate)
    except Exception:
        # Fallback when psutil is unavailable: representative per-route source-IP probes (default
        # route, CGNAT/Tailscale, each RFC1918 block, IPv6) — no-send routing lookups. Less complete
        # than getifaddrs but still catches the common interfaces.
        try:
            for target in ("192.0.2.1", "100.64.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1"):
                source = _probe_local_source_ip(socket.AF_INET, target)
                if source:
                    ids.add(source)
            source6 = _probe_local_source_ip(socket.AF_INET6, "2001:db8::1")
            if source6:
                ids.add(source6.split("%", 1)[0])
        except Exception:
            pass
    return frozenset(identity.casefold() for identity in ids)


def _ssh_config_hostname(alias: str) -> Optional[str]:
    """Best-effort ``HostName`` for an ``ssh`` alias from ``~/.ssh/config`` (no network).

    Deliberately minimal and conservative: only top-level ``Host`` blocks whose whitespace-separated
    patterns contain the alias as an EXACT token (no ``*``/``?`` wildcard expansion) contribute a
    ``HostName``. Anything unparsed, oversized, or wildcard-only yields ``None`` — the caller then
    fails closed. This exists so fleet aliases (``Host jim`` / ``HostName 192.168.3.184``) resolve to
    a classifiable address, and so a ``HostName 127.0.0.1`` alias is correctly seen as loopback.
    """
    try:
        config = Path("~/.ssh/config").expanduser()
        if not config.is_file() or config.stat().st_size > _MAX_SSH_CONFIG_BYTES:
            return None
        target = alias.casefold()
        matched = False
        result: Optional[str] = None
        for raw_line in config.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # ssh_config accepts `Keyword Value` (whitespace) and `Keyword=Value`.
            if "=" in line and line.split("=", 1)[0].strip().isalpha():
                key, _, value = line.partition("=")
            else:
                parts = line.split(None, 1)
                key, value = parts[0], (parts[1] if len(parts) > 1 else "")
            key = key.strip().casefold()
            value = value.strip()
            if key == "host":
                if matched and result:
                    return result
                matched = target in {pattern.casefold() for pattern in value.split()}
            elif key == "hostname" and matched and value:
                result = value
        return result
    except Exception:
        return None


def _extract_ssh_host(destination: str) -> Optional[str]:
    """Bare host from an ssh destination token, or ``None`` when it is ambiguous/unparseable.

    Handles ``user@host``, ``ssh://user@host:port/...`` URIs and ``[ipv6]:port`` brackets. Any
    shell-expansion marker makes the true host runtime-dependent -> ``None`` (fail closed)."""
    if not destination or any(marker in destination for marker in _SSH_DEST_AMBIGUOUS_MARKERS):
        return None
    host = destination
    if host.startswith("ssh://"):
        host = host[len("ssh://"):].split("/", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if not host:
        return None
    if host.startswith("["):  # bracketed IPv6 literal, optionally with :port
        end = host.find("]")
        if end == -1:
            return None
        return host[1:end] or None
    if host.count(":") == 1:  # host:port (URI/scp form); bare IPv6 has >1 colon
        left, right = host.split(":")
        if right.isdigit():
            host = left
    return host or None


def _ssh_destination_is_remote(destination: str, *, _depth: int = 0) -> bool:
    """True only when *destination* is PROVABLY a host other than the calling gateway.

    IP literals are classified with ``ipaddress`` (no network): a global/private address that is not
    loopback/link-local/unspecified/multicast/reserved and not one of this host's own identities is
    remote. Non-IP names are remote only when an ``ssh_config`` ``HostName`` resolves them to a
    remote address; a bare name we cannot resolve without DNS is ambiguous and returns False. Every
    "not sure" path returns False so the lifecycle payload stays scanned/blocked (fail closed)."""
    host = _extract_ssh_host(destination)
    if host is None or _depth > 3:
        return False
    host = host.strip().rstrip(".")
    if not host:
        return False
    lowered = host.casefold()
    local = _local_host_identities()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        # Collapse IPv4-mapped IPv6 (`::ffff:192.168.9.9`) to the IPv4 it dials, so an alternate
        # spelling of this host's own address cannot masquerade as a different host.
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        if (address.is_loopback or address.is_link_local or address.is_unspecified
                or address.is_multicast or address.is_reserved):
            return False
        # Compare as parsed addresses so zero-compression / mapped forms of an own-IP collapse
        # together; fall back to a string compare for any non-IP identities.
        for identity in local:
            try:
                if ipaddress.ip_address(identity) == address:
                    return False
            except ValueError:
                continue
        return str(address).casefold() not in local
    if lowered in local or lowered == "localhost" or lowered.endswith(".localhost"):
        return False
    mapped = _ssh_config_hostname(lowered)
    if mapped and mapped.casefold() != lowered:
        return _ssh_destination_is_remote(mapped, _depth=_depth + 1)
    # A real hostname we can only resolve via DNS: ambiguous by contract -> fail closed.
    return False


def _ssh_o_key(value: str) -> str:
    """Extract the ``-o`` keyword exactly as ssh's config tokenizer sees it, casefolded.

    ssh_config accepts ``Keyword Value``, ``Keyword=Value``, a single leading ``=`` separator that
    ssh discards (``-o=HostName=x`` sets HostName), and quotes around the keyword
    (``-o '"HostName" 127.0.0.1'``). Strip a leading ``=``/whitespace run, take the first token, then
    strip surrounding quotes — so `HostName` is recognized in every spelling (fail-closed: mangled
    forms ssh would reject over-block harmlessly)."""
    first = re.split(r"[=\s]", re.sub(r"^[=\s]+", "", value), 1)[0]
    return first.strip("\"'").casefold()


def _ssh_short_cluster(token: str, next_token: str) -> tuple[bool, bool]:
    """Decode a single-dash ssh short-option cluster with getopt bundling semantics.

    Returns ``(is_fail_closed, consumes_next_token)``. ssh bundles no-arg flags ahead of an arg-taking
    letter (``-tqoHostName=x`` == ``-t -q -o HostName=x``); the arg is the remainder of the token after
    that letter, or the next token when nothing is attached. ``is_fail_closed`` is True when the
    arg-taking letter is ``o`` with a host-redirect OR local-exec keyword, or is ``J``/``F`` — i.e.
    anything that makes the destination not provably the sole executor."""
    if not token.startswith("-") or token.startswith("--") or token == "-":
        return False, False
    cluster = token[1:]
    for position, char in enumerate(cluster):
        if char not in _SSH_ARG_OPTION_CHARS:
            continue  # a bundled no-arg flag; keep scanning
        attached = cluster[position + 1:]
        value = attached if attached else next_token
        consumes_next = not attached
        if char == "o":
            return _ssh_o_key(value) in _SSH_FAILCLOSED_O_KEYS, consumes_next
        return char in _SSH_HOST_OVERRIDE_SHORT_CHARS, consumes_next
    return False, False  # cluster of no-arg flags only


def _ssh_parse_invocation(
    segment: list[str], ssh_index: int
) -> tuple[Optional[str], list[int], bool]:
    """``(destination, remote_command_operand_indices, fail_closed)`` for an ssh *segment*.

    Fully getopt-parses the invocation: options and their consumed values are separated from bare
    operands. The first operand is the destination; the rest are the remote command — and ONLY those
    operand indices are returned for masking, so trailing OPTIONS (``-o LocalCommand=…`` runs on the
    caller; ``-o HostName=…`` redirects the host) are never masked and keep facing the regex.
    Options are parsed both before and after the destination because a permuting getopt reorders
    them. ``fail_closed`` is True when any option redirects the host or runs locally (``-o`` host/
    local-exec key, ``-J``/``-F``). Returns ``(None, [], ...)`` when no destination operand is found.
    """
    index = ssh_index + 1
    operand_indices: list[int] = []
    fail_closed = False
    end_of_options = False
    while index < len(segment):
        token = segment[index]
        if not end_of_options and token == "--":
            end_of_options = True
            index += 1
            continue
        if not end_of_options and token.startswith("-") and token != "-":
            is_fail_closed, consumes_next = _ssh_short_cluster(
                token, segment[index + 1] if index + 1 < len(segment) else ""
            )
            fail_closed = fail_closed or is_fail_closed
            index += 2 if (consumes_next and index + 1 < len(segment)) else 1
            continue
        operand_indices.append(index)
        index += 1
    if not operand_indices:
        return None, [], fail_closed
    return segment[operand_indices[0]], operand_indices[1:], fail_closed


def _mask_remote_ssh_command_payloads(text: str) -> str:
    """Replace the remote-command payload of a provably-remote ``ssh`` call with a neutral token.

    Mirrors ``_mask_data_sink_arguments``: masking can only ever REMOVE a would-be match, and it
    only fires when the destination is provably a different host, so it can only ALLOW what the plain
    regex would block — never block more. The local component of a compound command, and everything
    for a loopback/local/ambiguous destination, is left verbatim and stays scanned. Any failure
    returns the original text unchanged (fail closed)."""
    try:
        changed = False
        lines_out: list[str] = []
        for line in text.splitlines() or [text]:
            try:
                tokens = _shlex_tokens(line)
            except ValueError:
                lines_out.append(line)
                continue
            rebuilt: list[str] = []
            for segment in _split_segments(tokens, keep_controls=True):
                index = _executed_command_index(segment)
                if index is not None and _executable_name(segment[index]) in _SSH_EXECUTABLES:
                    destination, remote_indices, fail_closed = _ssh_parse_invocation(segment, index)
                    if (destination and remote_indices and not fail_closed
                            and _ssh_destination_is_remote(destination)):
                        # Mask ONLY the remote-command operands — never option tokens/values, so a
                        # `-o LocalCommand=…` payload (runs on the caller) stays visible and blocked.
                        remote_set = set(remote_indices)
                        emitted = False
                        for position, tok in enumerate(segment):
                            if position in remote_set:
                                if not emitted:
                                    rebuilt.append("arg")
                                    emitted = True
                            else:
                                rebuilt.append(tok)
                        changed = True
                        continue
                rebuilt.extend(segment)
            lines_out.append(" ".join(rebuilt))
        return "\n".join(lines_out) if changed else text
    except Exception:
        # Fail closed: leave the command unmasked so it stays fully scanned. Log the unexpected
        # failure (mirrors the referenced-script walk fallback) so a real bug here isn't invisible.
        logger.warning("remote-ssh masking failed; scanning command unmasked", exc_info=True)
        return text


def _lifecycle_command_scan_with_data_exemption(text: str) -> bool:
    """Lifecycle scan exempting matches inside data arguments AND inside the remote-command payload
    of a provably-remote ssh call: cheap regex first (no-match pays nothing), then re-scan with those
    payloads masked; only a surviving match blocks."""
    if not contains_gateway_lifecycle_command(text):
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    masked = _mask_remote_ssh_command_payloads(normalized)
    return contains_gateway_lifecycle_command(_mask_data_sink_arguments(masked))


def _direct_lifecycle_scan(command: str) -> bool:
    """Pure-string direct scans: lifecycle regex (data-exempted) + submit."""
    return (
        _lifecycle_command_scan_with_data_exemption(command)
        or contains_launchctl_submit_command(command)
    )


# --- path handling ----------------------------------------------------------------------------

def _resolve_lenient(path: Path) -> Path:
    """``path.resolve(strict=False)``, falling back to *path* on OSError (unreadable/long) or
    ValueError (embedded NUL from decoded binary tokenized as a path) — never crash the guard."""
    try:
        return path.resolve(strict=False)
    except (OSError, ValueError):
        return path


def _is_cloud_placeholder_path(path: Path) -> bool:
    """True for paths inside a macOS FileProvider-backed subtree. ``O_NONBLOCK`` does not make
    regular-file reads non-blocking, so opening an evicted placeholder can wait indefinitely for
    hydration before any command timeout starts: identify the boundary from the path alone."""
    parts = path.parts
    return any(
        parts[index - 1] == "Library" and part in _CLOUD_PLACEHOLDER_MARKERS
        for index, part in enumerate(parts)
        if index
    )


def _on_cloud_path(path: Path) -> bool:
    """Lexical OR resolved cloud check: covers direct cloud paths and local symlinks into one."""
    return _is_cloud_placeholder_path(path) or _is_cloud_placeholder_path(_resolve_lenient(path))


def _expand_candidate_path(candidate: str) -> Optional[Path]:
    """Sanitize a tokenized path candidate at the ingestion boundary. Tokens from shlex-splitting
    arbitrary (possibly binary-decoded) text can carry NUL or junk that each downstream ``Path`` op
    rejects differently (ValueError, RuntimeError when HOME is unset under launchd, OSError); reject
    once here. ``None`` = not a real path, nothing to scan.

    Every OS-facing ``Path`` operation downstream (``expanduser``, ``os.open``, ``resolve``) raises a
    *different* exception for the same junk (``ValueError: embedded null byte``, ``RuntimeError: Could not
    determine home directory`` when HOME is unset under launchd, OSError for over-long paths). Rejecting
    here — once, before any OS call — is the whole-class fix; catching per-syscall was the whack-a-mole that
    produced #76762, #77703, #77780, and #78256.
    """
    if not candidate or "\x00" in candidate:
        return None
    try:
        return Path(candidate).expanduser()
    except (ValueError, RuntimeError, OSError):
        return None


def _resolved_or_nothing(candidate: str, cwd: Optional[str]) -> Iterator[Path]:
    """Yield *candidate* anchored on *cwd* (or the process cwd) when it is a real path."""
    path = _expand_candidate_path(candidate)
    if path is None:
        return
    if not path.is_absolute():
        try:
            path = Path(cwd or Path.cwd()) / path
        except OSError:
            # Path.cwd() can raise when the process cwd was deleted.
            return
    yield path


def _resolve_script_path(script_path: str) -> Optional[Path]:
    """Resolve a cron ``script`` value the way ``cron.scheduler`` does (relative paths live under
    ``<HERMES_HOME>/scripts/``) so the guard scans the file that will actually run."""
    from hermes_constants import get_hermes_home

    raw = _expand_candidate_path(script_path)
    if raw is None:
        return None
    if raw.is_absolute():
        return raw
    try:
        return get_hermes_home() / "scripts" / raw
    except (RuntimeError, OSError):
        # get_hermes_home() falls back to Path.home(), which raises when neither HERMES_HOME nor
        # HOME is resolvable (launchd/systemd) — same ingestion contract: nothing to scan.
        return None


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path is not None and path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


# --- referenced-script discovery --------------------------------------------------------------

def _iter_option_values(segment: list[str], start: int, option: str) -> Iterator[str]:
    """Yield values given to *option*, in both ``--opt v`` and ``--opt=v`` form."""
    prefix = option + "="
    for position in range(start + 1, len(segment)):
        token = segment[position]
        if token == option and position + 1 < len(segment):
            yield segment[position + 1]
        elif token.startswith(prefix):
            yield token[len(prefix):]


def _references_at(segment: list[str], index: int, cwd: Optional[str]) -> Iterator[Path]:
    """Yield the scripts the token at *index* executes, if any."""
    if index >= len(segment):
        return
    executable = segment[index]
    executable_name = _executable_name(executable)

    if executable_name in {".", "source"}:
        if len(segment) > index + 1:
            yield from _resolved_or_nothing(segment[index + 1], cwd)
        return

    if executable_name in _SHELL_EXECUTABLES:
        arguments = segment[index + 1 :]
        arg_index = 0
        while arg_index < len(arguments):
            argument = arguments[arg_index]
            if argument == "--":
                arg_index += 1
                break
            if argument in _SHELL_COMMAND_FLAGS:
                break
            if argument in _SHELL_OPTIONS_WITH_VALUES:
                arg_index += 2
                continue
            if argument.startswith("-"):
                arg_index += 1
                continue
            break
        if arg_index < len(arguments) and arguments[arg_index] not in _SHELL_COMMAND_FLAGS:
            yield from _resolved_or_nothing(arguments[arg_index], cwd)
        return

    # A bare "/" is pathlib's division operator in Python sources, not an executable; resolving it
    # hits the filesystem root and fails the regular-file check, hard-blocking innocent .py scripts.
    if executable.strip("/") and ("/" in executable or executable.endswith((".sh", ".bash", ".zsh"))):
        yield from _resolved_or_nothing(executable, cwd)


def _iter_referenced_shell_scripts(command: str, *, cwd: Optional[str] = None) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell. Each segment is read at the
    original token AND at the peeled wrapper target — additive on purpose: peeling must never REMOVE
    a reference (a local ``./timeout`` is a script, not the coreutils wrapper)."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        yield from _references_at(segment, index, cwd)
        peeled = _peel_transparent_prefixes(segment, index)
        if peeled != index:
            yield from _references_at(segment, peeled, cwd)


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` (and ``su -c`` / ``env -S``) for recursive
    scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        # Read at the ORIGINAL token: peeling past `su`/`env` would discard the command option.
        for option in _STRING_COMMAND_OPTIONS.get(_executable_name(segment[index]), ()):
            yield from _iter_option_values(segment, index, option)
        index = _executed_command_index(segment)
        if index is None or _executable_name(segment[index]) not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in _SHELL_COMMAND_FLAGS:
                yield arguments[arg_index + 1]
                break


# --- referenced-script reading ----------------------------------------------------------------

def _has_binary_magic(data: bytes) -> bool:
    """True when *data* starts with a known compiled-binary signature. Deliberately narrower than
    "contains a NUL": ``bash`` still executes a NUL-bearing script, so a padded script must not
    bypass the scan. A shebang always wins (interpreted, never binary). Extensions are not
    consulted: a suffixless script must still be scanned and fail closed if oversized."""
    if data.startswith(b"#!"):
        return False
    return data.startswith(_BINARY_MAGICS)


def _read_referenced_script(
    path: Path, *, max_bytes: Optional[int] = None
) -> tuple[Optional[str], bool]:
    """Read a referenced script without racing SQLite connection lifecycle.

    The registry check must cover the complete ``open``/``read``/``close``
    sequence. A separate ``has_live_connection`` check would leave a race in
    which another thread opens SQLite after the check but before this function
    closes its descriptor, cancelling that connection's POSIX locks.
    """
    from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access

    try:
        with offline_file_access(path, what="read referenced script"):
            return _read_referenced_script_unlocked(path, max_bytes=max_bytes)
    except LiveConnectionError:
        return None, True
    except (OSError, ValueError):
        # Invalid path values, including embedded NULs, are not scripts.
        return None, False


def _read_referenced_script_unlocked(
    path: Path, *, max_bytes: Optional[int] = None
) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads.

    Shared choke point for every local script read, so the cloud-placeholder refusal lives here: a
    FileProvider path is never opened — not even to check hydration — because an evicted
    placeholder's ``open()`` can hang preflight. Lexical check: direct paths; resolved: symlinks.
    ``max_bytes`` lowers the per-file cap to what the calling walk can still afford.

    See #88052.
    """
    byte_limit = _capped_read_limit(max_bytes)
    if _on_cloud_path(path):
        return None, True
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        # OSError: unreadable/missing/over-long. ValueError: embedded NUL in *path*. Either is
        # "nothing to scan" — never crash the guard.
        return None, False
    try:
        # ValueError: an embedded NUL byte in *path* itself — a binary's decoded bytes tokenized into a
        # bogus script path by the recursion (#77703).
        metadata = os.fstat(descriptor)
        # Directories are not scripts. Docker Desktop writes ``fpath=(~/.docker/completions …)`` into
        # ``~/.zshrc``; the walk then treats that dir as a referenced script and used to fail-closed,
        # blocking ``source ~/.zshrc`` (#86753). Devices/sockets stay fail-closed.
        if not stat.S_ISREG(metadata.st_mode):
            # Directories are not scripts (`fpath=(~/.docker/completions …)` in ~/.zshrc must not
            # block `source ~/.zshrc`). Devices/sockets stay fail-closed.
            return None, not stat.S_ISDIR(metadata.st_mode)
        # Sniff a small prefix first: compiled binaries are never shell scripts, so skip them
        # WITHOUT reading the rest or feeding decoded garbage into the recursion.
        # Deliberately NOT keyed on the mere presence of a NUL byte (#77927): bash executes a text script
        # straight past an embedded NUL, so NUL-bearing text must fall through to the magic-number check +
        # NUL-strip below.
        data = os.read(descriptor, _BINARY_SNIFF_BYTES)
        if _has_binary_magic(data):
            return None, False
        # A regular file whose size already exceeds the cap fails closed without reading it (the
        # walk budget can be far below 1 MiB).
        if metadata.st_size > byte_limit:
            return None, True
        # Read the remainder (bounded); loop because os.read may return short.
        while len(data) <= byte_limit:
            chunk = os.read(descriptor, byte_limit + 1 - len(data))
            if not chunk:
                break
            data += chunk
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    if _has_binary_magic(data):
        return None, False
    # Size check BEFORE NUL stripping: stripping shrinks the buffer and would let an oversized file
    # slip under the threshold past this fail-closed branch.
    if len(data) > byte_limit:
        return None, True
    if b"\x00" in data:
        data = data.replace(b"\x00", b"")
    return data.decode("utf-8", errors="replace"), False


def _sanitize_remote_script_text(
    text: Optional[str], *, max_bytes: Optional[int] = None
) -> tuple[Optional[str], bool]:
    """Apply the local-read contract to text from an untrusted ``read_remote_script`` callback: NUL
    means binary (nothing to scan, checked first); oversized fails closed. Size compares re-encoded
    *bytes* (matching the ``head -c`` wire bound): a >1 MiB multibyte file truncated at the byte cap
    decodes to fewer chars, and a char count would scan instead of failing.

    The recursion boundary must not trust its callbacks: any backend (SSH, Modal, Daytona, or a future one)
    can hand back raw binary bytes decoded as text, or arbitrarily large output. Enforced here rather than
    inside each callback so the guarantee holds for every callback, not just the ones we hardened. See
    #76762, #77703.
    """
    if not text or "\x00" in text:
        return None, False
    byte_limit = _capped_read_limit(max_bytes)
    if len(text) > byte_limit:
        return None, True  # chars <= bytes: over the cap without encoding
    if len(text.encode("utf-8", errors="replace")) > byte_limit:
        return None, True
    return text, False


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded scanner. Non-regular/oversized inputs fail closed via a
    lifecycle-shaped sentinel; missing/unreadable paths stay empty so scheduler validation reports
    them."""
    resolved = _resolve_script_path(script_path)
    if resolved is None:
        return ""
    script_text, unsafe = _read_referenced_script(resolved)
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


# --- recursive walk ---------------------------------------------------------------------------

def _contains_unsafe_gateway_action(
    command: str, *, cwd: Optional[str], depth: int, visited: set[Path], budget: _LifecycleScanBudget,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    # Charge BEFORE _direct_lifecycle_scan: every scan in it tokenizes with shlex.
    if not budget.charge_text(command):
        return _budget_exhausted("text", depth)
    if _direct_lifecycle_scan(command):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    def recurse(text: str, cwd: Optional[str]) -> bool:
        return _contains_unsafe_gateway_action(
            text, cwd=cwd, depth=depth + 1, visited=visited, budget=budget,
            read_remote_script=read_remote_script,
        )

    for payload in _iter_shell_command_payloads(command):
        if recurse(payload, cwd):
            return True

    for script_path in _iter_referenced_shell_scripts(command, cwd=cwd):
        # Do not touch a FileProvider path even to discover whether the file is hydrated.
        if _on_cloud_path(script_path):
            return True
        resolved = _resolve_lenient(script_path)
        if resolved in visited:
            continue
        if not budget.charge_path():
            return _budget_exhausted("paths", depth)
        visited.add(resolved)
        # Never read more than the walk can still afford to tokenize; a file larger than the
        # remainder fails closed exactly like an oversized one.
        script_text, unsafe = _read_referenced_script(script_path, max_bytes=budget.bytes_remaining)
        if unsafe:
            return True
        if script_text is None and read_remote_script is not None:
            # Local path missing; the remote backend's output crosses the same trust boundary as a
            # local read — sanitize identically (binary skip + size fail-closed).
            if not budget.charge_remote_read():
                return _budget_exhausted("remote reads", depth)
            script_text, unsafe = _sanitize_remote_script_text(
                read_remote_script(str(script_path)), max_bytes=budget.bytes_remaining
            )
            if unsafe:
                return True
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's directory, not the cwd.
        if recurse(script_text, _resolve_script_directory(str(resolved)) or cwd):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str, *, cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts.

    Total by construction: never raises. Direct scans are pure string ops; the referenced-script
    walk (filesystem, remote backends, shlex on decoded bytes) is best-effort defense-in-depth — an
    unexpected failure is logged and treated as "walk found nothing".

    This is the contract #76762 established ("a guarded path must never crash the guard") enforced at the
    boundary instead of per-syscall: a guard crash propagates out of ``tools/terminal_tool.py`` and breaks
    every terminal command until the gateway restarts (#77780, #78256), which is strictly worse than either
    verdict.
    """
    try:
        return _contains_unsafe_gateway_action(
            command, cwd=cwd, depth=0, visited=set(), budget=_LifecycleScanBudget(),
            read_remote_script=read_remote_script,
        )
    except Exception:
        logger.warning(
            "lifecycle guard referenced-script walk failed; "
            "falling back to direct-scan verdict",
            exc_info=True,
        )
        try:
            return _direct_lifecycle_scan(command)
        except Exception:
            # If even the data-argument masker fails, fall to raw regex + submit scan: stay total.
            return contains_gateway_lifecycle_command(command) or contains_launchctl_submit_command(command)


def check_gateway_lifecycle(prompt: Optional[str], script: Optional[str] = None) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a gateway-lifecycle
    command. The script is read from disk and concatenated with the prompt so a command cannot slip
    through by being split across the two. Callers let the ``ValueError``-shaped exception
    propagate."""
    combined = prompt or ""
    python_script = False
    if script:
        resolved_script = _resolve_script_path(script)
        # Attribute the refusal correctly: not a lifecycle command, but a cloud path never opened.
        if resolved_script is not None and _on_cloud_path(resolved_script):
            raise GatewayLifecycleBlocked(
                # Attribute the refusal correctly: the script is not known to contain a lifecycle command —
                # it lives on a cloud-synced FileProvider path (iCloud Drive / ~/Library/CloudStorage) that
                # the guard refuses to open because an evicted placeholder can hang preflight indefinitely
                # (#88052). Fail closed with the real reason instead of implying a dangerous lifecycle
                # command.
                "Blocked: the cron script lives on a cloud-synced path "
                "(iCloud Drive / ~/Library/CloudStorage). Opening an "
                "evicted FileProvider placeholder can hang the guard's "
                "preflight scan indefinitely, so it is refused without "
                "being read. Move the script to a local, non-cloud path "
                "(e.g. ~/.hermes/scripts/) and recreate the job."
            )
        python_script = resolved_script is not None and resolved_script.suffix == ".py"
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if python_script:
        # Python runs via the interpreter, never a POSIX shell, and the shell reference walk is a
        # false-positive generator on Python sources (pathlib "/" resolves to the filesystem root).
        # The regex still scans the full text; non-regular/oversized files fail closed (sentinel).
        # The data-exemption masker tokenizes with shlex, so it is charged against the walk budget.
        # The direct command regex below still scans the full text, so a literal `hermes gateway restart`
        # embedded in a .py script is still blocked. See #77131, #78398.
        if not _LifecycleScanBudget().charge_text(combined):
            unsafe = _budget_exhausted("text", 0)
        else:
            unsafe = _lifecycle_command_scan_with_data_exemption(combined)
    else:
        unsafe = contains_gateway_lifecycle_command_or_referenced_script(
            combined, cwd=_resolve_script_directory(script) if script else None
        )
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
