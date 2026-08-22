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

This is a defence-in-depth layer.  ``tools/terminal_tool.py`` blocks direct
commands and shell scripts they reference when ``_HERMES_GATEWAY=1``. It also
rejects ``launchctl submit`` in gateway sessions because launchd treats that
primitive as a persistent KeepAlive job, not a one-shot task. ``hermes gateway
stop|restart`` separately refuse to self-target from inside the gateway.
Blocking cron specs at creation time as well means the agent gets an immediate,
informative rejection instead of scheduling a job that will only fail
(silently) when it fires.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


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
    # `submit` and `bootstrap` are included alongside the direct verbs
    # (kickstart/etc.): `launchctl submit -l ai.hermes.gateway-<suffix> --
    # <helper-script>` (or `launchctl bootstrap gui/<uid> <plist>`) creates
    # a NEW keepalive job wrapping an arbitrary helper, which is how a
    # blocked direct restart/kill gets laundered into a persistent restart
    # loop instead (#62891) — same foot-gun, indirect shape. Neutral-label
    # submissions that dodge this text anchor are caught separately by
    # `contains_launchctl_submit_command` (execution-aware, label-independent).
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill / kill targeting the hermes gateway process. Both
    # token orders because real reproductions show both.
    r"|(?:p?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:p?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


# A backslash immediately followed by a newline is a POSIX shell line
# continuation — the shell joins the two lines before parsing. Every branch
# above uses `[^\n]*` between its verb and the gateway identifier so the
# match can't span unrelated lines of a longer cron prompt/script, but that
# also means a real multi-line shell invocation split across continuation
# lines (e.g. `launchctl submit \` / `  -l ai.hermes.gateway-... \` / `  -- ...`,
# the exact reported shape in #62891) would otherwise slip past. Collapse
# continuations to a single space before matching, mirroring what the shell
# itself does, rather than loosening `[^\n]*` and risking false positives
# across genuinely separate lines.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern."""
    if not text:
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return bool(_GATEWAY_LIFECYCLE_PATTERN.search(normalized))


_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_HEREDOC_NON_SHELL_EXECUTABLES = frozenset(
    {"node", "nodejs", "deno", "bun", "ruby", "perl", "php", "lua", "r", "rscript"}
)
_PYTHON_EXECUTABLE = re.compile(r"(?i)^python(?:\d+(?:\.\d+)*)?$")
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")


# Directory names that sit directly under a `Library` path component and
# mark a FileProvider-backed subtree: `Mobile Documents` is iCloud Drive;
# `CloudStorage` hosts every third-party FileProvider domain (Dropbox,
# OneDrive, Google Drive, Box, ...) on modern macOS.
_CLOUD_PLACEHOLDER_MARKERS = frozenset({"Mobile Documents", "CloudStorage"})


def _is_cloud_placeholder_path(path: Path) -> bool:
    """Return True for paths inside a macOS FileProvider-backed subtree.

    ``O_NONBLOCK`` does not make regular-file reads non-blocking.  Opening an
    evicted FileProvider placeholder below ``~/Library/Mobile Documents``
    (iCloud Drive) or ``~/Library/CloudStorage`` (Dropbox / OneDrive /
    Google Drive and other third-party providers) can therefore wait
    indefinitely for hydration.  The lifecycle guard runs before a terminal
    command's timeout starts, so it must identify this boundary from path
    metadata and fail closed without opening the file.
    """
    parts = path.parts
    return any(
        parts[index - 1] == "Library" and part in _CLOUD_PLACEHOLDER_MARKERS
        for index, part in enumerate(parts)
        if index
    )

# Executables whose arguments are DATA, not commands: search patterns, SQL
# statements, log filters. None of these can execute their argument text, so
# a lifecycle-shaped string inside their arguments (a grep pattern hunting
# for `systemctl restart hermes-gateway` in syslog, a SQL LIKE literal over a
# restart-events table) is diagnostics, not a lifecycle command. Deliberately
# conservative: no `awk` (system()), no `sed` (`s///e`), no `echo`/`printf`
# (routinely piped into a shell), no `mysql` (`\\!` and `system` escapes).
_DATA_SINK_EXECUTABLES = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ag", "ack", "journalctl", "sqlite3", "psql"}
)
# Argument shapes that can smuggle execution back INTO a data sink: command
# and process substitution anywhere, sqlite3 dot-commands (`.shell ...`),
# psql backslash escapes (`\! ...`). Any hit disables masking for the whole
# segment — fail closed to the plain regex verdict.
_UNSAFE_DATA_ARG_MARKERS = ("`", "$(", "<(", ">(", "\\!")
# A data sink piped into a shell/interpreter can feed matched lines straight
# to execution (`grep 'systemctl restart hermes-gateway' f | sh`); never mask
# such a line.
_PIPE_TO_INTERPRETER = re.compile(
    r"\|\s*&?\s*(?:sudo\s+)?(?:sh|bash|dash|ksh|zsh|xargs|eval|source)\b"
)

# Executable-image magic numbers: ELF, PE/COFF, Mach-O (universal + thin,
# both endiannesses). A referenced file starting with one of these is a
# compiled binary, never a shell script — don't read or scan it at all.
_BINARY_MAGIC_PREFIXES = (
    b"\x7fELF",
    b"MZ",
    b"\xca\xfe\xba\xbe",
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
)
_BINARY_SNIFF_BYTES = 4096




_ReadRemoteScriptFn = Callable[[str], Optional[str]]


def _iter_single_quote_logical_lines(command: str) -> Iterator[str]:
    """Yield physical lines joined only across valid single-quoted strings.

    Shell permits literal newlines inside single quotes.  Keeping those lines
    together lets ``shlex`` retain argument boundaries without changing the
    guard's established conservative handling of multiline double quotes.
    """
    masked = list(command)
    start = 0
    single_quoted = False
    double_quoted = False
    escaped = False
    comment = False
    word_start = True

    for index, char in enumerate(command):
        if comment:
            if char == "\n":
                comment = False
                word_start = True
                yield "".join(masked[start : index + 1])
                start = index + 1
            else:
                masked[index] = " "
            continue
        if single_quoted:
            if char == "'":
                single_quoted = False
            continue
        if escaped:
            escaped = False
            word_start = False
            continue
        if char == "\\":
            escaped = True
            word_start = False
            continue
        if double_quoted:
            if char == '"':
                double_quoted = False
            if char == "\n":
                # Preserve the historical line-wise fallback for multiline
                # double quotes, whose contents can execute substitutions.
                yield "".join(masked[start : index + 1])
                start = index + 1
            continue
        if char == "#" and word_start:
            comment = True
            masked[index] = " "
            continue
        if char == "'":
            single_quoted = True
            word_start = False
            continue
        if char == '"':
            double_quoted = True
            word_start = False
            continue
        if char == "\n":
            yield "".join(masked[start : index + 1])
            start = index + 1
            word_start = True
            continue
        if char.isspace() or char in _CONTROL_CHARS:
            word_start = True
        else:
            word_start = False

    if start < len(command):
        yield "".join(masked[start:])


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments, honoring quotes and comments."""
    normalized = command.replace("\\\n", "")
    for logical_line in _iter_single_quote_logical_lines(normalized):
        physical_lines = logical_line.splitlines() or [logical_line]
        try:
            lexer = shlex.shlex(
                logical_line,
                posix=True,
                punctuation_chars=";&|()",
            )
            lexer.whitespace_split = True
            # Real shell comments were masked by the logical-line pass.  A
            # hash inside a word (``tag#literal``) remains ordinary data.
            lexer.commenters = ""
            token_batches = [list(lexer)]
        except ValueError:
            # Malformed quoting remains conservative: retain the previous
            # best-effort scan of each physical line in the failed chunk.
            token_batches = []
            for line in physical_lines:
                try:
                    lexer = shlex.shlex(
                        line,
                        posix=True,
                        punctuation_chars=";&|()",
                    )
                    lexer.whitespace_split = True
                    lexer.commenters = ""
                    token_batches.append(list(lexer))
                except ValueError:
                    continue

        for tokens in token_batches:
            segment: list[str] = []
            for token in tokens:
                if token and set(token) <= _CONTROL_CHARS:
                    if segment:
                        yield segment
                        segment = []
                    continue
                segment.append(token)
            if segment:
                yield segment


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    for index, token in enumerate(segment):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            continue
        return index
    return None


def _classify_heredoc_receiver(receiver: Optional[str]) -> str:
    """Classify a heredoc receiver for referenced-shell traversal."""
    if not receiver or "/" in receiver:
        # An arbitrary path can be named ``python3`` while actually being a
        # shell script. Only a plain command name is eligible for exemption.
        return "unknown"
    name = receiver.lower()
    if name in _SHELL_EXECUTABLES:
        return "shell"
    if _PYTHON_EXECUTABLE.fullmatch(name) or name in _HEREDOC_NON_SHELL_EXECUTABLES:
        return "non_shell"
    return "unknown"


def _heredoc_receiver(fragment: str) -> Optional[str]:
    """Return the executable receiving a heredoc in one shell segment."""
    receiver: Optional[str] = None
    for segment in _iter_command_segments(fragment):
        index = _command_token_index(segment)
        if index is not None:
            receiver = segment[index]
    return receiver


def _mask_inert_shell_text(text: str) -> str:
    """Neutralize quoted text and comments before shell-syntax regexes.

    Quote contents become ``x`` rather than whitespace so a quoted argument
    still occupies one token (important for ``hash -p '/path' name``).  Newline
    positions are retained for command-boundary-aware patterns.
    """
    masked = list(text)
    quote: Optional[str] = None
    escaped = False
    comment = False
    word_start = True

    for index, char in enumerate(text):
        if comment:
            if char == "\n":
                comment = False
                word_start = True
            else:
                masked[index] = " "
            continue
        if quote is not None:
            if char == "\n":
                continue
            masked[index] = "x"
            if quote == '"' and escaped:
                escaped = False
                continue
            if quote == '"' and char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if escaped:
            escaped = False
            word_start = False
            continue
        if char == "\\":
            escaped = True
            word_start = False
            continue
        if char in {"'", '"'}:
            quote = char
            word_start = False
            masked[index] = "x"
            continue
        if char == "#" and word_start:
            comment = True
            masked[index] = " "
            continue
        if char == "\n" or char.isspace() or char in _CONTROL_CHARS:
            word_start = True
        else:
            word_start = False

    return "".join(masked)


def _heredoc_receiver_is_overridden(prefix: str, receiver: Optional[str]) -> bool:
    """Return whether earlier syntax overrides a plain receiver command.

    Shell functions and aliases take precedence over PATH lookup. A command
    named ``python3`` is therefore not evidence of Python when the same outer
    line defines or aliases that name before the heredoc invocation.
    """
    if not receiver or "/" in receiver:
        return True
    syntax = _mask_inert_shell_text(prefix)
    name = re.escape(receiver)
    function_pattern = re.compile(
        rf"(?:^|[;&|\n]\s*)(?:"
        rf"function\s+{name}(?:\s*\(\s*\))?"
        rf"|{name}\s*\(\s*\)"
        rf")(?=\s|$)"
    )
    alias_pattern = re.compile(
        rf"(?:^|[;&|\n]\s*)alias\s+{name}\s*="
    )
    path_assignment_pattern = re.compile(
        r"(?:^|[;&|\n]\s*)(?:export\s+)?PATH\s*="
    )
    hash_override_pattern = re.compile(
        rf"(?:^|[;&|\n]\s*)(?:builtin\s+)?hash\s+-p\s+\S+\s+{name}(?:\s|$)"
    )
    dynamic_shell_state_pattern = re.compile(
        r"(?:^|[;&|\n]\s*)(?:(?:source|\.)\s+|eval(?:\s|$))"
    )
    return bool(
        function_pattern.search(syntax)
        or alias_pattern.search(syntax)
        or path_assignment_pattern.search(syntax)
        or hash_override_pattern.search(syntax)
        or dynamic_shell_state_pattern.search(syntax)
    )


def _iter_heredoc_declarations(
    line: str,
    *,
    prior_outer_text: str = "",
) -> Iterator[tuple[str, bool, str, bool]]:
    """Yield heredoc metadata for one outer-shell line.

    The final flag records whether any part of the delimiter word was quoted.
    Only quoted non-shell heredocs are provably inert shell data: unquoted
    bodies still perform command substitution before reaching the receiver.
    """
    quote: Optional[str] = None
    escaped = False
    segment_start = 0
    index = 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        if char in _CONTROL_CHARS:
            segment_start = index + 1
            index += 1
            continue
        if not line.startswith("<<", index) or line.startswith("<<<", index):
            index += 1
            continue

        cursor = index + 2
        strip_tabs = False
        if cursor < len(line) and line[cursor] == "-":
            strip_tabs = True
            cursor += 1
        while cursor < len(line) and line[cursor] in " \t":
            cursor += 1
        if cursor >= len(line) or line[cursor] in "\r\n;&|()<>":
            index += 2
            continue

        delimiter_chars: list[str] = []
        delimiter_quoted = False
        delimiter_quote: Optional[str] = None
        ansi_c_quote = False
        unsupported_ansi_escape = False
        while cursor < len(line):
            current = line[cursor]
            if delimiter_quote is not None:
                if current == delimiter_quote:
                    delimiter_quote = None
                    ansi_c_quote = False
                    cursor += 1
                    continue
                if ansi_c_quote and current == "\\":
                    # Bash $'...' applies ANSI-C escape decoding. Rather than
                    # guess the resulting delimiter, leave escaped forms to
                    # the conservative outer-shell scanner below.
                    unsupported_ansi_escape = True
                if (
                    delimiter_quote == '"'
                    and current == "\\"
                    and cursor + 1 < len(line)
                    and line[cursor + 1] in {'"', "\\", "$", "`"}
                ):
                    cursor += 1
                    current = line[cursor]
                delimiter_chars.append(current)
                cursor += 1
                continue
            if current.isspace() or current in ";&|()<>":
                break
            if (
                current == "$"
                and cursor + 1 < len(line)
                and line[cursor + 1] in {"'", '"'}
            ):
                delimiter_quoted = True
                delimiter_quote = line[cursor + 1]
                ansi_c_quote = delimiter_quote == "'"
                cursor += 2
                continue
            if current in {"'", '"'}:
                delimiter_quoted = True
                delimiter_quote = current
                cursor += 1
                continue
            if current == "\\" and cursor + 1 < len(line):
                delimiter_quoted = True
                cursor += 1
                current = line[cursor]
            delimiter_chars.append(current)
            cursor += 1

        if delimiter_quote is not None or unsupported_ansi_escape:
            # Malformed or escape-bearing ANSI-C quoted delimiter: leave the
            # line to the existing conservative shell scanner rather than
            # guessing its resulting delimiter.
            index += 2
            continue
        delimiter = "".join(delimiter_chars)

        if delimiter:
            receiver = _heredoc_receiver(line[segment_start:index])
            receiver_kind = (
                "unknown"
                if _heredoc_receiver_is_overridden(
                    prior_outer_text + line[:index], receiver
                )
                else _classify_heredoc_receiver(receiver)
            )
            yield (
                delimiter,
                strip_tabs,
                receiver_kind,
                delimiter_quoted,
            )
        index = max(cursor, index + 2)


def _split_outer_shell_and_heredocs(
    command: str,
) -> tuple[str, list[tuple[str, str, bool]]]:
    """Separate outer shell text from heredoc bodies for shell-only traversal.

    Declarations are consumed in shell order. The original command remains
    untouched for the direct lifecycle scan. Unterminated recognized heredocs
    consume the remainder as their body; malformed declarations fall back to
    the existing conservative outer-shell scan.
    """
    lines = command.splitlines(keepends=True)
    if not lines:
        return command, []

    outer_lines: list[str] = []
    heredocs: list[tuple[str, str, bool]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_index += 1
        # Backslash-newline is removed before shell parsing, including inside
        # a delimiter word (``<<P\\`` + ``Y`` declares delimiter ``PY``).
        # Join only outer-shell continuation lines; heredoc body lines are
        # consumed below without this normalization.
        while line_index < len(lines):
            if line.endswith("\\\r\n"):
                line = line[:-3] + lines[line_index]
            elif line.endswith("\\\n"):
                line = line[:-2] + lines[line_index]
            else:
                break
            line_index += 1
        prior_outer_text = "".join(outer_lines)
        outer_lines.append(line)
        declarations = list(
            _iter_heredoc_declarations(
                line,
                prior_outer_text=prior_outer_text,
            )
        )
        for delimiter, strip_tabs, receiver_kind, delimiter_quoted in declarations:
            body_lines: list[str] = []
            while line_index < len(lines):
                candidate = lines[line_index].rstrip("\r\n")
                if strip_tabs:
                    candidate = candidate.lstrip("\t")
                if candidate == delimiter:
                    line_index += 1
                    break
                body_lines.append(lines[line_index])
                line_index += 1
            heredocs.append(
                (receiver_kind, "".join(body_lines), delimiter_quoted)
            )
    return "".join(outer_lines), heredocs


def _iter_backtick_payloads(text: str) -> Iterator[str]:
    """Yield legacy command substitutions from an expanding heredoc body.

    Heredoc bodies are not parsed as ordinary shell quoting: in an unquoted
    heredoc, every unescaped backtick can start/end command substitution even
    when surrounded by quote-looking data. Preserve escaped characters inside
    each payload and ignore an unmatched opener, which the shell itself treats
    as a syntax error rather than executable content.
    """
    payload: Optional[list[str]] = None
    escaped = False
    for char in text:
        if escaped:
            if payload is not None:
                payload.extend(("\\", char))
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "`":
            if payload is None:
                payload = []
            else:
                yield "".join(payload)
                payload = None
            continue
        if payload is not None:
            payload.append(char)


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: the label of a submitted/bootstrapped job is
    chosen by whoever writes it, so a neutral name (``ai.hermes.svc-reload-tmp``)
    defeats any label-anchored regex (#62891, second reproduction). Both verbs
    register a NEW persistent launchd job (``submit`` jobs get KeepAlive
    semantics; ``bootstrap`` loads an arbitrary plist), which is never safe to
    do from inside the gateway process.
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        if Path(segment[index]).name == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _mask_data_sink_arguments(text: str) -> str:
    """Replace data-sink executables' arguments with a neutral placeholder.

    The lifecycle regex is command-shaped, but it cannot tell an EXECUTED
    ``systemctl restart hermes-gateway`` from the same characters appearing
    as *data* — a grep/rg pattern, a journalctl filter, a SQL string literal
    passed to sqlite3/psql. Those diagnostics commands were being rejected
    (false positives blocking legitimate cron prompts), e.g.::

        grep -c 'systemctl restart hermes-gateway' /var/log/syslog
        sqlite3 db "SELECT msg FROM log WHERE msg LIKE '%systemctl restart hermes-gateway%'"

    This masker shell-tokenizes each line and, for command segments whose
    executable is a known data sink (``_DATA_SINK_EXECUTABLES``), replaces
    every argument with ``arg``. The caller then re-runs the lifecycle regex
    on the masked text: a match that survives masking sits OUTSIDE any data
    argument and is a real command.

    Strictly fail-closed: masking is skipped (leaving the original,
    regex-matching text in place) whenever the line pipes into a shell or
    interpreter, any argument carries an execution-capable marker
    (substitution, sqlite3 ``.``-commands, psql ``\\!``), or the line cannot
    be tokenized at all. Masking can therefore only ever ALLOW a command the
    plain regex would have blocked — never block one it would have allowed —
    so it runs solely as a second-pass exemption check.
    """
    lines_out: list[str] = []
    changed = False
    for line in text.splitlines() or [text]:
        if _PIPE_TO_INTERPRETER.search(line):
            lines_out.append(line)
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            lines_out.append(line)
            continue

        segments: list[list[str]] = []
        current: list[str] = []
        for token in tokens:
            if token and set(token) <= _CONTROL_CHARS:
                segments.append(current)
                segments.append([token])
                current = []
                continue
            current.append(token)
        segments.append(current)

        rebuilt: list[str] = []
        for segment in segments:
            if not segment:
                continue
            index = _command_token_index(segment)
            if index is not None and Path(segment[index]).name in _DATA_SINK_EXECUTABLES:
                arguments = segment[index + 1 :]
                if not any(
                    argument.startswith(".")
                    or any(marker in argument for marker in _UNSAFE_DATA_ARG_MARKERS)
                    for argument in arguments
                ):
                    changed = True
                    rebuilt.extend(segment[: index + 1])
                    rebuilt.extend("arg" for _ in arguments)
                    continue
            rebuilt.extend(segment)
        lines_out.append(" ".join(rebuilt))
    if not changed:
        return text
    return "\n".join(lines_out)


def _lifecycle_command_scan_with_data_exemption(text: str) -> bool:
    """Lifecycle-regex scan that exempts matches living inside data arguments.

    Two-pass: the cheap regex first (the overwhelmingly common no-match case
    pays nothing extra); on a raw match, re-scan with data-sink arguments
    masked out. Only a match that survives masking — i.e. one in actual
    command position — blocks.
    """
    if not contains_gateway_lifecycle_command(text):
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return contains_gateway_lifecycle_command(_mask_data_sink_arguments(normalized))


def _direct_lifecycle_scan(command: str) -> bool:
    """Pure-string direct scans: lifecycle regex (data-exempted) + submit."""
    return _lifecycle_command_scan_with_data_exemption(
        command
    ) or contains_launchctl_submit_command(command)


def _expand_candidate_path(candidate: str) -> Optional[Path]:
    """Sanitize a tokenized path candidate at the ingestion boundary.

    Candidate tokens come from shlex-splitting arbitrary command text —
    including text recursively decoded from binaries or remote reads — so
    they can carry NUL bytes or other junk no real filesystem path can
    contain. Every OS-facing ``Path`` operation downstream (``expanduser``,
    ``os.open``, ``resolve``) raises a *different* exception for the same
    junk (``ValueError: embedded null byte``, ``RuntimeError: Could not
    determine home directory`` when HOME is unset under launchd, OSError
    for over-long paths). Rejecting here — once, before any OS call — is
    the whole-class fix; catching per-syscall was the whack-a-mole that
    produced #76762, #77703, #77780, and #78256.

    Returns ``None`` for candidates that cannot be a real path (nothing to
    scan), otherwise the ``expanduser()``-expanded ``Path``.
    """
    if not candidate or "\x00" in candidate:
        return None
    try:
        return Path(candidate).expanduser()
    except (ValueError, RuntimeError, OSError):
        return None


def _resolve_terminal_script_path(candidate: str, cwd: Optional[str]) -> Optional[Path]:
    path = _expand_candidate_path(candidate)
    if path is None:
        return None
    if not path.is_absolute():
        try:
            path = Path(cwd or Path.cwd()) / path
        except OSError:
            # Path.cwd() can raise when the process cwd was deleted.
            return None
    return path


def _iter_referenced_shell_scripts(
    command: str,
    *,
    cwd: Optional[str] = None,
    explicit_only: bool = False,
) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell.

    ``explicit_only`` retains only ``sh script`` and ``source script`` forms.
    It is used for quoted non-shell source bodies: explicit shell execution
    remains protected, while generic slash-looking language syntax is not
    promoted into an executable path.
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        executable = segment[index]
        executable_name = Path(executable).name

        if executable_name in {".", "source"}:
            if len(segment) > index + 1:
                resolved = _resolve_terminal_script_path(segment[index + 1], cwd)
                if resolved is not None:
                    yield resolved
            continue

        if executable_name in _SHELL_EXECUTABLES:
            arguments = segment[index + 1 :]
            arg_index = 0
            while arg_index < len(arguments):
                argument = arguments[arg_index]
                if argument == "--":
                    arg_index += 1
                    break
                if argument in {"-c", "--command"}:
                    break
                if argument in _SHELL_OPTIONS_WITH_VALUES:
                    arg_index += 2
                    continue
                if argument.startswith("-"):
                    arg_index += 1
                    continue
                break
            if arg_index < len(arguments) and arguments[arg_index] not in {
                "-c",
                "--command",
            }:
                resolved = _resolve_terminal_script_path(arguments[arg_index], cwd)
                if resolved is not None:
                    yield resolved
            continue

        if explicit_only and not (
            "/" in executable
            or executable.endswith((".sh", ".bash", ".zsh"))
        ):
            continue

        # A bare "/" token is pathlib's division operator in Python sources
        # (e.g. `Path.home() / ".hermes"`), not an executable reference.
        # Resolving it walks to the filesystem root and fails the
        # regular-file check below, hard-blocking innocent .py scripts
        # (#77131). Skip pure-separator tokens.
        if executable.strip("/"):
            if "/" in executable or executable.endswith((".sh", ".bash", ".zsh")):
                resolved = _resolve_terminal_script_path(executable, cwd)
                if resolved is not None:
                    yield resolved


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` for recursive scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None or Path(segment[index]).name not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in {"-c", "--command"}:
                yield arguments[arg_index + 1]
                break


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path is not None and path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


def _read_referenced_script(path: Path) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads.

    This is the shared choke point for every local script read the guard
    performs (the terminal walk in ``_contains_unsafe_gateway_action`` AND
    the cron-script scan in ``_read_script_for_scanning``), so the
    cloud-placeholder refusal lives here: a FileProvider path must never be
    opened — not even to discover whether the file is hydrated — because an
    evicted placeholder's ``open()`` can hang preflight indefinitely
    (#88052). The lexical check covers direct cloud paths; the resolved
    check covers local launchers that are symlinks into a cloud subtree.
    """
    if _is_cloud_placeholder_path(path):
        return None, True
    try:
        resolved = path.resolve(strict=False)
    except (OSError, ValueError):
        # OSError: unreadable/long paths. ValueError: embedded NUL byte
        # from a binary's decoded contents tokenized as a path — a
        # guarded path must never crash the guard (#76762).
        resolved = path
    if _is_cloud_placeholder_path(resolved):
        return None, True
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        # OSError: unreadable / missing / over-long paths. ValueError: an
        # embedded NUL byte in *path* itself — a binary's decoded bytes
        # tokenized into a bogus script path by the recursion (#77703). A
        # guarded read must never crash the guard, so treat either as
        # "nothing to scan" (mirrors the resolve() ValueError guard below).
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            # Directories are not scripts. Docker Desktop writes
            # ``fpath=(~/.docker/completions …)`` into ``~/.zshrc``; the
            # walk then treats that dir as a referenced script and used
            # to fail-closed, blocking ``source ~/.zshrc`` (#86753).
            # Devices/sockets stay fail-closed.
            if stat.S_ISDIR(metadata.st_mode):
                return None, False
            return None, True
        # Sniff a small prefix first: files that are clearly compiled
        # binaries (executable magic, or NUL bytes in the head) are never
        # shell scripts, so skip them WITHOUT reading the rest — reading a
        # megabyte of machine code just to discard it wastes the guard's
        # budget and (pre-#77703) fed decoded garbage into the recursion.
        data = os.read(descriptor, _BINARY_SNIFF_BYTES)
        if data.startswith(_BINARY_MAGIC_PREFIXES) or b"\x00" in data:
            return None, False
        # Read the remainder (bounded). Loop because os.read may return
        # short for non-regular-file-backed descriptors.
        while len(data) <= _MAX_REFERENCED_SCRIPT_BYTES:
            chunk = os.read(
                descriptor, _MAX_REFERENCED_SCRIPT_BYTES + 1 - len(data)
            )
            if not chunk:
                break
            data += chunk
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    # A NUL byte in the first chunk means this is a binary (ELF/Mach-O/
    # PE), not a shell script — scanning its decoded contents would
    # tokenize machine code and feed junk paths into the recursion
    # (including a `ValueError: embedded null byte` from Path.resolve,
    # #76762). Treat it as "nothing to scan" rather than unsafe: a binary
    # executed by the user is not a referenced *shell script*.
    if b"\x00" in data:
        return None, False
    if len(data) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return data.decode("utf-8", errors="replace"), False


def _sanitize_remote_script_text(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Apply the local-read contract to text from a ``read_remote_script`` callback.

    The recursion boundary must not trust its callbacks: any backend (SSH,
    Modal, Daytona, or a future one) can hand back raw binary bytes decoded
    as text, or arbitrarily large output. Mirror
    ``_read_referenced_script``'s semantics exactly — NUL bytes mean binary
    (nothing to scan, checked first, #77703), oversized text fails closed
    like an oversized local file (#76762) — so remote and local reads can
    never diverge again. The size check re-encodes to compare *bytes*
    (matching the local read and the ``head -c`` wire bound): a >1 MiB
    multibyte file truncated at the byte cap decodes to fewer characters
    than bytes, and a character-count check would scan the truncated text
    instead of failing closed. Enforced here rather than inside each
    callback so the guarantee holds for every callback, not just the ones
    we hardened.
    """
    if not text:
        return None, False
    if "\x00" in text:
        return None, False
    if len(text.encode("utf-8", errors="replace")) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return text, False


def _contains_unsafe_gateway_action(
    command: str,
    *,
    cwd: Optional[str],
    depth: int,
    visited: set[Path],
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
    explicit_shell_only: bool = False,
) -> bool:
    if _direct_lifecycle_scan(command):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    # Heredoc bodies are input to a receiving command, not additional outer
    # shell lines. Keep the direct scan above on the original text, but only
    # run shell-specific traversal over the outer command and bodies that are
    # actually consumed by a shell. Unknown receivers remain fail-closed by
    # taking the shell path.
    outer_command, heredocs = _split_outer_shell_and_heredocs(command)
    for payload in _iter_shell_command_payloads(outer_command):
        if _contains_unsafe_gateway_action(
            payload,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
            explicit_shell_only=explicit_shell_only,
        ):
            return True

    for receiver_kind, body, delimiter_quoted in heredocs:
        if not body:
            continue
        # A quoted delimiter suppresses shell expansion, so a clearly
        # non-shell receiver gets inert source/data. Unquoted bodies still
        # perform command substitution before stdin delivery and must retain
        # the conservative shell walk (e.g. $(sh ./restart.sh)).
        if receiver_kind == "non_shell" and delimiter_quoted:
            if _contains_unsafe_gateway_action(
                body,
                cwd=cwd,
                depth=depth + 1,
                # Restricted source-body traversal must not poison the full
                # outer-shell walk. Keep cycle protection local to this probe.
                visited=set(visited),
                read_remote_script=read_remote_script,
                explicit_shell_only=True,
            ):
                return True
            continue
        if not delimiter_quoted:
            for payload in _iter_backtick_payloads(body):
                if _contains_unsafe_gateway_action(
                    payload,
                    cwd=cwd,
                    depth=depth + 1,
                    visited=visited,
                    read_remote_script=read_remote_script,
                    explicit_shell_only=explicit_shell_only,
                ):
                    return True
        if _contains_unsafe_gateway_action(
            body,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
            explicit_shell_only=explicit_shell_only,
        ):
            return True

    for script_path in _iter_referenced_shell_scripts(
        outer_command,
        cwd=cwd,
        explicit_only=explicit_shell_only,
    ):
        # Do not touch a FileProvider path even to discover whether the file
        # is hydrated. The lexical check covers direct cloud paths; the
        # resolved check below covers local launchers that are symlinks into
        # a cloud subtree. _read_referenced_script repeats both checks as the
        # shared choke point, so every caller stays covered even if this
        # walk-level short-circuit is bypassed.
        if _is_cloud_placeholder_path(script_path):
            return True
        try:
            resolved = script_path.resolve(strict=False)
        except (OSError, ValueError):
            # OSError: unreadable/long paths. ValueError: embedded NUL byte
            # from a binary's decoded contents tokenized as a path — a
            # guarded path must never crash the guard (#76762).
            resolved = script_path
        if _is_cloud_placeholder_path(resolved):
            return True
        if explicit_shell_only:
            try:
                # A command-shaped path in an otherwise inert non-shell body
                # is retained to cover ambient receiver shadowing. Preserve
                # the exemption for local source-data FIFOs/directories; a
                # missing path is still eligible for a remote backend read.
                if script_path.exists() and not script_path.is_file():
                    continue
            except OSError:
                pass
        if resolved in visited:
            continue
        visited.add(resolved)
        script_text, unsafe = _read_referenced_script(script_path)
        if unsafe:
            return True
        if script_text is None and read_remote_script is not None:
            # Local path missing; try the remote backend if one is available.
            # The callback's output crosses the same trust boundary as a
            # local read — sanitize it identically before it enters the
            # recursion (binary skip + size fail-closed).
            script_text, unsafe = _sanitize_remote_script_text(
                read_remote_script(str(script_path))
            )
            if unsafe:
                return True
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's
        # directory, not the original command's cwd.
        script_dir = _resolve_script_directory(str(resolved)) or cwd
        if _contains_unsafe_gateway_action(
            script_text,
            cwd=script_dir,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
            explicit_shell_only=explicit_shell_only,
        ):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str,
    *,
    cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts.

    Total by construction: this function returns a verdict for *every*
    input and never raises. The direct scans below are pure string
    operations; the referenced-script walk touches the filesystem, remote
    backends, and shlex on arbitrary decoded bytes, so it is best-effort
    defense-in-depth — any unexpected failure inside it is logged and
    treated as "walk found nothing" rather than crashing the caller.

    This is the contract #76762 established ("a guarded path must never
    crash the guard") enforced at the boundary instead of per-syscall: a
    guard crash propagates out of ``tools/terminal_tool.py`` and breaks
    every terminal command until the gateway restarts (#77780, #78256),
    which is strictly worse than either verdict.
    """
    try:
        # Includes the direct regex/submit scans at depth 0.
        return _contains_unsafe_gateway_action(
            command,
            cwd=cwd,
            depth=0,
            visited=set(),
            read_remote_script=read_remote_script,
        )
    except Exception:
        logger.warning(
            "lifecycle guard referenced-script walk failed; "
            "falling back to direct-scan verdict",
            exc_info=True,
        )
        # Pure string scans of the top-level command — cannot raise.
        try:
            return _direct_lifecycle_scan(command)
        except Exception:
            # The data-argument masker tokenizes arbitrary text; if even
            # that fails, fall to the raw regex + submit scan so the guard
            # stays total.
            return contains_gateway_lifecycle_command(
                command
            ) or contains_launchctl_submit_command(command)




def _resolve_script_path(script_path: str) -> Optional[Path]:
    """Resolve a cron ``script`` value the same way the scheduler does.

    The scheduler (``cron.scheduler``) resolves a bare/relative script path
    under ``<HERMES_HOME>/scripts/`` and only accepts absolute paths as-is.
    We MUST mirror that here so the guard scans the file that will actually
    run — otherwise a job whose script lives at the scheduler's real location
    (``~/.hermes/scripts/restart.sh``) but is passed as the bare name
    ``restart.sh`` would read as a nonexistent relative path and silently
    scan prompt-only content, letting the command through.

    Returns ``None`` for values that cannot be a real path (NUL bytes,
    unexpandable ``~``) — the same ingestion contract as
    ``_expand_candidate_path``; such a value can never name a file the
    scheduler would execute, so there is nothing to scan.
    """
    from hermes_constants import get_hermes_home

    raw = _expand_candidate_path(script_path)
    if raw is None:
        return None
    if raw.is_absolute():
        return raw
    try:
        return get_hermes_home() / "scripts" / raw
    except (RuntimeError, OSError):
        # get_hermes_home() falls back to Path.home(), which raises when
        # neither HERMES_HOME nor HOME is resolvable (launchd/systemd
        # environments) — same ingestion contract: nothing to scan.
        return None


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded terminal-script scanner.

    Non-regular or oversized inputs fail closed by returning a lifecycle-shaped
    sentinel, while missing/unreadable/unresolvable paths remain empty so
    ordinary scheduler path validation can report them.
    """
    resolved = _resolve_script_path(script_path)
    if resolved is None:
        return ""
    script_text, unsafe = _read_referenced_script(resolved)
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


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
    python_script = False
    if script:
        resolved_script = _resolve_script_path(script)
        if resolved_script is not None:
            try:
                real_script = resolved_script.resolve(strict=False)
            except (OSError, ValueError):
                real_script = resolved_script
            if _is_cloud_placeholder_path(resolved_script) or _is_cloud_placeholder_path(
                real_script
            ):
                # Attribute the refusal correctly: the script is not known to
                # contain a lifecycle command — it lives on a cloud-synced
                # FileProvider path (iCloud Drive / ~/Library/CloudStorage)
                # that the guard refuses to open because an evicted
                # placeholder can hang preflight indefinitely (#88052).
                # Fail closed with the real reason instead of implying a
                # dangerous lifecycle command.
                raise GatewayLifecycleBlocked(
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
        # Python is executed by the interpreter, never through a POSIX
        # shell: the shell-script reference walk is a false-positive
        # generator on Python sources (pathlib's "/" operator resolves to
        # the filesystem root and trips the regular-file check, blocking
        # every innocent .py cron script, #77131). The direct command
        # regex below still scans the full text, so a literal
        # `hermes gateway restart` embedded in a .py script is still
        # blocked. Non-regular/oversized script files still fail closed
        # via the lifecycle-shaped sentinel in _read_script_for_scanning.
        unsafe = _lifecycle_command_scan_with_data_exemption(combined)
    else:
        script_dir = _resolve_script_directory(script) if script else None
        unsafe = contains_gateway_lifecycle_command_or_referenced_script(
            combined,
            cwd=script_dir,
        )
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
