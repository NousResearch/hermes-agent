from __future__ import annotations

import logging

import fnmatch
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import DashboardGovernanceContext
from .models import AccessDecision, EffectiveAccess

_SHELL_OPERATOR_RE = re.compile(r"(;|&&|\|\||\||`|\$\(|<\(|>\(|\s[<>]{1,2}\s|\d[<>])")

# Backticks and process substitution hide a nested command from segment
# parsing, so they stay blocked outright for governed users. Plain operators
# (;, &&, ||, |) are fine: every segment's argv0 is checked against the CLI
# allowlist below. Command substitution $(...) is NOT hard-blocked anymore:
# its inner command is extracted and segment-checked recursively (25-08-2026,
# to stop routine agent patterns like $(date +%F) failing for governed users).
_SHELL_SUBSTITUTION_RE = re.compile(r"(`|<\(|>\()")

_CMD_SUBSTITUTION_MARK = "__HERMES_SUBST__"

# Domain-wide-delegation CLIs. With `--as <email>` they act as ANY
# @synthwave.solutions mailbox; without it they run on the owner's own OAuth
# token (reads AND sends land as Michael). Hard requirement 29-08-2026: a
# governed non-admin may only drive their OWN account, so every call to one
# of these must carry `--as <their identity email>`; a missing or different
# subject is refused. Enforced here (not via grants) so a grant rewrite can
# never widen it: adding a role to the exempt set below takes a commit and a
# review, which is the point. Admins (bootstrap admins, owner/admin role) are
# exempt. admin_delivery joined them on 02-09-2026 on Michael's explicit
# decision, asked as "net zoals wij als admins": a delivery lead reads every
# colleague's mail, Chat and calendar, and sends as them too.
_DWD_CLIS = frozenset({"gchat", "gmail", "gws-hermes", "gdrive-dwd", "gdrive_dwd.py"})
_DWD_ADMIN_ROLES = frozenset({"owner", "admin", "admin_delivery"})
# Sentinel: no identity binding for this caller (admin or governance off).
_DWD_UNRESTRICTED = None
# Admins who may act as any colleague EXCEPT the listed accounts. In code for
# the same reason as the admin roles: only a commit can change it. 15-10-2026
# (Michael: "hij mag in iedereens email dwd behalve de mijne").
_DWD_ADMIN_EXCLUDED_SUBJECTS = {
    "marcel@synthwave.solutions": frozenset({"michael@synthwave.solutions"}),
}


class DwdExclusion(frozenset):
    """An admin caller that may act as anyone except these subjects."""


def dwd_identity_for(access: EffectiveAccess | None):
    """The identity a caller's DWD CLI calls must be bound to, or
    _DWD_UNRESTRICTED for admins. An unknown email binds to '' (fail closed).
    An admin with excluded subjects gets a DwdExclusion instead."""
    if access is None:
        return _DWD_UNRESTRICTED
    if "bootstrap_admin" in tuple(access.grant_sources or ()):
        return _DWD_UNRESTRICTED
    email = str(getattr(access.subject, "email", "") or "").strip().lower()
    if frozenset(access.roles or ()) & _DWD_ADMIN_ROLES:
        excluded = _DWD_ADMIN_EXCLUDED_SUBJECTS.get(email)
        return DwdExclusion(excluded) if excluded else _DWD_UNRESTRICTED
    return email


def _dwd_subjects(tokens: list[str]) -> list[str]:
    """Every `--as` subject in one segment, including argparse abbreviations
    (`--a`) and repeats: argparse takes the LAST one, so all must pass."""
    found = []
    for i, tok in enumerate(tokens):
        flag, eq, value = tok.partition("=")
        if flag not in ("--a", "--as"):
            continue
        if not eq:
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
        found.append(value.strip().lower())
    return found


def _dwd_subject(tokens: list[str]) -> str:
    """The `--as` subject in one segment ('' when absent)."""
    subjects = _dwd_subjects(tokens)
    return subjects[0] if subjects else ""


def _check_dwd_identity(segments: list[list[str]], identity: str) -> AccessDecision:
    for tokens in segments:
        argv0 = _segment_argv0(tokens)
        base = os.path.basename(argv0) if argv0 else ""
        if base not in _DWD_CLIS:
            continue
        subjects = _dwd_subjects(tokens)
        if not subjects or not all(subjects):
            return AccessDecision(False, "dwd_identity_required", detail=base)
        for subject in subjects:
            if isinstance(identity, DwdExclusion):
                if subject in identity:
                    return AccessDecision(False, "dwd_identity_excluded", detail=f"{base} --as {subject}")
                continue
            if not identity or subject != identity:
                return AccessDecision(False, "dwd_identity_mismatch", detail=f"{base} --as {subject}")
    return AccessDecision(True, "arguments_allowed")


# ── Blacklist accounts: structure is never a reason to refuse ───────────────
# 14-09-2026 (Michael: "blacklisted users kunnen overal bij, mijn tech team
# kan helemaal niks met de super agent zo"). For a default-allow account the
# strict parser above was the blocker: a `for` loop, a `{ }` group, backticks
# or an unparseable heredoc all came back as cli_compound_command_not_allowed
# or cli_shell_operator_not_allowed (Stephen's asset scan, Vansh's Power BI
# session check). Such an account is closed by its EXPLICIT blacklist only, so
# the lenient gate keeps exactly those checks, made robust against structure:
#   * a denied path anywhere in the line (regex over the raw string);
#   * a denied command name at every position where a word can be EXECUTED:
#     each segment's command word after assignments, redirects, control heads
#     and exec-forwarders (sudo, env, xargs, nohup, time, ...), the forwarders
#     themselves, the word after find -exec, interpreter targets, and the text
#     handed to sh -c, eval, $( ), backticks and process substitution. A word a
#     command merely mentions (a commit message, a grep pattern, a package
#     name, VAR=value) is not a command and does not refuse (sweep 14-09);
#   * the DWD identity binding wherever a DWD CLI is such a command word;
#   * the HERMES_DWD_IDENTITY tamper guard.
# Whitelist and legacy accounts keep the strict parser unchanged.
_TOKEN_SHELL_PUNCT = "`'\"()$;|&{}<>\n"
_EXEC_FORWARDERS = frozenset({
    "sudo", "doas", "nohup", "setsid", "stdbuf", "nice", "ionice", "time", "timeout",
    "watch", "chroot", "strace", "ltrace", "xargs", "parallel", "exec", "command",
    "builtin", "env",
})
# forwarder options that consume the next word, so it is not mistaken for the
# command: sudo -u USER, xargs -n N, nice -n N, timeout -k D, env -u NAME ...
_FORWARDER_OPTS_WITH_ARG = frozenset({"-u", "-g", "-n", "-P", "-L", "-s", "-d", "-a", "-E", "-k", "-C", "-o", "-f", "-c", "-p"})
_FORWARDER_POSITIONALS = {"timeout": 1, "chroot": 1}
_SHELL_NAMES = frozenset({"sh", "bash", "dash", "zsh", "ksh"})
_FIND_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_FUNCTION_DEF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\)$")


def _lenient_tokens(command_s: str) -> list[str]:
    """Best-effort tokens of a command string. shlex first; when shell
    structure defeats it, whitespace-split so a command word still surfaces."""
    try:
        raw = shlex.split(command_s, posix=True)
    except ValueError:
        raw = command_s.replace("\n", " ").split()
    tokens = []
    for tok in raw:
        tok = tok.strip(_TOKEN_SHELL_PUNCT)
        if tok:
            tokens.append(tok)
    return tokens


def _lenient_segments(command_s: str) -> list[list[str]]:
    """Raw tokens per segment (punctuation kept so `name()` is still visible);
    callers strip _TOKEN_SHELL_PUNCT per token."""
    try:
        return [list(seg) for seg in _split_shell_segments(command_s)]
    except ValueError:
        try:
            return [shlex.split(command_s, posix=True)]
        except ValueError:
            return [command_s.replace("\n", " ").split()]


def _lenient_fragments(command_s: str) -> list[str]:
    """Nested shell text the shell would run as commands: backticks, $( ) and
    process substitution."""
    frags = re.findall(r"`([^`]*)`", command_s)
    try:
        _, inners = _extract_cmd_substitutions(command_s.replace("`", " "))
        frags += inners
    except ValueError:
        frags += re.findall(r"\$\(([^()]*)\)", command_s)
    frags += re.findall(r"[<>]\(([^()]*)\)", command_s)
    return [frag for frag in frags if frag.strip()]


def _lenient_heredocs(command: str) -> tuple[str, list[str]]:
    """Like _strip_heredocs, but hands back EVERY body, quoted delimiter or
    not: for the lenient gate a body fed to a shell (`bash <<'SH'`) is a list
    of commands whichever way its delimiter is quoted."""
    if "<<" not in command:
        return command, []
    lines = command.split("\n")
    kept: list[str] = []
    bodies: list[str] = []
    index = 0
    quote = ""
    while index < len(lines):
        line = lines[index]
        index += 1
        matches, quote = _heredoc_openers(line, quote)
        openers = [(match.group(3) or match.group(4), match.group(1) == "-") for match in matches]
        for match in reversed(matches):
            line = line[:match.start()] + " " + line[match.end():]
        kept.append(line)
        for delimiter, strip_tabs in openers:
            body: list[str] = []
            while index < len(lines):
                current = lines[index]
                if current.strip() == delimiter or (strip_tabs and current.lstrip("\t") == delimiter):
                    index += 1
                    break
                body.append(current)
                index += 1
            bodies.append("\n".join(body))
    return "\n".join(kept), bodies


def _lenient_command_words(command_s: str, depth: int = 0) -> list[tuple[str, list[str]]]:
    """(word, following tokens) for every word this command can EXECUTE."""
    if depth > 4:
        return []
    words: list[tuple[str, list[str]]] = []
    head, bodies = _lenient_heredocs(command_s)
    for raw_segment in _lenient_segments(head):
        segment = [t.strip(_TOKEN_SHELL_PUNCT) for t in raw_segment]
        skip = 0
        argv0_found = False
        for index, raw_tok in enumerate(raw_segment):
            tok = segment[index]
            if skip:
                skip -= 1
                continue
            if not tok:
                continue
            if _REDIRECT_TOKEN_RE.match(raw_tok):
                skip = 1
                continue
            tail = [t for t in segment[index + 1:] if t]
            if not argv0_found:
                # `name()` opens a function body: the words inside are the
                # commands, the name is not.
                if _ENV_ASSIGNMENT_RE.match(tok) or tok in _SHELL_CONTROL_HEADS or _FUNCTION_DEF_RE.match(raw_tok):
                    continue
                base = os.path.basename(tok)
                if base in _EXEC_FORWARDERS:
                    words.append((tok, tail))
                    skip = _FORWARDER_POSITIONALS.get(base, 0)
                    continue
                if tok.startswith("-"):
                    if tok in _FORWARDER_OPTS_WITH_ARG:
                        skip = 1
                    continue
                argv0_found = True
                words.append((tok, tail))
                if base in _SHELL_NAMES and "-c" in tail:
                    code = tail[tail.index("-c") + 1:]
                    if code:
                        words += _lenient_command_words(code[0], depth + 1)
                elif base == "eval":
                    words += _lenient_command_words(" ".join(tail), depth + 1)
                continue
            if tok in _FIND_EXEC_FLAGS and tail:
                words.append((tail[0], tail[1:]))
        words += [(target, []) for target in _interpreter_targets([t for t in segment if t])]
    for frag in _lenient_fragments(head):
        words += _lenient_command_words(frag, depth + 1)
    if bodies and any(os.path.basename(word) in _SHELL_NAMES for word, _ in words):
        for body in bodies:
            words += _lenient_command_words(body, depth + 1)
    return words


def _check_cli_command_lenient(command_s: str, grants, dwd_identity=_DWD_UNRESTRICTED, inbox: str = "") -> AccessDecision:
    denied_path = _check_denied_paths(command_s, grants, inbox)
    if not denied_path.allowed:
        return denied_path
    words = _lenient_command_words(command_s)
    if grants.cli_denied_commands:
        from .models import grant_matches
        for word, _tail in words:
            base = os.path.basename(word)
            if grant_matches(grants.cli_denied_commands, word) or (base and grant_matches(grants.cli_denied_commands, base)):
                return AccessDecision(False, "cli_command_denied", detail=word)
    if dwd_identity is not _DWD_UNRESTRICTED:
        tamper = _check_identity_env_tamper(command_s)
        if not tamper.allowed:
            return tamper
        for word, tail in words:
            if os.path.basename(word) in _DWD_CLIS:
                dwd_decision = _check_dwd_identity([[word, *tail]], dwd_identity)
                if not dwd_decision.allowed:
                    return dwd_decision
    return AccessDecision(True, "arguments_allowed")


def _command_substitution_end(command: str, start: int) -> int:
    """Find the closing shell parenthesis, respecting each nested quote scope."""
    index, depth, quote, word_start = start, 1, "", True
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = ""
        elif char == "\\" and index + 1 < len(command) and (
                not quote or command[index + 1] in '$`"\\\n'):
            if command[index + 1] != "\n":
                word_start = False
            index += 2
            continue
        elif char == quote and quote:
            quote = ""
        elif char in "\"'" and not quote:
            quote = char
        elif command.startswith("$(", index):
            index = _command_substitution_end(command, index + 2) + 1
            word_start = False
            continue
        elif not quote:
            if char == "#" and word_start:
                newline = command.find("\n", index)
                if newline < 0:
                    break
                index = newline
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        word_start = not quote and (char.isspace() or char in ";|&()<>")
        index += 1
    raise ValueError("unbalanced command substitution")


def _extract_cmd_substitutions(command: str, *, shell_quotes: bool = True) -> tuple[str, list[str]]:
    """Extract only substitutions the shell actually executes.

    Single-quoted and escaped payloads are data, including Python/JS source.
    Double quotes still expand backticks and $(...), but not process
    substitutions. Bare heredoc bodies do not treat quote marks specially.
    Unsupported active backticks/process substitutions remain refused.
    """
    out: list[str] = []
    inners: list[str] = []
    i, n = 0, len(command)
    quote, word_start = "", True
    while i < n:
        char = command[i]
        if quote == "'":
            if char == "'":
                quote = ""
        elif char == "\\" and i + 1 < n and (
                (shell_quotes and not quote) or command[i + 1] in '$`"\\\n'):
            out.append(command[i:i + 2])
            if command[i + 1] != "\n":
                word_start = False
            i += 2
            continue
        elif shell_quotes and char == quote and quote:
            quote = ""
        elif shell_quotes and char in "\"'" and not quote:
            quote = char
        elif shell_quotes and not quote and char == "#" and word_start:
            newline = command.find("\n", i)
            end = n if newline < 0 else newline
            out.append(command[i:end])
            i = end
            continue
        elif char == "`" or (not quote and (command.startswith("<(", i) or command.startswith(">(", i))):
            raise ValueError("unsupported active shell substitution")
        elif command.startswith("$(", i):
            end = _command_substitution_end(command, i + 2)
            inners.append(command[i + 2:end])
            out.append(_CMD_SUBSTITUTION_MARK)
            i = end + 1
            word_start = False
            continue
        out.append(char)
        word_start = not quote and (char.isspace() or char in ";|&()<>")
        i += 1
    return "".join(out), inners

# Heredoc opener: << or <<- plus a delimiter word, quoted or bare. The
# lookarounds keep <<< (herestring) out, and the trailing check keeps a stray
# "a << b" inside a quoted string from being read as one.
_HEREDOC_OPEN_RE = re.compile(
    r"(?<!<)<<(-?)(?!<)\s*(?:([\'\"])([A-Za-z_][A-Za-z0-9_]*)\2|([A-Za-z_][A-Za-z0-9_]*))(?=\s|$)"
)


def _heredoc_openers(line: str, quote: str) -> tuple[list[Any], str]:
    """Locate real redirection headers, not quoted or commented examples."""
    matches = []
    index = 0
    word_start = True
    while index < len(line):
        char = line[index]
        if char == "\\" and quote != "'":
            index += 2
            word_start = False
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            word_start = False
        elif char == "#" and word_start:
            break
        else:
            match = _HEREDOC_OPEN_RE.match(line, index)
            if match:
                matches.append(match)
                index = match.end()
                word_start = False
                continue
            word_start = char.isspace() or char in ";|&()<>"
        index += 1
    return matches, quote


def _strip_heredocs(command: str) -> tuple[str, list[str]]:
    """Remove heredoc bodies before segmentation, returning the remaining
    command plus the bodies the shell would still expand.

    A heredoc body is stdin DATA for a command word that is itself checked, not
    a list of commands: `python3 - <<'PY' ... PY` is one python3 call. Without
    this, any ; | && inside the script split the body into fake segments whose
    first word was matched against cli.commands and denied, so an ordinary
    python heredoc failed for every governed user. Reported by Hrishikesh
    Oemraw on 28 Aug 2026 ("everything I ask gives governance"), on
    `python3 - <<'PY' ... d=json.load(r); print('URL',u) ... PY`, denied as
    cli_command_not_allowed (print(URL,u)).

    This grants nothing new: `python3 -c '<anything>'` was already opaque to
    the gate. A BARE delimiter (<<PY) is still expanded by the shell, so those
    bodies are handed back for the caller's substitution checks; a quoted
    delimiter (<<'PY') is literal and is dropped.
    """
    if "<<" not in command:
        return command, []
    lines = command.split("\n")
    kept: list[str] = []
    expanded: list[str] = []
    index = 0
    quote = ""
    while index < len(lines):
        line = lines[index]
        index += 1
        matches, quote = _heredoc_openers(line, quote)
        openers = [
            (match.group(3) or match.group(4), match.group(2) is None, match.group(1) == "-")
            for match in matches
        ]
        for match in reversed(matches):
            line = line[:match.start()] + " " + line[match.end():]
        kept.append(line)
        for delimiter, expands, strip_tabs in openers:
            body: list[str] = []
            while index < len(lines):
                current = lines[index]
                if current.strip() == delimiter or (strip_tabs and current.lstrip("\t") == delimiter):
                    index += 1
                    break
                body.append(current)
                index += 1
            if expands:
                expanded.append("\n".join(body))
    return "\n".join(kept), expanded


# Shell builtins that carry no execution surface of their own; they may appear
# as a segment head without an allowlist entry (export CLOUDSDK_CONFIG=...; ...).
_SHELL_BUILTINS = frozenset({
    "export", "cd", "set", "unset", "true", "false", "test", "[", "[[", "pwd",
    "wait", "exit", "read", "umask", "ulimit", "echo", "printf",
    # Compound syntax is recognized separately and refused when command
    # constraints apply; these names never grant its nested executables.
    "for", "while", "until", "do", "done", "if", "then", "else", "elif", "fi",
    "case", "esac", "select", "in", "time", "{", "}", "!", "break", "continue",
    "return", "local", "declare", "shift",
})

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REDIRECT_TOKEN_RE = re.compile(r"^\d*(>>|>|<|>&|<&|&>>|&>)\d*$")


def _strip_shell_comments(command: str) -> str:
    """Remove shell comments without losing the newline command boundary.

    shlex's built-in comment handling consumes that newline, merging the
    next executable into the previous command's arguments. Quoted/escaped
    hashes and hashes inside a word are ordinary data and must survive.
    """
    out = []
    quote = ""
    word_start = True
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\" and quote != "'" and index + 1 < len(command):
            out.append(command[index:index + 2])
            if command[index + 1] != "\n":
                word_start = False
            index += 2
            continue
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            word_start = False
        elif char == "#" and word_start:
            index = command.find("\n", index)
            if index < 0:
                break
            continue
        else:
            word_start = char.isspace() or char in ";|&()<>"
        out.append(char)
        index += 1
    return "".join(out)


def _split_shell_segments(command: str) -> list[list[str]]:
    """Split a shell command on ;, &&, ||, |, & and newlines into token lists,
    respecting quoting. Raises ValueError on unparseable input."""
    lex = shlex.shlex(_strip_shell_comments(command), posix=True, punctuation_chars=";|&\n")
    lex.commenters = ""
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    segments: list[list[str]] = [[]]
    for token in lex:
        if token and all(char in ";|&\n" for char in token):
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [seg for seg in segments if seg]


# Words that introduce another command without consuming arguments of their
# own. Without this, `command gchat --as someone-else` read as a call to
# `command` and slipped past both the allowlist and the identity binding.
_COMMAND_WRAPPERS = frozenset({"command", "builtin", "exec", "nohup", "time"})
_SHELL_CONTROL_HEADS = frozenset({
    "for", "while", "until", "do", "done", "if", "then", "else", "elif", "fi",
    "case", "esac", "select", "in", "function", "coproc", "{", "}", "!",
})


def _segment_argv0(tokens: list[str]) -> str:
    """First real command word of a segment: skips VAR=val prefixes,
    redirection operators plus their targets, an `env` prefix, and wrapper
    words such as `command` that just introduce the real command."""
    skip_next = False
    saw_env = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if _REDIRECT_TOKEN_RE.match(token):
            skip_next = True
            continue
        if _ENV_ASSIGNMENT_RE.match(token):
            continue
        if token == "env" and not saw_env:
            saw_env = True
            continue
        if token in _COMMAND_WRAPPERS:
            continue
        return token
    return ""


@dataclass(frozen=True)
class ToolIdentity:
    name: str
    toolset: str = ""
    mcp_server: str = ""
    mcp_tool: str = ""


def _contains(values: frozenset[str], value: str) -> bool:
    return "*" in values or bool(value and value in values)


def _entry_toolset(registry: Any, tool_name: str) -> str:
    try:
        entry = registry.get_entry(tool_name)
        toolset = getattr(entry, "toolset", "") if entry is not None else ""
        if toolset:
            return str(toolset)
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("_entry_toolset: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
    try:
        return str(registry.get_toolset_for_tool(tool_name) or "")
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("_entry_toolset: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return ""


def identify_tool(tool_name: str, registry: Any) -> ToolIdentity:
    toolset = _entry_toolset(registry, tool_name)
    if toolset.startswith("mcp-"):
        server = toolset[4:]
        local = tool_name
        # Native registration uses mcp__<sanitized server>__<tool>. Keep
        # legacy names readable too, but derive the server from the trusted
        # registry toolset rather than guessing at underscore boundaries.
        safe_server = re.sub(r"[^A-Za-z0-9_]", "_", server)
        for prefix in (f"mcp__{safe_server}__", f"mcp_{server}_", f"mcp_{safe_server}_"):
            if tool_name.startswith(prefix):
                local = tool_name[len(prefix):]
                break
        return ToolIdentity(name=tool_name, toolset=toolset, mcp_server=server, mcp_tool=local)
    return ToolIdentity(name=tool_name, toolset=toolset)


def _mcp_tool_allowed(access: EffectiveAccess, identity: ToolIdentity) -> bool:
    from .models import grant_matches
    grants = access.grants
    server_keys = {identity.mcp_server, identity.toolset}
    if any(key and grant_matches(access.deny.mcp_servers, key) for key in server_keys):
        return False
    for key in ("*", identity.mcp_server, identity.toolset):
        denied = access.deny.mcp_tools.get(key, frozenset())
        if grant_matches(denied, identity.name) or grant_matches(denied, identity.mcp_tool):
            return False
    server_allowed = any(key and access.allows("mcp_servers", key) for key in server_keys)
    if not server_allowed:
        return False
    for key in ("*", identity.mcp_server, identity.toolset):
        allowed_names = grants.mcp_tools.get(key)
        if allowed_names and ("*" in allowed_names or identity.name in allowed_names or identity.mcp_tool in allowed_names):
            return True
    return False


def decide_tool_access(access: EffectiveAccess | None, tool_name: str, registry: Any) -> AccessDecision:
    """Return whether *tool_name* may be exposed/executed for access.

    Modes:
    - no context/off/report_only: allow for compatibility; report_only auditing
      can be layered later without hiding schemas or blocking calls.
    - enforce: require explicit tool, toolset or MCP grant.
    """
    if access is None or access.mode != "enforce":
        return AccessDecision(True, "governance_inactive")
    if tool_name == OPENCODE_TOOL_NAME:
        return _decide_opencode_tool_access(access, registry)
    identity = identify_tool(tool_name, registry)
    from .models import grant_matches
    if grant_matches(access.deny.tools, tool_name) or (identity.toolset and grant_matches(access.deny.toolsets, identity.toolset)):
        return AccessDecision(False, "explicit_deny", detail=tool_name)
    if (access.access_mode or access.access_level) and (access.access_level or "user") == "user" and tool_name in {"terminal", "execute_code"}:
        return AccessDecision(False, "access_level_not_allowed", detail=tool_name)
    if access.host_execution_restricted() and tool_name in {"terminal", "execute_code"}:
        # A host process can derive paths/environment names internally. Human
        # and AI review cannot turn string inspection into process isolation.
        return AccessDecision(False, "host_execution_conflicts_with_resource_deny", detail=tool_name)
    if access.allows("tools", tool_name):
        if identity.mcp_server and not _mcp_tool_allowed(access, identity):
            return AccessDecision(False, "tool_not_allowed")
        return AccessDecision(True, "tool_allowed")
    if identity.mcp_server:
        if _mcp_tool_allowed(access, identity):
            return AccessDecision(True, "mcp_tool_allowed")
        return AccessDecision(False, "tool_not_allowed")
    if identity.toolset and access.allows("toolsets", identity.toolset):
        return AccessDecision(True, "toolset_allowed")
    return AccessDecision(False, "tool_not_allowed")



# ── opencode_task ─────────────────────────────────────────────────────────────
#
# opencode_task starts a host process (OpenCode) that edits files with its own
# tools, so it is governed exactly like terminal: whoever may not run
# `opencode run` in that workdir through terminal may not reach it through this
# tool either. The tool is never granted on its own; it rides the terminal
# grant plus the opencode CLI grant (15-10-2026).

OPENCODE_TOOL_NAME = "opencode_task"
OPENCODE_CLI_PROBE = "opencode run"


def _opencode_cli_decision(access: EffectiveAccess) -> AccessDecision:
    if not (access.access_mode or access.access_level):
        return AccessDecision(True, "opencode_cli_allowed")
    if not access.grants.cli_commands:
        return AccessDecision(False, "opencode_cli_not_allowed")
    decision = _check_cli_command(OPENCODE_CLI_PROBE, access.grants, dwd_identity_for(access),
                                  lenient=access.access_mode == "blacklist")
    if not decision.allowed:
        return AccessDecision(False, "opencode_cli_not_allowed", detail=decision.reason)
    return AccessDecision(True, "opencode_cli_allowed")


def _decide_opencode_tool_access(access: EffectiveAccess, registry: Any) -> AccessDecision:
    from .models import grant_matches
    if grant_matches(access.deny.tools, OPENCODE_TOOL_NAME):
        return AccessDecision(False, "explicit_deny", detail=OPENCODE_TOOL_NAME)
    terminal = decide_tool_access(access, "terminal", registry)
    if not terminal.allowed:
        return AccessDecision(False, terminal.reason, detail=OPENCODE_TOOL_NAME)
    return _opencode_cli_decision(access)


def _opencode_argument_decision(access: EffectiveAccess, args: dict[str, Any], session_id: str) -> AccessDecision:
    """Workdir, task text and optional test command, each checked as terminal would."""
    workdir = str(args.get("workdir") or "").strip()
    if not workdir or not os.path.isabs(os.path.expanduser(workdir)):
        return AccessDecision(False, "opencode_workdir_required", detail=workdir)
    cli = _opencode_cli_decision(access)
    if not cli.allowed:
        return cli
    probe = decide_tool_argument_access(access, "terminal", {"command": OPENCODE_CLI_PROBE, "workdir": workdir},
                                        session_id=session_id)
    if not probe.allowed:
        return probe
    # OpenCode writes inside the workdir, so the write rules apply to it too.
    write = decide_tool_argument_access(access, "write_file",
                                        {"path": os.path.join(workdir, ".opencode-task-probe")},
                                        session_id=session_id)
    if not write.allowed:
        return AccessDecision(False, write.reason, detail=workdir)
    from .models import grant_matches
    for candidate in _PATHLIKE_RE.findall(str(args.get("task") or "")):
        if len(candidate) < 2:
            continue
        canonical = _resolve_candidate_path(candidate)
        if secret_path_hard_denied(candidate) or secret_path_hard_denied(canonical):
            return AccessDecision(False, SECRET_PATH_DENY_REASON, detail=candidate)
        if access.access_mode == "blacklist" and access.configuration_denies_file(candidate, canonical):
            return AccessDecision(False, "file_denied_glob", detail=candidate)
        if (grant_matches(access.deny.file_read_roots | access.deny.file_write_roots, canonical, path=True)
                or _matches_denied_glob(candidate, access.deny.file_denied_globs)
                or _matches_denied_glob(canonical, access.deny.file_denied_globs)):
            return AccessDecision(False, "explicit_deny", detail=candidate)
        if (access.access_mode != "blacklist" and _matches_denied_glob(candidate, access.grants.file_denied_globs)
                and not _matches_denied_glob(candidate, access.grants.file_allow_globs)):
            return AccessDecision(False, "file_denied_glob", detail=candidate)
    test_command = str(args.get("test_command") or "").strip()
    if test_command:
        test = decide_tool_argument_access(access, "terminal", {"command": test_command, "workdir": workdir},
                                           session_id=session_id)
        if not test.allowed:
            return AccessDecision(False, test.reason, detail=test.detail or test_command)
    return AccessDecision(True, "arguments_allowed")

def tool_allowed_for_context(ctx: DashboardGovernanceContext | None, tool_name: str, registry: Any) -> AccessDecision:
    from .context import policy_contexts
    try:
        contexts = policy_contexts(ctx)
        first = AccessDecision(True, "governance_inactive")
        for index, bounded_ctx in enumerate(contexts):
            continuation = _continuation_scope_decision(bounded_ctx, index)
            if not continuation.allowed:
                return continuation
            decision = _tool_allowed_for_single_context(bounded_ctx, tool_name, registry)
            if not decision.allowed:
                return decision
            if index == 0:
                first = decision
        return first
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("tool_allowed_for_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return AccessDecision(False, "continuation_policy_unavailable")


def _continuation_scope_decision(ctx, index):
    if index and ctx.access.mode == "enforce" and not ctx.access.is_profile_allowed(ctx.active_profile):
        return AccessDecision(False, "profile_not_allowed")
    if index and ctx.access.mode == "enforce" and (ctx.access.access_mode or ctx.access.access_level) and not ctx.access.has_permission("chat:use"):
        return AccessDecision(False, "chat_not_allowed")
    return AccessDecision(True, "continuation_scope_allowed")


def _as_bot_denial(ctx, decision):
    """Mark a refusal that came from the bot's own capability list, so it can
    be filed as a request to widen that bot (not the person)."""
    reason = str(getattr(decision, "reason", "") or "")
    if reason.startswith("bot_"):
        return decision
    profile = str(getattr(ctx, "active_profile", "") or "")
    detail = str(getattr(decision, "detail", "") or "")
    return AccessDecision(False, "bot_" + reason, detail=f"{profile}:{detail}" if profile else detail)


def _live_tool_scope_decision(ctx):
    from .context import workspace_allowed_for_context
    if not workspace_allowed_for_context(ctx):
        return AccessDecision(False, "workspace_access_revoked")
    ceiling = getattr(ctx, "bot_access_ceiling", None)
    if ceiling is not None:
        checker = getattr(ctx, "bot_access_check", None)
        try:
            if not callable(checker) or checker() is not True:
                return AccessDecision(False, "bot_access_revoked")
        except Exception as _boundary_error:
            logging.getLogger(__name__).error("_live_tool_scope_decision: operation failed",
                exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
            return AccessDecision(False, "bot_access_revoked")
    return AccessDecision(True, "tool_scope_allowed")


def _tool_allowed_for_validated_scope(ctx, tool_name, registry):
    """Per-tool ceilings; private to a freshly checked runtime call or projection."""
    ceiling = getattr(ctx, "bot_access_ceiling", None)
    if ceiling is not None:
        if tool_name in {"terminal", OPENCODE_TOOL_NAME} and not ceiling.grants.cli_commands:
            return AccessDecision(False, "bot_cli_not_selected")
        bounded = decide_tool_access(ceiling, tool_name, registry)
        if not bounded.allowed:
            return _as_bot_denial(ctx, bounded)
    return decide_tool_access(ctx.access if ctx is not None else None, tool_name, registry)


def _tool_allowed_for_single_context(ctx, tool_name, registry):
    scope = _live_tool_scope_decision(ctx)
    if not scope.allowed:
        return scope
    return _tool_allowed_for_validated_scope(ctx, tool_name, registry)


def _projection_contexts_allowed(contexts):
    return all(_continuation_scope_decision(ctx, index).allowed
               and _live_tool_scope_decision(ctx).allowed
               for index, ctx in enumerate(contexts))


def tool_projection_scope_allowed(ctx: DashboardGovernanceContext | None) -> bool:
    """Recheck live ceilings before reusing an already assembled schema list.

    The schema cache fingerprint already includes immutable grants/denies. Live
    callbacks cannot be represented by that fingerprint and are never cached.
    """
    from .context import policy_contexts
    try:
        return _projection_contexts_allowed(policy_contexts(ctx))
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("tool_projection_scope_allowed: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return False


@dataclass(frozen=True)
class ToolSchemaProjection:
    names: frozenset[str]
    scope_allowed: bool


def project_tool_names_for_context(ctx, tool_names, registry) -> ToolSchemaProjection:
    """Project schemas using one local envelope expansion, never execution rights.

    Read live workspace/bot membership before and after the projection. Keep all
    original per-tool and continuation ceilings; if membership is revoked while
    building the list, discard the entire result. Nothing survives this call.
    Runtime dispatch and argument checks use their independent live paths.
    """
    from .context import policy_contexts
    try:
        contexts = policy_contexts(ctx)
        if not _projection_contexts_allowed(contexts):
            return ToolSchemaProjection(frozenset(), False)
        allowed = set()
        for name in tool_names:
            try:
                if all(_tool_allowed_for_validated_scope(bounded_ctx, name, registry).allowed
                       for bounded_ctx in contexts):
                    allowed.add(name)
            except Exception as _boundary_error:
                # Match a failed single-tool decision without exposing it.
                logging.getLogger(__name__).error("project_tool_names_for_context: operation failed",
                    exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
                continue
        if not _projection_contexts_allowed(contexts):
            return ToolSchemaProjection(frozenset(), False)
        return ToolSchemaProjection(frozenset(allowed), True)
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("project_tool_names_for_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return ToolSchemaProjection(frozenset(), False)


def _resolve_candidate_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(os.path.expanduser(raw)).resolve(strict=False))
    except (OSError, RuntimeError):
        return os.path.abspath(os.path.expanduser(raw))


def _path_within_roots(path: str, roots: frozenset[str]) -> bool:
    if not path or not roots or "*" in roots:
        return True
    candidate = _resolve_candidate_path(path)
    for root in roots:
        root_path = _resolve_candidate_path(root)
        if root_path and (candidate == root_path or candidate.startswith(root_path.rstrip(os.sep) + os.sep)):
            return True
    return False


def _matches_denied_glob(path: str, globs: frozenset[str]) -> bool:
    if not path or not globs:
        return False
    candidate = _resolve_candidate_path(path)
    raw = str(path)
    return any(fnmatch.fnmatch(candidate, pattern) or fnmatch.fnmatch(raw, pattern) for pattern in globs)


# WebUI uploads and large composer pastes land in
# <hermes home>/webui/attachments/<session id>/ (upstream #2319: outside the
# workspace by design). That directory sits under the generic `**/.hermes/**`
# secret boundary, so a governed user who pasted a long text saw
# "file_denied_glob" on the .md the WebUI had just written from their own
# paste, and an access request was raised per paste (Hrishikesh, 14-09-2026).
# The inbox of the session a turn runs under IS that person's message, not a
# secret: reading it needs no grant. Every other session's inbox stays governed,
# and a symlink or `..` inside the inbox resolves to its real target first.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{6,64}")


def _attachment_inbox_root() -> str:
    from hermes_cli.synthpulse_settings import setting
    override = str(setting("webui.attachment_dir", "") or "").strip()
    if override:
        return _resolve_candidate_path(override)
    try:
        from hermes_cli.config import get_hermes_home
        return _resolve_candidate_path(str(Path(get_hermes_home()) / "webui" / "attachments"))
    except (ImportError, OSError, RuntimeError):
        return ""


def session_attachment_inbox(session_id: Any) -> str:
    """Resolved inbox directory of one WebUI session, or "" when unknown."""
    sid = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(sid):
        return ""
    root = _attachment_inbox_root()
    return os.path.join(root, sid) if root else ""


def _in_session_inbox(path: Any, inbox: str) -> bool:
    if not path:
        return False
    candidate = _resolve_candidate_path(path)
    if inbox and (candidate == inbox or candidate.startswith(inbox.rstrip(os.sep) + os.sep)):
        return True
    return _in_own_state_cache(candidate)


# The turn's own technical output lands under <hermes home>/cache/: the live
# transcript of a subagent it delegated (cache/delegation/live/<id>/task-N.log),
# a tool result too large for the context (cache/spillover) and terminal
# output that spilled (cache/terminal-output). All of it sits under the
# generic `**/.hermes/**` secret boundary, so an agent that wanted to know
# why its own worker timed out got "file_denied_glob" on the log it had just
# written itself (Yaser, IvCB repair, 21-09-2026), and admins kept approving
# one-off allow_globs per log file. That output is the caller's own work, not
# a secret: reading it needs no grant. Only the running turn's state
# directory qualifies (the context-local Hermes home, so a profile never
# reads another profile's cache), writes stay governed, and a symlink or
# `..` inside the cache resolves to its real target first.
_OWN_STATE_CACHE_SUBDIRS = ("delegation", "spillover", "terminal-output")


def _own_state_cache_roots() -> tuple[str, ...]:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home()) / "cache"
    except (ImportError, OSError):
        return ()
    return tuple(_resolve_candidate_path(str(base / sub)) for sub in _OWN_STATE_CACHE_SUBDIRS)


def _in_own_state_cache(path: Any) -> bool:
    if not path:
        return False
    candidate = _resolve_candidate_path(path)
    for root in _own_state_cache_roots():
        if candidate == root or candidate.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def _command_id(command: Any) -> tuple[str, str]:
    raw = str(command or "").strip()
    if not raw:
        return "", ""
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        parts = raw.split()
    if not parts:
        return "", ""
    argv0 = parts[0]
    return argv0, os.path.basename(argv0)


def _interpreter_targets(tokens: list[str]) -> list[str]:
    """Directly named interpreter scripts/modules, never inline code text.

    This recognizes ordinary Python, Node and shell invocation operands. It
    does not attempt to inspect arbitrary program text or resolve aliases.
    """
    skip_next = False
    saw_env = False
    words = []
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if _REDIRECT_TOKEN_RE.match(token):
            skip_next = True
            continue
        if _ENV_ASSIGNMENT_RE.match(token):
            continue
        if token == "env" and not saw_env:
            saw_env = True
            continue
        if token in _COMMAND_WRAPPERS:
            continue
        words = tokens[index:]
        break
    if not words:
        return []
    name = os.path.basename(words[0])
    python = bool(re.fullmatch(r"(?:python|pypy)(?:[0-9]+(?:\.[0-9]+)*)?", name))
    node = name in {"node", "nodejs"}
    shell = name in {"sh", "bash", "dash", "zsh", "ksh"}
    if not (python or node or shell):
        return []
    targets = []
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            return targets + words[index + 1:index + 2]
        if not word.startswith("-"):
            return targets + [word]
        if word == "-":
            return targets
        # -c / --eval / -e / -p contain program text, not a script operand.
        if (python and word.startswith("-c")) or (shell and word.startswith("-") and not word.startswith("--") and "c" in word[1:]):
            return targets
        if node and (word in {"--eval", "--print"} or word.startswith(("--eval=", "--print=", "-e", "-p"))):
            return targets
        if python and word.startswith("-m"):
            return targets + ([word[2:]] if len(word) > 2 else words[index + 1:index + 2])
        if node and (word.startswith(("--require=", "--import=", "--loader=")) or (word.startswith("-r") and not word.startswith("--") and len(word) > 2)):
            targets.append(word.split("=", 1)[1] if "=" in word else word[2:])
            index += 1
            continue
        if node and word in {"--require", "-r", "--loader", "--import"}:
            targets.extend(words[index + 1:index + 2])
            index += 2
            continue
        if (python and word in {"-W", "-X"}) or (shell and word in {"-o", "-O"}):
            index += 2
        else:
            index += 1
    return targets


def _skill_names(args: Mapping[str, Any] | dict[str, Any]) -> list[str]:
    """Every skill a skill_view/skill_manage call targets.

    skill_manage's advertised shape is a batch: {"operations": [{"name":
    ..., "action": ...}, ...]} with no top-level name. Reading only
    args["name"] judged an empty string, so every batch edit was refused with
    skill_manage_not_allowed, for admins with `*` as well (Yaser and Michaël,
    22-09-2026). The legacy flat shape (top-level name) stays supported.
    """
    names: list[str] = []
    top = str(args.get("name") or "").strip()
    if top:
        names.append(top)
    operations = args.get("operations")
    if isinstance(operations, (list, tuple)):
        for op in operations:
            if isinstance(op, Mapping):
                name = str(op.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
    return names


def _skill_name_allowed(values: frozenset[str], name: Any) -> bool:
    skill = str(name or "").strip()
    if not skill:
        return False
    if "*" in values or skill in values:
        return True
    # Skill trees expose category-prefixed names ("synthwave/opencode",
    # "autonomous-ai-agents/opencode"); grants list bare names. Match on the
    # final path segment so one grant covers every category alias.
    return skill.rsplit("/", 1)[-1] in values


# A path anywhere in a command line, quoted or not: `cat ~/.hermes/x.json`,
# `python3 -c "open('/home/.../key.json')"`, `cp a b` all expose one.
_PATHLIKE_RE = re.compile(r"(?:~|\.{0,2}/)[A-Za-z0-9._~/@+-]*")


def _check_denied_paths(command_s: str, grants, inbox: str = "") -> AccessDecision:
    """Apply the file denied-globs to shell commands too.

    The read_file/write_file tools honoured denied_globs, the terminal did
    not, so `cat ~/.hermes/gmail-dwd-sa.json` handed a governed user the
    domain-wide-delegation key that the file tools refused them (found
    29-08-2026). One path per denial keeps the message actionable, and the
    per-person allow_globs exception still opens a single file.
    """
    if not grants.file_denied_globs:
        return AccessDecision(True, "arguments_allowed")
    for candidate in _PATHLIKE_RE.findall(command_s):
        if len(candidate) < 2 or _in_session_inbox(candidate, inbox):
            continue
        if _matches_denied_glob(candidate, grants.file_denied_globs) \
                and not _matches_denied_glob(candidate, grants.file_allow_globs):
            return AccessDecision(False, "file_denied_glob", detail=candidate)
    return AccessDecision(True, "arguments_allowed")


# ── Built-in hard deny: the SynthPulse secret mount (plan 4.3) ──────────────
# Pair tokens for A2A, the memory hub actor key and other installation secrets
# are mounted under this root, only into the containers that need them. The
# agent's own file tools run in such a container (the A2A adapter lives in the
# engine), so they never read, search, write or patch below it: for every
# caller, administrators included, in every governance mode and also without a
# governance context. This is deliberately not a policy `deny.files` entry:
# that would make host_execution_restricted() true and close the terminal,
# while terminals on client installations run without this mount anyway
# (go-live precondition of C8). Not configurable, so no setting can weaken it.
SECRET_PATH_ROOTS: tuple[str, ...] = ("/run/synthpulse/secrets",)
SECRET_PATH_DENY_REASON = "secret_path_hard_deny"
_FILE_TOOL_NAMES = frozenset({"read_file", "search_files", "write_file", "patch"})
# The V4A headers the patch tool accepts (tools/patch_parser.py), matched as
# leniently as the parser itself.
_V4A_FILE_HEADER_RE = re.compile(r"^\s*\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+?)\s*$", re.MULTILINE)
_V4A_MOVE_HEADER_RE = re.compile(r"^\s*\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+?)\s*$", re.MULTILINE)


def _secret_roots() -> list[str]:
    """Each configured root lexically and, when it is a link, by its target."""
    roots: list[str] = []
    for root in SECRET_PATH_ROOTS:
        for candidate in (os.path.normpath(root), _realpath(root)):
            if candidate and candidate not in roots:
                roots.append(candidate)
    return roots


def _realpath(path: str) -> str:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return ""


def _secret_path_candidates(raw: Any, *, likely_only: bool = False) -> list[str]:
    """Absolute spellings a file tool could resolve ``raw`` to.

    A relative path is joined with the process cwd and with an absolute
    TERMINAL_CWD (the file tools resolve against the task's terminal cwd), and
    every candidate is also taken through its symlinks. ``likely_only`` keeps
    one base for a relative path (TERMINAL_CWD when set, else the process cwd),
    for checks where an extra candidate would refuse too much.
    """
    text = str(raw if raw is not None else "").strip()
    if not text:
        return []
    try:
        expanded = os.path.expanduser(text)
    except (OSError, RuntimeError):
        expanded = text
    lexical: list[str] = []
    if os.path.isabs(expanded):
        lexical.append(os.path.normpath(expanded))
    else:
        bases: list[str] = []
        terminal_cwd = os.getenv("TERMINAL_CWD", "").strip()
        if terminal_cwd and os.path.isabs(terminal_cwd):
            bases.append(terminal_cwd)
        if not bases or not likely_only:
            try:
                bases.append(os.getcwd())
            except OSError:
                pass
        lexical.extend(os.path.normpath(os.path.join(base, expanded)) for base in bases)
    candidates: list[str] = []
    for path in lexical:
        for candidate in (path, _realpath(path)):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _path_at_or_below(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def secret_path_hard_denied(raw: Any) -> bool:
    """True when ``raw`` names the secret mount or anything below it."""
    roots = _secret_roots()
    return any(_path_at_or_below(candidate, root)
               for candidate in _secret_path_candidates(raw) for root in roots)


def _search_reaches_secret_mount(raw: Any) -> bool:
    """A search rooted above an existing mount would descend into it."""
    roots = [root for root in _secret_roots() if os.path.lexists(root)]
    return any(_path_at_or_below(root, candidate)
               for candidate in _secret_path_candidates(raw, likely_only=True) for root in roots)


def _file_tool_paths(tool_name: str, args: Mapping[str, Any]) -> list[Any]:
    paths: list[Any] = []
    if tool_name == "search_files" or "path" in args:
        paths.append(args.get("path") or ".")
    if tool_name == "patch":
        patch_text = args.get("patch")
        if isinstance(patch_text, str) and "***" in patch_text:
            paths.extend(match.group(1) for match in _V4A_FILE_HEADER_RE.finditer(patch_text))
            for match in _V4A_MOVE_HEADER_RE.finditer(patch_text):
                paths.extend((match.group(1), match.group(2)))
    return paths


def secret_path_decision(tool_name: str, args: Any) -> AccessDecision | None:
    """The built-in refusal for a file tool call that reaches the mount."""
    if tool_name not in _FILE_TOOL_NAMES:
        return None
    if not isinstance(args, Mapping):
        return None
    for path in _file_tool_paths(tool_name, args):
        if secret_path_hard_denied(path):
            return AccessDecision(False, SECRET_PATH_DENY_REASON, detail=str(path))
        if tool_name == "search_files" and _search_reaches_secret_mount(path):
            return AccessDecision(False, SECRET_PATH_DENY_REASON, detail=str(path))
    return None


def _check_identity_env_tamper(command_s: str) -> AccessDecision:
    """Refuse any attempt to touch the identity the child processes carry.

    HERMES_DWD_IDENTITY is what pins the Google CLIs to the caller's own
    account. Unsetting or rewriting it is never a legitimate need, and naming
    it at all is the only way to try.
    """
    if "HERMES_DWD_IDENTITY" in command_s:
        return AccessDecision(False, "dwd_identity_tamper", detail="HERMES_DWD_IDENTITY")
    return AccessDecision(True, "arguments_allowed")


def _check_expanded_heredoc_body(body: str, grants, dwd_identity=_DWD_UNRESTRICTED, inbox: str = "") -> AccessDecision:
    """A bare-delimiter heredoc is still expanded by the shell, so anything the
    shell would RUN inside it stays gated: backticks and process substitution
    are refused, and each $(...) is checked as a command. The literal text
    itself is data and is not segmented."""
    if _SHELL_SUBSTITUTION_RE.search(body):
        return AccessDecision(False, "cli_shell_operator_not_allowed")
    try:
        _, inners = _extract_cmd_substitutions(body, shell_quotes=False)
    except ValueError:
        return AccessDecision(False, "cli_shell_operator_not_allowed")
    for inner in inners:
        inner_decision = _check_cli_command(inner, grants, dwd_identity, inbox)
        if not inner_decision.allowed:
            return inner_decision
    return AccessDecision(True, "arguments_allowed")


def _check_cli_command(command_s: str, grants, dwd_identity=_DWD_UNRESTRICTED, inbox: str = "", *, lenient: bool = False) -> AccessDecision:
    """Validate one shell command string against the CLI grants: hard-block
    backticks/process substitution, recursively validate $(...) contents, and
    check every segment's argv0 against the allowlist. With ``lenient`` (a
    blacklist account) only the explicit blacklist, the DWD identity binding
    and the tamper guard apply; see _check_cli_command_lenient."""
    if lenient:
        return _check_cli_command_lenient(command_s, grants, dwd_identity, inbox)
    command_s, heredoc_bodies = _strip_heredocs(command_s)
    for body in heredoc_bodies:
        body_decision = _check_expanded_heredoc_body(body, grants, dwd_identity, inbox)
        if not body_decision.allowed:
            return body_decision
    try:
        command_s, inners = _extract_cmd_substitutions(command_s)
    except ValueError:
        return AccessDecision(False, "cli_shell_operator_not_allowed")
    for inner in inners:
        inner_decision = _check_cli_command(inner, grants, dwd_identity, inbox)
        if not inner_decision.allowed:
            return inner_decision
    denied_path = _check_denied_paths(command_s, grants, inbox)
    if not denied_path.allowed:
        return denied_path
    decision = _cli_compound_decision(command_s, grants, dwd_identity)
    if not decision.allowed:
        return decision
    decision = _cli_identity_decision(command_s, grants, dwd_identity)
    if not decision.allowed:
        return decision
    decision = _cli_deny_decision(command_s, grants, dwd_identity)
    if not decision.allowed:
        return decision
    decision = _cli_allow_decision(command_s, grants, dwd_identity)
    if not decision.allowed:
        return decision
    return AccessDecision(True, "arguments_allowed")


def _cli_compound_decision(command_s, grants, dwd_identity) -> AccessDecision:
    if grants.cli_denied_commands or (grants.cli_commands and "*" not in grants.cli_commands):
        # shlex identifies ordinary command chains; it is not a shell AST.
        # A compound head can hide a second executable in the same segment.
        # With hard command constraints, unknown structure must remain denied.
        # The review-only caller uses this same result to require a human.
        try:
            segments = _split_shell_segments(command_s)
        except ValueError:
            return AccessDecision(False, "cli_shell_operator_not_allowed")
        for tokens in segments:
            head = _segment_argv0(tokens)
            if head in _SHELL_CONTROL_HEADS or any(char in head for char in "(){}"):
                return AccessDecision(False, "cli_compound_command_not_allowed")
    return AccessDecision(True, "arguments_allowed")


def _cli_identity_decision(command_s, grants, dwd_identity) -> AccessDecision:
    if dwd_identity is not _DWD_UNRESTRICTED:
        tamper = _check_identity_env_tamper(command_s)
        if not tamper.allowed:
            return tamper
        # Identity binding runs regardless of the allowlist: even a wildcard
        # CLI grant must not let a non-admin act as someone else.
        try:
            dwd_segments = _split_shell_segments(command_s)
        except ValueError:
            return AccessDecision(False, "cli_shell_operator_not_allowed")
        dwd_decision = _check_dwd_identity(dwd_segments, dwd_identity)
        if not dwd_decision.allowed:
            return dwd_decision
    return AccessDecision(True, "arguments_allowed")


def _cli_deny_decision(command_s, grants, dwd_identity) -> AccessDecision:
    # A per-person CLI deny is subtracted into cli_denied_commands and has to
    # bite even when the role hands out a wildcard, otherwise the only way to
    # keep one command away from an administrator is to take the wildcard back
    # and enumerate every other command. Checked before the allowlist so deny
    # wins, and independently of it so "*" does not skip past it.
    if grants.cli_denied_commands:
        try:
            denied_segments = _split_shell_segments(command_s)
        except ValueError:
            return AccessDecision(False, "cli_shell_operator_not_allowed")
        for tokens in denied_segments:
            seg_argv0 = _segment_argv0(tokens)
            seg_base = os.path.basename(seg_argv0) if seg_argv0 else ""
            if not seg_argv0 or seg_argv0 == _CMD_SUBSTITUTION_MARK:
                continue
            from .models import grant_matches
            if grant_matches(grants.cli_denied_commands, seg_argv0) or grant_matches(grants.cli_denied_commands, seg_base):
                return AccessDecision(False, "cli_command_denied", detail=seg_argv0)
            for target in _interpreter_targets(tokens):
                if target and (grant_matches(grants.cli_denied_commands, target)
                               or grant_matches(grants.cli_denied_commands, os.path.basename(target))):
                    return AccessDecision(False, "cli_command_denied", detail=target)
    return AccessDecision(True, "arguments_allowed")


def _cli_allow_decision(command_s, grants, dwd_identity) -> AccessDecision:
    if grants.cli_commands and "*" not in grants.cli_commands:
        try:
            segments = _split_shell_segments(command_s)
        except ValueError:
            return AccessDecision(False, "cli_shell_operator_not_allowed")
        if not segments:
            return AccessDecision(False, "cli_command_not_allowed")
        for tokens in segments:
            seg_argv0 = _segment_argv0(tokens)
            seg_base = os.path.basename(seg_argv0) if seg_argv0 else ""
            if not seg_argv0 or seg_argv0 == _CMD_SUBSTITUTION_MARK:
                continue
            if seg_argv0 in _SHELL_BUILTINS or seg_base in _SHELL_BUILTINS:
                continue
            if seg_argv0 not in grants.cli_commands and seg_base not in grants.cli_commands:
                return AccessDecision(False, "cli_command_not_allowed", detail=seg_argv0)
    return AccessDecision(True, "arguments_allowed")



def cli_command_requires_manual_approval(access: EffectiveAccess, command: str) -> bool:
    """Use the command permission parser for the current policy's review floor.

    This is a review requirement, never a permission grant. Any matching
    executable (including a nested substitution) or unsupported parse needs
    a human; the caller must still perform the full hard checks first.
    """
    from .models import GrantSet
    required = access.grants.cli_approval_commands
    if not required or "bootstrap_admin" in access.grant_sources:
        return False
    review_selectors = GrantSet(cli_denied_commands=required)
    return not _check_cli_command(command, review_selectors, lenient=access.access_mode == "blacklist").allowed


def decide_tool_argument_access(access: EffectiveAccess | None, tool_name: str, args: dict[str, Any], *, session_id: str = "") -> AccessDecision:
    hard = secret_path_decision(tool_name, args)
    if hard is not None:
        return hard
    if access is None or access.mode != "enforce":
        return AccessDecision(True, "governance_inactive")
    if tool_name == OPENCODE_TOOL_NAME:
        return _opencode_argument_decision(access, args, session_id)
    inbox = session_attachment_inbox(session_id)
    if tool_name in {"read_file", "search_files"}:
        if _in_own_state_cache(args.get("path") or "."):
            return AccessDecision(True, "own_state_cache_allowed")
        if _in_session_inbox(args.get("path") or ".", inbox):
            return AccessDecision(True, "session_attachment_allowed")
    handler = _ARGUMENT_HANDLERS.get(tool_name)
    if handler is None:
        return AccessDecision(True, "arguments_allowed")
    return handler(access, tool_name, args, inbox)


def _decide_skill_arguments(access, tool_name, args, inbox) -> AccessDecision:
    from .models import grant_matches
    grants = access.grants
    dim = "skills_view" if tool_name == "skill_view" else "skills_manage"
    for name in _skill_names(args):
        if grant_matches(getattr(access.deny, dim), name) or grant_matches(getattr(access.deny, dim), name.rsplit("/", 1)[-1]):
            return AccessDecision(False, "explicit_deny", detail=name)
        if tool_name == "skill_view" and (grant_matches(access.deny.skills_load, name) or
                                          grant_matches(access.deny.skills_load, name.rsplit("/", 1)[-1])):
            return AccessDecision(False, "explicit_deny", detail=name)
        if (access.access_mode or access.access_level) and not access.allows(dim, name):
            return AccessDecision(False, "skill_not_allowed", detail=name)
    if tool_name == "skill_view":
        if not _skill_name_allowed(grants.skills_view, args.get("name")):
            return AccessDecision(False, "skill_not_allowed", detail=str(args.get("name") or ""))
    elif tool_name == "skill_manage":
        names = _skill_names(args)
        if not names:
            return AccessDecision(False, "skill_manage_not_allowed", detail="")
        for name in names:
            if not _skill_name_allowed(grants.skills_manage, name):
                return AccessDecision(False, "skill_manage_not_allowed", detail=name)
    return AccessDecision(True, "arguments_allowed")


def _decide_file_arguments(access, tool_name, args, inbox) -> AccessDecision:
    from .models import grant_matches
    grants = access.grants
    path = str(args.get("path") or ".")
    canonical = _resolve_candidate_path(path)
    dim = "file_read_roots" if tool_name in {"read_file", "search_files"} else "file_write_roots"
    if access.access_mode == "blacklist" and access.configuration_denies_file(path, canonical):
        return AccessDecision(False, "file_denied_glob", detail=path)
    if (_matches_denied_glob(path, access.deny.file_denied_globs)
            or _matches_denied_glob(canonical, access.deny.file_denied_globs)
            or grant_matches(getattr(access.deny, dim), canonical, path=True)):
        return AccessDecision(False, "explicit_deny", detail=path)
    if (access.access_mode or access.access_level) and not access.allows(dim, canonical, path=True):
        return AccessDecision(False, "file_root_not_allowed", detail=path)
    if tool_name in {"read_file", "search_files"}:
        path = args.get("path") or "."
        if _matches_denied_glob(str(path), grants.file_denied_globs) \
                and not _matches_denied_glob(str(path), grants.file_allow_globs):
            return AccessDecision(False, "file_denied_glob", detail=str(path))
        if grants.file_read_roots and not _path_within_roots(str(path), grants.file_read_roots):
            return AccessDecision(False, "file_read_root_not_allowed", detail=str(path))
    elif tool_name in {"write_file", "patch"}:
        path = args.get("path")
        if path:
            if _matches_denied_glob(str(path), grants.file_denied_globs) \
                    and not _matches_denied_glob(str(path), grants.file_allow_globs):
                return AccessDecision(False, "file_denied_glob", detail=str(path))
            if grants.file_write_roots and not _path_within_roots(str(path), grants.file_write_roots):
                return AccessDecision(False, "file_write_root_not_allowed", detail=str(path))
    return AccessDecision(True, "arguments_allowed")


def _decide_terminal_arguments(access, tool_name, args, inbox) -> AccessDecision:
    from .models import grant_matches
    grants = access.grants
    if tool_name == "terminal" and (access.access_mode or access.access_level):
        if not grants.cli_commands:
            return AccessDecision(False, "cli_command_not_allowed")
        workdir = _resolve_candidate_path(args.get("workdir") or ".")
        if not access.allows("cli_workdir_roots", workdir, path=True):
            return AccessDecision(False, "cli_workdir_not_allowed", detail=workdir)
        if not access.has_permission("terminal:use"):
            return AccessDecision(False, "terminal_not_allowed")
        for candidate in _PATHLIKE_RE.findall(str(args.get("command") or "")):
            if _in_session_inbox(candidate, inbox):
                continue
            canonical = _resolve_candidate_path(candidate)
            if access.access_mode == "blacklist" and access.configuration_denies_file(candidate, canonical):
                return AccessDecision(False, "file_denied_glob", detail=candidate)
            if (grant_matches(access.deny.file_read_roots | access.deny.file_write_roots, canonical, path=True)
                    or _matches_denied_glob(candidate, access.deny.file_denied_globs)
                    or _matches_denied_glob(canonical, access.deny.file_denied_globs)):
                return AccessDecision(False, "explicit_deny", detail=candidate)
    command = args.get("command")
    command_s = str(command or "")
    decision = _check_cli_command(command_s, grants, dwd_identity_for(access), inbox=inbox,
                                  lenient=access.access_mode == "blacklist")
    if not decision.allowed:
        return decision
    workdir = args.get("workdir")
    if workdir and grants.cli_workdir_roots and not _path_within_roots(str(workdir), grants.cli_workdir_roots):
        return AccessDecision(False, "cli_workdir_not_allowed", detail=str(workdir))
    return AccessDecision(True, "arguments_allowed")


def _decide_code_arguments(access, tool_name, args, inbox) -> AccessDecision:
    grants = access.grants
    # 15-10-2026: execute_code skipped the file denies entirely, so a
    # path the terminal refuses was one open() away. Same string-level
    # check as the lenient terminal gate; not process isolation.
    code_s = str(args.get("code") or "")
    denied_path = _check_denied_paths(code_s, grants, inbox)
    if not denied_path.allowed:
        return denied_path
    identity = dwd_identity_for(access)
    if identity is not _DWD_UNRESTRICTED:
        tamper = _check_identity_env_tamper(code_s)
        if not tamper.allowed:
            return tamper
    if isinstance(identity, DwdExclusion):
        lowered = code_s.lower()
        for subject in identity:
            if subject in lowered:
                return AccessDecision(False, "dwd_identity_excluded", detail=f"execute_code {subject}")
    return AccessDecision(True, "arguments_allowed")


_ARGUMENT_HANDLERS = {
    "skill_view": _decide_skill_arguments,
    "skill_manage": _decide_skill_arguments,
    "read_file": _decide_file_arguments,
    "search_files": _decide_file_arguments,
    "write_file": _decide_file_arguments,
    "patch": _decide_file_arguments,
    "terminal": _decide_terminal_arguments,
    "execute_code": _decide_code_arguments,
}


def tool_arguments_allowed_for_context(ctx: DashboardGovernanceContext | None, tool_name: str, args: dict[str, Any]) -> AccessDecision:
    from .context import policy_contexts
    try:
        # Built in and context free: holds for ungoverned turns as well.
        hard = secret_path_decision(tool_name, args)
        if hard is not None:
            return hard
        contexts = policy_contexts(ctx)
        first = AccessDecision(True, "governance_inactive")
        for index, bounded_ctx in enumerate(contexts):
            decision = _tool_arguments_for_single_context(bounded_ctx, tool_name, args)
            if not decision.allowed:
                return decision
            if index == 0:
                first = decision
        return first
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("tool_arguments_allowed_for_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return AccessDecision(False, "continuation_policy_unavailable")


def _tool_arguments_for_single_context(ctx, tool_name, args):
    from .context import workspace_allowed_for_context
    candidate = ""
    if tool_name in {"read_file", "search_files", "write_file", "patch"}:
        candidate = _resolve_candidate_path(args.get("path") or ".")
        # The running session's own attachment inbox is that person's own
        # message, not a workspace: it lives under the state directory and is
        # already carved out below (session_attachment_allowed). A registered
        # workspace whose root happens to contain the state directory (a stray
        # entry at /home, 20-09-2026) must not turn every upload into
        # workspace_access_revoked. The selected workspace root is still checked.
        if tool_name in {"read_file", "search_files"} and _in_session_inbox(
                candidate, session_attachment_inbox(getattr(ctx, "session_id", ""))):
            candidate = ""
    elif tool_name in {"terminal", OPENCODE_TOOL_NAME}:
        candidate = _resolve_candidate_path(args.get("workdir") or getattr(ctx, "workspace_path", "") or ".")
    if not workspace_allowed_for_context(ctx, candidate):
        return AccessDecision(False, "workspace_access_revoked")
    ceiling = getattr(ctx, "bot_access_ceiling", None)
    if ceiling is not None:
        checker = getattr(ctx, "bot_access_check", None)
        try:
            if not callable(checker) or checker() is not True:
                return AccessDecision(False, "bot_access_revoked")
        except Exception as _boundary_error:
            logging.getLogger(__name__).error("_tool_arguments_for_single_context: operation failed",
                exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
            return AccessDecision(False, "bot_access_revoked")
        if tool_name in {"terminal", OPENCODE_TOOL_NAME} and not ceiling.grants.cli_commands:
            return AccessDecision(False, "bot_cli_not_selected")
        if tool_name == "skill_view":
            name = str(args.get("name") or "")
            if "*" not in ceiling.grants.skills_load and name not in ceiling.grants.skills_load:
                return AccessDecision(False, "bot_skill_not_selected",
                                      detail=f"{getattr(ctx, 'active_profile', '') or ''}:{name}")
        bounded = decide_tool_argument_access(ceiling, tool_name, args, session_id=getattr(ctx, "session_id", ""))
        if not bounded.allowed:
            return _as_bot_denial(ctx, bounded)
    decision = decide_tool_argument_access(ctx.access if ctx is not None else None, tool_name, args,
                                          session_id=getattr(ctx, "session_id", ""))
    root = getattr(ctx, "project_workspace", "") if ctx is not None else ""
    if not root or tool_name not in {"read_file", "search_files", "write_file", "patch"}:
        return decision
    raw = str(args.get("path") or ".")
    # Grant only an explicit absolute file path. Relative paths retain the
    # normal engine policy because a tool backend may resolve its cwd differently.
    path = Path(os.path.expanduser(raw))
    workspace = Path(root)
    if not path.is_absolute() or not workspace.is_absolute():
        return decision
    try:
        lexical = Path(os.path.abspath(path))
        lexical.relative_to(workspace)
    except (ValueError, OSError):
        return decision
    denied = AccessDecision(False, "project_file_access_denied", detail=raw)
    try:
        # No project scope follows symlinks, including symlinked ancestors.
        if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            return denied
        path.resolve().relative_to(workspace.resolve())
        check = getattr(ctx, "project_access_check", None)
        if not callable(check) or check(str(path), tool_name in {"write_file", "patch"}) is not True:
            return denied
    except Exception as _boundary_error:
        logging.getLogger(__name__).error("_tool_arguments_for_single_context: operation failed",
            exc_info=(RuntimeError, RuntimeError(type(_boundary_error).__name__), _boundary_error.__traceback__))
        return denied
    # Never override denied globs, tool permissions, or other argument denials.
    if decision.reason in {"file_read_root_not_allowed", "file_write_root_not_allowed"}:
        return AccessDecision(True, "project_file_scope_allowed")
    return decision
