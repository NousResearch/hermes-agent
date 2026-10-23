"""Shell completion script generation for hermes CLI. Walks the live argparse parser tree, so
completion scripts never go stale; no extra dependencies."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

# Flags whose value is a session (handle, id or title) and sessions actions that take one.
_SESSION_FLAGS = ("-r", "--resume", "-c", "--continue")
_SESSION_ACTIONS = ("delete", "rename")


def _session_command() -> str:
    """The command a Tab press runs: this interpreter on the stdlib-only completer, no Hermes import."""
    script = Path(__file__).with_name("session_complete.py")
    return f"{shlex.quote(sys.executable)} -I -S {shlex.quote(str(script))}"


def _walk(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Recursively extract subcommands and flags from a parser."""
    flags: list[str] = []
    subcommands: dict[str, Any] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # _choices_actions has one entry per canonical name (aliases omitted).
            for pseudo in action._choices_actions:
                name = pseudo.dest
                subparser = action.choices.get(name)
                if name in subcommands or subparser is None:
                    continue
                subcommands[name] = {**_walk(subparser), "help": _clean(pseudo.help or "")}
        elif action.option_strings:
            flags.extend(o for o in action.option_strings if o.startswith("-"))
    return {"flags": flags, "subcommands": subcommands}


def _clean(text: str, maxlen: int = 60) -> str:
    """Strip shell-unsafe characters and truncate."""
    return text.replace("'", "").replace('"', "").replace("\\", "")[:maxlen]


# Profile actions that take a profile name as their next argument.
_PROFILE_NAME_ACTIONS = ("use", "delete", "show", "alias", "rename", "export")


def _sorted_subcommands(parser: argparse.ArgumentParser) -> list[tuple[str, dict[str, Any]]]:
    return sorted(_walk(parser)["subcommands"].items())


def generate_bash(parser: argparse.ArgumentParser) -> str:
    subcommands = _sorted_subcommands(parser)
    top_cmds = " ".join(cmd for cmd, _ in subcommands)
    cases: list[str] = []
    for cmd, info in subcommands:
        if cmd == "profile" and info["subcommands"]:
            # Complete actions, then profile names for actions that accept a profile argument.
            subcmds = " ".join(sorted(info["subcommands"]))
            cases.append(
                f"        profile)\n"
                f"            case \"$prev\" in\n"
                f"                profile)\n"
                f"                    COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n"
                f"                    return\n"
                f"                    ;;\n"
                f"                {'|'.join(_PROFILE_NAME_ACTIONS)})\n"
                f"                    COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n"
                f"                    return\n"
                f"                    ;;\n"
                f"            esac\n"
                f"            ;;")
        elif info["subcommands"] or info["flags"]:
            words = " ".join(sorted(info["subcommands"]) if info["subcommands"] else info["flags"])
            cases.append(
                f"        {cmd})\n"
                f"            COMPREPLY=($(compgen -W \"{words}\" -- \"$cur\"))\n"
                f"            return\n"
                f"            ;;")
    cases_str = "\n".join(cases)
    return f"""# Hermes Agent bash completion
# Add to ~/.bashrc:
#   eval "$(hermes completion bash)"

_hermes_profiles() {{
    local profiles_dir="$HOME/.hermes/profiles"
    local profiles="default"
    if [ -d "$profiles_dir" ]; then
        for f in "$profiles_dir"/*/; do
            [ -d "$f" ] && profiles="$profiles $(basename "$f")"
        done
    fi
    echo "$profiles"
}}

_hermes_sessions() {{
    {_session_command()} "$1" "${{COMP_WORDS[@]}}" 2>/dev/null | cut -f1
}}

# True when the word being completed is the session argument of `sessions delete|rename`, wherever
# options (-p NAME, -y) sit between the words.
_hermes_session_arg() {{
    local i w cmd= act= n=0
    for ((i=1; i<COMP_CWORD; i++)); do
        w=${{COMP_WORDS[i]}}
        case "$w" in
            -p|--profile) ((i++)); continue ;;
            -*) continue ;;
        esac
        if [[ -z $cmd ]]; then cmd=$w; elif [[ -z $act ]]; then act=$w; else ((n++)); fi
    done
    [[ $cmd == sessions && $n -eq 0 ]] && case "$act" in {'|'.join(_SESSION_ACTIONS)}) return 0 ;; esac
    return 1
}}

_hermes_completion() {{
    local cur prev
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    # Complete profile names after -p / --profile
    if [[ "$prev" == "-p" || "$prev" == "--profile" ]]; then
        COMPREPLY=($(compgen -W "$(_hermes_profiles)" -- "$cur"))
        return
    fi

    # Session handles after -r/--resume, -c/--continue and as the argument of sessions delete|rename
    case "$prev" in
        {'|'.join(_SESSION_FLAGS)})
            COMPREPLY=($(compgen -W "$(_hermes_sessions "$cur")" -- "$cur"))
            return
            ;;
    esac
    if [[ $cur != -* ]] && _hermes_session_arg; then
        COMPREPLY=($(compgen -W "$(_hermes_sessions "$cur")" -- "$cur"))
        return
    fi

    if [[ $COMP_CWORD -ge 2 ]]; then
        case "${{COMP_WORDS[1]}}" in
{cases_str}
        esac
    fi

    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W "{top_cmds}" -- "$cur"))
    fi
}}

complete -F _hermes_completion hermes
"""


def _zsh_describe_lines(subcommands: dict[str, Any], indent: str) -> str:
    """One ``'name:help'`` line per subcommand, sorted, at the given indent."""
    return "\n".join(
        f"{indent}'{sc}:{_clean(subcommands[sc].get('help', ''))}'" for sc in sorted(subcommands))


def generate_zsh(parser: argparse.ArgumentParser) -> str:
    subcommands = _sorted_subcommands(parser)
    top_cmds_str = _zsh_describe_lines(dict(subcommands), " " * 16)
    sub_cases: list[str] = []
    for cmd, info in subcommands:
        if not info["subcommands"]:
            continue
        if cmd == "sessions":
            sub_str = _zsh_describe_lines(info["subcommands"], " " * 24)
            sub_cases.append(
                f"                sessions)\n"
                f"                    case ${{line[2]}} in\n"
                f"                        {'|'.join(_SESSION_ACTIONS)})\n"
                f"                            _hermes_sessions\n"
                f"                            ;;\n"
                f"                        *)\n"
                f"                            local -a sessions_cmds\n"
                f"                            sessions_cmds=(\n"
                f"{sub_str}\n"
                f"                            )\n"
                f"                            _describe 'sessions command' sessions_cmds\n"
                f"                            ;;\n"
                f"                    esac\n"
                f"                    ;;")
            continue
        if cmd == "profile":
            # Complete actions, then profile names for actions that accept a profile argument.
            sub_str = _zsh_describe_lines(info["subcommands"], " " * 24)
            sub_cases.append(
                f"                profile)\n"
                f"                    case ${{line[2]}} in\n"
                f"                        {'|'.join(_PROFILE_NAME_ACTIONS)})\n"
                f"                            _hermes_profiles\n"
                f"                            ;;\n"
                f"                        *)\n"
                f"                            local -a profile_cmds\n"
                f"                            profile_cmds=(\n"
                f"{sub_str}\n"
                f"                            )\n"
                f"                            _describe 'profile command' profile_cmds\n"
                f"                            ;;\n"
                f"                    esac\n"
                f"                    ;;")
        else:
            sub_str = _zsh_describe_lines(info["subcommands"], " " * 20)
            safe = cmd.replace("-", "_")
            sub_cases.append(
                f"                {cmd})\n"
                f"                    local -a {safe}_cmds\n"
                f"                    {safe}_cmds=(\n"
                f"{sub_str}\n"
                f"                    )\n"
                f"                    _describe '{cmd} command' {safe}_cmds\n"
                f"                    ;;")
    sub_cases_str = "\n".join(sub_cases)
    return f"""#compdef hermes
# Hermes Agent zsh completion
# Add to ~/.zshrc:
#   eval "$(hermes completion zsh)"

_hermes_profiles() {{
    local -a profiles
    profiles=(default)
    if [[ -d "$HOME/.hermes/profiles" ]]; then
        profiles+=($HOME/.hermes/profiles/*(N/:t))
    fi
    _describe 'profile' profiles
}}

_hermes_sessions() {{
    local -a sessions
    local row
    for row in ${{(f)"$({_session_command()} "$PREFIX" "${{words[@]}}" 2>/dev/null)"}}; do
        if [[ $row == *$'\t'* ]]; then
            sessions+=("${{row%%$'\t'*}}:${{row#*$'\t'}}")
        else
            sessions+=("$row")
        fi
    done
    _describe -V 'session' sessions
}}

_hermes() {{
    local context state line
    typeset -A opt_args

    _arguments -C \\
        '(-)'{{-h,--help}}'[Show help and exit]' \\
        '(-)'{{-V,--version}}'[Show version and exit]' \\
        '(-)'{{-p,--profile}}'[Profile name]:profile:_hermes_profiles' \\
        '(-r --resume)'{{-r,--resume}}'[Resume a session by handle, ID or title]:session:_hermes_sessions' \\
        '(-c --continue)'{{-c,--continue}}'[Continue the latest or a named session]::session:_hermes_sessions' \\
        '1:command:->commands' \\
        '*::arg:->args'

    case $state in
        commands)
            local -a subcmds
            subcmds=(
{top_cmds_str}
            )
            _describe 'hermes command' subcmds
            ;;
        args)
            case ${{line[1]}} in
{sub_cases_str}
            esac
            ;;
    esac
}}

compdef _hermes hermes
"""


def generate_fish(parser: argparse.ArgumentParser) -> str:
    subcommands = _sorted_subcommands(parser)
    top_cmds_str = " ".join(cmd for cmd, _ in subcommands)
    lines: list[str] = [
        "# Hermes Agent fish completion",
        "# Add to your config:",
        "#   hermes completion fish | source",
        "",
        "# Helper: list available profiles",
        "function __hermes_profiles",
        "    echo default",
        "    if test -d $HOME/.hermes/profiles",
        "        for d in $HOME/.hermes/profiles/*/",
        "            basename $d",
        "        end",
        "    end",
        "end",
        "",
        "# Helper: session handles and titles, newest first",
        "function __hermes_sessions",
        "    set -l cur (commandline -ct)  # quoted below: an empty token stays one (empty) argument",
        f"    {_session_command()} \"$cur\" (commandline -opc) 2>/dev/null",
        "end",
        "",
        "# Disable file completion by default",
        "complete -c hermes -f",
        "",
        "# Complete profile names after -p / --profile",
        "complete -c hermes -f -s p -l profile"
        " -d 'Profile name' -xa '(__hermes_profiles)'",
        "",
        "# Complete session handles after -r/--resume and -c/--continue",
        "complete -c hermes -f -n 'string match -qr -- \"^(-r|--resume|-c|--continue)$\" (commandline -opc)[-1]'"
        " -a '(__hermes_sessions)'",
        "",
        "# Top-level subcommands"]
    for cmd, info in subcommands:
        lines.append(
            f"complete -c hermes -f "
            f"-n 'not __fish_seen_subcommand_from {top_cmds_str}' "
            f"-a {cmd} -d '{_clean(info.get('help', ''))}'")
    lines += ["", "# Subcommand completions"]
    for cmd, info in subcommands:
        if not info["subcommands"]:
            continue
        lines.append(f"# {cmd}")
        for sc, sinfo in sorted(info["subcommands"].items()):
            lines.append(
                f"complete -c hermes -f "
                f"-n '__fish_seen_subcommand_from {cmd}' "
                f"-a {sc} -d '{_clean(sinfo.get('help', ''))}'")
        if cmd == "sessions":
            for action in _SESSION_ACTIONS:
                lines.append(
                    f"complete -c hermes -f "
                    f"-n '__fish_seen_subcommand_from {action}; "
                    f"and __fish_seen_subcommand_from sessions' "
                    f"-a '(__hermes_sessions)'")
        if cmd == "profile":  # profile names for the actions that take one
            for action in sorted(_PROFILE_NAME_ACTIONS):
                lines.append(
                    f"complete -c hermes -f "
                    f"-n '__fish_seen_subcommand_from {action}; "
                    f"and __fish_seen_subcommand_from profile' "
                    f"-a '(__hermes_profiles)' -d 'Profile name'")
    lines.append("")
    return "\n".join(lines)
