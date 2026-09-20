"""Shell completion script generation for hermes CLI. Walks the live argparse parser tree, so
completion scripts never go stale; no extra dependencies."""

from __future__ import annotations

import argparse
from typing import Any


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

_hermes() {{
    local context state line
    typeset -A opt_args

    _arguments -C \\
        '(-)'{{-h,--help}}'[Show help and exit]' \\
        '(-)'{{-V,--version}}'[Show version and exit]' \\
        '(-)'{{-p,--profile}}'[Profile name]:profile:_hermes_profiles' \\
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
        "# Disable file completion by default",
        "complete -c hermes -f",
        "",
        "# Complete profile names after -p / --profile",
        "complete -c hermes -f -s p -l profile"
        " -d 'Profile name' -xa '(__hermes_profiles)'",
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
        if cmd == "profile":  # profile names for the actions that take one
            for action in sorted(_PROFILE_NAME_ACTIONS):
                lines.append(
                    f"complete -c hermes -f "
                    f"-n '__fish_seen_subcommand_from {action}; "
                    f"and __fish_seen_subcommand_from profile' "
                    f"-a '(__hermes_profiles)' -d 'Profile name'")
    lines.append("")
    return "\n".join(lines)
