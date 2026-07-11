"""Shell completion script generation for hermes CLI.

Walks the live argparse parser tree to generate accurate, always-up-to-date
completion scripts — no hardcoded subcommand lists, no extra dependencies.

Supports bash, zsh, and fish.
"""

from __future__ import annotations

import argparse
from typing import Any


def _walk(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Recursively extract subcommands and flags from a parser.

    Uses _SubParsersAction._choices_actions to get canonical names (no aliases)
    along with their help text.
    """
    flags: list[str] = []
    subcommands: dict[str, Any] = {}

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # _choices_actions has one entry per canonical name; aliases are
            # omitted, which keeps completion lists clean.
            seen: set[str] = set()
            for pseudo in action._choices_actions:
                name = pseudo.dest
                if name in seen:
                    continue
                seen.add(name)
                subparser = action.choices.get(name)
                if subparser is None:
                    continue
                info = _walk(subparser)
                info["help"] = _clean(pseudo.help or "")
                subcommands[name] = info
        elif action.option_strings:
            flags.extend(o for o in action.option_strings if o.startswith("-"))

    return {"flags": flags, "subcommands": subcommands}


def _clean(text: str, maxlen: int = 60) -> str:
    """Strip shell-unsafe characters and truncate."""
    return text.replace("'", "").replace('"', "").replace("\\", "")[:maxlen]


def _flags_word(info: dict[str, Any]) -> str:
    """Return a space-separated option list for shell completion."""
    return " ".join(sorted(info.get("flags") or []))


# ---------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------

def generate_bash(parser: argparse.ArgumentParser) -> str:
    tree = _walk(parser)
    top_cmds = " ".join(sorted(tree["subcommands"]))

    cases: list[str] = []
    for cmd in sorted(tree["subcommands"]):
        info = tree["subcommands"][cmd]
        if cmd == "profile" and info["subcommands"]:
            # Profile subcommand: complete actions, then profile names for
            # actions that accept a profile argument.
            subcmds = " ".join(sorted(info["subcommands"]))
            flags = _flags_word(info)
            flag_block = _bash_flag_block(flags)
            profile_actions = "use delete show alias rename export"
            cases.append(
                f"        profile)\n"
                f"{flag_block}"
                f"            case \"$prev\" in\n"
                f"                profile)\n"
                f"                    COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n"
                f"                    return\n"
                f"                    ;;\n"
                f"                {profile_actions.replace(' ', '|')})\n"
                f"                    COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n"
                f"                    return\n"
                f"                    ;;\n"
                f"            esac\n"
                f"            ;;"
            )
        elif info["subcommands"]:
            subcmds = " ".join(sorted(info["subcommands"]))
            flags = _flags_word(info)
            flag_block = _bash_flag_block(flags)
            nested_cases = _bash_nested_flag_cases(cmd, info)
            nested_block = ""
            if nested_cases:
                nested_block = (
                    f"            if [[ $COMP_CWORD -ge 3 && \"$cur\" == -* ]]; then\n"
                    f"                case \"{cmd}:${{COMP_WORDS[2]}}\" in\n"
                    f"{nested_cases}\n"
                    f"                esac\n"
                    f"            fi\n"
                )
            cases.append(
                f"        {cmd})\n"
                f"{flag_block}"
                f"{nested_block}"
                f"            COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n"
                f"            return\n"
                f"            ;;"
            )
        elif info["flags"]:
            flags = _flags_word(info)
            cases.append(
                f"        {cmd})\n"
                f"            COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n"
                f"            return\n"
                f"            ;;"
            )

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


def _bash_flag_block(flags: str) -> str:
    if not flags:
        return ""
    return (
        f"            if [[ \"$cur\" == -* ]]; then\n"
        f"                COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n"
        f"                return\n"
        f"            fi\n"
    )


def _bash_nested_flag_cases(cmd: str, info: dict[str, Any]) -> str:
    cases: list[str] = []
    for subcmd in sorted(info.get("subcommands") or {}):
        subinfo = info["subcommands"][subcmd]
        flags = _flags_word(subinfo)
        if not flags:
            continue
        cases.append(
            f"                    {cmd}:{subcmd})\n"
            f"                        COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n"
            f"                        return\n"
            f"                        ;;"
        )
    return "\n".join(cases)


# ---------------------------------------------------------------------------
# Zsh
# ---------------------------------------------------------------------------

def generate_zsh(parser: argparse.ArgumentParser) -> str:
    tree = _walk(parser)

    top_cmds_lines: list[str] = []
    for cmd in sorted(tree["subcommands"]):
        help_text = _clean(tree["subcommands"][cmd].get("help", ""))
        top_cmds_lines.append(f"                '{cmd}:{help_text}'")
    top_cmds_str = "\n".join(top_cmds_lines)

    sub_cases: list[str] = []
    for cmd in sorted(tree["subcommands"]):
        info = tree["subcommands"][cmd]
        if not info["subcommands"]:
            continue
        safe = cmd.replace("-", "_")
        flag_block = _zsh_flag_block(cmd, safe, info)
        if cmd == "profile":
            # Profile subcommand: complete actions, then profile names for
            # actions that accept a profile argument.
            sub_lines: list[str] = []
            for sc in sorted(info["subcommands"]):
                sh = _clean(info["subcommands"][sc].get("help", ""))
                sub_lines.append(f"                        '{sc}:{sh}'")
            sub_str = "\n".join(sub_lines)
            sub_cases.append(
                f"                profile)\n"
                f"{flag_block}"
                f"                    case ${{line[2]}} in\n"
                f"                        use|delete|show|alias|rename|export)\n"
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
                f"                    ;;"
            )
        else:
            sub_lines = []
            for sc in sorted(info["subcommands"]):
                sh = _clean(info["subcommands"][sc].get("help", ""))
                sub_lines.append(f"                    '{sc}:{sh}'")
            sub_str = "\n".join(sub_lines)
            sub_cases.append(
                f"                {cmd})\n"
                f"{flag_block}"
                f"                    local -a {safe}_cmds\n"
                f"                    {safe}_cmds=(\n"
                f"{sub_str}\n"
                f"                    )\n"
                f"                    _describe '{cmd} command' {safe}_cmds\n"
                f"                    ;;"
            )
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


def _zsh_flag_block(cmd: str, safe: str, info: dict[str, Any]) -> str:
    flags = sorted(info.get("flags") or [])
    if not flags:
        return ""
    flag_lines = "\n".join(f"                        '{flag}'" for flag in flags)
    return (
        f"                    if [[ ${{words[CURRENT]}} == -* ]]; then\n"
        f"                        local -a {safe}_flags\n"
        f"                        {safe}_flags=(\n"
        f"{flag_lines}\n"
        f"                        )\n"
        f"                        _describe '{cmd} option' {safe}_flags\n"
        f"                        return\n"
        f"                    fi\n"
    )


# ---------------------------------------------------------------------------
# Fish
# ---------------------------------------------------------------------------

def generate_fish(parser: argparse.ArgumentParser) -> str:
    tree = _walk(parser)
    top_cmds = sorted(tree["subcommands"])
    top_cmds_str = " ".join(top_cmds)

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
        "# Top-level subcommands",
    ]

    for cmd in top_cmds:
        info = tree["subcommands"][cmd]
        help_text = _clean(info.get("help", ""))
        lines.append(
            f"complete -c hermes -f "
            f"-n 'not __fish_seen_subcommand_from {top_cmds_str}' "
            f"-a {cmd} -d '{help_text}'"
        )

    lines.append("")
    lines.append("# Subcommand completions")

    profile_name_actions = {"use", "delete", "show", "alias", "rename", "export"}

    for cmd in top_cmds:
        info = tree["subcommands"][cmd]
        _append_fish_flags(lines, cmd, info, condition=f"__fish_seen_subcommand_from {cmd}")
        if not info["subcommands"]:
            continue
        lines.append(f"# {cmd}")
        for sc in sorted(info["subcommands"]):
            sinfo = info["subcommands"][sc]
            sh = _clean(sinfo.get("help", ""))
            lines.append(
                f"complete -c hermes -f "
                f"-n '__fish_seen_subcommand_from {cmd}' "
                f"-a {sc} -d '{sh}'"
            )
            _append_fish_flags(
                lines,
                f"{cmd} {sc}",
                sinfo,
                condition=(
                    f"__fish_seen_subcommand_from {cmd}; "
                    f"and __fish_seen_subcommand_from {sc}"
                ),
            )
        # For profile subcommand, complete profile names for relevant actions
        if cmd == "profile":
            for action in sorted(profile_name_actions):
                lines.append(
                    f"complete -c hermes -f "
                    f"-n '__fish_seen_subcommand_from {action}; "
                    f"and __fish_seen_subcommand_from profile' "
                    f"-a '(__hermes_profiles)' -d 'Profile name'"
                )

    lines.append("")
    return "\n".join(lines)


def _append_fish_flags(
    lines: list[str], label: str, info: dict[str, Any], *, condition: str
) -> None:
    for flag in sorted(info.get("flags") or []):
        if flag.startswith("--"):
            lines.append(
                f"complete -c hermes -f -n '{condition}' "
                f"-l {flag[2:]} -d '{label} option'"
            )
        elif flag.startswith("-") and len(flag) == 2:
            lines.append(
                f"complete -c hermes -f -n '{condition}' "
                f"-s {flag[1:]} -d '{label} option'"
            )
