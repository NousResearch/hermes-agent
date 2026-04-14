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


# ---------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------

def generate_bash(parser: argparse.ArgumentParser) -> str:
    tree = _walk(parser)
    top_cmds = " ".join(sorted(tree["subcommands"]))

    cases: list[str] = []
    for cmd in sorted(tree["subcommands"]):
        info = tree["subcommands"][cmd]
        if info["subcommands"]:
            subcmds = " ".join(sorted(info["subcommands"]))
            cases.append(
                f"        {cmd})\n"
                f"            COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n"
                f"            return\n"
                f"            ;;"
            )
        elif info["flags"]:
            flags = " ".join(info["flags"])
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

_hermes_completion() {{
    local cur prev
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

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
        sub_lines: list[str] = []
        for sc in sorted(info["subcommands"]):
            sh = _clean(info["subcommands"][sc].get("help", ""))
            sub_lines.append(f"                    '{sc}:{sh}'")
        sub_str = "\n".join(sub_lines)
        safe = cmd.replace("-", "_")
        sub_cases.append(
            f"                {cmd})\n"
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

_hermes() {{
    local context state line
    typeset -A opt_args

    _arguments -C \\
        '(-h --help){{-h,--help}}[Show help and exit]' \\
        '(-V --version){{-V,--version}}[Show version and exit]' \\
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

_hermes "$@"
"""


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
        "# Disable file completion by default",
        "complete -c hermes -f",
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

    for cmd in top_cmds:
        info = tree["subcommands"][cmd]
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

    lines.append("")
    return "\n".join(lines)
