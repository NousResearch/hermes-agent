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
        if cmd == "profile" and info["subcommands"]:
            # Profile subcommand: complete actions, then profile names for
            # actions that accept a profile argument.
            subcmds = " ".join(sorted(info["subcommands"]))
            profile_actions = "use delete show alias rename export"
            cases.append(
                f"        profile)\n"
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


# ---------------------------------------------------------------------------
# Zsh
# ---------------------------------------------------------------------------

def generate_zsh(parser: argparse.ArgumentParser) -> str:
    tree = _walk(parser)

    # Top-level flag lines. The three classic flags (-h/-V/-p) keep their
    # long-standing verbatim specs (the repo's own tests assert them);
    # every other global option is extracted live from the parser so new
    # flags get completion automatically. Options with a value get a
    # generic :value: tag unless special-cased below
    # (--resume/-r/--continue/-c → session list). --profile/-p never
    # appears in parser._actions (consumed by main._apply_profile_override
    # before argparse runs), so it is kept manual here.
    flag_lines: list[str] = [
        "'(-)'{-h,--help}'[Show help and exit]'",
        "'(-)'{-V,--version}'[Show version and exit]'",
        "'(-)'{-p,--profile}'[Profile name]:profile:_hermes_profiles'",
    ]
    _legacy_flags = {"-h", "--help", "-V", "--version", "-p", "--profile"}
    for action in parser._actions:
        if not action.option_strings:
            continue
        opts = [o for o in action.option_strings if o.startswith("-")]
        if not opts:
            continue
        if set(opts) & _legacy_flags:
            continue  # already emitted verbatim above
        spec = "{" + ",".join(opts) + "}" if len(opts) > 1 else opts[0]
        help_text = _clean(action.help or "")
        # zsh _arguments: description must not contain ':' or ']' (they are
        # syntax); strip them to keep the spec parseable.
        help_text = help_text.replace(":", " ").replace("]", " ").replace("[", " ")
        desc = f"[{help_text}]" if help_text else ""
        takes_arg = not isinstance(
            action,
            (argparse._StoreTrueAction, argparse._StoreFalseAction,
             argparse._HelpAction, argparse._VersionAction),
        ) and getattr(action, "nargs", None) != 0
        if takes_arg:
            if set(opts) & {"--resume", "-r", "--continue", "-c"}:
                flag_lines.append(
                    f"'(-)'{spec}'{desc}:session:_hermes_sessions'"
                )
            else:
                flag_lines.append(f"'(-)'{spec}'{desc}:value:'")
        else:
            flag_lines.append(f"'(-)'{spec}'{desc}'")
    flags_str = (" \\\n        ").join(flag_lines)

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
        elif cmd == "sessions":
            # Sessions subcommand: complete actions, then session IDs for
            # actions whose first positional arg is a session_id
            # (delete, rename). Other actions keep the plain action list.
            sub_lines: list[str] = []
            for sc in sorted(info["subcommands"]):
                sh = _clean(info["subcommands"][sc].get("help", ""))
                sub_lines.append(f"                        '{sc}:{sh}'")
            sub_str = "\n".join(sub_lines)
            sub_cases.append(
                f"                sessions)\n"
                f"                    case ${{line[2]}} in\n"
                f"                        delete|rename)\n"
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
                f"                    ;;"
            )
        else:
            sub_lines = []
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

_hermes_profiles() {{
    local -a profiles
    profiles=(default)
    if [[ -d "$HOME/.hermes/profiles" ]]; then
        profiles+=($HOME/.hermes/profiles/*(N/:t))
    fi
    _describe 'profile' profiles
}}

_hermes_sessions() {{
    local -a descs values
    local db="$HOME/.hermes/state.db"
    if [[ -n "$HERMES_HOME" ]]; then
        db="$HERMES_HOME/state.db"
    fi
    if [[ ! -r "$db" ]]; then
        _message 'no session database'
        return 1
    fi
    local now=$(date +%s)
    local rel diff
    while IFS='|' read -r title ws ts id; do
        diff=$(( now - ts ))
        if (( diff < 60 )); then
            rel='just now'
        elif (( diff < 3600 )); then
            rel="$(( diff / 60 ))m ago"
        elif (( diff < 86400 )); then
            rel="$(( diff / 3600 ))h ago"
        else
            rel="$(( diff / 86400 ))d ago"
        fi
        if [[ -n "$ws" ]]; then
            ws="${{ws:t}}"
        else
            ws='—'
        fi
        descs+=("$title  [$ws]  $rel  [$id]")
        values+=("$id")
    done < <(sqlite3 -cmd '.timeout 500' -separator '|' "$db" "SELECT title, COALESCE(NULLIF(git_repo_root,''), cwd, ''), CAST(COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id), s.started_at) AS INTEGER), id FROM (SELECT s.* FROM sessions s WHERE s.title IS NOT NULL ORDER BY COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id), s.started_at) DESC LIMIT 50) s" 2>/dev/null)
    compadd -l -d descs -a values
}}

_hermes() {{
    local context state line
    typeset -A opt_args

    _arguments -C \\
{flags_str} \\
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
