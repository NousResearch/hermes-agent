#!/usr/bin/env python3
"""factory_admission_hook.py — pont Hermes `pre_tool_call` -> gate d'admission.

Ce script est le point d'intégration runtime de la porte d'admission worktree
(HER-95). Il se branche via le mécanisme de shell-hooks générique déjà chargé
par `cli.py`, `hermes_cli/main.py` et `gateway/run.py`
(`agent.shell_hooks.register_from_config`) — aucun changement de core n'est
requis. Déclaration opt-in dans `cli-config.yaml` (profile-aware) :

    hooks:
      pre_tool_call:
        - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
          command: >-
            python3 /ABS/scripts/factory_admission_hook.py
            --registry /ABS/registry --agent default

    # Profil métier (ex. hermes-immo) : le refus de domaine est AUTOMATIQUE car
    # --profile/--domain-prefixes vivent dans la config du profil, pas dans un
    # flag que l'appelant doit se souvenir de passer.
    hooks:
      pre_tool_call:
        - matcher: "terminal|patch|write_file|str_replace_editor|apply_patch"
          command: >-
            python3 /ABS/scripts/factory_admission_hook.py
            --registry /ABS/registry --agent hermes-immo
            --profile hermes-immo --domain-prefixes JYI,HER

Protocole (voir `agent/shell_hooks.py`) : la charge utile arrive en JSON sur
stdin ; pour bloquer un tool AVANT son exécution, on émet sur stdout
`{"decision": "block", "reason": "..."}` (traduit par `_parse_response` en
`{"action": "block", "message": "..."}`, la forme que
`get_pre_tool_call_block_message` remonte au dispatcher de tools).

Posture : LECTURE SEULE. Le hook n'écrit jamais d'owner et ne persiste jamais le
PID éphémère de ce subprocess (cf. blocker identité process) — il ne fait
qu'interroger `evaluate_admission_guard`. Fail-open advisory si l'infra est
absente/anormale (pas de registry, pas un dépôt git) ; fail-closed (block) sur
un conflit d'occupation avéré ou une violation de domaine métier.
"""

import argparse
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factory_lane  # noqa: E402  (résolu via le sys.path ci-dessus)

# Événements build-capable pour lesquels le gate a un sens. Le matcher du hook
# filtre déjà côté Hermes ; cette liste est une défense en profondeur si le hook
# est câblé sans matcher.
_MUTATING_TOOLS = frozenset({
    "terminal", "patch", "write_file", "str_replace_editor", "apply_patch",
    "edit_file", "create_file", "delete_file", "move_file",
})

_PATH_AFFECTING_SHELL_COMMANDS = frozenset({
    "apply_patch", "cat", "chmod", "chown", "cp", "dd", "install", "ln",
    "mkdir", "mktemp", "mv", "perl", "python", "python3", "rm", "ruby",
    "sed", "sh", "tee", "touch", "truncate", "zsh",
})
_HARD_READONLY_SHELL_COMMANDS = frozenset({"echo", "false", "printf", "pwd", "true", ":"})
_SHELL_CONTROL_TOKENS = frozenset({"&&", "||", ";", "|", "&", "(", ")", "<", ">", ">>"})
_MAX_REPARSED_SHELL_DEPTH = 4


def _emit_block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))


def _has_active_shell_expansion(text):
    """Detect shell expansions that survive into execution, never quoted data.

    ``shlex`` deliberately removes quote context, so it cannot tell ``'$HOME'``
    (literal) from ``"$HOME"`` (expanded). This small scanner keeps that
    distinction and treats escaped ``$``/backticks as literal. It is a guard,
    not a shell interpreter: uncertainty remains a reason to refuse a mutable
    target rather than attempt expansion in the hook.
    """
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if quote == "'":
            if char == "'":
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "`":
            return True
        if char != "$" or index + 1 >= len(text):
            index += 1
            continue
        next_char = text[index + 1]
        if next_char == "(":
            return True
        if next_char == "{" or next_char == "_" or next_char.isalnum() or next_char in "?*@$#!-":
            return True
        index += 1
    return False


def _scan_dollar_paren_end(command, start):
    """Return the offset after a balanced active ``$(...)`` substitution."""
    depth = 1
    quote = None
    index = start + 2
    while index < len(command):
        char = command[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(command):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "\\" and index + 1 < len(command):
            index += 2
            continue
        if command.startswith("$(", index):
            depth += 1
            index += 2
            continue
        if char == ")":
            depth -= 1
            index += 1
            if depth == 0:
                return index
            continue
        index += 1
    return None


def _scan_backtick_end(command, start):
    """Return the offset after an active backtick substitution."""
    index = start + 1
    while index < len(command):
        if command[index] == "\\" and index + 1 < len(command):
            index += 2
            continue
        if command[index] == "`":
            return index + 1
        index += 1
    return None


def _mask_active_command_substitutions(command):
    """Replace active substitution bodies before tokenizing shell syntax.

    ``shlex`` otherwise splits ``$(date)`` on its parentheses, losing the fact
    that the resulting word is dynamic.  Single-quoted syntax stays untouched;
    unmatched substitutions remain unparseable and must be rejected by callers.
    """
    marker = "__HERMES_DYNAMIC_SUBSTITUTION__"
    result = []
    quote = None
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\" and index + 1 < len(command):
            result.append(command[index:index + 2])
            index += 2
            continue
        if quote == "'":
            result.append(char)
            if char == "'":
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            result.append(char)
            quote = char
            index += 1
            continue
        if command.startswith("$(", index):
            end = _scan_dollar_paren_end(command, index)
            if end is None:
                return None
            result.append(marker)
            index = end
            continue
        if char == "`":
            end = _scan_backtick_end(command, index)
            if end is None:
                return None
            result.append(marker)
            index = end
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _unquote_shell_token(token):
    """Return the one shell-quoted word a re-parsing wrapper will execute."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _shell_segments(command):
    """Return shell command segments, or ``None`` when syntax is ambiguous."""
    masked_command = _mask_active_command_substitutions(command)
    if masked_command is None:
        return None
    try:
        lexer = shlex.shlex(masked_command, posix=False, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None

    segments = []
    current = []
    for token in [*tokens, ";"]:
        if token in _SHELL_CONTROL_TOKENS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    return segments


def _reparsed_shell_programs(command, depth=0):
    """Return literal programs evaluated by ``sh -c``/``eval`` recursively.

    ``None`` means that a nested reparse is ambiguous or exceeds the fixed
    budget, so callers fail closed rather than silently omit a target.
    """
    if depth >= _MAX_REPARSED_SHELL_DEPTH:
        return None
    segments = _shell_segments(command)
    if segments is None:
        return None
    programs = []
    for segment in segments:
        command_name = os.path.basename(segment[0].strip("'\""))
        operands = segment[1:]
        if command_name in {"sh", "bash", "dash", "ksh", "zsh"}:
            script = None
            for index, operand in enumerate(operands):
                if operand == "-c":
                    if index + 1 >= len(operands):
                        return None
                    script = _unquote_shell_token(operands[index + 1])
                    break
            if script is None:
                continue
            programs.append(script)
        elif command_name == "eval":
            if not operands:
                return None
            programs.append(" ".join(_unquote_shell_token(operand) for operand in operands))
        else:
            continue
        nested = _reparsed_shell_programs(programs[-1], depth + 1)
        if nested is None:
            return None
        programs.extend(nested)
    return programs


def _reparsed_shell_target_is_dynamic(command_name, operands, depth):
    """Inspect code evaluated by ``sh -c``/``eval`` with the same path policy.

    The outer shell can safely single-quote a script, but that quote disappears
    before a re-parsing wrapper evaluates it. Treat only dynamic nested path
    effects as unsafe, preserving harmless ``printf $(date)`` scripts.
    """
    if command_name in {"sh", "bash", "dash", "ksh", "zsh"}:
        for index, operand in enumerate(operands):
            if operand == "-c" and index + 1 < len(operands):
                return _terminal_has_unresolved_dynamic_target(
                    _unquote_shell_token(operands[index + 1]), depth + 1,
                )
        return "-c" in operands
    if command_name == "eval":
        if not operands:
            return True
        return _terminal_has_unresolved_dynamic_target(
            " ".join(_unquote_shell_token(operand) for operand in operands), depth + 1,
        )
    return False


def _terminal_has_unresolved_dynamic_target(command, depth=0):
    """True when terminal inspection cannot resolve a potentially mutable path.

    Simple variable and command/backtick substitutions are blocked only when
    they can select a cwd (``cd``/``pushd``), a git ``-C`` target, a redirection
    target, or an operand of a command that may write. Commands outside a tiny
    hard read-only allowlist are conservatively treated as write-capable,
    preventing wrappers such as ``env`` or ``sudo`` from hiding the real
    command. Literal single-quoted dollars remain safe data.
    """
    if depth >= _MAX_REPARSED_SHELL_DEPTH:
        return True
    marker = "__HERMES_DYNAMIC_SUBSTITUTION__"
    masked_command = _mask_active_command_substitutions(command)
    if masked_command is None:
        return _has_active_shell_expansion(command)
    try:
        lexer = shlex.shlex(masked_command, posix=False, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return _has_active_shell_expansion(command)

    segment = []
    redirection_target = False
    for token in [*tokens, ";"]:
        if redirection_target:
            if token in _SHELL_CONTROL_TOKENS:
                return True
            if marker in token or _has_active_shell_expansion(token):
                return True
            redirection_target = False
        if token in _SHELL_CONTROL_TOKENS:
            if segment:
                if marker in segment[0] or _has_active_shell_expansion(segment[0]):
                    # A dynamic command word can resolve to a write-capable
                    # command or wrapper, so it is never a safe readonly call.
                    return True
                command_name = os.path.basename(segment[0].strip("'\""))
                operands = segment[1:]
                if _reparsed_shell_target_is_dynamic(command_name, operands, depth):
                    return True
                dynamic_operand = any(
                    marker in value or _has_active_shell_expansion(value)
                    for value in operands
                )
                if dynamic_operand:
                    if command_name in {"cd", "pushd"}:
                        return True
                    if command_name == "git":
                        for index, value in enumerate(operands):
                            if value == "-C" and index + 1 < len(operands):
                                if _has_active_shell_expansion(operands[index + 1]):
                                    return True
                            if value.startswith("-C") and _has_active_shell_expansion(value[2:]):
                                return True
                    if (
                        command_name in _PATH_AFFECTING_SHELL_COMMANDS
                        or command_name not in _HARD_READONLY_SHELL_COMMANDS
                    ):
                        return True
            segment = []
            redirection_target = token in {"<", ">", ">>"}
        else:
            segment.append(token)
    return False


def _path_anchor(path, base):
    """Return an existing directory from which git can resolve a target path.

    File mutation tools commonly target a not-yet-created file.  Walk upward to
    the first existing ancestor instead of falling back to the gateway cwd.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = path if os.path.isabs(path) else os.path.join(base, path)
    candidate = os.path.realpath(candidate)
    while not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    if os.path.isfile(candidate):
        candidate = os.path.dirname(candidate)
    return candidate


def _target_directories(payload):
    """Yield effective tool targets before the process cwd, without duplicates."""
    cwd = payload.get("cwd") or os.getcwd()
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}

    raw_targets = []
    workdir = tool_input.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        raw_targets.append(workdir)

    # File mutation tools use one of these names across Hermes adapters.
    for key in ("path", "file_path", "target_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            raw_targets.append(value)

    # Codex-style apply_patch transports one or more paths under ``changes``.
    # Treat every declared path as a first-class mutation target.
    changes = tool_input.get("changes")
    if isinstance(changes, list):
        for change in changes:
            if not isinstance(change, dict):
                continue
            for key in ("path", "file_path", "target_path"):
                value = change.get(key)
                if isinstance(value, str) and value.strip():
                    raw_targets.append(value)

    # A terminal command can address a foreign worktree without setting
    # ``workdir`` (``git -C /abs/path``, ``cd /abs/path``, ``touch /abs/file``).
    # Inspect path-shaped shell tokens as a defence in depth.  This is not shell
    # execution or expansion; malformed commands simply contribute no targets.
    command = tool_input.get("command")
    if payload.get("tool_name") == "terminal" and isinstance(command, str):
        nested_programs = _reparsed_shell_programs(command)
        if nested_programs is None:
            return
        commands_to_scan = [command, *nested_programs]
        for shell_command in commands_to_scan:
            try:
                lexer = shlex.shlex(shell_command, posix=True, punctuation_chars=";&|()<>")
                lexer.whitespace_split = True
                lexer.commenters = ""
                tokens = list(lexer)
            except ValueError:
                continue
            shell_operators = {"&&", "||", ";", "|", "&", "(", ")", "<", ">", ">>"}
            for index, token in enumerate(tokens):
                candidate = token.split("=", 1)[-1] if "=" in token else token
                path_shaped = os.path.isabs(candidate) or candidate.startswith(("./", "../"))
                relative_argument = (
                    index > 0
                    and token not in shell_operators
                    and not token.startswith("-")
                )
                if path_shaped or relative_argument:
                    raw_targets.append(candidate)

    raw_targets.append(cwd)
    seen = set()
    for raw in raw_targets:
        anchor = _path_anchor(raw, cwd)
        if anchor is not None and anchor not in seen:
            seen.add(anchor)
            yield anchor


def main(argv=None):
    parser = argparse.ArgumentParser(prog="factory_admission_hook")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--domain-prefixes")
    # Optionnel : restreint le gate à un sous-ensemble de tools même sans matcher.
    parser.add_argument("--only-mutating", action="store_true")
    args = parser.parse_args(argv)

    # 1) Charge utile stdin — illisible => fail-open (ne jamais casser un tool).
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    if payload.get("hook_event_name") != "pre_tool_call":
        return 0
    if args.only_mutating and payload.get("tool_name") not in _MUTATING_TOOLS:
        return 0

    session = payload.get("session_id") or ""

    command = (payload.get("tool_input") or {}).get("command")
    if payload.get("tool_name") == "terminal" and isinstance(command, str):
        if (
            _terminal_has_unresolved_dynamic_target(command)
            or _reparsed_shell_programs(command) is None
        ):
            _emit_block("unresolved shell expansion can affect a worktree target")
            return 0

    # 2) Root absent => fail-open advisory. Existing but malformed or
    # unreadable registry state is ambiguous, so it blocks fail-closed.
    try:
        root = factory_lane._readonly_registry_root(args.registry)
    except factory_lane.RegistryError as exc:
        _emit_block(str(exc))
        return 0
    except Exception:
        _emit_block("registry root cannot be inspected safely")
        return 0
    if root is None:
        return 0

    try:
        # Evaluate every effective target.  The gateway process cwd is only a
        # fallback: terminal(workdir=...) and absolute file paths must not bypass
        # admission when the session itself started elsewhere.
        for target_dir in _target_directories(payload):
            worktree_real = factory_lane._git_toplevel_or_none(target_dir)
            if worktree_real is None:
                worktree_real = os.path.realpath(target_dir)
            allowed, reason = factory_lane.evaluate_admission_guard(
                root, worktree_real, args.agent, session,
                profile=args.profile, domain_prefixes=args.domain_prefixes,
            )
            if not allowed:
                _emit_block(reason or "worktree admission denied")
                return 0
    except factory_lane.RegistryError as exc:
        # An owner scan that cannot prove the registry's integrity is not an
        # ownerless scan. Refuse before the mutable tool can observe a hidden
        # competing worktree claim.
        _emit_block(str(exc))
        return 0
    except Exception:
        # The advisory path remains fail-open only for unexpected errors outside
        # the secure registry scan. Registry scan/read errors are handled just
        # above and fail closed.
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
