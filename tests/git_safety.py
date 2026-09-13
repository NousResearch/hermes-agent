"""Classify accidental Git mutations before tests can touch their checkout."""
import os
import shlex
from pathlib import Path

_WRAPPERS = {"env", "nohup", "setsid", "timeout", "sudo", "xargs", "nice", "ionice", "stdbuf", "flock"}

# Guard accidental checkout writes by update tests, not arbitrary shell code.
_MUTATIONS = {
    "pull", "reset", "stash", "checkout", "switch", "restore",
    "clean", "rebase", "merge", "cherry-pick", "revert", "apply", "am",
}
_TARGET_OPTIONS = {"--git-dir", "--work-tree"}
_VALUE_OPTIONS = _TARGET_OPTIONS | {"-C", "-c", "--namespace", "--super-prefix"}

def _command_name(token):
    return str(token).replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")

def _git_argv_tail(cmd):
    if isinstance(cmd, (list, tuple)):
        tokens = [os.fsdecode(t) for t in cmd]
    else:
        try:
            tokens = shlex.split(os.fsdecode(cmd), posix=os.name != "nt")
        except ValueError:
            return None
        if os.name == "nt":
            tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
    if not tokens:
        return None
    head = _command_name(tokens[0])
    if head == "git":
        return tokens[1:]
    if head in {"sh", "bash", "zsh", "dash", "cmd", "powershell", "pwsh"}:
        for index, token in enumerate(tokens[1:], 1):
            if token.lower() in {"-c", "-lc", "/c", "-command"} and index + 1 < len(tokens):
                return _git_argv_tail(tokens[index + 1])
        return None
    if head in _WRAPPERS:
        for index, token in enumerate(tokens[1:], 1):
            if _command_name(token) == "git":
                return tokens[index + 1:]
    return None

def _git_verb_and_targets(tail, kwargs):
    cwd = Path(kwargs.get("cwd") or os.getcwd())
    env = kwargs.get("env")
    if env is None:
        env = os.environ
    explicit = {key: env[value] for key, value in
                (("--git-dir", "GIT_DIR"), ("--work-tree", "GIT_WORK_TREE")) if env.get(value)}
    index = 0
    while index < len(tail):
        token = tail[index]
        index += 1
        if not token.startswith("-"):
            targets = [cwd, *(cwd / value for value in explicit.values())]
            return token, targets, tail[index:]
        name, separator, value = token.partition("=")
        if token.startswith("-C") and token != "-C":
            name, value, separator = "-C", token[2:], "="
        if name in _VALUE_OPTIONS:
            if not separator:
                if index == len(tail):
                    break
                value = tail[index]
                index += 1
            if name == "-C":
                cwd = cwd / value
            elif name in _TARGET_OPTIONS:
                explicit[name] = value
    return None, [], []

def blocked_git_mutation(cmd, kwargs, protected_roots):
    tail = _git_argv_tail(cmd)
    if tail is None:
        return None
    verb, targets, after = _git_verb_and_targets(tail, kwargs or {})
    if verb not in _MUTATIONS:
        return None
    if verb == "stash" and after and after[0] in {"list", "show"}:
        return None
    for target in targets:
        try:
            resolved = target.resolve()
            if any(resolved.is_relative_to(Path(root).resolve()) for root in protected_roots):
                return verb
        except (OSError, ValueError):
            return verb
    return None
