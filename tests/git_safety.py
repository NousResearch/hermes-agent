"""Classify accidental Git mutations before tests can touch their checkout."""
import os
import shlex
from pathlib import Path

_WRAPPERS = {"env", "nohup", "setsid", "timeout", "sudo", "xargs", "nice", "ionice", "stdbuf", "flock"}

# Guard accidental checkout writes by update tests, not arbitrary shell code.
_MUTATIONS = {
    "pull", "reset", "stash", "checkout", "switch", "restore",
    "clean", "rebase", "merge", "cherry-pick", "revert", "apply", "am",
    "commit", "add", "rm", "update-ref", "branch", "worktree", "tag",
}
_TARGET_OPTIONS = {"--git-dir", "--work-tree"}
# `git config` flags that answer a query; any of them makes the call read-only.
_CONFIG_READ_FLAGS = {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l"}
# `git config` options whose value is the next argv entry, so it is not the key.
_CONFIG_VALUE_OPTIONS = {"-f", "--file", "--blob", "--type", "--default", "--comment", "--value", "--url"}
# `git config <subcommand> <key>` (Git 2.46+): the key follows the subcommand.
_CONFIG_WRITE_SUBCOMMANDS = {"set", "unset", "rename-section", "remove-section"}
_VALUE_OPTIONS = _TARGET_OPTIONS | {"-C", "-c", "--namespace", "--super-prefix"}

def _command_name(token):
    return str(token).replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")

def _git_argv_tail(cmd):
    if cmd is None:
        # Popen(args=None, executable=...) and shell-less spawns without argv: nothing to classify.
        return None
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
    try:
        cwd = Path(kwargs.get("cwd") or os.getcwd())
    except FileNotFoundError:
        # The process cwd was deleted (deferred kanban cleanup, #33774); git itself will
        # resolve relative to whatever it finds, and nothing under a dead dir is protected.
        cwd = Path("/nonexistent-deleted-cwd")
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

def _config_writes_url_rewrite(after):
    """True when `git config <after>` writes a url.<base>.insteadOf/pushInsteadOf key."""
    positional = []
    index = 0
    while index < len(after):
        arg = after[index]
        index += 1
        if arg in _CONFIG_READ_FLAGS:
            return False
        if arg.startswith("-"):
            if arg in _CONFIG_VALUE_OPTIONS:
                index += 1
            continue
        positional.append(arg)
    if positional and positional[0] in {"get", "list"}:
        return False
    if positional and positional[0] in _CONFIG_WRITE_SUBCOMMANDS:
        positional = positional[1:]
    key = positional[0].lower() if positional else ""
    return key.startswith("url.") and key.endswith((".insteadof", ".pushinsteadof"))

def blocked_git_mutation(cmd, kwargs, protected_roots):
    tail = _git_argv_tail(cmd)
    if tail is None:
        return None
    verb, targets, after = _git_verb_and_targets(tail, kwargs or {})
    if verb == "config":
        # Querying config is safe; writing URL rewrites into the checkout is not.
        if not _config_writes_url_rewrite(after):
            return None
    elif verb not in _MUTATIONS:
        return None
    if verb == "stash" and after and after[0] in {"list", "show"}:
        return None
    if verb == "worktree" and after and after[0] == "list":
        return None
    if verb == "branch" and (not after or after[0] in {"--list", "--show-current", "-a", "-r", "-v", "-vv"}):
        return None
    if verb == "tag" and (not after or any(arg in {"--list", "-l"} for arg in after)):
        # --merged/--contains can precede --list. Do not let a combined
        # listing and write option bypass the checkout guard.
        if not any(arg in {"-d", "--delete", "-f", "--force", "-a", "--annotate", "-s", "--sign", "-u", "--local-user"} for arg in after):
            return None
    for target in targets:
        try:
            resolved = target.resolve()
            if any(resolved.is_relative_to(Path(root).resolve()) for root in protected_roots):
                return verb
        except (OSError, ValueError):
            return verb
    return None
