"""Security-sensitive configuration and environment policy enforcement.

Guards security-policy keys (approvals.*, security.*, command_allowlist*,
hooks.*, hooks_auto_accept) and execution-critical environment variables from
unauthorized mutation via `hermes config set`/`unset` and related APIs
(#81101, #104697).
"""

from __future__ import annotations

import os
import platform
import re
import sys
from typing import Optional

_IS_WINDOWS = platform.system() == "Windows"
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Env var names that influence how the next subprocess executes — never writable through
# ``save_env_value``: dynamic loader (LD_*/DYLD_*: attacker code loads before main()),
# interpreter init (PYTHON*, NODE_*: Hermes restarts through them), PATH (fix tool lookup
# with absolute paths instead), git rewrites (fire on every plugin install/update),
# implicitly-invoked commands (BROWSER/EDITOR/VISUAL/PAGER = RCE on next $EDITOR), SHELL,
# and Hermes runtime-location / security-policy flags (config.yaml is the supported surface).
#
# ``HERMES_*`` overall is NOT blocked — many integration credentials use that prefix
# (HERMES_LANGFUSE_PUBLIC_KEY, HERMES_SPOTIFY_CLIENT_ID, ...). The denylist is name-by-name so
# it cannot break provider setup wizards. Enforced on *write* only: pre-existing/out-of-band
# ``.env`` values keep working; the dashboard's writable surface just cannot escalate.

# Whole families whose every member steers execution or config injection, matched by prefix
# because enumeration cannot cover unbounded names (GIT_CONFIG_KEY_17 / GIT_CONFIG_VALUE_17).
_ENV_VAR_NAME_DENY_PREFIXES: tuple[str, ...] = (
    "LD_", "DYLD_",
    # PARAMETERS/COUNT/KEY_*/VALUE_* inject config pairs; GLOBAL/SYSTEM/NOSYSTEM redirect the
    # config sources _subprocess_compat already nulls for the same reason.
    "GIT_CONFIG_",
)

_ENV_VAR_NAME_DENYLIST: frozenset[str] = frozenset({
    # Loader / linker (the LD_/DYLD_ prefixes cover the family; kept name-by-name for clarity)
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_DEBUG",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
    # Python / Node — init-time injection beyond the loader paths
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PYTHONEXECUTABLE", "PYTHONNOUSERSITE", "PYTHONBREAKPOINT", "PYTHONCASEOK",
    "NODE_OPTIONS", "NODE_PATH",
    # Other interpreter / toolchain injection (same class as PYTHONPATH / NODE_OPTIONS)
    "PERL5OPT", "PERL5LIB", "PERLLIB", "RUBYOPT", "RUBYLIB", "CLASSPATH",
    "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS",
    "GOFLAGS", "RUSTFLAGS",
    # General / git — executed helpers, repo/config redirection, and template hooks
    "PATH", "SHELL", "BROWSER", "EDITOR", "VISUAL", "PAGER", "MANPAGER",
    "GIT_SSH_COMMAND", "GIT_EXEC_PATH", "GIT_SHELL",
    "GIT_SSH", "GIT_ASKPASS", "SSH_ASKPASS", "SUDO_ASKPASS",
    "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_PAGER", "GIT_EXTERNAL_DIFF",
    "GIT_PROXY_COMMAND", "GIT_TEMPLATE_DIR", "GIT_DIR",
    # Shell init files / interactive hooks — sourced before or during execution
    "BASH_ENV", "ENV", "ZDOTDIR", "PROMPT_COMMAND", "VIMINIT", "EXINIT",
    # Hermes runtime location
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_CONFIG", "HERMES_ENV",
    "HERMES_CONFIG_PATH", "HERMES_ENV_PATH",
    # MCP catalog trust root; package-manager wrappers may still set it in the process env.
    "HERMES_OPTIONAL_MCPS",
    # Local ACP subprocess selection (executable/argv authority).
    "HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS",
    # Security policy / approval-routing context — set via their dedicated controls only.
    "HERMES_YOLO_MODE", "HERMES_ACCEPT_HOOKS", "HERMES_REDACT_SECRETS",
    "HERMES_INTERACTIVE", "HERMES_EXEC_ASK", "HERMES_GATEWAY_SESSION",
    "HERMES_CRON_SESSION", "HERMES_SINGLE_QUERY_SESSION",
    "HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM"})


def _env_var_policy_name(key: str, *, is_windows: Optional[bool] = None) -> str:
    """Name used for env policy comparisons: Windows env names are case-insensitive, POSIX not.
    The override keeps both semantics testable on any host."""
    windows = _IS_WINDOWS if is_windows is None else is_windows
    return key.upper() if windows else key


def validate_env_var_name_for_write(key: str) -> None:
    """Validate an env name before a generic persistence write (exposed for batch callers)."""
    if not _ENV_VAR_NAME_RE.match(key):
        raise ValueError(f"Invalid environment variable name: {key!r}")
    policy_name = _env_var_policy_name(key)
    if policy_name in _ENV_VAR_NAME_DENYLIST or policy_name.startswith(_ENV_VAR_NAME_DENY_PREFIXES):
        raise ValueError(
            f"Environment variable {key!r} is on the writer denylist. "
            "Names that influence subprocess execution (LD_PRELOAD, PYTHONPATH, PATH, EDITOR, ...) "
            "or Hermes runtime location and security policy (HERMES_HOME, HERMES_YOLO_MODE, ...) "
            "cannot be persisted via the env writer. If you really need this, edit ~/.hermes/.env "
            "directly.")


# Security-sensitive config keys that `hermes config set`/`unset` refuses to
# mutate without an explicit operator override. ``config.yaml`` IS the security
# policy (approvals.mode, command_allowlist, security.*, executable hooks), the
# config cache is mtime-keyed, and a write takes effect mid-session — so the
# sanctioned CLI path must not let an agent silently weaken the approval gate,
# redaction, or the hook runner's consent the way it could by writing to the
# file directly. Mirrors the file-tool deny in tools/file_tools.py
# (_check_sensitive_path). The class follows the config consumers that grant,
# suppress, or execute authority: ``hooks.*`` registers shell commands the hook
# runner executes via subprocess.Popen (agent/shell_hooks.py) and
# ``hooks_auto_accept`` is the consent flag its ``_resolve_effective_accept``
# treats as auto-approval, so both belong here (#104697 review round 7, F2).
# See #81101.
_SENSITIVE_CONFIG_KEYS = (
    "approvals.",
    "security.",
    "command_allowlist",
    "hooks.",
    "hooks_auto_accept",
)


def _is_sensitive_config_key(key: str) -> bool:
    """Return whether *key* targets a security-sensitive config section.

    A bare section name (``approvals``) is guarded too, since ``config set
    approvals off --force`` would replace the whole mapping with a scalar.
    """
    normalized = key.strip().lower()
    return any(
        normalized == prefix.rstrip(".")
        or normalized.startswith(prefix)
        for prefix in _SENSITIVE_CONFIG_KEYS
    )


# Handler frames that may authorize a security-policy write. One entry per sanctioned
# human-input surface; the path is a suffix so a source checkout, an installed tree and a
# test shim all match. Two tables, deliberately different:
#
#   STAMP  — who may call grant_operator_policy_write() on its OWN frame.
#   WRITE  — whose frame satisfies the writer's caller-chain check.
#
# The gateway /approvals handler is trusted to stamp on its own frame; the interactive
# REPL handler is not, because an agent can call it unbound. Its write is authorized by
# the mixin frame, but the grant must come from process_command above it (see
# tools.approval_context._is_sanctioned_policy_stamp_caller), which an unbound call has
# no way to fabricate.
_SANCTIONED_STAMP_FRAMES: dict[str, tuple[str, ...]] = {
    "_handle_approvals_command": ("gateway/slash_commands.py",),
    "_tui_policy_write": ("tui_gateway/methods_config_set.py",),
    "_set_approval_mode": ("gateway/session_tui_config.py",),
}

_SANCTIONED_WRITE_FRAMES: dict[str, tuple[str, ...]] = {
    "_handle_approvals_command": ("gateway/slash_commands.py", "hermes_cli/cli_commands_mixin.py"),
    "_tui_policy_write": ("tui_gateway/methods_config_set.py",),
    "_set_approval_mode": ("gateway/session_tui_config.py",),
}


def is_sanctioned_stamp_frame(name: str, path: str) -> bool:
    """Whether *name*/*path* may stamp an operator-policy grant on its own frame."""
    return any(path.endswith(suffix) for suffix in _SANCTIONED_STAMP_FRAMES.get(name, ()))


def is_sanctioned_write_frame(name: str, path: str) -> bool:
    """Whether *name*/*path* satisfies the writer's sanctioned-caller check."""
    return any(path.endswith(suffix) for suffix in _SANCTIONED_WRITE_FRAMES.get(name, ()))


def _policy_write_authorized() -> bool:
    """Whether THIS call may mutate a security-policy config key.

    Two conditions, both checked here at write time:
    1. operator grant: the caller runs inside the one-shot scope stamped ONLY by a
       sanctioned human-actor handler — the gateway ``/approvals`` command (after its
       enabled-admin check) or the interactive REPL ``/approvals``. No ambient context
       (TTY, platform env) participates: an interactive CLI's TTY is shared by the
       agent in that same process, so context alone cannot distinguish a human
       keystroke from agent-executed Python (#104697 review round 3, Flash BLOCKER 2).
    2. sanctioned caller chain: the live call stack must contain a sanctioned
       operator handler frame between the write and the process entry. A forged
       grant without a handler frame — the #104059 attack shape, casual
       import-and-call from agent Python — is refused. Frame inspection is not
       cryptographically sound (a determined attacker can fake frames); it
       forecloses the realistic forge, and the residual is disclosed in the PR.
    """
    try:
        from tools.approval_context import is_operator_policy_write
    except (ImportError, AttributeError):
        return False
    if not is_operator_policy_write():
        return False
    # Walk frames directly (co_name/co_filename) instead of inspect.stack(): stack() reads
    # each frame's source file from disk, which the test harness flags as real-home file I/O
    # and which costs a disk read per frame on the write path.
    frame = sys._getframe(1)
    while frame is not None:
        path = (frame.f_code.co_filename or "").replace(os.sep, "/")
        if is_sanctioned_write_frame(frame.f_code.co_name, path):
            return True
        frame = frame.f_back
    return False


def _refuse_sensitive_config_key(key: str, *, hint: str) -> None:
    """Print a refusal for a security-sensitive key write and exit non-zero."""
    print(
        f"✗ Cannot {hint} '{key}': this key controls security policy "
        f"(approvals / security / command_allowlist / executable hooks) and "
        f"cannot be changed without explicit operator confirmation.",
        file=sys.stderr,
    )
    if key == "approvals.mode":
        print(
            "  Use the /approvals slash command (manual|smart|off) to change "
            "the approval mode.",
            file=sys.stderr,
        )
    else:
        print(
            "  Security-policy keys cannot be changed via `hermes config set` "
            "(with or without --force).",
            file=sys.stderr,
        )
        print(
            "  Edit config.yaml directly (operator), or use the dedicated "
            "slash command where one exists (`/approvals` for approvals.mode).",
            file=sys.stderr,
        )
    sys.exit(1)


def check_policy_write(key: str, hint: str = "set") -> None:
    """Refuse security-sensitive key writes without operator authorization."""
    if _is_sensitive_config_key(key):
        if hint == "unset" or not _policy_write_authorized():
            _refuse_sensitive_config_key(key, hint=hint)
