"""A local session's client environment: the allowlisted values a shell-launched client (classic CLI,
Ink TUI) sends as ``session.create``'s ``client_env``. Main's ``hermes`` process read them from its own
shell; the one gateway now runs every session, so they are frozen with the session's policy and bound
for its turns only, never written to the gateway's ``os.environ``. Desktop sends none: its sessions
keep the gateway's environment, and so does a session created before ``client_env`` existed.

Plain names and string values only. Behaviour names are read through :func:`session_env`; the
toolchain names (``PATH``, an activated venv, nvm/pyenv/conda) reach that session's tool subprocesses
through :func:`tool_env`. Values are never logged.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar

from tools.code_execution_env import is_secret_env_name

# What main's CLI read per process and the gateway otherwise reads daemon-wide.
BEHAVIOUR_ENV = ('HERMES_EPHEMERAL_SYSTEM_PROMPT', 'HERMES_PREFILL_MESSAGES_FILE', 'HERMES_INFERENCE_PROVIDER',
                 'HERMES_ENABLE_PROJECT_PLUGINS')
TOOL_ENV = ('PATH', 'VIRTUAL_ENV', 'GOPATH', 'GOROOT', 'CARGO_HOME', 'RUSTUP_HOME', 'JAVA_HOME')
TOOL_ENV_PREFIXES = ('CONDA_', 'NODE_', 'NVM_', 'PYENV_')
# ``terminal.client_env_passthrough`` names more non-secret toolchain variables for a profile.
CONFIG_KEY = 'client_env_passthrough'
_MAX_NAMES = 256
_MAX_BYTES = 256 * 1024

_BOUND: ContextVar = ContextVar('session_client_env', default=None)


def configured_names(config) -> frozenset:
    """The profile's extra tool names (``terminal.client_env_passthrough``)."""
    terminal = config.get('terminal') if isinstance(config, dict) else None
    raw = terminal.get(CONFIG_KEY) if isinstance(terminal, dict) else None
    return frozenset(name.strip() for name in raw if isinstance(name, str) and name.strip()) if isinstance(raw, list) else frozenset()


def _travels(name: str) -> bool:
    """A toolchain name may travel unless it is a credential by execute_code's secret policy (even when
    listed: ``NODE_AUTH_TOKEN`` matches ``NODE_``; secrets reach tools through ``terminal.env_passthrough``)
    or a Hermes control name (a delegated child's markers, Kanban pins)."""
    return not is_secret_env_name(name) and not name.startswith('HERMES_')


def allowed(name: str, extra: frozenset = frozenset()) -> bool:
    if name in BEHAVIOUR_ENV:
        return True
    return _travels(name) and (name in TOOL_ENV or name.startswith(TOOL_ENV_PREFIXES) or name in extra)


def collect_client_env(environ=None, config=None) -> dict[str, str]:
    """Client side: this shell's allowlisted values (the profile's extra names included)."""
    environ = os.environ if environ is None else environ
    extra = configured_names(config)
    return {name: value for name, value in environ.items() if isinstance(value, str) and allowed(name, extra)}


def accept_client_env(value, config) -> dict[str, str]:
    """Gateway side: the validated map, filtered again against the session profile's own allowlist.
    Raises ``ValueError`` for anything but a bounded flat string-to-string map."""
    if (not isinstance(value, dict) or len(value) > _MAX_NAMES
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items())
            or sum(len(k) + len(v) for k, v in value.items()) > _MAX_BYTES):
        raise ValueError('client_env must be a map of variable names to string values')
    extra = configured_names(config)
    return {name: value[name] for name in sorted(value) if allowed(name, extra)}


def retry_matches(committed_json, params) -> bool:
    """Whether a retried ``session.create`` carries the client environment its committed session froze.
    The committed (normalised) map decides, never the current allowlist: every stored name must come back
    with its value; names the gateway filtered out at creation do not count, whatever the rules are now."""
    if committed_json is None:
        return 'client_env' not in params
    value = params.get('client_env')
    return isinstance(value, dict) and all(value.get(k) == v for k, v in json.loads(committed_json).items())


@contextmanager
def client_env_scope(env):
    """Bind one session's client environment (``None``: the gateway's own) for this context."""
    token = _BOUND.set(env)
    try:
        yield
    finally:
        _BOUND.reset(token)


def bound_client_env():
    return _BOUND.get()


def session_env(name: str, default: str = '') -> str:
    """A behaviour value for the session executing here: its client's (an unexported name is unset,
    as it was for main's process), else the gateway's own environment."""
    env = _BOUND.get()
    if env is None:
        return os.getenv(name, default)
    return env.get(name, default)


def tool_env() -> dict[str, str]:
    """The toolchain values this session's tool subprocesses start with (empty when unbound)."""
    env = _BOUND.get() or {}
    return {name: value for name, value in env.items() if name not in BEHAVIOUR_ENV and _travels(name)}


def toolchain_environ(base=None) -> dict[str, str]:
    """The environment this session's tool subprocesses start from (``base``: the gateway's). Unbound
    (messaging, cron, Desktop): unchanged. A shell client's session: its toolchain replaces the gateway's,
    and a built-in toolchain name it did not export is unset, as in main's process: one client's venv
    never reaches another's session through the daemon it started."""
    env = dict(os.environ if base is None else base)
    if _BOUND.get() is None:
        return env
    kept = {k: v for k, v in env.items() if k == 'PATH' or not (k in TOOL_ENV or k.startswith(TOOL_ENV_PREFIXES))}
    return kept | tool_env()
