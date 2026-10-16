"""A local session's client environment: the allowlisted values a shell-launched client (classic CLI,
Ink TUI) sends as ``session.create``'s ``client_env``. Main's ``hermes`` process read them from its own
shell; the one gateway now runs every session, so they are frozen with the session's policy and bound
for its turns only, never written to the gateway's ``os.environ``. Desktop sends none: its sessions
keep the gateway's environment, and so does a session created before ``client_env`` existed.

Plain names and string values only. Behaviour names are read through :func:`session_env`; the
toolchain names (``PATH``, an activated venv, nvm/pyenv/conda) reach that session's tool subprocesses
through :func:`toolchain_environ`. Values naming this host (toolchain paths, the prefill file, the
project-plugins check of the client's folder) apply only when the gateway proved the client is on its
host (``local``); otherwise the session keeps the gateway's and is told so. Values are never logged.
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
# Behaviour values that name the client's host: a path, or the opt-in that inspects the client's folder.
HOST_BEHAVIOUR = ('HERMES_PREFILL_MESSAGES_FILE', 'HERMES_ENABLE_PROJECT_PLUGINS')
TOOL_ENV = ('PATH', 'VIRTUAL_ENV', 'GOPATH', 'GOROOT', 'CARGO_HOME', 'RUSTUP_HOME', 'JAVA_HOME')
TOOL_ENV_PREFIXES = ('CONDA_', 'NODE_', 'NVM_', 'PYENV_')
# ``terminal.client_env_passthrough`` names more non-secret toolchain variables for a profile.
CONFIG_KEY = 'client_env_passthrough'
# The session-start notice of a session whose client the gateway could not place on its own host.
NOT_LOCAL_WARNING = ("this session uses the gateway's own PATH, toolchain and prefill file: the client's environment "
                     'was not applied because the client is not attached to this gateway locally.')
_MAX_NAMES = 256
_MAX_BYTES = 256 * 1024

# (env, local) for the session executing here; None outside a shell client's session.
_BOUND: ContextVar = ContextVar('session_client_env', default=None)


def configured_names(config) -> frozenset:
    """The profile's extra tool names (``terminal.client_env_passthrough``)."""
    terminal = config.get('terminal') if isinstance(config, dict) else None
    raw = terminal.get(CONFIG_KEY) if isinstance(terminal, dict) else None
    return frozenset(name.strip() for name in raw if isinstance(name, str) and name.strip()) if isinstance(raw, list) else frozenset()


def _travels(name: str, secret=None) -> bool:
    """A toolchain name may travel unless it is a credential by execute_code's secret policy (even when
    listed: ``NODE_AUTH_TOKEN`` matches ``NODE_``; secrets reach tools through ``terminal.env_passthrough``)
    or a Hermes control name (a delegated child's markers, Kanban pins). ``secret``: a session's frozen
    copy of that policy's substrings."""
    hidden = is_secret_env_name(name) if secret is None else any(s in name.upper() for s in secret)
    return not hidden and not name.startswith('HERMES_')


def client_env_rules(config, *, local=True) -> dict:
    """The normalisation rules (plain data) a session's map is filtered by, frozen with it: an exact retry
    is judged by these, never by the profile's allowlist or the secret policy as they are later. A client
    not proven ``local`` keeps only the values that name nothing on this host (prompt, provider request)."""
    from tools.code_execution_env import _SECRET_SUBSTRINGS
    return {'behaviour': [n for n in BEHAVIOUR_ENV if local or n not in HOST_BEHAVIOUR],
            'names': sorted(set(TOOL_ENV) | configured_names(config)) if local else [],
            'prefixes': list(TOOL_ENV_PREFIXES) if local else [], 'secret': list(_SECRET_SUBSTRINGS)}


def _kept(name: str, rules: dict) -> bool:
    return name in rules['behaviour'] or (_travels(name, rules['secret']) and (
        name in rules['names'] or name.startswith(tuple(rules['prefixes']))))


def collect_client_env(environ=None, config=None) -> dict[str, str]:
    """Client side: this shell's allowlisted values (the profile's extra names included)."""
    environ = os.environ if environ is None else environ
    rules = client_env_rules(config)
    return {name: value for name, value in environ.items() if isinstance(value, str) and _kept(name, rules)}


def normalise_client_env(value, rules) -> dict[str, str]:
    """The validated map filtered by ``rules`` (``None``: unfiltered, for a map frozen before rules were).
    Raises ``ValueError`` for anything but a bounded flat string-to-string map."""
    if (not isinstance(value, dict) or len(value) > _MAX_NAMES
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items())
            or sum(len(k) + len(v) for k, v in value.items()) > _MAX_BYTES):
        raise ValueError('client_env must be a map of variable names to string values')
    return {name: value[name] for name in sorted(value) if rules is None or _kept(name, rules)}


def accept_client_env(value, config, *, local=True) -> dict[str, str]:
    """Gateway side: the map filtered by the session profile's own current rules."""
    return normalise_client_env(value, client_env_rules(config, local=local))


def retry_matches(committed: dict, request: dict) -> bool:
    """Whether a retried ``session.create`` (``request``: its params) carries the client environment the
    committed session (``committed``: its stored policy) froze. The complete retried map, normalised by
    the rules frozen with the session, must equal the stored one: an added name those rules keep, a
    missing stored name or a changed value is a different request; one they drop never counts. Staging
    and the transactional commit both judge by this. A session committed without a map (Desktop, one
    from before the field) still matches only a retry that sends none."""
    stored = committed.get('client_env_json')
    if stored is None or 'client_env' not in request:
        return stored is None and 'client_env' not in request
    rules = committed.get('client_env_rules_json')
    try:
        sent = normalise_client_env(request['client_env'], None if rules is None else json.loads(rules))
    except ValueError:
        return False
    return sent == json.loads(stored)


@contextmanager
def client_env_scope(env, *, local=True):
    """Bind one session's client environment (``None``: the gateway's own) for this context; ``local``
    is the gateway's own proof, frozen in the session's policy, that the client runs on this host."""
    token = _BOUND.set(None if env is None else (env, bool(local)))
    try:
        yield
    finally:
        _BOUND.reset(token)


def bound_client_env():
    bound = _BOUND.get()
    return None if bound is None else bound[0]


def session_env(name: str, default: str = '') -> str:
    """A behaviour value for the session executing here: its client's (an unexported name is unset,
    as it was for main's process), else the gateway's own environment, which a client not attached
    locally also keeps for the values that name this host."""
    bound = _BOUND.get()
    if bound is None or (name in HOST_BEHAVIOUR and not bound[1]):
        return os.getenv(name, default)
    return bound[0].get(name, default)


def tool_env() -> dict[str, str]:
    """The toolchain values this session's tool subprocesses start with (empty when unbound or remote)."""
    bound = _BOUND.get()
    if bound is None or not bound[1]:
        return {}
    return {name: value for name, value in bound[0].items() if name not in BEHAVIOUR_ENV and _travels(name)}


def toolchain_environ(base=None) -> dict[str, str]:
    """The environment this session's tool subprocesses start from (``base``: the gateway's). Unbound
    (messaging, cron, Desktop) or a client not attached locally: unchanged. A local shell client's
    session: its toolchain replaces the gateway's, and a built-in toolchain name it did not export is
    unset, as in main's process: one client's venv never reaches another's session through the daemon
    it started."""
    env = dict(os.environ if base is None else base)
    bound = _BOUND.get()
    if bound is None or not bound[1]:
        return env
    kept = {k: v for k, v in env.items() if k == 'PATH' or not (k in TOOL_ENV or k.startswith(TOOL_ENV_PREFIXES))}
    return kept | tool_env()
