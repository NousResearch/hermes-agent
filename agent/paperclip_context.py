"""Launch-only Paperclip run context. Never put these values in policy JSON or os.environ.

A turn takes an immutable copy through ContextVar binding; child builders (tools/environments/
local.py's ``_make_run_env``, tools/code_execution_env.py's ``_build_child_env``) remove ALL
inherited PAPERCLIP_* names before applying that copy last, so neither the daemon's own ambient
environment nor any ``env_passthrough``/``client_env`` allowlist can smuggle or shadow it.

This ContextVar is bound once per Paperclip-run session, across the CLI process -> shared gateway
RPC boundary (``hermes_cli.gateway_chat.paperclip_context_policy`` sends ``collect_context()`` as
``session.create``'s ``paperclip_context``; ``gateway.session_policy.bind_paperclip_context`` validates
it and keeps the actual values ONLY in the owning authority's in-memory registry, for this authority's
lifetime, exactly like ``bind_launch_key`` keeps an API key out of the durable receipt -- the frozen
``LocalSessionPolicy`` carries just an opaque ``paperclip_context_ref`` (a session slot plus a
non-reversible fingerprint), never the values themselves, so they never reach ``asdict(policy)`` or the
session's SQLite receipt. ``gateway.session_policy.policy_scope`` resolves that ref back through the
authority and binds the result for every turn via ``LocalSessionPolicy.paperclip_scope``, mirroring the
existing ``client_env`` mechanism. A restarted authority's registry is empty, so a session that carried
a context runs its turns with none afterward -- the same fail-closed disposition as a session a Paperclip
wake never launched (TUI, Desktop, ACP, bot rooms, messaging/cron), which sends or stores none to begin
with.

Accepted risk, by design (MON-1027 review): ``paperclip_context`` is a generic ``session.create`` field,
so any client already inside this gateway's local trust tier -- the tier ``api_key`` and ``client_env``
already rely on, since reaching ``session.create`` at all requires the private control socket's ticket --
can assert an arbitrary Paperclip identity for that session's terminal/execute_code children.
``validate_context`` checks only the shape (allowlisted names, bounded size, all-or-nothing required
subset), never that the asserted identity belongs to the caller: there is no cryptographic binding, and
none is added here. The trust boundary is unchanged (a caller able to reach this RPC locally is already
trusted with a real model API key and an arbitrary ``client_env``); this adds a field inside that
boundary, not a new one. Pinned by ``tests/gateway/test_paperclip_context.py``.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os

NAMES = frozenset({'PAPERCLIP_API_URL', 'PAPERCLIP_API_KEY', 'PAPERCLIP_AGENT_ID',
                   'PAPERCLIP_COMPANY_ID', 'PAPERCLIP_RUN_ID', 'PAPERCLIP_TASK_ID',
                   'PAPERCLIP_WAKE_REASON', 'PAPERCLIP_WAKE_COMMENT_ID'})
_REQUIRED = NAMES - {'PAPERCLIP_TASK_ID', 'PAPERCLIP_WAKE_REASON', 'PAPERCLIP_WAKE_COMMENT_ID'}
_BOUND: ContextVar = ContextVar('paperclip_run_context', default=None)


def collect_context(environ=None):
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if k in NAMES and isinstance(v, str) and v}


def validate_context(value):
    # Never include offending names/values in exceptions: input is secret-bearing.
    if (not isinstance(value, dict) or set(value) - NAMES
            or any(not isinstance(v, str) or not v or '\x00' in v for v in value.values())
            or sum(len(v) for v in value.values()) > 64 * 1024
            or (value and not _REQUIRED <= set(value))):
        raise ValueError('invalid Paperclip run context')
    return dict(value)


@contextmanager
def paperclip_scope(value):
    values = validate_context(value)
    # Reuse the bounded, profile-scoped hard-egress registry. Opaque run keys have
    # no guaranteed regex shape, and background/log readers can outlive this turn.
    # This registry is redaction-only: it cannot supply a credential to a tool.
    if values.get('PAPERCLIP_API_KEY'):
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(values['PAPERCLIP_API_KEY'])
    token = _BOUND.set(values)
    try:
        yield
    finally:
        _BOUND.reset(token)


def child_environ(base=None):
    """The only sanctioned source of PAPERCLIP_* for a child process.

    Strips any PAPERCLIP_* already in *base* (daemon ambient env, an inherited passthrough rule)
    and overlays the current turn's bound copy, so a child always sees exactly this turn's run
    identity, never a stale or foreign one. Child env builders must call this LAST, after every
    other env-assembly step, so nothing added afterward can re-introduce a raw PAPERCLIP_* value.
    """
    env = {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith('PAPERCLIP_')}
    return env | dict(_BOUND.get() or {})


def context_identity():
    """Opaque in-memory kernel identity. Never log or persist it."""
    return hashlib.sha256(json.dumps(_BOUND.get() or {}, sort_keys=True).encode()).digest()
