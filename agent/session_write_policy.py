"""T3 PR97786 — Session write policy: per-turn task-local authorization for file/terminal mutations.

The SessionWritePolicy is evaluated ONCE at agent initialization from the canonical authority
provenance (see ``agent.self_improvement_policy.evaluate``), retained on the AIAgent instance,
and bound into a ContextVar at the actual turn-execution seam in ``agent.turn_facade``.

Design invariants (C1, C2, C16):

  * Authority is initialized ONCE per AIAgent — never re-derived from the current environment
    at mutation time (C7).
  * The ContextVar is bound at the actual per-turn façade execution seam — never as a
    process-global monkeypatch on ``agent.conversation_loop.run_conversation``.
  * ``session_write_policy_scope`` installs/restores via tokens in a guaranteed finally path.
  * Malformed / missing / stale / incompatible protected authority FAILS CLOSED (C2): a
    default-session fallback would silently bypass a protected session, so any anomaly
    raises rather than downgrading.
  * Concurrent protected turns remain task-local: overlapping turns must each observe
    their own policy (C16).

This module is intentionally import-free of any runtime helpers that would force a load
order dependency on the AIAgent class.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass(frozen=True)
class TerminalPolicy:
    """Terminal enforcement shape consumed by ``tools/terminal_tool.py``.

    ``mutation_capable_expressions`` is the exact set of bash expressions whose execution
    would mutate filesystem / state and which the protected-DENY path must reject. Adding
    entries here is a contract change; consumers must keep their read-only command
    repertoire working (e.g. ``ls``, ``cat``, ``grep`` must remain allowed).
    """
    mutation_capable_expressions: frozenset = field(default_factory=lambda: frozenset({
        "rm", "mv", "cp", "mkdir", "rmdir", "touch", "chmod", "chown", "chgrp",
        "ln", "tee", ">", ">>", "sed -i", "patch", "git add", "git commit",
        "git push", "git checkout -b", "git reset", "git clean", "npm install",
        "pip install", "uv pip install",
    }))


@dataclass(frozen=True)
class SessionWritePolicy:
    """Per-turn write authorization. ``protected=True`` means the session is in a
    protected-evaluation mode (e.g. background review, cron suggestions, sub-agent
    delegation) and tool mutations must be denied.

    ``deny_all`` is a coarse-granularity override: if True, all file writes and
    mutation-capable terminal expressions are denied regardless of any sub-permission
    that may have been granted upstream. ``protected=True`` MUST imply ``deny_all=True``
    on construction; this is enforced by ``deny_all`` factory below.
    """
    protected: bool = False
    deny_all: bool = False
    file_writer: str = "normal"  # "normal" | "restricted"
    terminal_policy: TerminalPolicy = field(default_factory=TerminalPolicy)

    def __post_init__(self) -> None:
        # C2: malformed / inconsistent authority fails closed at construction.
        if self.protected and not self.deny_all:
            raise ValueError(
                "SessionWritePolicy: protected=True requires deny_all=True (fail-closed)"
            )
        if self.file_writer not in {"normal", "restricted"}:
            raise ValueError(
                f"SessionWritePolicy: unknown file_writer={self.file_writer!r} (fail-closed)"
            )

    @classmethod
    def default(cls) -> "SessionWritePolicy":
        """Normal / allow policy — the baseline for untrusted sessions."""
        return cls(protected=False, deny_all=False, file_writer="normal")

    @classmethod
    def deny_all(cls) -> "SessionWritePolicy":
        """Protected-DENY policy — fails closed for both file and terminal mutations."""
        return cls(protected=True, deny_all=True, file_writer="restricted", terminal_policy=TerminalPolicy())

    def with_terminal_policy(self, policy: TerminalPolicy) -> "SessionWritePolicy":
        return SessionWritePolicy(
            protected=self.protected,
            deny_all=self.deny_all,
            file_writer=self.file_writer,
            terminal_policy=policy,
        )


_DEFAULT_POLICY: SessionWritePolicy = SessionWritePolicy.default()

# The active per-turn policy. Bound at TurnFacadeMixin.run_conversation seam in
# ``agent/turn_facade.py``. Reading outside the bound scope returns the default.
_current_session_write_policy: ContextVar[SessionWritePolicy] = ContextVar(
    "hermes_session_write_policy", default=_DEFAULT_POLICY
)


def get_current_session_write_policy() -> SessionWritePolicy:
    """Return the bound per-turn policy (or default outside any turn)."""
    return _current_session_write_policy.get()


def require_turn_policy() -> SessionWritePolicy:
    """Return the bound policy; raise if the default fallback is in effect AND the caller
    requires a non-default policy. Used by mutation paths that MUST refuse to run under
    a default session (e.g. background review fork tools invoked outside the protected
    turn scope)."""
    policy = _current_session_write_policy.get()
    if policy is _DEFAULT_POLICY:
        raise RuntimeError(
            "SessionWritePolicy: no per-turn policy bound; refuse to operate "
            "outside an explicit turn scope (fail-closed C2/C16)"
        )
    return policy


@contextmanager
def session_write_policy_scope(policy: SessionWritePolicy) -> Iterator[SessionWritePolicy]:
    """Bind ``policy`` as the active per-turn SessionWritePolicy for the duration of the
    block. Restoration is via ContextVar token reset in a guaranteed finally path so that
    success, exception, and cancellation paths all restore the prior value (C11/C12/C13).

    Concurrent / nested / overlapping scopes are supported: ContextVars are task-local
    by construction, so two threads each binding a different policy observe their own.
    """
    if policy is None:
        # C2: None is treated as "no authority" — refuse to bind.
        raise ValueError("SessionWritePolicy: cannot bind None (fail-closed C2)")
    token: Token = _current_session_write_policy.set(policy)
    try:
        yield policy
    finally:
        _current_session_write_policy.reset(token)


__all__ = [
    "SessionWritePolicy",
    "TerminalPolicy",
    "get_current_session_write_policy",
    "require_turn_policy",
    "session_write_policy_scope",
]
