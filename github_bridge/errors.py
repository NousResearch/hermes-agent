"""Typed exceptions for the GitHub Bridge."""
from __future__ import annotations


class BridgeError(Exception):
    """Base class for all bridge errors."""


# --- Configuration / Auth --------------------------------------------------
class ConfigMissing(BridgeError):
    """Required config or credential is missing."""


class ConfigInvalid(BridgeError):
    """Config is present but malformed."""


class AuthError(BridgeError):
    """GitHub App authentication failed (JWT mint or token exchange)."""


class TokenExpired(BridgeError):
    """Installation token is past its TTL."""


# --- Policy / safety -------------------------------------------------------
class PolicyViolation(BridgeError):
    """A safety or governance gate was tripped."""


class ProtectedBranch(BridgeError):
    """Attempted to write to a protected branch (main/master/release/*)."""


class UnsafeRemote(BridgeError):
    """Remote URL is not HTTPS api.github.com or contains credentials."""


class ScopeViolation(BridgeError):
    """Path or content is outside the bridge's allowlist."""


class SecretLeak(BridgeError):
    """Secret pattern detected in staged content or commit message."""


# --- State / file-safety ---------------------------------------------------
class InboxStateCorrupt(BridgeError):
    """The persisted state file is unreadable or unparseable.

    Per Issue #19 ChatGPT review: ``InboxStateStore.load()`` MUST fail
    closed. A silent reset to an empty dedup set would let an attacker
    (or a half-written file from a crashed prior run) replay every
    intent since the last good save.

    The store is required to *preserve evidence* by renaming the
    unreadable file to ``<path>.corrupt.<unix-ts>`` before raising.
    """


class InsecurePath(BridgeError):
    """A state/audit/lock path is unsafe (symlink, wrong owner, etc.)."""


class UntrustedCommenter(BridgeError):
    """A commenter's GitHub login is not in the operator's trust set.

    Per Issue #19 ChatGPT review: App-ID alone is not a proof of
    trusted commenter. The envelope must carry the human/bot login
    that posted the comment, and the bridge must cross-check it
    against an operator-managed allowlist.
    """


# --- Git / push ------------------------------------------------------------
class GitError(BridgeError):
    """A git operation failed."""


class PushRejected(BridgeError):
    """GitHub rejected the push (403, 422, conflict)."""


class NetworkUnreachable(BridgeError):
    """Cannot reach api.github.com."""


# --- Lifecycle -------------------------------------------------------------
class Revoked(BridgeError):
    """Token or App has been revoked; operator action required."""


class OperatorActionRequired(BridgeError):
    """A gate cannot be cleared autonomously; the operator must act.

    The single ONE operator step in the bridge design is encoded here.
    The CLI surfaces this as a structured error so the caller can prompt
    the operator without re-running the bridge.
    """

    def __init__(self, step: str, instructions: str):
        self.step = step
        self.instructions = instructions
        super().__init__(f"{step}: {instructions}")
