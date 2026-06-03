"""Opt-in registry of known outbound ACP transports.

Mirror (reversed) of ``acp_registry/agent.json``: instead of advertising
Hermes *as* an agent, this records how to **launch** a known external ACP
agent as a subprocess.  Entries are explicit and opt-in — there is no
auto-discovery and no entry is enabled by default for any runtime path.

Two safety invariants are enforced *by construction* here (see design §2.7):

* **env is allowlisted.**  Only the keys named in an entry's ``env_allowlist``
  are forwarded to the child process.  No ``ANTHROPIC_*`` / ``OPENAI_*`` /
  ``HERMES_*`` credential keys are forwarded by default — the external CLI is
  expected to manage its own auth.
* **no implicit credentials.**  Opting a specific credential env-key in
  requires editing an entry's ``env_allowlist`` explicitly (a Filip-approval
  gate in higher phases), never a wildcard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


# A conservative base allowlist that every transport inherits.  These are the
# non-secret keys a child process needs to find binaries and render output
# correctly.  Deliberately excludes anything that could carry a credential.
_BASE_ENV_ALLOWLIST: tuple[str, ...] = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "NO_PROXY",
    "no_proxy",
)


@dataclass(frozen=True)
class TransportSpec:
    """How to launch a known external ACP agent.

    Attributes:
        name:          Registry key (e.g. ``"claude"``).
        command:       Executable to spawn.
        args:          Static args appended after ``command`` (ACP-mode flags).
        env_allowlist: Extra env keys (beyond the base allowlist) forwarded to
                       the child.  Credential keys belong here only behind an
                       explicit approval gate.
        auth_required: ``"external"`` means the CLI manages its own login
                       out-of-band; Hermes never forwards credentials for it.
        supports_load_session: Known ACP ``session/load`` support.  When
                       ``False`` the runner (Phase 2) falls back to a fresh
                       session + history-as-prefix.
        notes:         Free-form known-quirks string for operators.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()
    auth_required: str = "external"
    supports_load_session: bool = False
    notes: str = ""

    def resolve_env(self, base_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """Return the allowlisted env dict to hand to ``spawn_agent_process``.

        Only keys in the base allowlist + this spec's ``env_allowlist`` that are
        actually present in *base_env* (default: the live process env) are
        included.  Everything else — including all credential variables — is
        dropped.
        """
        source = os.environ if base_env is None else base_env
        allowed = set(_BASE_ENV_ALLOWLIST) | set(self.env_allowlist)
        return {k: v for k, v in source.items() if k in allowed}


# Opt-in entries only.  Each records command + ACP flags + known quirks.
# NOTE: no credential env keys are listed — every backend manages its own auth.
_DEFAULT_TRANSPORTS: tuple[TransportSpec, ...] = (
    TransportSpec(
        name="claude",
        command="claude",
        args=("--acp",),
        supports_load_session=False,
        notes="Anthropic Claude Code ACP mode; manages its own credentials.",
    ),
    TransportSpec(
        name="codex",
        command="codex",
        args=("acp",),
        supports_load_session=False,
        notes="OpenAI Codex ACP mode; manages its own credentials.",
    ),
    TransportSpec(
        name="gemini-cli",
        command="gemini",
        args=("--experimental-acp",),
        supports_load_session=False,
        notes="Google Gemini CLI ACP mode; manages its own credentials.",
    ),
    TransportSpec(
        name="hermes-acp-sibling",
        command="hermes",
        args=("acp",),
        supports_load_session=True,
        notes="A sibling Hermes instance as an ACP server (acp_adapter); "
        "persists sessions to its own state.db so session/load works.",
    ),
)


class TransportRegistry:
    """Lookup table for opt-in outbound transports.

    The registry never auto-discovers binaries on ``PATH`` and never enables a
    transport for a runtime path — it only answers "given a name, how would I
    launch it and which env keys are safe to forward".
    """

    def __init__(self, specs: Optional[List[TransportSpec]] = None):
        self._specs: Dict[str, TransportSpec] = {}
        for spec in specs if specs is not None else _DEFAULT_TRANSPORTS:
            self.register(spec)

    def register(self, spec: TransportSpec) -> None:
        """Add or replace a transport entry."""
        self._specs[spec.name] = spec

    def resolve(self, name: str) -> TransportSpec:
        """Return the spec for *name* or raise ``KeyError`` if unknown.

        Deny-by-default: an unregistered name is never silently launched.
        """
        try:
            return self._specs[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise KeyError(
                f"Unknown ACP transport {name!r}; known: {sorted(self._specs)}"
            ) from exc

    def get(self, name: str) -> Optional[TransportSpec]:
        """Return the spec for *name* or ``None`` (no raise)."""
        return self._specs.get(name)

    def names(self) -> List[str]:
        return sorted(self._specs)


# Module-level default registry for convenience.
DEFAULT_REGISTRY = TransportRegistry()
