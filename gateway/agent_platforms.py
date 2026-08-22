"""Per-agent platform bindings: one platform identity per agent.

The single-gateway-multi-agent work gives one gateway process N agents
(``AgentProfile`` registry + declarative routes).  Platform adapters,
however, are keyed by platform — one connection per platform, one platform
identity shared by every agent.  For identity-bearing platforms like Buzz
(Block's Nostr-based human+agent workspace), that collapses the value of
the registry: each agent SHOULD appear in the workspace as its own member,
with its own key, its own mention gate, and its own DM inbox.

This module is the bridge.  Config surface (config.yaml)::

    gateway:
      agents:
        chip:
          home_dir: ~/.hermes/agents/chip
          buzz:
            nsec_env: CHIP_BUZZ_NSEC     # NAME of the env var, never the key
            channels:                     # optional — defaults inherit from
              - <channel-uuid>            #   gateway.platforms.buzz.extra
            require_mention: true
        scout:
          buzz:
            nsec_env: SCOUT_BUZZ_NSEC

Because ``load_agent_registry`` forwards unknown agent keys into
``AgentProfile.config_overrides``, the ``buzz:`` block needs no changes to
``agent/profile.py`` — it arrives here as ``config_overrides["buzz"]``.

Design notes — why ``nsec_env`` (a name) rather than an inline secret or a
per-platform token list:

* It matches ``AgentProfile.api_key_env``: the multi-agent branch's existing
  idiom is "config names an env var; the value never appears in config.yaml".
* It matches ``gateway.multiplex_profiles``' fail-closed secret handling:
  the value may live in the process environment / secret scope, or in the
  agent's own ``<home>/.env`` (the agent-registry analogue of a profile
  home).  Resolution checks the secret scope first and the agent's ``.env``
  second, and NEVER falls back to the platform's shared credential
  (``BUZZ_PRIVATE_KEY``) — a per-agent connection that silently borrowed the
  shared key would impersonate the wrong workspace member.

Routing: each binding's adapter is given
``set_routing_context(routes=[], default_agent=<agent_id>)`` — the
connection identity IS the routing decision.  A message that arrives on
agent A's connection was addressed to agent A's member key (mention or DM),
so the gateway-level ``routes`` table is deliberately not consulted for
these adapters.  The ``select_agent`` plugin hook still runs (an explicit
override stays possible).

Eventual unification with ``multiplex_profiles`` (which solves the adjacent
problem — per-*profile* platform credentials) is out of scope here and
tracked in the PR description; both systems deliberately share the
fingerprint salt so a credential claimed by either one is refused by the
other.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig

logger = logging.getLogger(__name__)

# Platforms that support per-agent identities.  Buzz first: on Nostr an
# identity is just a keypair, so "give every agent its own account" is a
# config entry rather than a per-bot app registration.  Other platforms can
# opt in later by teaching their adapter a ``<secret>_env`` extra.
SUPPORTED_AGENT_PLATFORMS: Tuple[str, ...] = ("buzz",)

# Per-platform name of the credential key inside the agent's platform block.
_CREDENTIAL_ENV_KEYS: Dict[str, str] = {"buzz": "nsec_env"}

# Extra keys that carry the credential *name* into the adapter.
_PRIVATE_KEY_ENV_EXTRA = "private_key_env"
_ENV_FILE_EXTRA = "env_file"

# Salt shared with GatewayRunner._adapter_credential_fingerprint so the
# multiplex same-token guard and this module's same-key guard see identical
# fingerprints for identical secrets.
_FINGERPRINT_SALT = "hermes-mux:"


@dataclass
class AgentPlatformBinding:
    """One agent's claim on one platform identity."""

    agent_id: str
    platform_value: str
    secret_env: str
    env_file: Optional[Path]
    config: PlatformConfig = field(repr=False)

    @property
    def platform(self) -> Platform:
        return Platform(self.platform_value)


def build_agent_platform_bindings(
    registry: Dict[str, Any],
    gateway_config: Any = None,
) -> Tuple[List[AgentPlatformBinding], List[str]]:
    """Derive per-agent platform bindings from the AgentProfile registry.

    Returns ``(bindings, problems)`` where *problems* are operator-facing
    warnings (never containing secret values).  Agents without a platform
    block are silently skipped — per-agent identity is opt-in.
    """
    bindings: List[AgentPlatformBinding] = []
    problems: List[str] = []
    claimed_env_names: Dict[Tuple[str, str], str] = {}

    for agent_id in sorted(registry or {}):
        profile = registry[agent_id]
        overrides = getattr(profile, "config_overrides", None) or {}
        for platform_value in SUPPORTED_AGENT_PLATFORMS:
            block = overrides.get(platform_value)
            if block is None:
                continue
            if not isinstance(block, dict):
                problems.append(
                    f"agents.{agent_id}.{platform_value} ignored: expected a "
                    f"mapping, got {type(block).__name__}"
                )
                continue
            cred_key = _CREDENTIAL_ENV_KEYS[platform_value]
            secret_env = str(block.get(cred_key) or "").strip()
            if not secret_env:
                problems.append(
                    f"agents.{agent_id}.{platform_value} ignored: "
                    f"'{cred_key}' is required (the NAME of the env var "
                    f"holding this agent's key — never the key itself)"
                )
                continue
            owner = claimed_env_names.get((platform_value, secret_env))
            if owner is not None:
                problems.append(
                    f"agents.{agent_id}.{platform_value} ignored: {cred_key} "
                    f"{secret_env!r} is already claimed by agent {owner!r} — "
                    f"one key cannot be two workspace members"
                )
                continue
            claimed_env_names[(platform_value, secret_env)] = agent_id

            base_extra: Dict[str, Any] = {}
            platforms = getattr(gateway_config, "platforms", None) or {}
            base_pc = platforms.get(Platform(platform_value))
            if base_pc is not None and isinstance(
                getattr(base_pc, "extra", None), dict
            ):
                base_extra = dict(base_pc.extra)

            extra = dict(base_extra)
            extra.update(
                {k: v for k, v in block.items() if k != cred_key}
            )
            extra[_PRIVATE_KEY_ENV_EXTRA] = secret_env
            extra["agent_id"] = agent_id

            env_file: Optional[Path] = None
            try:
                env_file = Path(profile.resolved_home) / ".env"
            except Exception:
                env_file = None
            if env_file is not None:
                extra[_ENV_FILE_EXTRA] = str(env_file)

            bindings.append(
                AgentPlatformBinding(
                    agent_id=agent_id,
                    platform_value=platform_value,
                    secret_env=secret_env,
                    env_file=env_file,
                    config=PlatformConfig(enabled=True, extra=extra),
                )
            )

    return bindings, problems


def resolve_binding_secret(binding: AgentPlatformBinding) -> str:
    """Resolve a binding's credential VALUE.  Never log the return value.

    Order: gateway secret scope / process environment (by name), then the
    agent's own ``<home>/.env``.  Deliberately no fallback to any shared
    platform credential — see the module docstring.
    """
    value = ""
    try:
        from agent.secret_scope import get_secret

        try:
            value = (get_secret(binding.secret_env, "") or "").strip()
        except Exception:
            value = ""
    except Exception:
        value = os.getenv(binding.secret_env, "").strip()
    if not value and binding.env_file is not None:
        try:
            from agent.secret_scope import load_env_file

            value = (
                load_env_file(Path(binding.env_file).expanduser()).get(
                    binding.secret_env
                )
                or ""
            ).strip()
        except Exception:
            value = ""
    return value


def secret_fingerprint(secret: str) -> Optional[str]:
    """Log-safe fingerprint of a credential (salted hash, never the value)."""
    if not isinstance(secret, str) or not secret.strip():
        return None
    return hashlib.sha256(
        (_FINGERPRINT_SALT + secret.strip()).encode("utf-8")
    ).hexdigest()[:16]
