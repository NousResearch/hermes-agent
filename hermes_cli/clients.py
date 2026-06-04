"""Declarative client (bot) registry for multi-tenant Hermes deployments.

Each client bot maps to a Hermes profile under
``$HERMES_HOME/profiles/<profile>/`` — the ``default`` profile is the root
itself. Profiles are auto-discovered from the filesystem at container boot
(see :mod:`hermes_cli.container_boot`); this registry adds a *declarative
control plane* on top so deploy/onboarding tooling has a single source of
truth for which profiles should exist, in which environment (``prod`` /
``dev``), and how they are configured. It exists to replace hardcoded
profile lists in deploy scripts.

Design notes
------------
* **Secrets never live here.** Each client names an env var via
  ``telegram_token_ref``; the actual token is resolved at runtime from the
  host's secret store (e.g. ``$HERMES_HOME/secrets/<name>.env``), never from
  this file. This keeps the registry safe to version in a public repo while
  client identities/tokens stay on the host.
* **Soft isolation.** ``isolation: shared`` (default) runs the profile inside
  the shared gateway container; ``isolation: container`` marks a client for
  its own container. The on-disk layout (``profiles/<name>/``,
  ``secrets/<name>.env``) is identical either way, so graduating a client to
  full isolation is a one-flag change, not a data migration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # PyYAML is a hard dep of the project; guard only for odd import envs.
    import yaml
except Exception:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]

# Profile/client names become directory names and s6 service slots, so keep
# them to a conservative slug: lowercase alphanumerics and dashes, 2-40 chars,
# no leading/trailing dash.
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$")
VALID_ENVS = frozenset({"prod", "dev"})
VALID_ISOLATION = frozenset({"shared", "container"})

# Env var pointing at the active registry file on a host. Tooling may also
# pass an explicit path; this is just the zero-config default.
REGISTRY_ENV_VAR = "HERMES_CLIENTS_REGISTRY"


class RegistryError(ValueError):
    """Raised when the client registry is missing or structurally invalid.

    The message aggregates every problem found in one pass so a misconfigured
    registry surfaces all its issues at once rather than one-per-run.
    """


@dataclass(frozen=True)
class Client:
    """One client bot = one Hermes profile."""

    name: str
    profile: str
    env: str
    telegram_token_ref: str | None = None
    model: str | None = None
    tier: str = "standard"
    isolation: str = "shared"

    @property
    def is_default(self) -> bool:
        """The ``default`` profile is the gateway root, not a subdir."""
        return self.profile == "default"


@dataclass(frozen=True)
class Registry:
    clients: tuple[Client, ...]

    def for_env(self, env: str) -> tuple[Client, ...]:
        """Clients targeted at ``env`` (``prod`` or ``dev``), in file order."""
        return tuple(c for c in self.clients if c.env == env)

    def profiles_for_env(self, env: str) -> tuple[str, ...]:
        """Profile names for ``env`` — the data-driven replacement for the
        hardcoded profile lists deploy scripts used to carry."""
        return tuple(c.profile for c in self.for_env(env))

    def get(self, name: str) -> Client | None:
        for c in self.clients:
            if c.name == name:
                return c
        return None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.clients)


def parse_registry(data: Any) -> Registry:
    """Validate a already-loaded registry mapping and build a :class:`Registry`.

    ``data`` is the parsed YAML/JSON document: ``{"clients": [ {...}, ... ]}``.
    Raises :class:`RegistryError` listing every problem found.
    """
    issues: list[str] = []

    if not isinstance(data, dict):
        raise RegistryError(
            f"registry must be a mapping with a 'clients' list, got {type(data).__name__}"
        )

    raw_clients = data.get("clients")
    if raw_clients is None:
        raise RegistryError("registry is missing the top-level 'clients' list")
    if not isinstance(raw_clients, list):
        raise RegistryError(
            f"'clients' must be a list, got {type(raw_clients).__name__}"
        )

    clients: list[Client] = []
    seen_names: set[str] = set()
    seen_profiles: set[str] = set()

    for i, entry in enumerate(raw_clients):
        where = f"clients[{i}]"
        if not isinstance(entry, dict):
            issues.append(f"{where} must be a mapping, got {type(entry).__name__}")
            continue

        name = str(entry.get("name") or "").strip()
        if not name:
            issues.append(f"{where} is missing required field 'name'")
            continue
        if not _NAME_RE.match(name):
            issues.append(
                f"{where} name {name!r} is invalid (use lowercase letters, "
                "digits and dashes; 2-40 chars; no leading/trailing dash)"
            )
        if name in seen_names:
            issues.append(f"{where} duplicate client name {name!r}")
        seen_names.add(name)

        profile = str(entry.get("profile") or name).strip()
        if profile in seen_profiles:
            issues.append(f"{where} profile {profile!r} is already used by another client")
        seen_profiles.add(profile)

        env = str(entry.get("env") or "").strip()
        if env not in VALID_ENVS:
            issues.append(
                f"{where} ({name}) has invalid env {env!r}; expected one of {sorted(VALID_ENVS)}"
            )

        isolation = str(entry.get("isolation") or "shared").strip()
        if isolation not in VALID_ISOLATION:
            issues.append(
                f"{where} ({name}) has invalid isolation {isolation!r}; "
                f"expected one of {sorted(VALID_ISOLATION)}"
            )

        token_ref = entry.get("telegram_token_ref")
        token_ref = str(token_ref).strip() if token_ref else None
        # Every named bot needs its own Telegram token; the root 'default'
        # profile may inherit the base gateway config, so it's exempt.
        if profile != "default" and not token_ref:
            issues.append(
                f"{where} ({name}) is missing 'telegram_token_ref' — each bot "
                "needs its own token (name the env var, not the secret itself)"
            )

        model = entry.get("model")
        model = str(model).strip() if model else None
        tier = str(entry.get("tier") or "standard").strip()

        clients.append(
            Client(
                name=name,
                profile=profile,
                env=env,
                telegram_token_ref=token_ref,
                model=model,
                tier=tier,
                isolation=isolation,
            )
        )

    if issues:
        raise RegistryError(
            "invalid client registry:\n  - " + "\n  - ".join(issues)
        )

    return Registry(clients=tuple(clients))


def resolve_registry_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the registry file path.

    Precedence: explicit ``path`` arg → ``$HERMES_CLIENTS_REGISTRY``. Raises
    :class:`RegistryError` if neither is set (there is intentionally no
    silent default so tooling can't act on the wrong file).
    """
    if path:
        return Path(path)
    env_path = os.environ.get(REGISTRY_ENV_VAR, "").strip()
    if env_path:
        return Path(env_path)
    raise RegistryError(
        f"no registry path given and ${REGISTRY_ENV_VAR} is unset"
    )


def load_registry(path: str | os.PathLike[str] | None = None) -> Registry:
    """Load and validate the registry from ``path`` (or the env default)."""
    if yaml is None:  # pragma: no cover
        raise RegistryError("PyYAML is required to load the client registry")
    registry_path = resolve_registry_path(path)
    if not registry_path.is_file():
        raise RegistryError(f"registry file not found: {registry_path}")
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"could not parse {registry_path}: {exc}") from exc
    return parse_registry(data)


def profile_dir(client: Client, hermes_root: str | os.PathLike[str]) -> Path:
    """On-disk profile directory for a client.

    The ``default`` client *is* the root; named profiles live under
    ``<root>/profiles/<profile>/``. Identical regardless of ``isolation``.
    """
    root = Path(hermes_root)
    if client.is_default:
        return root
    return root / "profiles" / client.profile


def secret_env_path(client: Client, hermes_root: str | os.PathLike[str]) -> Path:
    """Per-client secret env file (host-side, never committed)."""
    return Path(hermes_root) / "secrets" / f"{client.name}.env"
