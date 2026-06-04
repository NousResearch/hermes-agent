"""Idempotent client (bot) onboarding — append to the registry + scaffold.

Turns "add a new client bot" into one safe, re-runnable command:

  python -m hermes_cli.add_client acme --env prod

It (1) appends a validated entry to the client registry, (2) creates the
profile directory under the data volume, and (3) drops a 0600 secret-env
stub for the bot's Telegram token. Re-running for an existing client is a
no-op for the registry and only fills in any missing scaffolding, so it is
safe to use in deploy automation.

Secrets are never written with real values — the stub names the env var
(``telegram_token_ref``) and leaves the operator to fill it in.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from hermes_cli.clients import (
    Client,
    RegistryError,
    parse_registry,
    profile_dir,
    resolve_registry_path,
    secret_env_path,
)


def derive_token_ref(name: str) -> str:
    """Default env-var name for a client's Telegram token (e.g. acme -> ACME_TG_TOKEN)."""
    return name.strip().upper().replace("-", "_") + "_TG_TOKEN"


def build_entry(
    name: str,
    env: str,
    *,
    profile: str | None = None,
    token_ref: str | None = None,
    model: str | None = None,
    tier: str = "standard",
    isolation: str = "shared",
) -> dict[str, Any]:
    """Build a registry entry, omitting fields left at their defaults."""
    entry: dict[str, Any] = {"name": name, "env": env}
    if profile and profile != name:
        entry["profile"] = profile
    effective_profile = profile or name
    # The root 'default' profile may inherit the base config; everything else
    # is its own bot and needs a token ref.
    if effective_profile != "default":
        entry["telegram_token_ref"] = token_ref or derive_token_ref(name)
    elif token_ref:
        entry["telegram_token_ref"] = token_ref
    if model:
        entry["model"] = model
    if tier and tier != "standard":
        entry["tier"] = tier
    if isolation and isolation != "shared":
        entry["isolation"] = isolation
    return entry


def load_doc(path: Path) -> dict[str, Any]:
    """Load the registry document, or an empty one if the file doesn't exist."""
    if not path.exists():
        return {"clients": []}
    if yaml is None:  # pragma: no cover
        raise RegistryError("PyYAML is required to edit the client registry")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {"clients": []}
    if not isinstance(data, dict) or not isinstance(data.get("clients"), list):
        raise RegistryError(f"{path} is not a valid registry (need a 'clients' list)")
    return data


def add_entry_to_doc(doc: dict[str, Any], entry: dict[str, Any]) -> bool:
    """Append ``entry`` unless its name already exists.

    Returns True if added, False if the name was already present (no-op).
    """
    existing = {c.get("name") for c in doc["clients"] if isinstance(c, dict)}
    if entry["name"] in existing:
        return False
    doc["clients"].append(entry)
    return True


def write_doc(path: Path, doc: dict[str, Any]) -> None:
    """Validate then atomically write the registry (temp file + rename)."""
    # Re-validate the whole document so we never persist a broken registry.
    parse_registry(doc)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def scaffold_client(client: Client, hermes_root: str | os.PathLike[str]) -> list[str]:
    """Create the profile dir and a 0600 secret stub if missing.

    Returns a list of human-readable actions taken (empty if nothing was
    missing — i.e. the call was a full no-op).
    """
    actions: list[str] = []
    if not client.is_default:
        pdir = profile_dir(client, hermes_root)
        if not pdir.exists():
            pdir.mkdir(parents=True, exist_ok=True)
            actions.append(f"created profile dir {pdir}")

    if client.telegram_token_ref:
        secret = secret_env_path(client, hermes_root)
        secret.parent.mkdir(parents=True, exist_ok=True)
        if not secret.exists():
            secret.write_text(
                f"# Telegram token for client '{client.name}'.\n"
                f"# Fill in the value; this file is host-only and never committed.\n"
                f"{client.telegram_token_ref}=\n",
                encoding="utf-8",
            )
            os.chmod(secret, 0o600)
            actions.append(f"created secret stub {secret} (0600)")
    return actions


def add_client(
    name: str,
    env: str,
    *,
    registry_path: str | os.PathLike[str] | None = None,
    hermes_root: str | os.PathLike[str] = "/opt/data",
    profile: str | None = None,
    token_ref: str | None = None,
    model: str | None = None,
    tier: str = "standard",
    isolation: str = "shared",
    out=sys.stdout,
) -> bool:
    """Add (or reconcile) a client. Returns True if the registry was changed."""
    path = resolve_registry_path(registry_path)
    doc = load_doc(path)
    entry = build_entry(
        name, env, profile=profile, token_ref=token_ref,
        model=model, tier=tier, isolation=isolation,
    )
    added = add_entry_to_doc(doc, entry)

    # Validate the resulting doc and get the typed client back for scaffolding.
    registry = parse_registry(doc)
    client = registry.get(name)
    assert client is not None  # just added or pre-existing

    if added:
        write_doc(path, doc)
        print(f"added '{name}' ({env}) to {path}", file=out)
    else:
        print(f"'{name}' already in {path} — reconciling scaffolding only", file=out)

    actions = scaffold_client(client, hermes_root)
    for a in actions:
        print(f"  {a}", file=out)
    if not added and not actions:
        print("  nothing to do (already fully provisioned)", file=out)

    if client.telegram_token_ref:
        print(
            f"\nNext: set {client.telegram_token_ref} in "
            f"{secret_env_path(client, hermes_root)}, then redeploy/restart "
            f"the {env} gateway.",
            file=out,
        )
    return added


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-add-client",
        description="Idempotently onboard a client bot into the registry + scaffold its profile.",
    )
    parser.add_argument("name", help="client name (lowercase slug; becomes the profile dir)")
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--registry", help="path to clients.yaml (default: $HERMES_CLIENTS_REGISTRY)")
    parser.add_argument("--hermes-root", default="/opt/data", help="data volume root")
    parser.add_argument("--profile", help="profile dir name (default: same as client name)")
    parser.add_argument("--token-ref", help="env-var name for the Telegram token (default: <NAME>_TG_TOKEN)")
    parser.add_argument("--model", help="override model for this client")
    parser.add_argument("--tier", default="standard")
    parser.add_argument("--isolation", default="shared", choices=["shared", "container"])
    args = parser.parse_args(argv)

    try:
        add_client(
            args.name, args.env,
            registry_path=args.registry,
            hermes_root=args.hermes_root,
            profile=args.profile,
            token_ref=args.token_ref,
            model=args.model,
            tier=args.tier,
            isolation=args.isolation,
        )
    except (RegistryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
