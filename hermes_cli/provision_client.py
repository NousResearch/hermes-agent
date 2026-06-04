"""One-command client-bot provisioning, in-container, no host access.

A new "sibling" client bot is just a new Hermes profile: the gateway runs a
supervised per-profile s6 service that ``hermes profile create`` registers and
starts live (see hermes_cli/profiles.py). This orchestrator wraps the whole
flow behind a single deterministic call so a *cheap* model (e.g. DeepSeek),
driven by the ``provision-client`` skill, can run it reliably:

  1. record the client in the registry + drop a 0600 token secret stub
     (in-volume; never touches the host);
  2. write the Telegram token into the secret if supplied;
  3. guard: refuse to *activate* a bot whose token is still empty (a cheap
     model shouldn't spin up a dead bot);
  4. create + launch the profile via ``hermes profile create`` (cloning a
     template profile so the bot inherits a known config — e.g. a cheap
     default model — instead of starting blank).

It is idempotent: re-running for an already-created profile is a no-op for
both the registry and the profile.

Out of scope (host-level, deliberately not automated here): creating sibling
*volumes* / *containers* (client_split, compose_gen) — those need host Docker
and stay a separate, human-approved step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from hermes_cli.add_client import add_client
from hermes_cli.clients import load_registry, profile_dir, secret_env_path

Runner = Callable[[Sequence[str]], None]


def _default_runner(argv: Sequence[str]) -> None:
    subprocess.run(list(argv), check=True)


def token_value(secret_path: Path, token_ref: str) -> str | None:
    """Return the non-empty value of ``token_ref`` in an env file, else None."""
    if not secret_path.is_file():
        return None
    prefix = f"{token_ref}="
    for line in secret_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return value or None
    return None


def write_token(secret_path: Path, token_ref: str, value: str) -> None:
    """Set ``token_ref`` to ``value`` in the secret env file (0600), preserving
    any other lines."""
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{token_ref}="
    lines: list[str] = []
    found = False
    if secret_path.is_file():
        for line in secret_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(prefix):
                lines.append(f"{token_ref}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{token_ref}={value}")
    secret_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(secret_path, 0o600)


def profile_is_created(pdir: Path) -> bool:
    """A profile is 'created' once ``hermes profile create`` has seeded its
    config.yaml (add_client only makes the bare directory)."""
    return (pdir / "config.yaml").is_file()


def build_create_command(name: str, *, clone_from: str | None, description: str | None) -> list[str]:
    cmd = ["hermes", "profile", "create", name]
    if clone_from:
        cmd += ["--clone", "--clone-from", clone_from]
    if description:
        cmd += ["--description", description]
    return cmd


def provision_client(
    name: str,
    env: str = "prod",
    *,
    token: str | None = None,
    model: str | None = None,
    clone_from: str | None = None,
    description: str | None = None,
    registry_path: str | os.PathLike[str] | None = None,
    hermes_root: str | os.PathLike[str] = "/opt/data",
    require_token: bool = True,
    runner: Runner = _default_runner,
    out=sys.stdout,
) -> bool:
    """Provision (or reconcile) a client bot. Returns True if the profile was
    newly created."""
    # 1. Registry entry + profile dir + secret stub (idempotent, in-volume).
    add_client(
        name, env, model=model,
        registry_path=registry_path, hermes_root=hermes_root, out=out,
    )

    registry = load_registry(registry_path)
    client = registry.get(name)
    assert client is not None
    secret = secret_env_path(client, hermes_root)

    # 2. Write the token if provided.
    if token and client.telegram_token_ref:
        write_token(secret, client.telegram_token_ref, token)
        print(f"  wrote token into {secret}", file=out)

    # 3. Guard: don't activate a bot with no token.
    has_token = bool(
        client.telegram_token_ref and token_value(secret, client.telegram_token_ref)
    )
    if require_token and client.telegram_token_ref and not has_token:
        raise ValueError(
            f"refusing to activate '{name}': {client.telegram_token_ref} is not "
            f"set in {secret}. Pass --token, fill the secret, or use "
            "--allow-empty-token to stage without starting the gateway."
        )

    # 4. Create + launch the profile, unless already created.
    pdir = profile_dir(client, hermes_root)
    if profile_is_created(pdir):
        print(f"  profile '{name}' already created — skipping (no-op)", file=out)
        return False

    cmd = build_create_command(name, clone_from=clone_from, description=description)
    print(f"  creating + launching profile: {' '.join(cmd)}", file=out)
    runner(cmd)
    print(f"provisioned '{name}' ({env}); gateway registered and started", file=out)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-provision-client",
        description="Provision a new client bot (registry + secret + live profile gateway).",
    )
    parser.add_argument("name")
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--token", help="Telegram bot token value (written to the secret, 0600)")
    parser.add_argument("--model", help="model slug recorded for this client (e.g. a cheap default)")
    parser.add_argument("--clone-from", help="template profile to clone config from")
    parser.add_argument("--description", help="role description for task routing")
    parser.add_argument("--registry", help="path to clients.yaml (default: $HERMES_CLIENTS_REGISTRY)")
    parser.add_argument("--hermes-root", default="/opt/data")
    parser.add_argument("--allow-empty-token", action="store_true",
                        help="stage the client without a token (does not start the gateway)")
    args = parser.parse_args(argv)

    try:
        provision_client(
            args.name, args.env,
            token=args.token, model=args.model,
            clone_from=args.clone_from, description=args.description,
            registry_path=args.registry, hermes_root=args.hermes_root,
            require_token=not args.allow_empty_token,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
