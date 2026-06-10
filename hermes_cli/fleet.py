"""In-container fleet client — Hermes' handle on host-level bot operations.

Hermes owns the fleet but lives in a container. Host operations (spinning up a
client's isolated container, stopping one, tailing its host-level logs) are
performed by SSHing to an unprivileged ``hermes-ops`` host user whose key is
locked to an allowlisted broker (``scripts/fleet_broker.py``). This module is
the thin client: it validates the subcommand locally, then hands it to the
broker over loopback SSH (the gateway container runs ``network_mode: host``, so
the host's sshd is reachable at 127.0.0.1).

Division of labor (keep it):
  * The smart, registry-aware work — editing ``clients.yaml`` and generating
    ``docker-compose.clients.yml`` + ``isolated.list`` — happens HERE, in the
    container, via the existing ``hermes_cli.compose_gen`` / ``client_split``
    (tested, registry-aware). ``hermes fleet generate`` runs that.
  * The host broker only runs bounded ``docker``/``compose`` against those
    generated artifacts. It never parses the registry.

Usage:
    python -m hermes_cli.fleet generate            # regen compose from registry
    python -m hermes_cli.fleet apply               # converge host to the compose
    python -m hermes_cli.fleet up <client>         # start one isolated bot
    python -m hermes_cli.fleet down <client>
    python -m hermes_cli.fleet restart <client>
    python -m hermes_cli.fleet status <client>
    python -m hermes_cli.fleet logs <client> [n]
    python -m hermes_cli.fleet ps
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# Loopback SSH identity the deploy drops onto the volume for Hermes.
FLEET_DIR = os.environ.get("HERMES_FLEET_DIR", "/opt/data/fleet")
SSH_KEY = os.environ.get("HERMES_FLEET_KEY", f"{FLEET_DIR}/id_ed25519")
KNOWN_HOSTS = os.environ.get("HERMES_FLEET_KNOWN_HOSTS", f"{FLEET_DIR}/known_hosts")
OPS_USER = os.environ.get("HERMES_FLEET_USER", "hermes-ops")
OPS_HOST = os.environ.get("HERMES_FLEET_HOST", "127.0.0.1")

# Host-side subcommands the broker accepts. Kept in lockstep with
# scripts/fleet_broker.py::ALLOWED so we fail fast in-container with a clear
# message instead of a generic broker denial.
REMOTE_SUBCOMMANDS = {"ps", "status", "logs", "up", "down", "restart", "apply"}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")


def _valid_slug(name: str) -> str:
    if not _SLUG.match(name):
        raise SystemExit(f"fleet: invalid client name {name!r}")
    return name


def build_ssh_argv(remote_args: Sequence[str]) -> list[str]:
    """Build the ssh argv that runs one broker subcommand. The forced command
    on the host ignores everything after the host except $SSH_ORIGINAL_COMMAND,
    which is exactly the joined remote_args."""
    return [
        "ssh",
        "-i", SSH_KEY,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        f"{OPS_USER}@{OPS_HOST}",
        " ".join(remote_args),
    ]


def _remote(remote_args: list[str]) -> int:
    if not Path(SSH_KEY).is_file():
        raise SystemExit(
            f"fleet: no broker key at {SSH_KEY} — the host fleet broker isn't "
            "provisioned yet (it is set up by the deploy). Cannot reach the host."
        )
    return subprocess.run(build_ssh_argv(remote_args), check=False).returncode


# ── In-container generation (registry → compose artifacts on the volume) ──

def cmd_generate(args: argparse.Namespace) -> int:
    """Regenerate docker-compose.clients.yml + isolated.list from the client
    registry, onto the shared volume where the broker's `apply`/`up` read them.
    This is the registry-aware half that must stay in-container."""
    from hermes_cli.compose_gen import main as compose_main

    data_root = args.data_root
    out = str(Path(data_root) / "docker-compose.clients.yml")
    argv = ["--data-root", data_root, "--output", out]
    if args.env:
        argv += ["--env", args.env]
    if args.registry:
        argv += ["--registry", args.registry]
    rc = compose_main(argv)
    if rc == 0:
        print(f"fleet: generated {out} (+ isolated.list) from the registry")
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes-fleet",
        description="Hermes' host-level fleet control (via the hermes-ops SSH broker).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="regen compose artifacts from the registry (in-container)")
    g.add_argument("--data-root", default="/opt/data",
                   help="host data volume root the isolated profiles live under")
    g.add_argument("--env", choices=["prod", "dev"], help="only this environment")
    g.add_argument("--registry", help="path to clients.yaml (default: $HERMES_CLIENTS_REGISTRY)")
    g.set_defaults(func=cmd_generate)

    # Remote (broker) subcommands — thin passthroughs with local validation.
    sub.add_parser("ps", help="list all fleet containers (host)")
    sub.add_parser("apply", help="converge the host to the generated compose")
    for name in ("up", "down", "restart", "status"):
        p = sub.add_parser(name, help=f"{name} one client's isolated container (host)")
        p.add_argument("client")
    lg = sub.add_parser("logs", help="tail one client's container logs (host)")
    lg.add_argument("client")
    lg.add_argument("lines", nargs="?", type=int, default=100)

    args = parser.parse_args(argv)

    if args.command == "generate":
        return args.func(args)

    # Build the remote command, validating client slugs before they leave.
    remote: list[str] = [args.command]
    if args.command in ("up", "down", "restart", "status"):
        remote.append(_valid_slug(args.client))
    elif args.command == "logs":
        remote.append(_valid_slug(args.client))
        if not (1 <= args.lines <= 2000):
            raise SystemExit("fleet: logs lines must be in 1..2000")
        remote.append(str(args.lines))
    return _remote(remote)


if __name__ == "__main__":
    raise SystemExit(main())
