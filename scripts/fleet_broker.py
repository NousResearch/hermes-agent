#!/usr/bin/env python3
"""Hermes fleet broker — the ONLY command an inbound `hermes-ops` SSH session
may run.

Installed on the host and wired as the forced command for the hermes-ops SSH
key (``command="/opt/hermes/fleet/fleet_broker.py" ssh-ed25519 ...`` in
authorized_keys). OpenSSH puts the client's requested command in
``$SSH_ORIGINAL_COMMAND`` and runs THIS script instead, so the key can do
exactly what this allowlist permits and nothing else — no shell, no scp, no
arbitrary docker.

Design rules (security crux — keep them):
  * stdlib only; runs under the host's system python3 (no venv on the host).
  * NEVER shell out with a string / shell=True. Every external call is a fixed
    argv list; the only interpolated values are client names that have passed
    a strict slug regex first.
  * The host stays dumb: it runs `docker` / `docker compose` against compose
    artifacts that the in-container Hermes generated onto the shared volume
    (where the registry + tested compose_gen live). The broker never parses the
    registry or builds compose itself.
  * Every invocation is appended to an audit log with the resolved argv.
  * Unknown subcommand, bad args, or anything unparseable → exit non-zero with
    a one-line reason; the original command is never executed.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Strict client/profile slug — must match hermes_cli.service_manager's
# validate_profile_name contract (lowercase, digit, dash; bounded length).
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")

CONFIG_PATH = os.environ.get("HERMES_FLEET_CONF", "/opt/hermes/fleet/fleet.conf")
AUDIT_LOG = os.environ.get("HERMES_FLEET_AUDIT", "/opt/hermes/fleet/audit.log")
DOCKER = shutil.which("docker") or "/usr/bin/docker"


def _load_conf() -> dict:
    """Read the small key=value config the deploy writes at install time.
    Resolves the host-side volume path (so the broker can find the compose
    artifacts Hermes generates) and the gateway container name."""
    conf = {
        "VOL_SRC": "/root/.hermes",
        "GATEWAY_CONTAINER": "hermes",
        "CLIENTS_COMPOSE": "",  # default derived from VOL_SRC below
    }
    try:
        for line in Path(CONFIG_PATH).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip()
    except OSError:
        pass
    if not conf.get("CLIENTS_COMPOSE"):
        conf["CLIENTS_COMPOSE"] = str(Path(conf["VOL_SRC"]) / "docker-compose.clients.yml")
    return conf


def _audit(event: str, detail: object = "") -> None:
    try:
        Path(AUDIT_LOG).parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        who = os.environ.get("SSH_CONNECTION", "?").split(" ")[0]
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} from={who} {event} {detail}\n")
    except OSError:
        pass


def _die(reason: str, code: int = 2) -> "NoReturn":  # type: ignore[name-defined]
    _audit("DENY", reason)
    print(f"fleet-broker: denied: {reason}", file=sys.stderr)
    sys.exit(code)


def _valid_slug(name: str) -> str:
    if not _SLUG.match(name):
        _die(f"invalid client name {name!r}")
    return name


def _run(argv: list[str]) -> int:
    """Run a fixed argv (never a shell string). Streams output to the SSH
    session. Returns the child's exit code."""
    _audit("RUN", json.dumps(argv))
    try:
        return subprocess.run(argv, check=False).returncode
    except FileNotFoundError as e:
        _die(f"missing binary: {e}", code=127)


# ── Allowlisted subcommands ──
# Each handler receives the already-tokenised args (subcommand stripped) and the
# resolved conf. Keep every docker invocation to a fixed, inspectable argv.

def _compose_base(conf: dict) -> list[str]:
    compose = conf["CLIENTS_COMPOSE"]
    if not Path(compose).is_file():
        _die(f"clients compose not found at {compose} "
             "(run compose_gen in-container first)")
    return [DOCKER, "compose", "-f", compose]


def cmd_ps(args: list[str], conf: dict) -> int:
    """List all fleet containers (host-wide docker ps, read-only)."""
    return _run([DOCKER, "ps", "-a", "--filter", "name=hermes",
                 "--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}"])


def cmd_status(args: list[str], conf: dict) -> int:
    """Inspect one client's container state (read-only)."""
    if len(args) != 1:
        _die("usage: status <client>")
    name = _valid_slug(args[0])
    return _run([DOCKER, "inspect", "-f",
                 "{{.Name}} running={{.State.Running}} "
                 "restarting={{.State.Restarting}} status={{.State.Status}}",
                 f"hermes-{name}"])


def cmd_logs(args: list[str], conf: dict) -> int:
    """Tail one client's container logs (read-only). Default 100 lines, max 2000."""
    if not args or len(args) > 2:
        _die("usage: logs <client> [lines]")
    name = _valid_slug(args[0])
    tail = "100"
    if len(args) == 2:
        if not args[1].isdigit() or not (1 <= int(args[1]) <= 2000):
            _die("lines must be an integer in 1..2000")
        tail = args[1]
    return _run([DOCKER, "logs", "--tail", tail, f"hermes-{name}"])


def cmd_up(args: list[str], conf: dict) -> int:
    """Bring up one client's isolated container from the generated compose."""
    if len(args) != 1:
        _die("usage: up <client>")
    name = _valid_slug(args[0])
    return _run([*_compose_base(conf), "up", "-d", f"hermes-{name}"])


def cmd_down(args: list[str], conf: dict) -> int:
    """Stop + remove one client's isolated container."""
    if len(args) != 1:
        _die("usage: down <client>")
    name = _valid_slug(args[0])
    return _run([*_compose_base(conf), "rm", "-sf", f"hermes-{name}"])


def cmd_restart(args: list[str], conf: dict) -> int:
    if len(args) != 1:
        _die("usage: restart <client>")
    name = _valid_slug(args[0])
    return _run([*_compose_base(conf), "restart", f"hermes-{name}"])


def cmd_apply(args: list[str], conf: dict) -> int:
    """Converge the whole fleet to the generated compose (spins up newly-added
    isolated clients, applies config changes). Operates ONLY on the compose
    file Hermes generated; it does not invent services."""
    if args:
        _die("usage: apply  (no args)")
    return _run([*_compose_base(conf), "up", "-d", "--remove-orphans"])


ALLOWED = {
    "ps": cmd_ps,
    "status": cmd_status,
    "logs": cmd_logs,
    "up": cmd_up,
    "down": cmd_down,
    "restart": cmd_restart,
    "apply": cmd_apply,
}


def parse(original: str) -> tuple[str, list[str]]:
    """Tokenise $SSH_ORIGINAL_COMMAND into (subcommand, args). Rejects anything
    that isn't a known subcommand. Shell metacharacters are not special — we
    never pass the result to a shell — but shlex still rejects unbalanced quotes."""
    try:
        tokens = shlex.split(original or "")
    except ValueError as e:
        _die(f"unparseable command ({e})")
    if not tokens:
        _die("no command (this key only runs `fleet` subcommands)")
    sub, rest = tokens[0], tokens[1:]
    # Tolerate a leading "fleet" / "hermes fleet" prefix from the client.
    if sub in ("fleet", "hermes") and rest:
        if sub == "hermes" and rest and rest[0] == "fleet":
            rest = rest[1:]
        if rest:
            sub, rest = rest[0], rest[1:]
    if sub not in ALLOWED:
        _die(f"unknown subcommand {sub!r}; allowed: {', '.join(sorted(ALLOWED))}")
    return sub, rest


def main() -> int:
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    sub, args = parse(original)
    conf = _load_conf()
    _audit("ACCEPT", json.dumps([sub, *args]))
    return ALLOWED[sub](args, conf)


if __name__ == "__main__":
    raise SystemExit(main())
