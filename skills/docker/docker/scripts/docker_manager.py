#!/usr/bin/env python3
"""Docker container and image manager — inspect, clean, and monitor Docker resources.

Usage:
    python docker_manager.py ps                          # list running containers
    python docker_manager.py ps --all                    # list all containers
    python docker_manager.py images                      # list images with sizes
    python docker_manager.py stats                       # one-shot resource snapshot
    python docker_manager.py inspect <name_or_id>        # inspect a container (pretty)
    python docker_manager.py logs <name_or_id>           # tail last 50 lines of logs
    python docker_manager.py logs <name_or_id> --lines 200
    python docker_manager.py clean                       # dry-run cleanup report
    python docker_manager.py clean --execute             # actually remove unused resources
    python docker_manager.py df                          # disk usage summary

No dependencies beyond Python stdlib and a running Docker daemon.
"""

import json
import subprocess
import sys
from datetime import datetime


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(args: list[str], check: bool = True, include_stderr: bool = False) -> str:
    """Run a docker command and return stdout (optionally combined with stderr)."""
    try:
        result = subprocess.run(
            ["docker"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )
        if include_stderr and result.stderr:
            # docker logs writes application output to stderr by design
            combined = "\n".join(filter(None, [result.stdout, result.stderr]))
            return combined.strip()
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Docker is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def _json(args: list[str]) -> list | dict:
    """Run a docker command that returns JSON and parse it."""
    raw = _run(args)
    if not raw:
        return []
    return json.loads(raw)


def _ago(iso: str) -> str:
    """Convert an ISO timestamp to a human-readable 'N ago' string."""
    try:
        dt = datetime.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")
        s = int((datetime.utcnow() - dt).total_seconds())
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return iso[:10] if iso else "unknown"


def _parse_json_lines(raw: str) -> list[dict]:
    """Parse docker output where each line is a separate JSON object."""
    result = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_ps(show_all: bool = False) -> None:
    """List containers in a readable table."""
    args = ["ps", "--format", "json"]
    if show_all:
        args.append("--all")

    raw = _run(args)
    if not raw:
        print("No containers found.")
        return

    containers = _parse_json_lines(raw)
    if not containers:
        print("No containers found.")
        return

    print(f"{'NAME':<24} {'IMAGE':<32} {'STATUS':<20} {'PORTS':<28} {'CREATED'}")
    print("-" * 115)
    for c in containers:
        print(
            f"{c.get('Names','')[:23]:<24} "
            f"{c.get('Image','')[:31]:<32} "
            f"{c.get('Status','')[:19]:<20} "
            f"{c.get('Ports','')[:27]:<28} "
            f"{c.get('CreatedAt','')[:16]}"
        )
    print(f"\nTotal: {len(containers)} container(s)")


def cmd_images() -> None:
    """List images with their sizes."""
    raw = _run(["images", "--format", "json"])
    if not raw:
        print("No images found.")
        return

    images = _parse_json_lines(raw)
    if not images:
        print("No images found.")
        return

    print(f"{'REPOSITORY':<36} {'TAG':<20} {'IMAGE ID':<14} {'SIZE':<12} {'CREATED'}")
    print("-" * 100)
    for img in images:
        print(
            f"{img.get('Repository','<none>')[:35]:<36} "
            f"{img.get('Tag','<none>')[:19]:<20} "
            f"{img.get('ID','')[:13]:<14} "
            f"{img.get('Size','')[:11]:<12} "
            f"{img.get('CreatedAt','')[:16]}"
        )
    print(f"\nTotal: {len(images)} image(s)")


def cmd_stats() -> None:
    """One-shot CPU/memory/network snapshot for all running containers."""
    raw = _run(["stats", "--no-stream", "--format", "json"])
    if not raw:
        print("No running containers.")
        return

    stats = _parse_json_lines(raw)
    if not stats:
        print("No running containers.")
        return

    print(f"{'NAME':<24} {'CPU %':<10} {'MEM USAGE':<20} {'MEM %':<10} {'NET I/O':<22} {'BLOCK I/O'}")
    print("-" * 105)
    for s in stats:
        print(
            f"{s.get('Name','')[:23]:<24} "
            f"{s.get('CPUPerc','0%')[:9]:<10} "
            f"{s.get('MemUsage','')[:19]:<20} "
            f"{s.get('MemPerc','')[:9]:<10} "
            f"{s.get('NetIO','')[:21]:<22} "
            f"{s.get('BlockIO','')}"
        )
    print(f"\nSnapshot of {len(stats)} running container(s)")


def cmd_inspect(name: str) -> None:
    """Pretty-print key details about a container."""
    data = _json(["inspect", name])
    if not data:
        print(f"Container '{name}' not found.")
        return

    c      = data[0]
    cfg    = c.get("Config", {})
    state  = c.get("State", {})
    net    = c.get("NetworkSettings", {})
    mounts = c.get("Mounts", [])

    print(f"=== {c.get('Name', '').lstrip('/')} ===")
    print(f"  ID:         {c.get('Id', '')[:12]}")
    print(f"  Image:      {cfg.get('Image', '')}")
    print(f"  Status:     {state.get('Status', '')}  (PID {state.get('Pid', 0)})")
    print(f"  Started:    {_ago(state.get('StartedAt', ''))}")
    print(f"  Cmd:        {' '.join(cfg.get('Cmd') or [])}")
    print(f"  Entrypoint: {' '.join(cfg.get('Entrypoint') or [])}")

    # Show env var keys only (never values — may contain secrets)
    env = [e for e in (cfg.get("Env") or []) if e]
    if env:
        print(f"\n  Env ({len(env)} vars — keys only):")
        for e in env[:8]:
            print(f"    {e.split('=')[0]}")
        if len(env) > 8:
            print(f"    ... and {len(env) - 8} more")

    # Port bindings
    ports = net.get("Ports", {})
    if ports:
        print("\n  Ports:")
        for container_port, bindings in ports.items():
            if bindings:
                for b in bindings:
                    print(f"    {b.get('HostIp','0.0.0.0')}:{b.get('HostPort')} -> {container_port}")
            else:
                print(f"    {container_port} (not published)")

    # Networks
    networks = net.get("Networks", {})
    if networks:
        print("\n  Networks:")
        for net_name, net_info in networks.items():
            print(f"    {net_name}: {net_info.get('IPAddress') or 'no IP'}")

    # Volume mounts
    if mounts:
        print("\n  Mounts:")
        for m in mounts:
            src = m.get("Source", "")[-40:]
            print(
                f"    [{m.get('Type','bind')}] ...{src} "
                f"-> {m.get('Destination','')} ({m.get('Mode','rw')})"
            )


def cmd_logs(name: str, lines: int = 50) -> None:
    """Tail the last N lines of a container's logs.

    Docker writes container output to stderr by design, so both
    stdout and stderr are captured and combined.
    """
    output = _run(
        ["logs", "--tail", str(lines), name],
        check=False,
        include_stderr=True,
    )
    if output:
        print(output)
    else:
        print(f"No logs for '{name}' (container not found or no output yet).")


def cmd_clean(execute: bool = False) -> None:
    """Report unused Docker resources and optionally remove them."""
    print("=== Docker Cleanup Report ===\n")

    stopped_raw = _run([
        "ps", "-a", "--filter", "status=exited",
        "--format", "{{.Names}}\t{{.ID}}\t{{.Status}}",
    ])
    stopped = [l for l in stopped_raw.splitlines() if l.strip()]
    print(f"Stopped containers: {len(stopped)}")
    for s in stopped[:5]:
        print(f"  {s}")
    if len(stopped) > 5:
        print(f"  ... and {len(stopped) - 5} more")

    dangling_raw = _run([
        "images", "--filter", "dangling=true",
        "--format", "{{.ID}}\t{{.Size}}",
    ])
    dangling = [l for l in dangling_raw.splitlines() if l.strip()]
    print(f"\nDangling images: {len(dangling)}")
    for d in dangling[:5]:
        print(f"  {d}")
    if len(dangling) > 5:
        print(f"  ... and {len(dangling) - 5} more")

    volumes_raw = _run([
        "volume", "ls", "--filter", "dangling=true",
        "--format", "{{.Name}}",
    ])
    volumes = [v for v in volumes_raw.splitlines() if v.strip()]
    print(f"\nUnused volumes: {len(volumes)}")
    for v in volumes[:5]:
        print(f"  {v}")
    if len(volumes) > 5:
        print(f"  ... and {len(volumes) - 5} more")

    if not execute:
        print("\n[Dry run] Add --execute to actually remove these resources.")
        print("         WARNING: volume removal is permanent and cannot be undone.")
        return

    print("\nCleaning up...")
    if stopped:
        print(f"  Containers: {_run(['container', 'prune', '-f'], check=False) or 'done'}")
    if dangling:
        print(f"  Images:     {_run(['image', 'prune', '-f'], check=False) or 'done'}")
    if volumes:
        print(f"  Volumes:    {_run(['volume', 'prune', '-f'], check=False) or 'done'}")
    if not (stopped or dangling or volumes):
        print("Nothing to clean up.")
    else:
        print("\nDone.")


def cmd_df() -> None:
    """Show Docker disk usage summary."""
    print(_run(["system", "df"]))


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "ps":
        cmd_ps(show_all=("--all" in args or "-a" in args))
    elif cmd == "images":
        cmd_images()
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "inspect":
        if len(args) < 2:
            print("Usage: docker_manager.py inspect <name_or_id>")
            sys.exit(1)
        cmd_inspect(args[1])
    elif cmd == "logs":
        if len(args) < 2:
            print("Usage: docker_manager.py logs <name_or_id> [--lines N]")
            sys.exit(1)
        lines = 50
        if "--lines" in args:
            idx = args.index("--lines")
            if idx + 1 < len(args):
                lines = int(args[idx + 1])
        cmd_logs(args[1], lines)
    elif cmd == "clean":
        cmd_clean(execute=("--execute" in args))
    elif cmd == "df":
        cmd_df()
    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
