"""Named execution-runtime catalog for bots/profiles.

First milestone of isolated per-bot runtimes: register local/docker/ssh targets
in ``terminal.runtimes``, assign one to the current profile, test connectivity
without starting a bot, and stop only that profile's containers.

Secrets never belong in the catalog — SSH keys stay in ``.env``. This module
does not provision cloud resources.
"""

from __future__ import annotations

import json
import re
import subprocess
from argparse import Namespace
from typing import Any

from hermes_cli.config import load_config, save_config
from hermes_cli.profiles import get_active_profile_name


RUNTIME_KINDS = ("local", "docker", "ssh")
_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
_SECRET_KEYS = frozenset({
    "ssh_key", "password", "token", "secret", "api_key", "private_key",
})
_CLOUD_KINDS = frozenset({"modal", "daytona", "vercel_sandbox"})


class RuntimeError_(ValueError):
    """User-facing catalog/assignment error."""


def validate_runtime_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not _NAME_RE.match(cleaned):
        raise RuntimeError_(
            f"Invalid runtime name {name!r}. Use a letter followed by "
            "letters, digits, hyphens, or underscores (max 64)."
        )
    return cleaned


def exportable_spec(spec: dict) -> dict:
    """Copy a catalog entry with secret-shaped keys stripped."""
    return {k: v for k, v in spec.items() if k not in _SECRET_KEYS}


def build_runtime_spec(
    kind: str,
    *,
    workspace: str = "",
    image: str = "",
    docker_volumes: list[str] | None = None,
    container_cpu: int | None = None,
    container_memory: int | None = None,
    ssh_host: str = "",
    ssh_user: str = "",
    ssh_port: int | None = None,
) -> dict[str, Any]:
    kind = (kind or "").strip().lower()
    if kind in _CLOUD_KINDS:
        raise RuntimeError_(
            f"{kind} is a cloud sandbox Hermes already supports via "
            f"`hermes config set terminal.backend {kind}`. "
            "`hermes runtime` does not provision or start billable cloud "
            "resources. For a user-managed cloud VM, register kind=ssh "
            "pointing at that host."
        )
    if kind not in RUNTIME_KINDS:
        raise RuntimeError_(
            f"Unknown runtime kind {kind!r}. First milestone kinds: "
            f"{', '.join(RUNTIME_KINDS)}."
        )
    spec: dict[str, Any] = {"kind": kind}
    if workspace.strip():
        spec["workspace"] = workspace.strip()
    if kind == "docker":
        if image.strip():
            spec["image"] = image.strip()
        if docker_volumes:
            spec["docker_volumes"] = list(docker_volumes)
        if container_cpu is not None:
            spec["container_cpu"] = int(container_cpu)
        if container_memory is not None:
            spec["container_memory"] = int(container_memory)
    if kind == "ssh":
        if ssh_host.strip():
            spec["ssh_host"] = ssh_host.strip()
        if ssh_user.strip():
            spec["ssh_user"] = ssh_user.strip()
        if ssh_port is not None:
            spec["ssh_port"] = int(ssh_port)
    return spec


def apply_spec_to_terminal(terminal: dict, spec: dict) -> None:
    """Project a catalog entry onto live ``terminal.*`` keys the backends already read."""
    kind = spec.get("kind") or "local"
    terminal["backend"] = kind
    if spec.get("workspace"):
        terminal["cwd"] = spec["workspace"]
    if kind == "docker":
        if spec.get("image"):
            terminal["docker_image"] = spec["image"]
        if "docker_volumes" in spec:
            terminal["docker_volumes"] = list(spec["docker_volumes"])
        if "container_cpu" in spec:
            terminal["container_cpu"] = spec["container_cpu"]
        if "container_memory" in spec:
            terminal["container_memory"] = spec["container_memory"]
    if kind == "ssh":
        for key in ("ssh_host", "ssh_user", "ssh_port"):
            if key in spec and spec[key] not in (None, ""):
                terminal[key] = spec[key]


def _terminal_section(config: dict | None = None) -> dict:
    cfg = config if config is not None else load_config()
    terminal = cfg.get("terminal")
    if not isinstance(terminal, dict):
        terminal = {}
        cfg["terminal"] = terminal
    runtimes = terminal.get("runtimes")
    if not isinstance(runtimes, dict):
        terminal["runtimes"] = {}
    return terminal


def catalog(config: dict | None = None) -> dict[str, dict]:
    terminal = _terminal_section(config)
    raw = terminal.get("runtimes") or {}
    return {str(name): dict(spec) for name, spec in raw.items() if isinstance(spec, dict)}


def assigned_name(config: dict | None = None) -> str:
    name = str(_terminal_section(config).get("runtime") or "").strip()
    return name


def get_spec(name: str, config: dict | None = None) -> dict:
    spec = catalog(config).get(name)
    if spec is None:
        raise RuntimeError_(f"Runtime {name!r} is not registered. Use `hermes runtime add`.")
    return spec


def persist_catalog(runtimes: dict[str, dict], assigned: str | None) -> None:
    config = load_config()
    terminal = _terminal_section(config)
    terminal["runtimes"] = {
        name: exportable_spec(spec) for name, spec in runtimes.items()
    }
    if assigned is None:
        assigned = str(terminal.get("runtime") or "")
    terminal["runtime"] = assigned
    save_config(config)


def register_runtime(name: str, spec: dict) -> dict:
    name = validate_runtime_name(name)
    cleaned = exportable_spec(spec)
    if cleaned.get("kind") not in RUNTIME_KINDS:
        raise RuntimeError_(f"Runtime {name!r} has invalid kind {cleaned.get('kind')!r}.")
    items = catalog()
    items[name] = cleaned
    persist_catalog(items, assigned_name())
    return cleaned


def remove_runtime(name: str) -> None:
    name = validate_runtime_name(name)
    items = catalog()
    if name not in items:
        raise RuntimeError_(f"Runtime {name!r} is not registered.")
    del items[name]
    current = assigned_name()
    persist_catalog(items, "" if current == name else current)


def assign_runtime(name: str) -> dict:
    name = validate_runtime_name(name)
    spec = get_spec(name)
    config = load_config()
    terminal = _terminal_section(config)
    apply_spec_to_terminal(terminal, spec)
    terminal["runtime"] = name
    save_config(config)
    return spec


def unassign_runtime() -> None:
    config = load_config()
    terminal = _terminal_section(config)
    terminal["runtime"] = ""
    save_config(config)


def _probe_docker() -> tuple[str, str]:
    import shutil

    if not shutil.which("docker"):
        return ("needs_setup", "Docker CLI not found — install Docker Desktop or docker-ce.")
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2,
        )
        if proc.returncode == 0:
            return ("ready", "")
        return ("needs_setup", "Docker daemon not reachable — start Docker and retry.")
    except subprocess.TimeoutExpired:
        return ("needs_setup", "Docker daemon not responding (timed out).")
    except Exception as exc:
        return ("unavailable", f"Docker probe failed: {exc}")


def _probe_ssh(spec: dict) -> tuple[str, str]:
    from hermes_cli.config import get_env_value

    def _val(key: str, env_var: str) -> str:
        value = spec.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
        try:
            return (get_env_value(env_var) or "").strip()
        except Exception:
            return ""

    host = _val("ssh_host", "TERMINAL_SSH_HOST")
    user = _val("ssh_user", "TERMINAL_SSH_USER")
    missing = [k for k, v in (("ssh_host", host), ("ssh_user", user)) if not v]
    if missing:
        return (
            "needs_setup",
            f"Set {', '.join(missing)} on the runtime (or TERMINAL_SSH_* in .env).",
        )
    return ("ready", f"{user}@{host}")


def probe_spec(spec: dict) -> tuple[str, str]:
    """Connectivity probe without starting a bot or creating a sandbox."""
    kind = str(spec.get("kind") or "local").strip().lower()
    if kind == "local":
        return ("ready", "local host")
    if kind == "docker":
        return _probe_docker()
    if kind == "ssh":
        return _probe_ssh(spec)
    return ("unavailable", f"Unsupported runtime kind {kind!r} for this milestone.")


def assigned_runtime_block() -> dict[str, str] | None:
    """Actionable block when the assigned runtime is registered but not ready."""
    try:
        config = load_config()
    except Exception:
        return None
    name = assigned_name(config)
    if not name:
        return None
    try:
        spec = get_spec(name, config)
    except RuntimeError_:
        return {
            "reason": (
                f"Assigned runtime {name!r} is missing from terminal.runtimes. "
                "Re-register it with `hermes runtime add` or `hermes runtime unassign`."
            ),
            "retry_hint": f"hermes runtime list  (then add or unassign {name})",
        }
    status, detail = probe_spec(spec)
    if status == "ready":
        return None
    extra = f" {detail}" if detail else ""
    return {
        "reason": (
            f"Runtime {name!r} ({spec.get('kind')}) is {status}.{extra} "
            "The bot was not started on the Desktop host as a fallback."
        ),
        "retry_hint": f"hermes runtime test {name}",
    }


def _profile_label() -> str:
    from tools.environments.docker import _sanitize_label_value

    return _sanitize_label_value(get_active_profile_name())


def list_profile_containers() -> list[dict[str, str]]:
    """Docker containers labeled for the active profile only."""
    from tools.environments.docker import find_docker

    docker = find_docker()
    if not docker:
        return []
    profile = _profile_label()
    try:
        proc = subprocess.run(
            [
                docker, "ps", "-a",
                "--filter", "label=hermes-agent=1",
                "--filter", f"label=hermes-profile={profile}",
                "--format", "{{.ID}}\t{{.Status}}\t{{.Image}}\t{{.Names}}",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append({
            "id": parts[0],
            "status": parts[1],
            "image": parts[2],
            "name": parts[3] if len(parts) > 3 else "",
            "profile": profile,
        })
    return rows


def _cleanup_vm(task_id: str, *, force_remove: bool = False) -> None:
    from tools.terminal_tool_lifecycle import cleanup_vm

    cleanup_vm(task_id, force_remove=force_remove)


def _find_docker() -> str | None:
    from tools.environments.docker import find_docker

    return find_docker()


def stop_profile_runtime(*, force_remove: bool = True) -> dict[str, Any]:
    """Stop this profile's execution env only — never sibling bots or unlabeled containers."""
    stopped = []
    _cleanup_vm("default", force_remove=force_remove)
    docker = _find_docker()
    for row in list_profile_containers():
        cid = row["id"]
        if docker:
            subprocess.run(
                [docker, "stop", cid], capture_output=True, timeout=20, check=False,
            )
            if force_remove:
                subprocess.run(
                    [docker, "rm", "-f", cid], capture_output=True, timeout=20, check=False,
                )
        stopped.append(cid)
    return {
        "profile": get_active_profile_name(),
        "stopped": stopped,
    }


def runtime_status_payload() -> dict[str, Any]:
    config = load_config()
    terminal = _terminal_section(config)
    name = assigned_name(config)
    spec = catalog(config).get(name) if name else None
    probe = probe_spec(spec) if spec else probe_spec({"kind": terminal.get("backend") or "local"})
    return {
        "profile": get_active_profile_name(),
        "assigned": name or None,
        "backend": terminal.get("backend") or "local",
        "workspace": terminal.get("cwd") or "",
        "image": spec.get("image") if spec else terminal.get("docker_image") or "",
        "limits": {
            "cpu": terminal.get("container_cpu"),
            "memory_mb": terminal.get("container_memory"),
        },
        "probe": {"status": probe[0], "detail": probe[1]},
        "containers": list_profile_containers(),
        "runtimes": {n: exportable_spec(s) for n, s in catalog(config).items()},
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_add(args: Namespace) -> int:
    volumes = [v.strip() for v in (getattr(args, "volume", None) or []) if str(v).strip()]
    spec = build_runtime_spec(
        args.kind,
        workspace=getattr(args, "workspace", "") or "",
        image=getattr(args, "image", "") or "",
        docker_volumes=volumes or None,
        container_cpu=getattr(args, "cpu", None),
        container_memory=getattr(args, "memory", None),
        ssh_host=getattr(args, "ssh_host", "") or "",
        ssh_user=getattr(args, "ssh_user", "") or "",
        ssh_port=getattr(args, "ssh_port", None),
    )
    register_runtime(args.name, spec)
    print(f"Registered runtime {args.name!r} (kind={spec['kind']}).")
    print("Test with `hermes runtime test` before assigning it to a bot.")
    return 0


def _cmd_list(args: Namespace) -> int:
    payload = runtime_status_payload()
    if getattr(args, "json", False):
        _print_json({"assigned": payload["assigned"], "runtimes": payload["runtimes"]})
        return 0
    items = payload["runtimes"]
    if not items:
        print("No named runtimes. Add one with `hermes runtime add NAME --kind docker`.")
        return 0
    assigned = payload["assigned"]
    print(f"Runtimes ({len(items)})  assigned={assigned or '(none)'}")
    for name, spec in sorted(items.items()):
        mark = "*" if name == assigned else " "
        bits = [spec.get("kind", "?")]
        if spec.get("workspace"):
            bits.append(spec["workspace"])
        if spec.get("ssh_host"):
            user = spec.get("ssh_user") or ""
            bits.append(f"{user}@{spec['ssh_host']}" if user else spec["ssh_host"])
        print(f"  {mark} {name}: {', '.join(bits)}")
    return 0


def _cmd_remove(args: Namespace) -> int:
    remove_runtime(args.name)
    print(f"Removed runtime {args.name!r}. Other bots were not stopped.")
    return 0


def _cmd_assign(args: Namespace) -> int:
    spec = assign_runtime(args.name)
    print(f"Assigned runtime {args.name!r} (backend={spec['kind']}) to this profile.")
    return 0


def _cmd_unassign(_args: Namespace) -> int:
    unassign_runtime()
    print("Cleared runtime assignment. Existing terminal.backend was left in place.")
    return 0


def _cmd_test(args: Namespace) -> int:
    name = (getattr(args, "name", None) or assigned_name() or "").strip()
    if not name:
        print("No runtime name given and none assigned. Pass a name or `hermes runtime assign` first.")
        return 1
    spec = get_spec(name)
    status, detail = probe_spec(spec)
    print(f"Runtime {name!r} ({spec.get('kind')}): {status}" + (f" — {detail}" if detail else ""))
    if status != "ready":
        print("Fix the target, then re-run `hermes runtime test`. A bot will not start on an unready runtime.")
        return 1
    return 0


def _cmd_status(args: Namespace) -> int:
    payload = runtime_status_payload()
    if getattr(args, "json", False):
        _print_json(payload)
        return 0
    print(f"Profile:    {payload['profile']}")
    print(f"Assigned:   {payload['assigned'] or '(none)'}")
    print(f"Backend:    {payload['backend']}")
    if payload["workspace"]:
        print(f"Workspace:  {payload['workspace']}")
    if payload["image"]:
        print(f"Image:      {payload['image']}")
    limits = payload["limits"]
    print(f"Limits:     cpu={limits.get('cpu')} memory_mb={limits.get('memory_mb')}")
    probe = payload["probe"]
    print(f"Probe:      {probe['status']}" + (f" — {probe['detail']}" if probe["detail"] else ""))
    containers = payload["containers"]
    if containers:
        print("Containers (this profile only):")
        for row in containers:
            print(f"  {row['id'][:12]}  {row['status']}  {row['image']}")
    return 0


def _cmd_stop(args: Namespace) -> int:
    if not getattr(args, "yes", False):
        print(
            "This stops/removes Hermes-labeled containers for the current profile only. "
            "Re-run with --yes to confirm. Cloud/billable hosts are never created here."
        )
        return 1
    result = stop_profile_runtime(force_remove=True)
    print(f"Stopped runtime for profile {result['profile']!r}.")
    if result["stopped"]:
        print("Removed containers: " + ", ".join(c[:12] for c in result["stopped"]))
    else:
        print("No running containers for this profile.")
    return 0


_ACTIONS = {
    "add": _cmd_add,
    "list": _cmd_list, "ls": _cmd_list,
    "remove": _cmd_remove, "rm": _cmd_remove,
    "assign": _cmd_assign,
    "unassign": _cmd_unassign,
    "test": _cmd_test,
    "status": _cmd_status,
    "stop": _cmd_stop,
}


def runtime_command(args: Namespace) -> int:
    """Entry point for ``hermes runtime``."""
    action = getattr(args, "runtime_action", None)
    handler = _ACTIONS.get(action)
    if handler is None:
        print("Usage: hermes runtime {add|list|remove|assign|unassign|test|status|stop}")
        print("Run 'hermes runtime --help' for details.")
        return 2
    try:
        return int(handler(args) or 0)
    except RuntimeError_ as exc:
        print(f"Error: {exc}")
        return 1
