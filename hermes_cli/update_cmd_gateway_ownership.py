"""Identify gateways safe to stop without interfering with systemd services."""

from pathlib import Path


def protected_gateway_identity(pid: int) -> dict:
    """Capture recovery evidence while a protected gateway is still observable."""
    from gateway.status import get_process_start_time
    from hermes_cli.update_fleet_scope import gateway_pid_home
    from contextlib import suppress
    from uuid import uuid4
    home = birth = None
    with suppress(Exception):
        home = gateway_pid_home(pid)
    with suppress(Exception):
        birth = get_process_start_time(pid)
    identity = {"kind": "gateway", "profile": None, "pid": pid, "protected_unmapped": True,
                "incident_id": uuid4().hex, "service_cgroup": _systemd_service_cgroup_for_pid(pid),
                "home": home, "birth": birth}
    # Repeated catch-up for the same live incarnation must tag its receipt with
    # the incident already kept in the host record, not invent a second debt.
    with suppress(Exception):
        from hermes_cli.update_host_obligation import read_host_obligation
        record = read_host_obligation() or {}
        for prior in (record.get("inventory") or {}).get("runtimes", []):
            if (isinstance(prior, dict) and prior.get("protected_unmapped") and prior.get("pid") == pid
                    and birth is not None and prior.get("birth") == birth and prior.get("home") == home
                    and prior.get("incident_id")):
                return dict(prior)
    return identity


def _classify_systemd_cgroup(text: str, *, initial_namespace: bool) -> bool | None:
    """Classify cgroup text given whether the reader sees the initial namespace."""
    systemd_paths: list[str] = []
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, cgroup_path = parts
        controller_names = controllers.split(",") if controllers else []
        if not (
            (hierarchy == "0" and not controllers)
            or "name=systemd" in controller_names
        ):
            continue
        if not cgroup_path.startswith("/") or cgroup_path == "/":
            return None
        systemd_paths.append(cgroup_path)

    if not systemd_paths:
        return None
    if any(
        component.endswith(".service")
        for cgroup_path in systemd_paths
        for component in cgroup_path.split("/")
        if component
    ):
        return True
    return False if initial_namespace else None


def _systemd_service_membership_for_pid(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> bool | None:
    """Return service membership; unknown or namespace-hidden ownership is None.

    The kernel assigns this inode to the initial cgroup namespace (Linux UAPI
    ``CGROUP_NS_INIT_INO``); /proc/<pid>/cgroup is relative to the reader's
    namespace. Failure to prove the initial namespace cannot prove a negative.
    """
    try:
        text = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    try:
        initial_namespace = (proc_root / "self" / "ns" / "cgroup").stat().st_ino == 0xEFFFFFFB
    except OSError:
        initial_namespace = False
    return _classify_systemd_cgroup(text, initial_namespace=initial_namespace)


def _systemd_service_cgroup_for_pid(pid: int, *, proc_root: Path = Path("/proc")) -> str | None:
    """Return a visible systemd service's cgroup path, if uniquely identified."""
    try:
        text = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return None
    services = set()
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if not ((hierarchy == "0" and not controllers) or "name=systemd" in controllers.split(",")):
            continue
        if not path.startswith("/") or path == "/":
            return None
        components = path.strip("/").split("/")
        for index, component in enumerate(components):
            if component.endswith(".service"):
                services.add("/" + "/".join(components[:index + 1]))
                break
    return next(iter(services)) if len(services) == 1 else None


def _select_update_manual_gateway_pids(
    candidate_pids,
    *,
    known_service_pids,
    systemd_supported: bool,
) -> list[int]:
    """Exclude known or visible services and PIDs with unknown Linux ownership."""
    from hermes_cli.update_cmd import _systemd_service_membership_for_pid
    from hermes_platform.host.facts import os_family

    service_pids = set(known_service_pids)
    manual_pids: list[int] = []
    for pid in dict.fromkeys(candidate_pids):
        if pid in service_pids:
            continue
        membership = _systemd_service_membership_for_pid(pid)
        if membership is True:
            continue
        if membership is None and (
            systemd_supported or os_family().startswith("linux")
        ):
            continue
        manual_pids.append(pid)
    return manual_pids
