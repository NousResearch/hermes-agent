#!/usr/bin/env python3
"""kubectl_manager — Inspect, diagnose, and summarize Kubernetes cluster resources.

Usage:
    python kubectl_manager.py summary                          # cluster-wide workload overview
    python kubectl_manager.py pods                             # pod status table (default namespace)
    python kubectl_manager.py pods --namespace production      # specific namespace
    python kubectl_manager.py pods --all-namespaces            # all namespaces
    python kubectl_manager.py nodes                            # node capacity and status
    python kubectl_manager.py events                           # recent warning events
    python kubectl_manager.py events --namespace production
    python kubectl_manager.py logs <pod>                       # tail last 50 lines
    python kubectl_manager.py logs <pod> --namespace NS --lines 200
    python kubectl_manager.py top                              # resource usage (needs metrics-server)
    python kubectl_manager.py diagnose <pod>                   # full pod health report
    python kubectl_manager.py diagnose <pod> --namespace NS

No dependencies beyond Python stdlib and kubectl in PATH.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _run(args: list[str], check: bool = True) -> str:
    """Run a kubectl command and return stdout."""
    try:
        result = subprocess.run(
            ["kubectl"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else ""
        if stderr:
            print(f"Error: {stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: kubectl is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def _json(args: list[str]) -> dict | list:
    """Run kubectl and parse JSON output."""
    raw = _run(args + ["-o", "json"])
    if not raw:
        return {}
    return json.loads(raw)


def _ago(iso: str) -> str:
    """Convert ISO timestamp to 'N ago' string."""
    try:
        ts = iso[:19].replace("Z", "")
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        s = int((datetime.now(timezone.utc) - dt).total_seconds())
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m"
        if s < 86400:
            return f"{s // 3600}h"
        return f"{s // 86400}d"
    except Exception:
        return iso[:10] if iso else "?"


def _ns_args(namespace: str | None, all_ns: bool) -> list[str]:
    """Build namespace flags for kubectl."""
    if all_ns:
        return ["--all-namespaces"]
    if namespace:
        return ["-n", namespace]
    return []


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_summary() -> None:
    """Cluster-wide workload overview."""
    print("=== Cluster Summary ===\n")

    # Nodes
    nodes_data = _json(["get", "nodes"])
    nodes = nodes_data.get("items", [])
    ready_nodes = sum(
        1 for n in nodes
        if any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in n.get("status", {}).get("conditions", [])
        )
    )
    print(f"Nodes:       {ready_nodes}/{len(nodes)} Ready")

    # Pods across all namespaces
    pods_data = _json(["get", "pods", "--all-namespaces"])
    pods = pods_data.get("items", [])
    running = sum(1 for p in pods if p.get("status", {}).get("phase") == "Running")
    pending = sum(1 for p in pods if p.get("status", {}).get("phase") == "Pending")
    failed  = sum(1 for p in pods if p.get("status", {}).get("phase") == "Failed")
    print(f"Pods:        {running} Running, {pending} Pending, {failed} Failed  ({len(pods)} total)")

    # Deployments
    dep_data = _json(["get", "deployments", "--all-namespaces"])
    deps = dep_data.get("items", [])
    dep_ready = sum(
        1 for d in deps
        if d.get("status", {}).get("readyReplicas", 0) == d.get("spec", {}).get("replicas", 1)
    )
    print(f"Deployments: {dep_ready}/{len(deps)} Ready")

    # Services
    svc_data = _json(["get", "services", "--all-namespaces"])
    svcs = svc_data.get("items", [])
    print(f"Services:    {len(svcs)}")

    # Namespaces
    ns_data = _json(["get", "namespaces"])
    namespaces = [n.get("metadata", {}).get("name", "") for n in ns_data.get("items", [])]
    print(f"Namespaces:  {', '.join(namespaces)}")

    # Recent warnings
    ev_raw = _run([
        "get", "events", "--all-namespaces",
        "--field-selector", "type=Warning",
        "--sort-by=.lastTimestamp",
    ], check=False)
    warning_lines = [l for l in ev_raw.splitlines() if l.strip() and "NAMESPACE" not in l]
    if warning_lines:
        print(f"\nWarning Events: {len(warning_lines)} recent warning(s)")
        for line in warning_lines[-3:]:
            print(f"  {line[:110]}")
    else:
        print("\nWarning Events: none")


def cmd_pods(namespace: str | None = None, all_ns: bool = False) -> None:
    """Pod status table."""
    args = ["get", "pods"] + _ns_args(namespace, all_ns)
    data = _json(args)
    pods = data.get("items", [])

    if not pods:
        ns_label = "all namespaces" if all_ns else (namespace or "default")
        print(f"No pods found in {ns_label}.")
        return

    show_ns = all_ns or namespace is None and len(
        set(p.get("metadata", {}).get("namespace", "") for p in pods)
    ) > 1

    if show_ns:
        print(f"{'NAMESPACE':<20} {'NAME':<44} {'STATUS':<12} {'READY':<8} {'RESTARTS':<10} {'AGE'}")
        print("-" * 110)
    else:
        print(f"{'NAME':<44} {'STATUS':<12} {'READY':<8} {'RESTARTS':<10} {'AGE'}")
        print("-" * 90)

    for p in pods:
        meta   = p.get("metadata", {})
        status = p.get("status", {})
        spec   = p.get("spec", {})

        name   = meta.get("name", "")[:43]
        ns     = meta.get("namespace", "")[:19]
        phase  = status.get("phase", "Unknown")[:11]
        age    = _ago(meta.get("creationTimestamp", ""))

        containers  = spec.get("containers", [])
        cs_statuses = status.get("containerStatuses", [])
        ready_count = sum(1 for c in cs_statuses if c.get("ready"))
        total_count = len(containers)
        restarts    = sum(c.get("restartCount", 0) for c in cs_statuses)

        ready_str    = f"{ready_count}/{total_count}"
        restarts_str = str(restarts)

        if show_ns:
            print(f"{ns:<20} {name:<44} {phase:<12} {ready_str:<8} {restarts_str:<10} {age}")
        else:
            print(f"{name:<44} {phase:<12} {ready_str:<8} {restarts_str:<10} {age}")

    print(f"\nTotal: {len(pods)} pod(s)")


def cmd_nodes() -> None:
    """Node capacity and readiness."""
    data = _json(["get", "nodes"])
    nodes = data.get("items", [])

    if not nodes:
        print("No nodes found.")
        return

    print(f"{'NAME':<40} {'STATUS':<10} {'ROLES':<20} {'VERSION':<16} {'AGE':<8} {'CPU':<8} {'MEMORY'}")
    print("-" * 120)

    for n in nodes:
        meta   = n.get("metadata", {})
        status = n.get("status", {})
        labels = meta.get("labels", {})

        name    = meta.get("name", "")[:39]
        age     = _ago(meta.get("creationTimestamp", ""))
        version = status.get("nodeInfo", {}).get("kubeletVersion", "")[:15]

        # Node roles from labels
        roles = [
            k.split("/")[-1]
            for k in labels
            if k.startswith("node-role.kubernetes.io/")
        ]
        role_str = ",".join(roles) if roles else "worker"

        # Ready condition
        conditions = status.get("conditions", [])
        ready_cond = next((c for c in conditions if c.get("type") == "Ready"), {})
        is_ready   = ready_cond.get("status") == "True"
        ready_str  = "Ready" if is_ready else "NotReady"

        # Capacity
        capacity = status.get("capacity", {})
        cpu_cap  = capacity.get("cpu", "?")
        mem_cap  = capacity.get("memory", "?")

        # Convert memory from Ki to Gi
        if mem_cap.endswith("Ki"):
            mem_gi = int(mem_cap[:-2]) / (1024 * 1024)
            mem_str = f"{mem_gi:.1f}Gi"
        else:
            mem_str = mem_cap

        print(f"{name:<40} {ready_str:<10} {role_str[:19]:<20} {version:<16} {age:<8} {cpu_cap:<8} {mem_str}")

    print(f"\nTotal: {len(nodes)} node(s)")


def cmd_events(namespace: str | None = None) -> None:
    """Recent warning events."""
    args = ["get", "events", "--field-selector", "type=Warning", "--sort-by=.lastTimestamp"]
    if namespace:
        args += ["-n", namespace]
    else:
        args += ["--all-namespaces"]

    raw = _run(args, check=False)
    lines = [l for l in raw.splitlines() if l.strip()]

    if not lines or all("NAMESPACE" in l or "LAST SEEN" in l for l in lines):
        print("No warning events found.")
        return

    # Print header + last 30 warnings
    data_lines = [l for l in lines if "NAMESPACE" not in l and "LAST SEEN" not in l]
    print(f"Recent Warning Events ({len(data_lines)} total, showing last 30):\n")
    for line in data_lines[-30:]:
        print(line[:120])


def cmd_logs(pod: str, namespace: str | None = None, lines: int = 50) -> None:
    """Tail pod logs (captures both stdout and stderr)."""
    args = ["logs", "--tail", str(lines), pod]
    if namespace:
        args += ["-n", namespace]

    try:
        result = subprocess.run(
            ["kubectl"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = "\n".join(filter(None, [result.stdout, result.stderr])).strip()
        if output:
            print(output)
        else:
            print(f"No logs for pod '{pod}'.")
    except FileNotFoundError:
        print("Error: kubectl is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)


def cmd_top() -> None:
    """Resource usage — requires metrics-server."""
    print("=== Node Resource Usage ===\n")
    node_out = _run(["top", "nodes"], check=False)
    if "error" in node_out.lower() or not node_out:
        print("metrics-server not available. Install with:")
        print("  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml")
    else:
        print(node_out)

    print("\n=== Pod Resource Usage (all namespaces) ===\n")
    pod_out = _run(["top", "pods", "--all-namespaces", "--sort-by=memory"], check=False)
    if pod_out and "error" not in pod_out.lower():
        lines = pod_out.splitlines()
        print("\n".join(lines[:30]))
        if len(lines) > 30:
            print(f"... and {len(lines) - 30} more pods")
    else:
        print("No pod metrics available.")


def cmd_diagnose(pod: str, namespace: str | None = None) -> None:
    """Full pod health report."""
    ns_args = ["-n", namespace] if namespace else []
    ns_label = namespace or "default"

    print(f"=== Diagnosing pod '{pod}' in namespace '{ns_label}' ===\n")

    # 1. Basic info
    data = _json(["get", "pod", pod] + ns_args)
    if not data or not data.get("metadata"):
        print(f"Pod '{pod}' not found in namespace '{ns_label}'.")
        return

    meta   = data.get("metadata", {})
    status = data.get("status", {})
    spec   = data.get("spec", {})

    phase = status.get("phase", "Unknown")
    print(f"Phase:    {phase}")
    print(f"Node:     {spec.get('nodeName', 'unscheduled')}")
    print(f"Age:      {_ago(meta.get('creationTimestamp', ''))}")
    print(f"IP:       {status.get('podIP', 'none')}")

    # 2. Container statuses
    containers    = spec.get("containers", [])
    cs_statuses   = status.get("containerStatuses", [])
    init_cs       = status.get("initContainerStatuses", [])

    print(f"\nContainers ({len(containers)}):")
    for cs in cs_statuses:
        name     = cs.get("name", "")
        ready    = "Ready" if cs.get("ready") else "NOT READY"
        restarts = cs.get("restartCount", 0)
        image    = cs.get("image", "")

        state     = cs.get("state", {})
        state_key = list(state.keys())[0] if state else "unknown"
        state_det = state.get(state_key, {})
        reason    = state_det.get("reason", "")
        msg       = state_det.get("message", "")

        print(f"  [{ready}] {name} — {image}")
        print(f"           State: {state_key}{' (' + reason + ')' if reason else ''}")
        if msg:
            print(f"           Message: {msg[:120]}")
        print(f"           Restarts: {restarts}")

        # Last state (crash info)
        last = cs.get("lastState", {})
        if last:
            last_key = list(last.keys())[0]
            last_det = last.get(last_key, {})
            last_reason = last_det.get("reason", "")
            last_exit   = last_det.get("exitCode", "")
            last_fin    = _ago(last_det.get("finishedAt", ""))
            print(f"           Last state: {last_key} (exit {last_exit}, reason: {last_reason}, {last_fin} ago)")

    if init_cs:
        print(f"\nInit Containers ({len(init_cs)}):")
        for cs in init_cs:
            ready = "Done" if cs.get("ready") else "NOT DONE"
            print(f"  [{ready}] {cs.get('name','')} — restarts: {cs.get('restartCount',0)}")

    # 3. Conditions
    conditions = status.get("conditions", [])
    if conditions:
        print("\nConditions:")
        for c in conditions:
            ctype  = c.get("type", "")
            cstat  = c.get("status", "")
            creason = c.get("reason", "")
            cmsg   = c.get("message", "")
            flag   = "✓" if cstat == "True" else "✗"
            detail = f" — {creason}: {cmsg[:80]}" if (creason or cmsg) and cstat != "True" else ""
            print(f"  {flag} {ctype}{detail}")

    # 4. Resource requests vs limits
    print("\nResource Requests/Limits:")
    for c in containers:
        res = c.get("resources", {})
        req = res.get("requests", {})
        lim = res.get("limits", {})
        print(f"  {c.get('name','')}:")
        print(f"    CPU:    requests={req.get('cpu','none')}  limits={lim.get('cpu','none')}")
        print(f"    Memory: requests={req.get('memory','none')}  limits={lim.get('memory','none')}")

    # 5. Recent events for this pod
    events_raw = _run([
        "get", "events",
        "--field-selector", f"involvedObject.name={pod}",
        "--sort-by=.lastTimestamp",
    ] + ns_args, check=False)

    event_lines = [
        l for l in events_raw.splitlines()
        if l.strip() and "LAST SEEN" not in l
    ]
    if event_lines:
        print(f"\nRecent Events ({len(event_lines)}):")
        for line in event_lines[-10:]:
            print(f"  {line[:115]}")
    else:
        print("\nRecent Events: none")

    # 6. Tail logs
    print("\nRecent Logs (last 20 lines):")
    cmd_logs(pod, namespace, lines=20)


# ─── Entry point ──────────────────────────────────────────────────────────────

def _get_flag(args: list[str], flag: str, default=None):
    """Extract a named flag value from args list."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    rest = args[1:]

    namespace = _get_flag(rest, "--namespace") or _get_flag(rest, "-n")
    all_ns    = "--all-namespaces" in rest or "-A" in rest

    if cmd == "summary":
        cmd_summary()

    elif cmd == "pods":
        cmd_pods(namespace=namespace, all_ns=all_ns)

    elif cmd == "nodes":
        cmd_nodes()

    elif cmd == "events":
        cmd_events(namespace=namespace)

    elif cmd == "logs":
        if not rest or rest[0].startswith("-"):
            print("Usage: kubectl_manager.py logs <pod> [--namespace NS] [--lines N]")
            sys.exit(1)
        pod   = rest[0]
        lines = int(_get_flag(rest, "--lines") or 50)
        cmd_logs(pod, namespace=namespace, lines=lines)

    elif cmd == "top":
        cmd_top()

    elif cmd == "diagnose":
        if not rest or rest[0].startswith("-"):
            print("Usage: kubectl_manager.py diagnose <pod> [--namespace NS]")
            sys.exit(1)
        cmd_diagnose(rest[0], namespace=namespace)

    else:
        print(f"Unknown command: '{cmd}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
