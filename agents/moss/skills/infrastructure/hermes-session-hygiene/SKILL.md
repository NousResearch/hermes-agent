---
name: hermes-session-hygiene
description: Detect, clean up, and prevent idle/duplicate Hermes CLI sessions and their orphaned MCP child processes. Covers the full lifecycle — scan, SIGTERM, verify, and prevention via bashrc alias.
adoption_status: provisional
---

# Hermes Session Hygiene — Detection, Cleanup, and Prevention

## When this skill applies

When you need to:
- Diagnose memory pressure from duplicate Hermes sessions or orphaned MCP children
- Clean up idle SSH sessions that are wasting ~400-700MB per idle Hermes CLI + its MCP triad
- Apply the "alias-to-refuse" prevention pattern to stop bare `hermes` invocations
- Verify the gateway is unaffected after cleanup

## Scope guard: not every zombie is a Hermes session

If the user says "zombie processes", first identify the process class before applying this cleanup playbook. Linux `Z` / `<defunct>` processes inside Docker app containers are different from idle Hermes CLI sessions:

- Hermes session hygiene applies when the parent process is a Hermes CLI or its MCP child stack.
- Container-local zombies usually need app/container operational triage, not `kill hermes` or MCP cleanup.
- Zombies are already dead; direct `kill` does nothing. You clear them by making the parent reap them.
- **The canonical fix for Docker Compose containers:** add `init: true` to the service definition. Docker injects tini as PID 1, which reaps all orphaned children automatically. A bare container restart without `init: true` will immediately recreate the same zombies.
- Before restarting anything, check whether a bigger source-of-truth/config drift exists. If runtime files are dirty, secret-bearing, or not committed to the intended repo, fix the handoff plan before doing a cosmetic restart.

Use this skill for Hermes-owned process trees. For service-specific zombies, load the service's operations skill as well (for example `postiz-self-hosting` for Postiz/Temporal containers).

### Zombie triage: mapping PIDs to containers

When investigating container-local zombies, follow this workflow to identify root cause:

```bash
# 1. Find zombie processes
ps aux | awk '$8 ~ /Z/ {print}'

# 2. Find each zombie's parent PID
ps -p <ZOMBIE_PID> -o ppid= | tr -d ' '

# 3. Check if the parent is inside a container (look at cgroup)
cat /proc/<PARENT_PID>/cgroup | head -1
# Output like: 0::/system.slice/docker-<CONTAINER_ID>.scope

# 4. Identify the container
docker inspect --format '{{.Name}} {{.Config.Cmd}}' <CONTAINER_ID>

# 5. Inside the container, PID 1 is the main process — check what it is
docker inspect <CONTAINER_ID> --format '{{.State.Pid}}'  # host PID of container's PID 1
# If PID 1 is node, python, or any non-init process, zombie reaping is broken

# 6. Use nsenter to inspect inside the container's PID namespace
sudo nsenter -t <HOST_PID> -p -- sh -c 'for d in /proc/[0-9]*/; do
  pid=$(basename $d); [ "$pid" = "1" ] 2>/dev/null || continue
  cat $d/cmdline 2>/dev/null | tr "\0" " "
done'

# 7. Fix: add init: true to the container's docker-compose service
```

**Root cause pattern:** When a Docker container's PID 1 is Node.js, Python, or a shell script (not tini/dumb-init), child processes that exit before the main process are never reaped. Standard `docker run` starts the command directly as PID 1 without init behaviour. Adding `init: true` tells Docker to run tini as PID 1, which spawns your command and reaps all orphans.

### References

See `references/container-zombie-triage.md` for a full worked example from the 2026-05-20 Postiz/Temporal zombie investigation.

## Root Cause

Upstream Hermes spawns MCP servers per-process. Every `hermes` or `hermes chat` invocation reads `~/.hermes/config.yaml`, finds the MCP servers (workspace-mcp, ms-365-mcp, nanobanana-mcp), and forks each as a child. There is no shared MCP pool, no IPC, and no on-demand loading.

The gateway daemon (`hermes gateway run --replace`) has its own `gateway.pid` and `gateway.lock` — these prevent gateway self-duplication, but do NOT prevent CLI sessions from spawning independent stacks.

On Sahil's KENSEI setup, multiple `hermes_cli.main gateway run --replace` processes can be normal: the default gateway plus specialist profile gateways run with different `HERMES_HOME` values under `~/.hermes/profiles/<profile>`. Do not classify these as stale duplicates until you inspect each PID's environment.

## Step 1: Detection (Live Scan)

```bash
# All Hermes Python processes (parent processes, sorted by RSS)
ps -eo pid,ppid,rss,stat,etime,tty,cmd | grep -E "hermes" | grep python | grep -v grep | sort -rn -k3

# All MCP children with their parent PIDs
ps -eo pid,ppid,rss,stat,etime,tty,cmd | grep -E "workspace-mcp|ms-365-mcp|nanobanana" | grep -v grep

# Gateway PID for cross-reference
cat /home/kensei/.hermes/gateway.pid 2>/dev/null || echo "no gateway"

# Distinguish default gateway from specialist profile gateways before cleanup
for pid in $(ps -eo pid=,cmd= | awk '/python -m hermes_cli.main gateway run --replace/ && !/awk/ {print $1}'); do
  printf '%s ' "$pid"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^HERMES_HOME=' || echo 'HERMES_HOME=?'
done
```

**How to read the output:**
- The default gateway process is PID 1's child (PPID=1), daemonised, no TTY, with `HERMES_HOME=/home/kensei/.hermes`.
- Specialist profile gateways are also PID 1's children and use `HERMES_HOME=/home/kensei/.hermes/profiles/<profile>`; keep them unless the specific profile is duplicated or stale.
- CLI session processes are children of bash/SSH (PPID=bash), attached to a TTY.
- Each Hermes Python process may have MCP children: workspace-mcp (via uvx), ms-365-mcp-server (Node), nanobanana-mcp (Node).
- Gateway children are correctly parented to their gateway PID. CLI children are parented to the CLI PID.

**Idle check:**

```bash
w
```

If all SSH sessions show 5h+ idle, the CLIs are doing nothing. Their MCP triads are also doing nothing.

## Step 2: Memory Assessment

For each idle CLI stack, sum RSS of:
- The Python Hermes process (~450MB)
- ms-365-mcp-server (~120MB Node)
- nanobanana-mcp (~70MB Node)
- workspace-mcp (~125MB Python via uvx)

Multiply by the number of idle generations. Typical: 2 idle CLIs = ~1.2-1.6GB reclaimable.

System memory check:

```bash
free -h
```

## Step 3: Safe Cleanup

**Never mass-pkill `hermes`** — that kills the live gateway too. Always target by specific PID.

```bash
# 1. Pre-flight: identity check
grep -q hermes /proc/<CLI_PID>/comm || { echo "PID identity mismatch, abort"; exit 1; }

# 2. Check for mid-write (atomic writes in progress)
ls /home/kensei/.hermes/memories/*.tmp 2>/dev/null && { echo "write in progress, wait"; exit 1; }

# 3. Graceful SIGTERM
kill -TERM <CLI_PID> <CLI_PID_2>

# 4. Bounded poll (up to 25s — covers upstream's 18s shutdown budget + slack)
for i in {1..25}; do
  kill -0 <CLI_PID> 2>/dev/null || kill -0 <CLI_PID_2> 2>/dev/null || break
  sleep 1
done

# 5. Report survivors (do NOT blanket-pkill)
for pid in <CLI_PID> <CLI_PID_2> <MCP_CHILD_1> <MCP_CHILD_2> ...; do
  kill -0 $pid 2>/dev/null && echo "survivor: $pid"
done

# 6. If survivors exist, escalate with manual confirmation
# kill -KILL <survivor-pids>
```

**Why this works:** Upstream Hermes has a proper shutdown chain:
- SIGTERM handler in `cli.py:~12797` — calls `agent.interrupt()`, waits `HERMES_SIGTERM_GRACE` (1.5s), raises KeyboardInterrupt
- `atexit _run_cleanup` in `cli.py:~12746`
- `shutdown_mcp_servers` in `mcp_tool.py:~3286` — per-server async shutdown, timeout 15s, then `_kill_orphaned_mcp_children` with 2s SIGKILL fallback
- Total budget: ~18.5s

## Step 4: Verify Cleanup

```bash
# No MCP children except gateway's
ps -eo pid,ppid,rss,cmd | grep -E "workspace-mcp|ms-365-mcp|nanobanana" | grep -v grep

# Gateway still owns port 8000
ss -tlnp | grep :8000

# Memory freed
free -h

# Cron still ticking
ls -la /home/kensei/.hermes/cron/.tick.lock
```

## Step 5: Prevention (alias-to-refuse)

Add to `~/.bashrc`:

```bash
hermes() {
  if [ $# -eq 0 ]; then
    echo "Bare 'hermes' forks a private MCP stack (~500MB). Use 'hermes chat --force' if you really want a new session, or talk to the gateway over Telegram." >&2
    return 1
  fi
  command hermes "$@"
}
```

This catches the muscle-memory pattern of just typing `hermes` (pressing Enter without a subcommand). It's the highest-impact low-effort prevention. The gateway handles persistent Telegram interaction, so you rarely need a CLI session at all.

**Do NOT use `pkill hermes`** — takes down the gateway. Always target by PID.

## References

- `references/mcp-lifecycle-analysis-2026-05-11.md` — live PID audit from May 2026 with exact command invocations
- `references/alias-to-refuse-pattern.md` — the bashrc alias implementation detail
