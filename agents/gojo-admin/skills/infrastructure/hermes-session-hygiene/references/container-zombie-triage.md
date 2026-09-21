# Container Zombie Triage — Worked Example

## Source session

2026-05-20: Sahil asked "investigate our system for zombie processes".
4 zombie processes found; all boot-time artifacts in Postiz and Temporal containers.

## Detection

```bash
# Find zombies
ps aux | awk '$8 ~ /Z/ {print}'
```

Output:
```
root    3046  [node] <defunct>
root    3188  [node] <defunct>
kensei  3362  [auto-setup.sh] <defunct>
root   15267  [node] <defunct>
```

## Map zombies to parents

```bash
for pid in 3046 3188 3362 15267; do
  ppid=$(ps -p $pid -o ppid= | tr -d ' ')
  echo "Zombie $pid -> PPID $ppid ($(ps -p $ppid -o comm= 2>/dev/null | tr -d ' '))"
done
```

Output:
```
Zombie 3046 -> PPID 2154 (node)     # Node.js = container PID 1
Zombie 3188 -> PPID 2154 (node)     # same parent
Zombie 15267 -> PPID 2154 (node)    # same parent
Zombie 3362 -> PPID 2155 (temporal-server)
```

## Check cgroup to identify containers

```bash
cat /proc/2154/cgroup
# 0::/system.slice/docker-24ac8af9fbb41c0d...scope
```

The cgroup contains the Docker container ID. Cross-reference:

```bash
docker inspect 24ac8af9fbb4 --format '{{.Name}} {{.State.Pid}} {{.Config.Cmd}}'
# /postiz 2154 [sh /patch-and-start.sh]
```

So PID 2154 IS the Postiz container's main process.

## Inspect container internals

```bash
CONTAINER_PID=$(docker inspect postiz --format '{{.State.Pid}}')

# Check what PID 1 is inside the container
sudo nsenter -t $CONTAINER_PID -p -- cat /proc/1/cmdline | tr '\0' ' '
# node /usr/local/bin/pnpm run pm2

# List all processes inside the container
sudo nsenter -t $CONTAINER_PID -p -- sh -c '
for d in /proc/[0-9]*/; do
  pid=$(basename $d)
  [ "$pid" -le "1" ] && continue
  stat=$(cat $d/stat 2>/dev/null | cut -d" " -f3)
  ppid=$(cat $d/stat 2>/dev/null | cut -d" " -f4)
  cmd=$(cat $d/cmdline 2>/dev/null | tr "\0" " " | head -c 60)
  [ -n "$stat" ] && echo "$pid $ppid $stat $cmd"
done 2>/dev/null | sort -n
'

# Look for Z state (zombie) inside the container
sudo nsenter -t $CONTAINER_PID -p -- sh -c '
for d in /proc/[0-9]*/; do
  stat=$(cat $d/stat 2>/dev/null | cut -d" " -f3)
  [ "$stat" = "Z" ] || continue
  pid=$(basename $d); ppid=$(cat $d/stat 2>/dev/null | cut -d" " -f4)
  echo "Zombie $pid child of $ppid"
done
'
```

## Root cause pattern

The container's command was `['sh', '/patch-and-start.sh']` which ends with:
```sh
exec pnpm run pm2
```

This makes Node.js (`node /usr/local/bin/pnpm run pm2`) the container's PID 1. Node.js does NOT reap zombie children (no SIGCHLD `wait()` loop). When pnpm spawns intermediate shells (`sh -c pnpm run pm2-run`, `prisma-db-push` subprocesses), those children exit and become zombies because PID 1 never collects their exit status.

## Canonical fix: `init: true`

Docker ships `docker-init` (tini). Enable it per-service:

```yaml
services:
  postiz:
    init: true
    # ... rest of config
```

This tells Docker to run tini as PID 1 instead of the application command. tini spawns your app as a child and reaps all orphaned/zombie processes automatically. After adding `init: true` and restarting, the container will still accumulate zombie children temporarily, but tini reaps them almost immediately.

**Verify Docker has init support:**
```bash
docker info | grep -i init
# Init Binary: docker-init
```

## What NOT to do

- `kill -9 <zombie_pid>` — zombies are already dead; SIGKILL does nothing.
- `pkill hermes` — kills the gateway, not the container zombies.
- Bare container restart without `init: true` — recreates the same zombies immediately.
- SIGCHLD to parent (`kill -CHLD <parent>`) — only works if the parent has a proper SIGCHLD handler; Node.js does not.

## When to accept and move on

A small stable count of zombies (1–5, all boot-age, no growth) is a low-priority aesthetic issue. Only act when:
- The count is growing over time
- PID table exhaustion is approaching (check `/proc/sys/kernel/pid_max`: typically 4 million)
- The parent process is about to be restarted anyway (piggyback the `init: true` fix)
