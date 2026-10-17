---
name: unraid-bypass-shfs-for-node-datadirs
description: Move/point heavy blockchain node datadirs off /mnt/user (shfs/FUSE) to the direct pool path (/mnt/cache or /mnt/<pool>) with safe stop, verify, switch, recreate steps; includes Portainer compose and docker run cases.
version: 1.0.0
author: scotthawk-maker
license: MIT
metadata:
  hermes:
    tags: [Unraid, Docker, shfs, FUSE, performance, blockchain, node]
    related_skills: [unraid-shfs-audit, unraid-container-repoint, unraid-container-limits, unraid-docker-cleanup]
---

# Goal
Reduce `shfs` overhead by ensuring node datadirs are accessed via the **direct pool mount** (e.g. `/mnt/cache/appdata/...` or `/mnt/<pool>/appdata/...`) instead of `/mnt/user/appdata/...`.

# Why
`/mnt/user` goes through Unraid `shfs` (FUSE). Large DB workloads with many files can cause high `shfs` CPU and sluggish I/O.

# Procedure

## 0) Pre-flight checks
Run:
```bash
cat /proc/mdstat | sed -n '1,160p'
pgrep -x mover && pgrep -ax mover || echo mover_not_running
find /mnt -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort

df -hT /mnt/cache /mnt/cache/appdata /mnt/user /mnt/user/appdata
```
- If mover is running, wait.
- If array/pool degraded and you can’t risk a long copy, consider postponing.

## 1) Identify source folders and pool destinations
Typical sources:
- `/mnt/user/appdata/<container-folder>`
Destinations:
- `/mnt/cache/appdata/<container-folder>` (if pool name is `cache`)
- or `/mnt/<poolname>/appdata/<container-folder>`

Check sizes on both paths:
```bash
du -sh /mnt/user/appdata/<folder>
du -sh /mnt/cache/appdata/<folder> || echo missing
```

### Key optimization (learned): if the destination already exists with the same size
On Unraid, `/mnt/user/appdata/...` may already be a user-share view of the same pool path. If sizes match and the destination exists, you can often **skip rsync** and just repoint binds.

## 2) Determine container management style
### A) Compose-managed (often Portainer)
Find compose files:
```bash
find /mnt/cache/appdata/portainer/compose -maxdepth 2 -type f -name docker-compose.yml -print
```
Then inspect labels to locate project + compose path:
```bash
docker inspect -f '{{.Name}} labels={{json .Config.Labels}}' <container> | head -c 5000
```
Common labels:
- `com.docker.compose.project`
- `com.docker.compose.project.config_files`

### B) Direct docker run
Check binds:
```bash
docker inspect -f '{{.Name}} Binds={{json .HostConfig.Binds}}' <container>
```
Also note network + published ports:
```bash
docker inspect -f '{{.Name}} Network={{.HostConfig.NetworkMode}} Ports={{json .NetworkSettings.Ports}} Cmd={{json .Config.Cmd}}' <container>
```

## 3) Stop containers (avoid DB corruption)
```bash
docker stop -t 120 <bitcoin> <doge> <tari>
```

## 4) Update mounts to direct pool paths

### 4A) Compose case (Portainer stacks)
1) **Back up** compose file:
```bash
cp -a /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml \
      /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml.bak.$(date +%Y%m%d-%H%M%S)
```
2) Edit volume lines, e.g.:
- From: `/mnt/user/appdata/bitcoin-node:/home/bitcoin/.bitcoin`
- To:   `/mnt/cache/appdata/bitcoin-node:/home/bitcoin/.bitcoin`

3) Redeploy with the correct project name:
```bash
docker compose -p <project> -f /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml up -d --force-recreate
```

### 4B) Direct docker run case (example: Tari)
1) Record existing settings (network, ports, command, image).
2) Remove old container:
```bash
docker rm -f <container>
```
3) Recreate with updated bind:
```bash
docker run -d \
  --name <container> \
  --network <network> \
  -p <hostport>:<containerport> \
  -v /mnt/cache/appdata/<path>:/var/tari/node \
  <image> <cmd...>
```

## 5) Verify
### Verify binds
```bash
for c in bitcoin-node dogecoin-node seahawks-tari; do
  docker inspect -f '{{.Name}} Binds={{json .HostConfig.Binds}}' $c
done
```
### Verify running
```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | egrep '^(NAMES|bitcoin-node|dogecoin-node|seahawks-tari)'
```
### Check shfs CPU and node CPU
```bash
ps -C shfs -o pid,%cpu,%mem,etime,cmd
ps -eo pid,comm,%cpu,%mem,etime,args --sort=-%cpu | head -n 15
```

# Rollback
- Compose: restore the backed-up compose file and rerun `docker compose up -d --force-recreate`.
- Direct run: recreate container with original `/mnt/user/...` bind.

# Notes / Pitfalls
- `pgrep -af mover` can match your own command line; use `pgrep -x mover` to check mover reliably.
- `docker compose` may need longer than 180s if it pulls images; rerun is idempotent.
- Even after bypassing shfs for node DBs, the system can still feel slow due to CPU-heavy syncing; consider CPU caps (`--cpus`) before memory caps.
- **Session-killing risk**: If your AI assistant (e.g. ollama) runs through one of the containers you're repointing, restarting that container will kill your active session. Always repoint the model-serving container LAST, or check first — it may already be on a direct path (e.g. ollama uses `/mnt/cache/ollama` rather than `/mnt/user/appdata/ollama`).
- **Non-standard paths**: Not all containers use `/mnt/user/appdata/<container-name>`. Some (like ollama) use entirely different paths like `/mnt/cache/ollama`. Always `docker inspect` the actual Binds before assuming the pattern.
- **Compose file location**: Docker labels (`com.docker.compose.project.config_files`) may show `/data/compose/80/docker-compose.yml` but the real filesystem path is `/mnt/cache/appdata/portainer/compose/80/docker-compose.yml`. Use `find` to locate the actual file if the label path doesn't resolve.
- **Same-data shortcut confirmed**: When `du -sh` shows identical sizes on `/mnt/user/appdata/<folder>` and `/mnt/cache/appdata/<folder>`, they are the same data (FUSE view of the pool) — you can skip rsync entirely and just repoint the bind mount.
