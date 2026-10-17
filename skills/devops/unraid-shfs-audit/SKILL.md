---
name: unraid-shfs-audit
description: Scan all Docker containers on Unraid for /mnt/user binds that go through shfs/FUSE, and flag candidates for repointing to direct pool paths. Reports current CPU usage of shfs process.
version: 1.0.0
author: scotthawk-maker
license: MIT
metadata:
  hermes:
    tags: [Unraid, Docker, shfs, FUSE, performance, audit]
    related_skills: [unraid-bypass-shfs-for-node-datadirs, unraid-container-repoint, unraid-container-limits, unraid-docker-cleanup]
---

# Goal
Identify Docker containers still binding volumes through `/mnt/user/...` (shfs/FUSE) so they can be repointed to direct pool paths (`/mnt/cache/...` or `/mnt/diskX/...`) for better performance.

# Why
`/mnt/user` routes through Unraid's shfs (FUSE filesystem). Heavy I/O workloads (databases, node datadirs, etc.) on shfs can cause high CPU and sluggish I/O. Bypassing shfs by using direct pool paths eliminates this overhead.

# Procedure

## 1) Scan all running containers for /mnt/user binds
```bash
echo "=== Containers with /mnt/user binds (going through shfs) ==="
docker ps --format '{{.Names}}' | while read c; do
  binds=$(docker inspect -f '{{json .HostConfig.Binds}}' "$c" 2>/dev/null)
  if echo "$binds" | grep -q "/mnt/user/"; then
    echo "  $c: $binds"
  fi
done
```

If nothing prints, all containers are already bypassing shfs.

## 2) Show containers already on direct pool paths
```bash
echo "=== Containers on direct pool paths (bypassing shfs) ==="
docker ps --format '{{.Names}}' | while read c; do
  binds=$(docker inspect -f '{{json .HostConfig.Binds}}' "$c" 2>/dev/null)
  if echo "$binds" | grep -qE "/mnt/(cache|disk[0-9]+)/"; then
    echo "  $c: $binds"
  fi
done
```

## 3) Check shfs CPU usage
```bash
ps -C shfs -o pid,%cpu,%mem,etime,cmd
```

Normal idle: ~0-1%. Sustained >5% suggests containers are hammering shfs.

## 4) Check container sizes on both paths
For each container flagged with `/mnt/user` binds, check if data already exists on the direct pool:
```bash
# Example for a container with /mnt/user/appdata/something
du -sh /mnt/user/appdata/<folder>
du -sh /mnt/cache/appdata/<folder> 2>/dev/null || echo "Not on cache yet"
```

If sizes match, the data is already on cache (user share is just a FUSE view) — no rsync needed, just repoint the bind.

## 5) Also check stopped containers
```bash
echo "=== Stopped containers on /mnt/user ==="
docker ps -a --format '{{.Names}}' | while read c; do
  status=$(docker inspect -f '{{.State.Status}}' "$c")
  if [ "$status" != "running" ]; then
    binds=$(docker inspect -f '{{json .HostConfig.Binds}}' "$c" 2>/dev/null)
    if echo "$binds" | grep -q "/mnt/user/"; then
      echo "  $c (STOPPED): $binds"
    fi
  fi
done
```

# Output interpretation
- **No /mnt/user binds found**: System is fully bypassing shfs. investigate shfs CPU separately.
- **/mnt/user binds found, matching /mnt/cache data exists**: Safe to repoint — just swap bind paths, no data copy needed.
- **/mnt/user binds found, no cache data**: Need to rsync data first, then repoint. Use the unraid-bypass-shfs-for-node-datadirs skill.