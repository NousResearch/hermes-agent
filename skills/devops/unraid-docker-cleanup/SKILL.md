---
name: unraid-docker-cleanup
description: Audit and clean orphaned Docker resources on Unraid — stopped containers, unused images, orphaned appdata dirs, dangling volumes, and build cache. Frees disk space on the cache pool.
version: 1.0.0
author: scotthawk-maker
license: MIT
metadata:
  hermes:
    tags: [Unraid, Docker, cleanup, disk-space, orphaned]
    related_skills: [unraid-shfs-audit, unraid-container-repoint]
---

# Goal
Free disk space on Unraid by removing orphaned Docker resources: stopped containers, unused images, orphaned appdata, dangling volumes, and build cache.

# Procedure

## 1) Inventory stopped containers
```bash
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | grep -v "Up "
```
Before removing, check their binds:
```bash
for c in <stopped_containers>; do
  docker inspect -f '{{.Name}} Binds={{json .HostConfig.Binds}}' $c
done
```
If the binds reference appdata dirs that should be kept (like blockchain data), ask the user before deleting.

## 2) Remove stopped containers (after confirming)
```bash
docker rm <stopped_container_names>
```

## 3) Identify orphaned appdata dirs
```bash
for d in /mnt/cache/appdata/*/; do
  name=$(basename "$d")
  found=$(docker ps -a --format '{{.Names}}' | grep -i "$name" 2>/dev/null)
  if [ -z "$found" ]; then
    size=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    echo "  ORPHANED: $name ($size)"
  fi
done
```
ALWAYS cross-reference with container binds — some appdata dirs have different names than their containers. Delete only dirs that have NO container reference.

## 4) Remove orphaned appdata (after confirming)
```bash
rm -rf /mnt/cache/appdata/<orphaned_dir>
```

## 5) Prune unused Docker images
```bash
docker image prune -a -f
```
Removes images not used by any running container.

## 6) Prune dangling volumes
```bash
# List first
docker volume ls -q | while read v; do
  used=$(docker ps -a --filter "volume=$v" --format '{{.Names}}' 2>/dev/null)
  if [ -z "$used" ]; then
    echo "  ORPHAN: $v"
  fi
done

# Remove after confirming
docker volume prune -f
```

## 7) Prune build cache
```bash
docker builder prune -f
```

## 8) Verify
```bash
docker system df
du -sh /mnt/cache/appdata/
```

# Pitfalls
- Blockchain data dirs (bitcoin-node, dogecoin-node, monero, etc.) can be 100+ GB — always confirm before deleting
- Some appdata dirs are referenced by container names that don't match the folder name (e.g., seahawks-mining contains tari, monero, p2pool subdirs)
- `docker volume prune` will NOT remove volumes referenced by stopped containers — remove containers first
- Named volumes like `open-webui` are used by active containers — check before pruning