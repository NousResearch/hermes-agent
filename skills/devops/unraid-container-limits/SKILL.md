---
name: unraid-container-limits
description: Apply and persist Docker container memory/CPU resource limits on Unraid. Re-applies limits after container recreations. Uses docker update for live application and provides persistent re-apply commands.
version: 1.0.0
author: scotthawk-maker
license: MIT
metadata:
  hermes:
    tags: [Unraid, Docker, resource-limits, memory, CPU, performance]
    related_skills: [unraid-container-repoint, unraid-docker-cleanup]
---

# Goal
Apply resource limits to all Docker containers on Unraid to prevent resource hogging. Limits are applied live via `docker update` but must be re-applied after any container recreation.

# Current Limits (ScottHawk system)

These are the tested and working limits for this specific system (Xeon E5-2620 v3, 62GB RAM, 12 threads):

| Container | Memory | Reservation | CPUs |
|-----------|--------|-------------|------|
| PortainerCE | 512m | 256m | 0.5 |
| unomp-redis | 512m | 256m | 0.5 |
| dogecoin-exporter | 256m | 128m | 0.5 |
| Dashboard-Logs | 256m | 128m | 0.5 |
| seahawks-p2pool | 256m | 128m | 0.5 |
| seahawks-proxy | 256m | 128m | 0.5 |
| seahawks-dashboard | 256m | 128m | 0.5 |
| obsidian | 1g | 512m | 1 |
| mcpo | 1g | 512m | 1 |
| public-pool | 1g | 512m | 1 |
| unomp-portal | 1g | 512m | 1 |
| seahawks-monero | 2g | 1g | 2 |
| open-webui | 2g | 1g | 1 |
| bitcoin-node | 4g | 2g | 2 |
| ollama | 8g | 4g | 4 |
| unsloth-studio | 16g | 8g | 4 |

# Procedure

## 1) Apply limits to all containers (live, no restart)

```bash
docker update --memory=512m --memory-reservation=256m --cpus=0.5 PortainerCE
docker update --memory=512m --memory-reservation=256m --cpus=0.5 unomp-redis
docker update --memory=256m --memory-reservation=128m --cpus=0.5 dogecoin-exporter
docker update --memory=256m --memory-reservation=128m --cpus=0.5 Dashboard-Logs
docker update --memory=256m --memory-reservation=128m --cpus=0.5 seahawks-p2pool
docker update --memory=256m --memory-reservation=128m --cpus=0.5 seahawks-proxy
docker update --memory=256m --memory-reservation=128m --cpus=0.5 seahawks-dashboard
docker update --memory=1g --memory-reservation=512m --cpus=1 obsidian
docker update --memory=1g --memory-reservation=512m --cpus=1 mcpo
docker update --memory=1g --memory-reservation=512m --cpus=1 public-pool
docker update --memory=1g --memory-reservation=512m --cpus=1 unomp-portal
docker update --memory=2g --memory-reservation=1g --cpus=2 seahawks-monero
docker update --memory=2g --memory-reservation=1g --cpus=1 open-webui
docker update --memory=4g --memory-reservation=2g --cpus=2 bitcoin-node
docker update --memory=8g --memory-reservation=4g --cpus=4 ollama
docker update --memory=16g --memory-reservation=8g --cpus=4 unsloth-studio
```

## 2) Verify limits applied

```bash
for c in PortainerCE unomp-redis dogecoin-exporter Dashboard-Logs seahawks-p2pool seahawks-proxy seahawks-dashboard obsidian mcpo public-pool unomp-portal seahawks-monero open-webui bitcoin-node ollama unsloth-studio; do
  echo "$(docker inspect -f '{{.Name}} mem={{.HostConfig.Memory}} reservation={{.HostConfig.MemoryReservation}} cpus={{.HostConfig.NanoCpus}}' $c)"
done
```

## 3) Verify all containers still running

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
```

# Pitfalls
- `docker update` limits are NOT persistent. If a container is recreated (docker rm + run, compose up --force-recreate, image update), limits must be re-applied.
- The "Your kernel does not support swap limit capabilities" warning is normal on Unraid — memory limits work, just no swap accounting.
- If a container gets OOM killed, check `docker inspect -f '{{.State.OOMKilled}}' <container>` and consider raising its memory limit.
- For compose-managed containers, limits can be added to docker-compose.yml under `deploy.resources.limits` for persistence.