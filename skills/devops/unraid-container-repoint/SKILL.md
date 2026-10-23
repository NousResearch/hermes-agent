---
name: unraid-container-repoint
description: Stop, remove, and recreate a Docker container on Unraid with updated volume bind mounts. Handles both direct docker-run and Portainer compose-managed containers. Preserves all original config (env, ports, network, restart policy, image).
version: 1.0.0
author: scotthawk-maker
license: MIT
metadata:
  hermes:
    tags: [Unraid, Docker, volumes, bind-mount, repoint]
    related_skills: [unraid-shfs-audit, unraid-bypass-shfs-for-node-datadirs, unraid-container-limits]
---

# Goal
Safely repoint a container's volume binds from `/mnt/user/...` (shfs/FUSE) to direct pool paths (`/mnt/cache/...`, `/mnt/diskX/...`) on Unraid, preserving all other configuration.

# Determine management style first

```bash
docker inspect -f '{{.Name}} compose={{index .Config.Labels "com.docker.compose.project"}} compose_config={{index .Config.Labels "com.docker.compose.project.config_files"}}' <container>
```

- If `compose=` is empty: **direct docker run** — use Procedure A
- If `compose=` has a value: **compose-managed** — use Procedure B

---

## Procedure A: Direct docker-run container

### 1) Capture full container config
```bash
docker inspect <container> --format '{{json .Config.Env}}'
docker inspect <container> --format '{{json .HostConfig.RestartPolicy}}'
docker inspect <container> --format '{{json .HostConfig.PortBindings}}'
docker inspect <container> --format '{{json .HostConfig.Binds}}'
docker inspect <container> --format '{{.Config.Image}}'
docker inspect <container> --format '{{.HostConfig.NetworkMode}}'
docker inspect <container> --format '{{json .Config.Cmd}}'
```

### 2) Stop and remove the container
```bash
docker stop -t 30 <container> && docker rm <container>
```

### 3) Recreate with updated binds
Replace `/mnt/user/appdata/...` with `/mnt/cache/appdata/...` (or `/mnt/diskX/appdata/...`) in the `-v` flags. Keep all other flags identical.

```bash
docker run -d \
  --name <container> \
  --network <network> \
  -e "KEY=VALUE" \
  -p <hostport>:<containerport> \
  -v /mnt/cache/appdata/<path>:/container/path:rw \
  <image>
```

### 4) Verify
```bash
docker inspect -f '{{.Name}} Binds={{json .HostConfig.Binds}} Status={{.State.Status}}' <container>
```

---

## Procedure B: Compose-managed (Portainer stack)

### 1) Locate the compose file
```bash
# From compose_config label, e.g. /data/compose/80/docker-compose.yml
# On Unraid, actual file is usually at:
find /mnt/cache/appdata/portainer/compose -name docker-compose.yml
```

### 2) Backup the compose file
```bash
cp -a /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml \
      /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml.bak.$(date +%Y%m%d-%H%M%S)
```

### 3) Edit volume paths in the compose file
```bash
sed -i 's|/mnt/user/appdata/|/mnt/cache/appdata/|g' \
  /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml
```

### 4) Stop, remove, and recreate with compose
```bash
docker stop -t 30 <container1> <container2>
docker rm <container1> <container2>
docker compose -p <project> -f /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml up -d --force-recreate
```

### 5) Verify
```bash
for c in <container1> <container2>; do
  docker inspect -f '{{.Name}} Binds={{json .HostConfig.Binds}} Status={{.State.Status}}' $c
done
```

### 6) Cleanup (optional, after verifying)
```bash
rm /mnt/cache/appdata/portainer/compose/<id>/docker-compose.yml.bak.*
```

---

# Re-applying resource limits after recreation
`docker update` resource limits are NOT persistent across container recreations. After running Procedure A step 3 or Procedure B step 4, re-apply the container's limits using the unraid-container-limits skill.

Quick reference for `docker run` recreation — add these flags:
```
--memory=<limit> --memory-reservation=<soft_limit> --cpus=<cores>
```

# Pitfalls
- **Don't skip the config capture** — if you rm a container without recording env vars, ports, network, etc., you'll have to reconstruct from memory.
- **Compose `version` attribute is obsolete** in newer docker compose — warning is harmless, can be removed.
- **Check data exists at the new path first** — if `/mnt/cache/appdata/<folder>` doesn't exist or sizes don't match, you may need to rsync data before repointing.
- **For containers running the current AI session** (e.g., ollama): restarting will kill the session. Repoint these LAST and warn the user they'll need to reconnect.
- **Resource limits reset on recreate** — always re-apply `docker update` limits after any container recreation. See unraid-container-limits skill.