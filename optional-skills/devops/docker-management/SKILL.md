---
name: docker-management
description: Manage Docker containers, images, volumes, networks, and Compose stacks — lifecycle ops, debugging, cleanup, and Dockerfile optimization.
version: 1.0.0
author: sprmn24
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [docker, containers, devops, infrastructure, compose, images, volumes, networks, debugging]
    category: devops
    requires_toolsets: [terminal]
---

# Docker Management

Manage Docker containers, images, volumes, networks, and Compose stacks using standard Docker CLI commands. No additional dependencies beyond Docker itself.

## When to Use

- Run, stop, restart, remove, or inspect containers
- Build, pull, push, tag, or clean up Docker images
- Work with Docker Compose (multi-service stacks)
- Manage volumes or networks
- Debug a crashing container or analyze logs
- Check Docker disk usage or free up space
- Review or optimize a Dockerfile

## Prerequisites

- Docker Engine installed and running
- User added to the `docker` group (or use `sudo`)
- Docker Compose v2 (included with modern Docker installations)

Quick check:

```bash
docker --version && docker compose version
```

## Quick Reference

| Task | Command |
|------|---------|
| Run container (background) | `docker run -d --name NAME IMAGE` |
| Stop + remove | `docker stop NAME && docker rm NAME` |
| View logs (follow) | `docker logs --tail 50 -f NAME` |
| Shell into container | `docker exec -it NAME /bin/sh` |
| List all containers | `docker ps -a` |
| Build image | `docker build -t TAG .` |
| Compose up | `docker compose up -d` |
| Compose down | `docker compose down` |
| Disk usage | `docker system df` |
| Cleanup dangling | `docker image prune && docker container prune` |

## Procedure

### 1. Identify the domain

Figure out which area the request falls into:

- **Container lifecycle** → run, stop, start, restart, rm, pause/unpause
- **Container interaction** → exec, cp, logs, inspect, stats
- **Image management** → build, pull, push, tag, rmi, save/load
- **Docker Compose** → up, down, ps, logs, exec, build, config
- **Volumes & networks** → create, inspect, rm, prune, connect
- **Troubleshooting** → log analysis, exit codes, resource issues

### 2. Container operations

**Run a new container:**

```bash
# Detached service with port mapping
docker run -d --name web -p 8080:80 nginx

# With environment variables
docker run -d -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=mydb --name db postgres:16

# With persistent data (named volume)
docker run -d -v pgdata:/var/lib/postgresql/data --name db postgres:16

# For development (bind mount source code)
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 --name dev my-app

# Interactive debugging (auto-remove on exit)
docker run -it --rm ubuntu:22.04 /bin/bash

# With resource limits and restart policy
docker run -d --memory=512m --cpus=1.5 --restart=unless-stopped --name app my-app
```

Key flags: `-d` detached, `-it` interactive+tty, `--rm` auto-remove, `-p` port (host:container), `-e` env var, `-v` volume, `--name` name, `--restart` restart policy.

**Manage running containers:**

```bash
docker ps                        # running containers
docker ps -a                     # all (including stopped)
docker stop NAME                 # graceful stop
docker start NAME                # start stopped container
docker restart NAME              # stop + start
docker rm NAME                   # remove stopped container
docker rm -f NAME                # force remove running container
docker container prune           # remove ALL stopped containers
```

**Interact with containers:**

```bash
docker exec -it NAME /bin/sh          # shell access (use /bin/bash if available)
docker exec NAME env                   # view environment variables
docker exec -u root NAME apt update    # run as specific user
docker logs --tail 100 -f NAME         # follow last 100 lines
docker logs --since 2h NAME            # logs from last 2 hours
docker cp NAME:/path/file ./local      # copy file from container
docker cp ./file NAME:/path/           # copy file to container
docker inspect NAME                    # full container details (JSON)
docker stats --no-stream               # resource usage snapshot
docker top NAME                        # running processes
```

### 3. Image management

```bash
# Build
docker build -t my-app:latest .
docker build -t my-app:prod -f Dockerfile.prod .
docker build --no-cache -t my-app .              # clean rebuild
DOCKER_BUILDKIT=1 docker build -t my-app .       # faster with BuildKit

# Pull and push
docker pull node:20-alpine
docker login ghcr.io
docker tag my-app:latest registry/my-app:v1.0
docker push registry/my-app:v1.0

# Inspect
docker images                          # list local images
docker history IMAGE                   # see layers
docker inspect IMAGE                   # full details

# Cleanup
docker image prune                     # remove dangling (untagged) images
docker image prune -a                  # remove ALL unused images (careful!)
docker image prune -a --filter "until=168h"   # unused images older than 7 days
```

### 4. Docker Compose

```bash
# Start/stop
docker compose up -d                   # start all services detached
docker compose up -d --build           # rebuild images before starting
docker compose down                    # stop and remove containers
docker compose down -v                 # also remove volumes (DESTROYS DATA)

# Monitoring
docker compose ps                      # list services
docker compose logs -f api             # follow logs for specific service
docker compose logs --tail 50          # last 50 lines all services

# Interaction
docker compose exec api /bin/sh        # shell into running service
docker compose run --rm api npm test   # one-off command (new container)
docker compose restart api             # restart specific service

# Validation
docker compose config                  # validate and view resolved config
```

**Minimal compose.yml example:**

```yaml
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 5. Volumes and networks

```bash
# Volumes
docker volume ls                       # list volumes
docker volume create mydata            # create named volume
docker volume inspect mydata           # details (mount point, etc.)
docker volume rm mydata                # remove (fails if in use)
docker volume prune                    # remove unused volumes

# Networks
docker network ls                      # list networks
docker network create mynet            # create bridge network
docker network inspect mynet           # details (connected containers)
docker network connect mynet NAME      # attach container to network
docker network disconnect mynet NAME   # detach container
docker network rm mynet                # remove network
docker network prune                   # remove unused networks
```

### 6. Disk usage and cleanup

Always start with a diagnostic before cleaning:

```bash
# Check what's using space
docker system df                       # summary
docker system df -v                    # detailed breakdown

# Targeted cleanup (safe)
docker container prune                 # stopped containers
docker image prune                     # dangling images
docker volume prune                    # unused volumes
docker network prune                   # unused networks

# Aggressive cleanup (confirm with user first!)
docker system prune                    # containers + images + networks
docker system prune -a                 # also unused images
docker system prune -a --volumes       # EVERYTHING — named volumes too
```

**Warning:** Never run `docker system prune -a --volumes` without confirming with the user. This removes named volumes with potentially important data.

## Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Container exits immediately | Main process finished or crashed | Check `docker logs NAME`, try `docker run -it --entrypoint /bin/sh IMAGE` |
| "port is already allocated" | Another process using that port | `docker ps` or `lsof -i :PORT` to find it |
| "no space left on device" | Docker disk full | `docker system df` then targeted prune |
| Can't connect to container | App binds to 127.0.0.1 inside container | App must bind to `0.0.0.0`, check `-p` mapping |
| Permission denied on volume | UID/GID mismatch host vs container | Use `--user $(id -u):$(id -g)` or fix permissions |
| Compose services can't reach each other | Wrong network or service name | Services use service name as hostname, check `docker compose config` |
| Build cache not working | Layer order wrong in Dockerfile | Put rarely-changing layers first (deps before source code) |
| Image too large | No multi-stage build, no .dockerignore | Use multi-stage builds, add `.dockerignore` |

## Verification

After any Docker operation, verify the result:

- **Container started?** → `docker ps` (check status is "Up")
- **Logs clean?** → `docker logs --tail 20 NAME` (no errors)
- **Port accessible?** → `curl -s http://localhost:PORT` or `docker port NAME`
- **Image built?** → `docker images | grep TAG`
- **Compose stack healthy?** → `docker compose ps` (all services "running" or "healthy")
- **Disk freed?** → `docker system df` (compare before/after)

## Dockerfile Optimization Tips

When reviewing or creating a Dockerfile, suggest these improvements:

1. **Multi-stage builds** — separate build environment from runtime to reduce final image size
2. **Layer ordering** — put dependencies before source code so changes don't invalidate cached layers
3. **Combine RUN commands** — fewer layers, smaller image
4. **Use .dockerignore** — exclude `node_modules`, `.git`, `__pycache__`, etc.
5. **Pin base image versions** — `node:20-alpine` not `node:latest`
6. **Run as non-root** — add `USER` instruction for security
7. **Use slim/alpine bases** — `python:3.12-slim` not `python:3.12`
