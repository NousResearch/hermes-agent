---
name: docker-compose-ops
description: "Manage and debug Docker Compose multi-container services."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [docker, compose, containers, devops, services, troubleshooting]
    category: devops
    related_skills: [systematic-debugging]
---

# Docker Compose Operations

Operate, inspect, and troubleshoot multi-container application stacks using modern Docker Compose (Compose V2).

## Core Principles & Safety Rules

1. **Volume Safety Law**: NEVER run `docker compose down -v` or prune volumes without explicit user confirmation. Destroying named or anonymous database volumes is irreversible.
2. **Pre-flight Validation First**: ALWAYS validate configuration syntax and environment variable substitution with `docker compose config` before executing `up`.
3. **Use Modern Compose V2 Syntax**: Use `docker compose` (space), not deprecated `docker-compose` (hyphen).
4. **Targeted Operations**: When debugging a multi-service stack, isolate operations to the specific failing service rather than churning the entire fleet.

---

## Workflow Checklist

| Phase | Goal | Key Command |
|---|---|---|
| 1. Pre-flight | Validate compose file & env vars | `docker compose config` |
| 2. Startup | Start services in background | `docker compose up -d [--build]` |
| 3. Status | Check state & health checks | `docker compose ps -a` |
| 4. Diagnostics | Tail logs & inspect failures | `docker compose logs -f --tail=100 <service>` |
| 5. In-Container | Run migrations or diagnostics | `docker compose exec <service> <command>` |
| 6. Teardown | Stop services preserving volumes | `docker compose down` |

---

## Phase 1: Pre-Flight & Configuration Verification

Before starting or modifying services, verify the Compose configuration and daemon state:

### 1. Verify Docker Daemon Connectivity
```bash
docker info > /dev/null 2>&1 || docker version
```
If the daemon is not reachable, advise the user to start Docker Desktop or the system Docker service.

### 2. Validate Compose YAML & Environment Variables
```bash
docker compose config --quiet
```
To view the fully resolved configuration with interpolated environment variables:
```bash
docker compose config
```

### 3. Check for Port Collisions on the Host
Before starting containers that bind host ports (e.g. `5432:5432`, `6379:6379`, `8080:80`):
- **Linux/macOS**: `lsof -i :<PORT> || ss -tulpn | grep :<PORT>`
- **Windows (PowerShell)**: `Get-NetTCPConnection -LocalPort <PORT> -ErrorAction SilentlyContinue`

---

## Phase 2: Lifecycle Management

### Start Services (Detached Mode)
```bash
# Start all services
docker compose up -d

# Start with forced rebuild of local Dockerfile images
docker compose up -d --build

# Start only specific services and their dependencies
docker compose up -d <service_name>
```

### Stop or Restart Services
```bash
# Stop containers without removing network or container definitions
docker compose stop [<service_name>]

# Gracefully restart a specific service (useful after config updates)
docker compose restart <service_name>

# Stop and remove containers and networks (persists volumes)
docker compose down
```

---

## Phase 3: Diagnostic Playbooks

When a service fails to start, crashes repeatedly (`CrashLoopBackOff`), or fails health checks:

### 1. Inspect Service Status & Exit Codes
```bash
docker compose ps -a
```
Look for:
- `Exit 1` / `Exit 137` (OOM killed) / `Exit 127` (command not found).
- Health status: `(healthy)`, `(unhealthy)`, or `(health: starting)`.

### 2. Tailing Logs
```bash
# Tail last 100 lines and follow logs for a failing service
docker compose logs --tail=100 -f <service_name>

# View timestamped logs across all services
docker compose logs --timestamps --tail=50
```

### 3. Inspect Container Health Details
For services with a configured `healthcheck`:
```bash
docker inspect --format='{{json .State.Health}}' $(docker compose ps -q <service_name>)
```

### 4. Execute Commands Inside Running Containers
```bash
# Run database migrations or CLI tools
docker compose exec <service_name> <command>

# Test network connectivity between services
docker compose exec <service_a> ping -c 3 <service_b>
docker compose exec <service_a> curl -I http://<service_b>:<port>
```

---

## Phase 4: Common Troubleshooting Scenarios

### Scenario A: Host Port Already Occupied
**Symptom**: `Bind for 0.0.0.0:<PORT> failed: port is already allocated`
**Remediation**:
1. Identify the occupying host process.
2. If another container holds the port, stop it: `docker ps --filter "publish=<PORT>"`.
3. If a local native service occupies the port, override the host port mapping in `.env` or `docker-compose.override.yml`:
   ```yaml
   services:
     db:
       ports:
         - "5433:5432" # Remap host port to 5433 while keeping container port 5432
   ```

### Scenario B: Database Not Ready on App Startup
**Symptom**: App container immediately crashes with `connection refused` to database.
**Remediation**:
Configure health-based startup dependencies in `docker-compose.yml`:
```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: app
      POSTGRES_PASSWORD: secretpassword
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    depends_on:
      db:
        condition: service_healthy
```

### Scenario C: Environment Variables Missing in Container
**Symptom**: Service reports missing credentials or database URLs.
**Remediation**:
1. Run `docker compose config` to check what Compose interpolated.
2. Verify the presence of `.env` in the same directory as `docker-compose.yml`.
3. Pass environment files explicitly if using custom profiles:
   ```bash
   docker compose --env-file .env.local up -d
   ```

---

## Phase 5: Safe Cleanup & Maintenance

```bash
# Clean stopped containers and unused networks without touching volumes
docker compose down --remove-orphans

# Reclaim space from dangling build caches safely
docker builder prune -f
```
