---
name: docker
description: Build, run, and manage Docker containers and images. Covers the full container lifecycle — images, containers, volumes, networks, Compose multi-service stacks, image optimization, and production hardening. Use when running isolated workloads, packaging applications, or spinning up local services without polluting the host system.
version: 1.0.0
author: dogiladeveloper
license: MIT
metadata:
  hermes:
    tags: [Docker, Containers, DevOps, Docker-Compose, Images, Volumes, Networking, Production, Isolation]
    related_skills: [github-pr-workflow, axolotl, vllm]
    homepage: https://github.com/dogiladeveloper
---

# Docker

Build, ship, and run applications in isolated containers.

## Quick Reference

| Action | Command |
|--------|---------|
| Run a container | `docker run -it ubuntu bash` |
| List running containers | `docker ps` |
| List all containers | `docker ps -a` |
| Stop a container | `docker stop <id>` |
| Remove a container | `docker rm <id>` |
| List images | `docker images` |
| Pull an image | `docker pull python:3.12-slim` |
| Build an image | `docker build -t myapp:latest .` |
| Remove an image | `docker rmi myapp:latest` |
| View logs | `docker logs -f <id>` |
| Exec into container | `docker exec -it <id> bash` |

## Helper Script

This skill includes `scripts/docker_manager.py` — a zero-dependency CLI tool
for inspecting and managing Docker resources.

```bash
python scripts/docker_manager.py ps                    # list running containers
python scripts/docker_manager.py ps --all              # all containers (including stopped)
python scripts/docker_manager.py images                # list images with sizes
python scripts/docker_manager.py stats                 # one-shot CPU/memory snapshot
python scripts/docker_manager.py inspect mycontainer   # pretty-print container details
python scripts/docker_manager.py logs mycontainer      # tail last 50 lines of logs
python scripts/docker_manager.py logs mycontainer --lines 200
python scripts/docker_manager.py clean                 # dry-run cleanup report
python scripts/docker_manager.py clean --execute       # remove stopped containers, dangling images, unused volumes
python scripts/docker_manager.py df                    # disk usage summary
```

---

## 1. Running Containers

### Basic run

```bash
# Run interactively and remove when done
docker run --rm -it python:3.12-slim python3

# Run in background (detached)
docker run -d --name myredis redis:7-alpine

# Run with port mapping (host:container)
docker run -d -p 8080:80 nginx:alpine

# Run with environment variables
docker run -d \
  -e DATABASE_URL=postgres://user:pass@db:5432/app \
  -e SECRET_KEY=mysecret \
  myapp:latest
```

### Resource limits

```bash
# Limit memory and CPU
docker run -d \
  --memory="512m" \
  --cpus="1.0" \
  --name myapp \
  myapp:latest

# Check resource usage live
docker stats
docker stats --no-stream  # one-shot snapshot
```

### Useful run flags

| Flag | Purpose |
|------|---------|
| `--rm` | Auto-remove container on exit |
| `-d` | Detached (background) |
| `-it` | Interactive + TTY |
| `--name NAME` | Give container a name |
| `-p HOST:CONT` | Port mapping |
| `-v HOST:CONT` | Volume mount |
| `--env-file .env` | Load env vars from file |
| `--network NAME` | Attach to network |
| `--restart unless-stopped` | Auto-restart policy |

---

## 2. Building Images

### Minimal Dockerfile

```dockerfile
FROM python:3.12-slim

# Install dependencies first (cached layer)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last (changes most often)
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build commands

```bash
# Build with a tag
docker build -t myapp:latest .

# Build with build args
docker build \
  --build-arg APP_VERSION=1.2.3 \
  --build-arg ENV=production \
  -t myapp:1.2.3 .

# Build and push in one shot (Docker Buildx)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myuser/myapp:latest \
  --push .

# Rebuild without cache
docker build --no-cache -t myapp:latest .
```

### Multi-stage build (smaller images)

```dockerfile
# Stage 1: build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: runtime (only copies compiled output)
FROM node:20-alpine AS runtime
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### .dockerignore

Always create `.dockerignore` to keep images lean:

```
.git
.gitignore
node_modules
__pycache__
*.pyc
.env
.env.*
*.log
dist
.DS_Store
README.md
tests/
docs/
```

---

## 3. Volumes & Persistent Storage

```bash
# Named volume (managed by Docker, survives container removal)
docker volume create mydata
docker run -d -v mydata:/data postgres:16-alpine

# Bind mount (host path ↔ container path)
docker run -d -v $(pwd)/data:/data myapp:latest

# Read-only bind mount
docker run -d -v $(pwd)/config:/config:ro myapp:latest

# List volumes
docker volume ls

# Inspect a volume (find where data lives on host)
docker volume inspect mydata

# Remove unused volumes
docker volume prune
```

---

## 4. Networking

```bash
# Create a custom bridge network
docker network create mynet

# Run containers on the same network (they can reach each other by name)
docker run -d --name db --network mynet postgres:16-alpine
docker run -d --name app --network mynet -e DATABASE_HOST=db myapp:latest

# List networks
docker network ls

# Inspect a network (see which containers are on it)
docker network inspect mynet

# Connect an already-running container to a network
docker network connect mynet mycontainer

# Remove unused networks
docker network prune
```

---

## 5. Docker Compose

Best for multi-service local development.

### Example `compose.yaml`

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/app
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    volumes:
      - .:/app          # live code reload in dev
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: app
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d app"]
      interval: 5s
      timeout: 5s
      retries: 5

  cache:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

### Compose commands

```bash
# Start all services (build if needed)
docker compose up -d

# Build and start (force rebuild)
docker compose up -d --build

# View logs (all services)
docker compose logs -f

# View logs (one service)
docker compose logs -f app

# Stop all services
docker compose down

# Stop and delete volumes (clean slate)
docker compose down -v

# Scale a service
docker compose up -d --scale worker=3

# Run one-off command in a service
docker compose run --rm app python manage.py migrate

# Restart a single service
docker compose restart app

# Check status
docker compose ps
```

---

## 6. Inspecting & Debugging

```bash
# View real-time logs
docker logs -f mycontainer

# Last 50 lines of logs
docker logs --tail 50 mycontainer

# Get a shell inside a running container
docker exec -it mycontainer bash

# Or sh for Alpine-based images
docker exec -it mycontainer sh

# Inspect everything about a container (JSON)
docker inspect mycontainer

# Get just the IP address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mycontainer

# Copy files in/out of a container
docker cp mycontainer:/app/logs/error.log ./error.log
docker cp ./config.json mycontainer:/app/config.json

# View filesystem changes in a container
docker diff mycontainer

# Check container resource usage
docker stats mycontainer --no-stream
```

---

## 7. Image Management

```bash
# List images with sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Remove dangling images (untagged)
docker image prune

# Remove ALL unused images
docker image prune -a

# Tag an image for a registry
docker tag myapp:latest myuser/myapp:v1.2.3
docker tag myapp:latest myuser/myapp:latest

# Push to Docker Hub
docker login
docker push myuser/myapp:v1.2.3
docker push myuser/myapp:latest

# Save image to a tar file (for offline transfer)
docker save myapp:latest | gzip > myapp.tar.gz

# Load image from tar
docker load < myapp.tar.gz

# Show image layers and sizes
docker history myapp:latest
```

---

## 8. Clean Up

```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove EVERYTHING unused at once (containers, images, networks, build cache)
docker system prune -a

# Check disk usage
docker system df
docker system df -v  # verbose (shows per-image/volume breakdown)
```

---

## 9. Production Hardening

Security best practices for production containers:

```dockerfile
FROM python:3.12-slim

# Install only what's needed, clean up in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

# Healthcheck so orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with security flags:

```bash
docker run -d \
  --read-only \                          # read-only root filesystem
  --cap-drop ALL \                       # drop all Linux capabilities
  --security-opt no-new-privileges \     # prevent privilege escalation
  --pids-limit 256 \                     # limit process count
  --memory="512m" \                      # memory cap
  --cpus="1.0" \                         # CPU cap
  --tmpfs /tmp:rw,noexec,nosuid \        # writable /tmp without exec
  --restart unless-stopped \
  --name myapp \
  myapp:latest
```

---

## 10. Common Patterns

### Wait for a dependency before starting

```bash
# In your entrypoint.sh
until pg_isready -h "$DB_HOST" -p "${DB_PORT:-5432}"; do
  echo "Waiting for database..."
  sleep 2
done
exec "$@"
```

### Pass secrets safely (not via ENV)

```bash
# Use Docker secrets (Swarm) or mount files
docker run -d \
  -v /run/secrets/db_password:/run/secrets/db_password:ro \
  myapp:latest

# In app code: read from file
import os
with open('/run/secrets/db_password') as f:
    db_password = f.read().strip()
```

### Health check for Compose dependency ordering

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy   # waits until db healthcheck passes
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5
```

### Run a quick one-off command without installing anything

```bash
# Run Python without installing Python locally
docker run --rm python:3.12-slim python3 -c "import sys; print(sys.version)"

# Run a database migration
docker run --rm \
  -e DATABASE_URL=postgres://user:pass@localhost:5432/app \
  --network host \
  myapp:latest python manage.py migrate

# Convert a file using ImageMagick without installing it
docker run --rm -v $(pwd):/work grande/imagemagick \
  convert /work/photo.png /work/photo.jpg
```

---

## Contributing

Skill authored by **dogiladeveloper**.

- GitHub: [github.com/dogiladeveloper](https://github.com/dogiladeveloper)
- Discord: `dogiladeveloper`
- Twitter/X: [@dogiladeveloper](https://twitter.com/dogiladeveloper)

Issues, improvements, and pull requests are welcome!
