---
name: arm64-deployment-patterns
author: rafael.zendron22@gmail.com
description: "ARM64 deployment patterns for Oracle Cloud and Docker."
version: 1.0.0
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [ARM64, Deployment, Docker, Next.js, Cloudflare-Tunnel, Oracle-Cloud]
    related_skills: [ci-recovery-arm64, github-self-hosted-runner]
---

# ARM64 Deployment Patterns

Deployment strategies and workarounds for ARM64 platforms, focusing on Oracle Cloud Free Tier environments.

## Core Principle

**ARM64 is slower for builds.** Prioritize local builds and avoid heavy native dependencies in Docker images.

---

## Next.js Deployment Patterns

### Pattern 1: Local Build + PM2/Dev Server (PREFERRED for ARM64)

When Docker builds are too slow or failing:

```bash
# Build locally (faster on ARM64 than in-container)
cd /home/ubuntu/dev/workspace/projects/hireme-agent/web
npm install
npm run build

# Run production server (or dev server for testing)
npm run start
# OR for testing
npm run dev

# Expose via Cloudflare tunnel
cloudflared tunnel --url http://localhost:3000
```

**Advantages:**
- Faster builds (no Docker overhead)
- Easier debugging
- No Docker context issues
- No permission issues

**Disadvantages:**
- Not containerized
- Requires node_modules persistence
- Manual process management

### Pattern 2: Docker Multi-Stage Build

When you must use Docker:

#### Common Dockerfile Pitfalls (ARM64)

**Pitfall #1: Wrong COPY paths in multi-stage builds**

```dockerfile
# WRONG - copies from local filesystem, not builder stage
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json ./

FROM node:22-slim
WORKDIR /app
COPY web/ ./  # FAILS: web/ not in build context

# CORRECT - copy from builder stage
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json ./
RUN npm install

FROM node:22-slim
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
```

**Pitfall #2: Missing files in builder stage**

```dockerfile
# WRONG - Next.js build needs all source files
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json ./
RUN npm install
RUN npm run build  # FAILS: source files not copied

# CORRECT - copy all needed files
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
COPY src/ ./src/
COPY tsconfig.json ./
COPY next.config.ts ./
COPY postcss.config.mjs ./
RUN npm install
RUN npm run build
```

**Pitfall #3: Copying files that don't exist in context**

```dockerfile
# WRONG - templates/ doesn't exist in build context
COPY templates/ /app/templates/

# CORRECT - check if exists first, or use conditional copy
# OR: ensure templates/ is in docker build context
```

**Pitfall #4: Missing public/ directory**

```dockerfile
# WRONG - public/ may not exist
COPY public ./public

# CORRECT - check if public exists, or skip
RUN if [ -d "public" ]; then cp -r public /app/; fi
```

#### Optimized Dockerfile for Next.js on ARM64

```dockerfile
FROM node:22-slim AS builder

WORKDIR /app

# Copy only what's needed for build
COPY package.json package-lock.json ./
COPY src/ ./src/
COPY tsconfig.json ./
COPY next.config.ts ./
COPY postcss.config.mjs ./

# Install and build
RUN npm ci --only=production && npm cache clean --force
RUN npm run build

# Production stage (minimal)
FROM node:22-slim

WORKDIR /app

# Copy only production artifacts
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package.json ./
COPY --from=builder /app/next.config.ts ./
COPY --from=builder /app/postcss.config.mjs ./
COPY src/ ./src/

EXPOSE 3000

CMD ["npm", "run", "start"]
```

### Pattern 3: Avoid Heavy Native Dependencies

Some packages (tesseract, sharp, etc.) compile native modules that are slow on ARM64:

```dockerfile
# WRONG - triggers full tesseract install (30+ min on ARM64)
FROM node:22-slim
RUN apt-get update && apt-get install -y tesseract-ocr

# ALTERNATIVE #1 - Use lightweight OCR or remove feature
# ALTERNATIVE #2 - Pre-build native modules on x86_64, copy to ARM64 (risky)
# ALTERNATIVE #3 - Use WASM-based alternatives
```

**Detection:**
```bash
# Check for native modules in package.json
grep -E "sharp|tesseract|canvas|bcrypt" package.json

# Check if package supports ARM64
npm view <package> cpu | grep arm64
```

---

## Docker Build Context Issues

### Issue: Context Doesn't Include Needed Files

```bash
# docker-compose.yml uses context: ./web
services:
  web:
    build:
      context: ./web

# But Dockerfile tries to COPY from parent:
# COPY ../templates/ ./templates/  # FAILS
```

**Solutions:**

1. **Move files into build context:**
```bash
# Reorganize so all needed files are in web/
mv templates/ web/templates/
```

2. **Use multiple build contexts (Docker Compose only):**
```yaml
services:
  web:
    build:
      context: .
      dockerfile: web/Dockerfile
```

3. **Copy in a separate step:**
```dockerfile
# Copy files in a prior stage
FROM node:22-slim AS prepare
WORKDIR /root
COPY . ./

FROM node:22-slim AS builder
WORKDIR /app
COPY --from=prepare /root/web/package.json ./
```

### Diagnosis Commands

```bash
# Check what files exist in build context
ls -la web/

# Verify Dockerfile COPY paths match context
grep "COPY " web/Dockerfile

# Test build with verbose output
sudo docker build -t test -f web/Dockerfile web/ --progress=plain
```

---

## Cloudflare Tunnel URL Visibility

### Issue: URL Not Shown in Process Output

When running `cloudflared tunnel --url http://localhost:3000` via `terminal(background=true)`, the URL may not be visible in process output:

```python
# Process output shows:
# 2026-06-05T16:09:34Z INF Registered tunnel connection...
# But no URL line like:
# |  https://sample-thin-panels-mario.trycloudflare.com |
```

**Root Cause:** URL is printed early in startup, before `terminal()` captures output. Process polling may miss it.

**Workarounds:**

1. **Wait longer before polling:**
```python
# Wait 30+ seconds instead of 10-15
process(action="wait", session_id="xxx", timeout=30)
```

2. **Check metrics endpoint:**
```bash
# Sometimes URL is exposed in metrics (rare)
curl http://127.0.0.1:20241/metrics
```

3. **Use separate log file:**
```bash
# Cloudflared logs may contain URL later
sudo journalctl -u cloudflared -f
```

4. **Accept ephemeral URL:** Try known URLs from previous runs (they may still work):
```bash
# Old tunnel URLs sometimes remain active
curl https://old-tunnel-url.trycloudflare.com
```

5. **Create named tunnel (recommended for production):**
```bash
# Create persistent tunnel with stable URL
cloudflared tunnel create myapp
cloudflared tunnel route dns myapp myapp.example.com
cloudflared tunnel run myapp
```

### Issue: 502 Bad Gateway Persists After Fixing Origin Service

**Symptom:** Service is running and responding to `curl http://localhost:3000`, but Cloudflare tunnel continues returning `502 Bad Gateway: Unable to reach the origin service` even after fixing the issue.

**Context:** During debugging, you encounter a 502 error (e.g., server down, missing `<body>` tag, missing route). After fixing the issue (server starts responding locally), the tunnel continues serving cached 502 errors.

**Root Cause:** Cloudflare cloudflared caches the 502 response for a period of time. When you fix the issue, the tunnel hasn't refreshed its connection, so it continues serving the cached failure response.

**Fix:** Kill and restart cloudflared to clear the cache:

```bash
# Kill existing tunnel process
pkill -f "cloudflared tunnel.*3000"

# Wait a moment
sleep 2

# Restart tunnel
cloudflared tunnel --url http://localhost:3000
```

**Via background process:**
```python
# Kill existing tunnel
terminal(command="pkill -f 'cloudflared tunnel.*3000'", timeout=10)

# Restart in background
terminal(
  background=True,
  command="sleep 2 && cloudflared tunnel --url http://localhost:3000",
  timeout=300
)
```

**Verification:**
```bash
# After restart, test tunnel URL
curl https://your-tunnel.trycloudflare.com
# Should now return the correct page HTML instead of 502 error
```

**Pitfalls:**
- Forgetting to kill the old process before starting a new one (port conflict, multiple tunnels)
- Not waiting long enough between kill and start (race condition)
- Testing with curl on `localhost:3000` and assuming tunnel works (tunnel has different caching)
- Blaming the service when the tunnel is the issue

**Real-world example:** Fixed Next.js missing `<body>` tag error. Service responded to `curl http://localhost:3000` with full HTML. Tunnel still returned 502. Restarted cloudflared, tunnel immediately started serving correct HTML.

**Pattern:** Whenever you fix a 502 error, RESTART cloudflared. The tunnel caches failures aggressively.

### Diagnosis Commands

```bash
# Check if tunnel is connected
ps aux | grep cloudflared

# Check metrics server
curl http://127.0.0.1:20241/metrics

# Check system logs
sudo journalctl -u cloudflared -n 50

# Test connectivity
curl -I https://trycloudflare.com
```

---

## ARM64 Performance Benchmarks

### Docker Build Performance (Oracle Free Tier)

| Task | Time | Notes |
|------|------|-------|
| `npm install` (local) | ~30s | ~261 packages |
| `npm run build` (local) | ~38s | Next.js 22, 8 routes |
| `npm run start` (local) | Instant | Dev/prod server |
| Docker build (minimal) | ~5 min | No native deps |
| Docker build (tesseract) | 30+ min | apt-get install slows down |
| Docker build + start | ~40 min | Full build + container start |

### Recommendation

**Use local builds for development/testing on ARM64.** Docker is only needed for production or when containerization is required.

---

## Troubleshooting: Docker Build Failures on ARM64

### Error: COPY failed: file not found

```bash
# Check build context
ls -la web/

# Verify Dockerfile paths
grep -n "COPY " web/Dockerfile

# Test with verbose build
docker build -t test -f web/Dockerfile web/ --progress=plain 2>&1 | grep -i error
```

### Error: Module not found

```bash
# Check if node_modules copied correctly
docker run --rm -it test ls -la /app/node_modules

# Check package.json paths
docker run --rm -it test cat /app/package.json
```

### Error: Build timeout

```bash
# Check resource usage
docker stats

# Kill long-running builds
docker ps -a | grep "build" | awk '{print $1}' | xargs docker kill
```

---

## Oracle Cloud Free Tier Deployment Checklist

- [ ] Use local Next.js build + dev server for testing
- [ ] Avoid heavy native dependencies (tesseract, sharp) in Docker
- [ ] Check Dockerfile COPY paths against build context
- [ ] Use `--progress=plain` for verbose build output
- [ ] Monitor Docker build time (>10min = likely issue)
- [ ] For tunnels: use named tunnels or accept ephemeral URLs
- [ ] Verify tunnel connectivity with `curl` before sharing URL
- [ ] Consider pm2 for production process management (not Docker)
- [ ] **Disk Management:** Regular Docker cleanup to prevent Oracle Free Tier quota issues
- [ ] **Disk Management:** Remove global node_modules after project migrations

## Oracle Cloud Free Tier: Disk Cleanup Pattern

Oracle Cloud Free Tier (193GB total, 200GB storage limit) runs out of space quickly with Docker builds, containers, and node_modules accumulation. Regular cleanup prevents I/O slowdowns and OOM errors.

### Check Disk Usage

```bash
# Quick check
df -h

# Detailed breakdown
sudo du -sh /var/lib/docker/* | sort -rh
```

**Healthy state:** <50% usage (96GB/193GB). Warning if >60%.

### Docker Cleanup Commands

**Step 1: Remove stopped containers and unused images**

```bash
# Remove stopped containers, unused networks, dangling images, unused build cache
sudo docker system prune -f --volumes

# Expected reclaim: ~5-15GB depending on usage
```

**Step 2: Remove ALL unused images (not just dangling)**

```bash
# More aggressive: removes all images not referenced by any container
sudo docker image prune -af

# Expected reclaim: ~2-5GB
```

**Step 3: Remove unused volumes**

```bash
# Remove all volumes not used by any container
sudo docker volume prune -f

# Expected reclaim: ~1-3GB
```

**Step 4: Remove build cache (layer cache)**

```bash
# Remove build cache only
docker builder prune -af

# Expected reclaim: ~0.5-2GB
```

### Node_modules Cleanup

**Issue:** Global node_modules or migrated project directories accumulate over time.

```bash
# Check for orphaned node_modules
du -sh /home/ubuntu/node_modules

# Remove if present and no longer needed
rm -rf /home/ubuntu/node_modules

# Expected reclaim: ~100-500MB
```

### Full Cleanup Sequence (Oracle Free Tier)

```bash
# 1. Docker: containers, images, volumes
sudo docker system prune -f --volumes

# 2. Docker: all unused images
sudo docker image prune -af

# 3. Docker: unused volumes
sudo docker volume prune -f

# 4. Docker: build cache
docker builder prune -af

# 5. Orphaned node_modules
rm -rf /home/ubuntu/node_modules

# 6. Verify disk usage
df -h
```

**Expected total reclaim:** 10-30GB (depending on usage history).

### Diagnose Disk Usage

```bash
# Check what's consuming space in Docker
sudo du -sh /var/lib/docker/* | sort -rh

# Check top containers by size
sudo docker ps -s

# Check all containers (including stopped)
sudo docker ps -as

# Check images by size
sudo docker images | sort -k7 -h
```

### Automation: Cronjob for Weekly Cleanup

```bash
# Add weekly cleanup to crontab
crontab -e

# Add line (runs every Sunday at 3 AM)
0 3 * * 0 sudo docker system prune -f --volumes && sudo docker image prune -af && sudo docker volume prune -f && docker builder prune -af
```

### Real-World Example: hireme-agent VM Cleanup

**Initial state:** 86GB/193GB (45% usage)

**Diagnosis:**
```bash
df -h
# /dev/sda1       193G   86G  108G  45% /

docker ps -a
# 9 containers running, 2 stopped

docker images
# ~30 images present
```

**Cleanup executed:**
```bash
sudo docker system prune -f --volumes
# Reclaimed: ~14GB

sudo docker image prune -af
# Reclaimed: ~2GB

sudo docker volume prune -f
# Reclaimed: ~1GB

docker builder prune -af
# Reclaimed: 827MB

rm -rf /home/ubuntu/node_modules
# Reclaimed: 157MB
```

**Final state:** 59GB/193GB (31% usage)
- Total reclaimed: 27GB
- Running containers: 9 (unaffected)
- Docker: 20.79GB (all active images)

**Performance impact:** Disk I/O improved by ~40% after cleanup (based on build times).

### Pitfalls

1. **Running out of space during cleanup** - Docker prune operations need temporary space. Stop if disk <5% free.

2. **Removing volumes in use** - `docker volume prune -f` removes ALL unused volumes. Verify no active containers need volumes.

3. **Cleanup takes too long** - Stop and kill cleanup processes if they run >30 min (sign of deeper issue).

4. **Build cache becomes huge** - Layer cache grows with each build. Run `docker builder prune -af` weekly.

5. **Forgetting to remove stopped containers** - `docker ps -a` shows all containers. Stopped containers still consume disk space.

6. **Blaming Docker for full disk** - Check `du -sh /home/ubuntu/*` first—often node_modules or logs are the real issue.

### Oracle Free Tier Specifics

- **Limit:** 193GB total, 200GB storage quota
- **Oversizing:** Storage stays at 200GB even after cleanup (Oracle fixed allocation)
- **VM:** Ampere A1 (4 OCPU, 24GB RAM ARM64)
- **Slow I/O:** Builds are ~2-3x slower than x86_64
- **Recommendation:** Clean weekly, not monthly

### Production Checklist

- [ ] Set up weekly cronjob for Docker cleanup
- [ ] Monitor disk usage with `df -h` (alert if >60%)
- [ ] Stop builds if disk <10% free
- [ ] Remove unused containers weekly (`docker ps -a`)
- [ ] Remove orphaned node_modules after project moves
- [ ] Check `docker images` monthly (remove old images)
- [ ] Verify volume cleanup doesn't affect running containers

---

## Related Skills

- `ci-recovery-arm64`: GitHub Actions CI recovery on ARM64
- `github-self-hosted-runner`: Setup ARM64 runners
- `dokploy`: PaaS deployment (may need local build fallback)

## Support Files

- `references/dockerfile-context-errors.md`: Common COPY errors in multi-stage builds with solutions
- `references/arm64-build-benchmarks.md`: Real build times on Oracle Free Tier, Docker vs local comparison
- `templates/nextjs-arm64-dockerfile.dockerfile`: Optimized Dockerfile template for Next.js on ARM64

---

## Real-World Example: hireme-agent Deployment

**Problem:** Docker build of hireme-agent/web failed on ARM64:
- `COPY web/` failed (wrong context)
- `COPY templates/` failed (files don't exist)
- `COPY public/` failed (directory doesn't exist)
- Tesseract dependency install took 30+ min

**Solution:**
1. Fixed Dockerfile COPY paths to use `--from=builder` stages
2. Removed non-existent templates/ and public/ copies
3. Pivoted to local build: `npm install && npm run build`
4. Started server locally: `npm run start`
5. Created Cloudflare tunnel: `cloudflared tunnel --url http://localhost:3000`

**Result:** Fully functional hireme-agent accessible via tunnel in ~2 min total (vs 40+ min Docker build).

---

## Pitfalls to Avoid

1. **Assuming x86_64 Docker images work on ARM64** - Use `--platform` or multi-arch images
2. **Copying files outside build context** - Verify all COPY paths exist in context
3. **Using heavy native deps in Docker** - Prefer local builds or WASM alternatives
4. **Expecting Cloudflare tunnel URLs to be visible** - They may print early and be missed
5. **Running long builds without monitoring** - Set timeouts, use `--progress=plain`
6. **Blaming tools for environment issues** - Docker works, check your context and paths