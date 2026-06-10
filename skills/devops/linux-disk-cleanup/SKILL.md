---
name: linux-disk-cleanup
description: "Audit and clean disk usage on Linux/WSL2 systems. Systematic methodology to find space hogs and safely reclaim disk space."
version: 1.0.0
tags: [linux, disk, cleanup, sysadmin, wsl2]
---

# Linux Disk Cleanup

Systematic approach to audit disk usage and identify reclaimable space on Linux/WSL2 systems.

## Step 1: Overview

```bash
df -h / /mnt/c 2>/dev/null
```

Shows total/used/free per mount point. Identify which filesystem is full.

## Step 2: Home Directory Breakdown

**PITFALL**: `du -sh ~/*` can timeout (30s+) on dirs with many small files (node_modules, .cache). Use targeted `for` loop instead:
```bash
for d in ~/.cache ~/.npm ~/.local ~/.ollama ~/.vscode-server ~/.hermes ~/.node-gyp ~/portfolio_dev ~/estacio-prep; do du -sh "$d" 2>/dev/null; done
```

For hidden dirs specifically:
```bash
du -sh ~/.cache ~/.npm ~/.local ~/.ollama ~/.vscode-server ~/.hermes ~/.node-gyp 2>/dev/null | sort -rh
```

Pitfall: `du` on dirs with many small files (node_modules, .cache) can timeout. Use specific paths instead of broad wildcards.

## Step 2b: System-Level Hogs (often missed)

```bash
# Ollama models — can be 7+ GB even if ~/.ollama shows 12KB (models live in /usr/local/lib)
du -sh /usr/local/lib/ollama 2>/dev/null
du -sh /usr/local/lib/*/ 2>/dev/null | sort -rh | head -5

# OpenCode session data — can be 2+ GB
du -sh ~/.local/share/opencode 2>/dev/null

# pnpm store — separate from cache
pnpm store path 2>/dev/null && pnpm store prune --dry-run 2>/dev/null
```

**Common hidden hogs:**
| Location | Typical Size | Safe to remove? | Notes |
|----------|-------------|----------------|-------|
| `/usr/local/lib/ollama/` | 2-10 GB | Yes | Ollama models. Remove with `sudo rm -rf /usr/local/lib/ollama` if no GPU/LLM use |
| `~/.local/share/opencode/` | 1-3 GB | Yes | OpenCode CLI session data, regenerates |
| `~/.local/share/pnpm/` | 1-3 GB | Prune only | `pnpm store prune` removes unused packages |
| `~/.local/share/uv/` | 50-100 MB | Yes | `uv cache clean` handles the cache, this is the data dir |

## Step 3: System Directories

```bash
du -sh /var/cache /var/log /var/tmp /tmp /snap 2>/dev/null | sort -rh
sudo du -sh /var/cache/apt /var/log/journal /var/lib/snapd /usr/local/lib 2>/dev/null | sort -rh
```

## Step 4: Largest Installed Packages

```bash
dpkg-query -W --showformat='${Installed-Size;10}\t${Package}\n' | sort -rn | head -20
```

Good for finding apps the user doesn't use (thunderbird, libreoffice, firefox on WSL2, etc).

## Step 5: Docker (Biggest Hidden Hog)

```bash
docker system df
docker ps --format "table {{.Names}}\t{{.Status}}"  # Check what's running
```

Docker images, build cache, and volumes can consume 10s of GB invisibly.

**PITFALL**: `docker system prune -a --volumes -f` only reclaims images not used by ANY container (even stopped ones). If all images have running containers, you may get only ~150MB back despite `docker system df` showing 13GB+ "reclaimable". The "reclaimable" count is misleading — it includes images referenced by stopped containers. To actually reclaim, stop/remove unused containers first.

**PITFALL**: `docker image prune -a -f --filter "until=48h"` returns 0B even when 13GB+ is "reclaimable". Time-based filters only match image creation time, not last-used time. Remove stopped containers explicitly (`docker rm <name>`) then run `docker image prune -a -f` WITHOUT filter for real results.

**ORPHANED VOLUMES** — When `docker system prune -a --volumes` returns 0B (all images tied to running containers), check for orphaned volumes manually:
```bash
docker system df -v 2>&1 | grep -A 50 "Local Volumes"
```
Volumes with `LINKS 0` belong to removed/stopped containers and are safe to delete. Example:
```bash
# Identify orphaned (LINKS=0) volumes
docker system df -v | awk '/Local Volumes/,0' | awk '$2 == 0 {print $1}'
# Remove them explicitly
docker volume rm <volume_name1> <volume_name2> ...
```
This recovered ~1.6GB in one session (SonarQube data/logs, old project postgres data, build caches from removed containers). These are NOT cleaned by `docker system prune` because prune only removes volumes not referenced by ANY container definition — but containers that were `docker rm`'d leave their volumes behind as orphans.

**PITFALL**: Docker Desktop on Linux eats 1.5-2GB RAM for SonarQube or similar heavy containers. Consider `docker stop <container>` when RAM is constrained, not just disk.

## Step 6: Cache Breakdown

```bash
du -sh ~/.cache/*/ 2>/dev/null | sort -rh | head -15
```

Common cache hogs and their safety:
| Cache | Safe to remove? | Notes |
|-------|----------------|-------|
| `~/.npm/_cacache` | Yes | `npm cache clean --force` |
| `~/.npm/_npx` | Yes | **Can be 2+ GB.** NPX cache stores full node_modules for every `npx` run. `rm -rf ~/.npm/_npx`. NOT cleaned by `npm cache clean`! |
| `~/.bun/install/cache/` | Yes | Bun package cache. Can accumulate multiple versions of same binary (e.g., opencode x86/musl/baseline variants). `rm -rf ~/.bun/install/cache/<pkg>-*` for specific packages |
| `~/.cache/yarn` | Yes | Can be 5-10GB. `rm -rf ~/.cache/yarn` |
| `~/.cache/puppeteer` | Yes | Regenerates on next puppeteer launch |
| `~/.cache/ms-playwright*` | Yes | Reinstall with `npx playwright install` if needed |
| `~/.cache/Cypress` | Yes | Reinstall with `npx cypress install` if needed |
| `~/.cache/camoufox` | Yes | Browser automation cache (Camoufox/Firefox). Can be 1.4GB+. Safe to remove, re-downloads on next use |
| `~/.cache/opencode` | Yes | Temporary session data |
| `~/.local/share/opencode` | Yes | OpenCode CLI session data, 1-3GB typical |
| `~/.cache/prisma` | Yes | Prisma engine cache. 50-125MB typical. `rm -rf ~/.cache/prisma`. Regenerates on next prisma generate |
| `~/.cache/uv` | Yes | `uv cache clean` |
| `~/.cache/snyk` | Yes | Snyk CLI cache. Can be 67MB+ |
| `~/.cache/google-chrome` | Yes | Browser cache |

## Hermes Directory Breakdown

Common hogs in `~/.hermes/`:

| Location | Typical Size | Safe to remove? | Notes |
|----------|-------------|----------------|-------|
| `~/.hermes/sessions/` | 1-3 GB | Yes | Session transcripts. Regenerate on new sessions |
| `~/.hermes/state-snapshots/` | 500MB-2GB | Yes | Pre-update snapshots. Safe to clear: `rm -rf ~/.hermes/state-snapshots/*` |
| `~/.hermes/hermes-agent/` | 1.5-2GB | No | Agent venv + runtime. Do NOT remove |
| `~/.hermes/checkpoints/` | 50-100MB | Yes | Old checkpoints |
| `~/.hermes/skills/` | 10-30MB | No | Skill definitions. Do NOT remove |
| `~/.hermes/logs/` | 5-15MB | Yes | Rotated logs |

Total `~/.hermes/` can easily reach 6-7 GB. Sessions + state-snapshots are the low-hanging fruit.
| `~/.cache/pip` | Yes | `pip cache purge`. Can accumulate 500MB+ |
| `<project>/.next` | Yes | Next.js build cache. `rm -rf <project>/.next`. Safe — regenerates on next build. Carreer-ops had 1GB in turbopack cache alone |
| `~/.cache/mozilla` | Low | Firefox profile data |

## Step 7: Old Kernels

```bash
uname -r  # Current kernel — DO NOT remove this one
dpkg -l 'linux-image-*' | grep '^ii' | awk '{print $2, $3}'
# Also check headers and modules:
dpkg -l 'linux-headers-*' 'linux-modules-*' 'linux-modules-extra-*' 2>/dev/null | grep '^ii' | awk '{print $2, $3}'
```

Remove old kernels with `sudo apt-get purge -y linux-image-X.X.X-XX-generic linux-headers-X.X.X-XX-generic linux-modules-X.X.X-XX-generic linux-modules-extra-X.X.X-XX-generic`.

**PITFALL**: `apt-get purge` for kernel packages can take 60-120s (dpkg triggers, initramfs regeneration). Always run in background with timeout 180s+ to avoid terminal timeout. Use `background: true` in Hermes terminal.

## Safe Cleanup Commands (Ordered by Impact)

```bash
# Docker — biggest impact, reclaim unused images + build cache + volumes
docker system prune -a --volumes # interactive, asks confirmation

# Force restart policy on all live containers (handles cases like 'unless-stopped' not auto-restarting)
docker update --restart unless-stopped $(docker ps -q)
```
# npm cache
npm cache clean --force

# apt cache
sudo apt clean

# Journal logs (keep last 100MB)
sudo journalctl --vacuum-size=100M

# UV (Python) cache
uv cache clean

# Prisma engine cache (50-125MB, regenerates on prisma generate)
rm -rf ~/.cache/prisma

# Camoufox browser automation cache (can be 1.4GB+)
rm -rf ~/.cache/camoufox

# Browser/tool caches
rm -rf ~/.cache/puppeteer
rm -rf ~/.cache/ms-playwright*
rm -rf ~/.cache/opencode
rm -rf ~/.cache/google-chrome

# Remove unused packages
sudo apt autoremove -y

# NPX cache (NOT cleaned by npm cache clean!)
rm -rf ~/.npm/_npx

# Bun binary variants (multiple arch builds of same package)
rm -rf ~/.bun/install/cache/oh-my-opencode-*

# Next.js build caches across projects
find ~/projetos -name ".next" -type d -exec du -sh {} + 2>/dev/null
# rm -rf ~/projetos/<project>/.next  # per-project, safe to remove

# pip cache
pip cache purge

# Ollama (if not using local LLMs — can be 7+ GB)
sudo rm -rf /usr/local/lib/ollama

# OpenCode session data (1-3 GB typical)
rm -rf ~/.local/share/opencode
```

## Presenting Results to User

Present as a prioritized table:
1. What can be cleaned
2. Size estimate
3. Risk level (zero/low/medium)
4. Command to run

Order by size descending. Docker and caches are usually the low-hanging fruit. Package removals (thunderbird, libreoffice) need user confirmation.
