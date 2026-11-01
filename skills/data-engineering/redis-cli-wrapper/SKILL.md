---
name: redis-cli-wrapper
description: "Redis CLI wrapper for Hermes Agent — key pattern scanning, memory analysis, TTL statistics, large key detection, INFO parsing."
version: "0.1.0"
author: "kenya"
license: "MIT"
platforms: [linux, macos, windows]
category: "data-engineering"
tags: [redis, cache, memory, keys, ttl, monitoring]
depends_on: []
compatibility:
  hermes: ">=0.18.0"
  claude-code: ">=1.0.0"
  codex: ">=1.0.0"
  opencode: ">=0.5.0"
maturity: "beta"
homepage: "https://github.com/NousResearch/hermes-agent"
repository: "https://github.com/NousResearch/hermes-agent"
---

# Redis CLI Wrapper

> Redis CLI wrapper for Hermes Agent — key pattern scanning, memory analysis, TTL statistics, large key detection, INFO parsing.

## Prerequisites
- Redis server accessible via network/Unix socket
- `redis-cli` in PATH (or `redis-py` for Python API)
- Python 3.10+ with `redis` package

## Installation
```bash
hermes skill install redis-cli-wrapper
# Or manual
pip install redis
```

## Configuration
| Environment Variable | Required | Description | Example |
|----------------------|----------|-------------|---------|
| `REDIS_URL` | Yes | Redis connection URL | `redis://localhost:6379/0` or `rediss://user:pass@host:6380/0` |
| `REDIS_PASSWORD` | No | Password if not in URL | `secret` |

## Usage
### redis_keys
Scan keys by pattern with optional count limit.

```bash
# Scan all keys matching pattern
hermes skill run redis-cli-wrapper redis_keys --pattern "user:*" --limit 100

# Count keys only
hermes skill run redis-cli-wrapper redis_keys --pattern "session:*" --count-only
```

### redis_memory
Analyze memory usage - largest keys, memory stats, fragmentation.

```bash
# Top 20 largest keys
hermes skill run redis-cli-wrapper redis_memory --top-keys 20

# Full memory stats
hermes skill run redis-cli-wrapper redis_memory --stats
```

### redis_ttl
TTL statistics and expiring keys analysis.

```bash
# TTL distribution
hermes skill run redis-cli-wrapper redis_ttl --distribution

# Keys expiring in next hour
hermes skill run redis-cli-wrapper redis_ttl --expiring-within 3600
```

### redis_info
Parse and display Redis INFO sections.

```bash
# All sections
hermes skill run redis-cli-wrapper redis_info

# Specific section
hermes skill run redis-cli-wrapper redis_info --section memory
```

## API / Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| `redis_keys` | Scan keys by pattern | `pattern: str, limit: int, count_only: bool` |
| `redis_memory` | Memory analysis | `top_keys: int, stats: bool` |
| `redis_ttl` | TTL analysis | `distribution: bool, expiring_within: int` |
| `redis_info` | INFO command parser | `section: str` |

## Examples
```bash
# Find large keys eating memory
hermes skill run redis-cli-wrapper redis_memory --top-keys 50 --format json

# Check session TTL health
hermes skill run redis-cli-wrapper redis_ttl --distribution --format table

# Quick memory overview
hermes skill run redis-cli-wrapper redis_info --section memory --format json
```

## Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| `Connection refused` | Redis not running | Start Redis or check host/port |
| `NOAUTH Authentication required` | Password needed | Set REDIS_PASSWORD or include in REDIS_URL |
| `LOADING Redis is loading the dataset` | Redis starting up | Wait for startup to complete |
| `OOM command not allowed` | Memory limit reached | Increase maxmemory or evict keys |

## Changelog
### v0.1.0 (2026-08-15)
- Initial release with key scanning, memory analysis, TTL stats, INFO parsing
- JSON/Table output formats