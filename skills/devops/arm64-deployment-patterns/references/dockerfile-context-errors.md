# Dockerfile Context Errors on ARM64

Common `COPY` failures in multi-stage Docker builds on ARM64 and their solutions.

## Common Errors

### COPY --from=builder with wrong path

```dockerfile
# WRONG — copies from build context root, not builder stage
COPY web/ /app/web/

# CORRECT — copies from the builder stage
COPY --from=builder /app/web/ /app/web/
```

### Non-existent directories

```dockerfile
# WRONG — templates/ doesn't exist in the build context
COPY templates/ /app/templates/

# CORRECT — verify directory exists before copying, or skip
RUN mkdir -p /app/templates
```

### Missing public/ directory

```dockerfile
# WRONG — public/ only exists in Next.js projects, not all frameworks
COPY public/ /app/public/

# CORRECT — conditional copy
COPY --chown=node:node package*.json ./
```

## ARM64-Specific Issues

### Base image architecture

```dockerfile
# WRONG — may pull x86 image on ARM64
FROM node:18

# CORRECT — explicitly specify platform
FROM --platform=linux/arm64 node:18
```

### apt-get timeouts on slow ARM64

```dockerfile
# Add retries and timeout handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
```

### Build context size

ARM64 VMs often have limited resources. Keep context small:

```bash
# Create .dockerignore
echo "node_modules\n.git\n*.log\ntmp/" > .dockerignore
```

## Quick Diagnosis

```bash
# Test build context
docker build --no-cache --progress=plain -t test-arm64 . 2>&1 | head -50

# Check what's in the context
tar -czf - . | docker run --rm -i alpine tar -tzf - | head -20
```
