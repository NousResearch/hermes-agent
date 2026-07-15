# Optimized Next.js Dockerfile for ARM64

Multi-stage Dockerfile template for deploying Next.js on ARM64 (Oracle Cloud Free Tier).

```dockerfile
# Stage 1: Dependencies
FROM --platform=linux/arm64 node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 2: Build
FROM --platform=linux/arm64 node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 3: Production
FROM --platform=linux/arm64 node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

## Usage

```bash
docker build -t nextjs-arm64 .
docker run -p 3000:3000 nextjs-arm64
```

## Notes

- Requires `output: 'standalone'` in `next.config.js`
- Skip Docker entirely on ARM64 for faster local builds:
  ```bash
  npm install && npm run build && npm start
  ```
