# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────
# TencentDB-Agent-Memory — Production Image
# ──────────────────────────────────────────────────────────────────────
# Multi-stage build (requires DOCKER_BUILDKIT=1):
#   1. deps-builder: install all deps (incl. native compile toolchain)
#   2. runtime:      slim image with only runtime deps + app
#
# 配置方式: 挂载 config.yaml 到 /data/config/tdai-gateway.yaml
#   docker run -v ./config.yaml:/data/config/tdai-gateway.yaml ...
#
# Build:
#   docker build -t tencentdb-agent-memory:latest .
#
# Package manager: npm (Node.js bundled). The repo ships pnpm-lock.yaml for
# local development, but inside the image we generate an npm lockfile on the
# fly to keep the image free of pnpm/corepack and to align with team CI/CD
# conventions. A small package.json patch is applied to:
#   - drop optional peerDependencies (`openclaw`, `node-llama-cpp`) which
#     pull in transitive packages that ship broken `workspace:*` refs to
#     unpublished sub-packages (e.g. @jimp/config-typescript).
#   - add `overrides` to short-circuit any straggler workspace refs.
# These tweaks only affect the image, not the source tree.
# ──────────────────────────────────────────────────────────────────────

# Stage 1: build/install deps (with toolchain for native bindings)
FROM node:22-slim AS deps-builder

# apt 源默认走 Debian 官方，公网可直接构建。内网构建加速：
#   docker build --build-arg APT_MIRROR=mirrors.tencent.com .
ARG APT_MIRROR=deb.debian.org
RUN if [ "$APT_MIRROR" != "deb.debian.org" ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi

# Native build toolchain (sqlite-vec, @node-rs/jieba etc.)
# 使用 BuildKit cache mount 缓存 apt 包，避免重复下载
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        make \
        g++ \
        ca-certificates

WORKDIR /build

# node:22-slim 自带 npm@10.9.8 有 arborist "edgesOut" crash（干净
# `npm install` 稳定报 `Cannot read properties of null (reading
# 'edgesOut')`）。升到 11 规避，不影响其它构建行为。
RUN npm install -g npm@11 --no-audit --no-fund

# Copy package metadata first to maximize Docker layer cache
COPY package.json ./

# Patch package.json for npm-friendly install:
#   1. Strip optional peerDependencies (openclaw, node-llama-cpp) — these
#      pull in @whiskeysockets/baileys → jimp@1.6.1, whose published manifest
#      contains a broken `workspace:*` ref to @jimp/config-typescript that
#      npm cannot resolve.
#   2. Add overrides as a defensive short-circuit in case any other
#      transitive dep reintroduces the same pattern.
RUN node -e " \
        const fs=require('fs'); \
        const p=JSON.parse(fs.readFileSync('package.json','utf8')); \
        delete p.peerDependencies; \
        delete p.peerDependenciesMeta; \
        p.overrides=Object.assign({},p.overrides,{ \
            '@jimp/config-typescript':'npm:dotenv@latest' \
        }); \
        fs.writeFileSync('package.json',JSON.stringify(p,null,2)); \
    "

# Generate lockfile + install production deps.
# --omit=optional 精简镜像体积，减少 CI push 超时风险（可选依赖如 mongodb、
# @clickhouse/client、kafkajs 等在网关镜像中不需要）。
# --ignore-scripts skips the openclaw postinstall patch (irrelevant in image).
# --legacy-peer-deps relaxes strict peer resolution for older transitive deps.
# 使用 BuildKit cache mount 缓存 npm 下载缓存，大幅加速重复构建
RUN --mount=type=cache,id=npm-cache,target=/root/.npm \
    npm install \
        --omit=dev \
        --omit=optional \
        --ignore-scripts \
        --legacy-peer-deps \
        --no-audit \
        --no-fund

# tsx 需要 esbuild 二进制（是 tsx 的 optionalDependency），--omit=optional 后会丢失。
# 单独安装 esbuild (--no-save 不修改 package.json)，esbuild ~10MB 远小于被省略的
# 可选依赖总大小（mongodb ~100MB+ 等）。
#
# 注意：这里**不能**加 --omit=optional —— esbuild 的平台特定二进制
# （@esbuild/linux-x64 / @esbuild/linux-arm64）本身就是 optionalDependency，
# 加了会导致运行时报 "The package @esbuild/linux-x64 could not be found"。
# 允许 dev 也不影响：esbuild 本身没有 devDep。
RUN --mount=type=cache,id=npm-cache,target=/root/.npm \
    npm install \
        --no-save \
        --ignore-scripts \
        --no-audit \
        --no-fund \
        esbuild

# Now copy the rest of the source
# 注意: .dockerignore 已排除 node_modules/.git/__tests__/docs 等
COPY . .

# 保留原始 package.json（COPY . 会覆盖 patch 后的版本，所以无需 git checkout）
# 注: .git 已被 .dockerignore 排除，原来的 git checkout 命令实际上总是失败的


# ──────────────────────────────────────────────────────────────────────
# Stage 2: runtime — slim image
# ──────────────────────────────────────────────────────────────────────
FROM node:22-slim AS runtime

ARG APT_MIRROR=deb.debian.org
RUN if [ "$APT_MIRROR" != "deb.debian.org" ]; then \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
      sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null || true; \
    fi

# Runtime essentials only (curl for HEALTHCHECK; tini for proper signal handling)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
        ca-certificates

WORKDIR /app

# Copy app from builder stage
COPY --from=deps-builder /build /app

# Data directory (mountable as PVC in K8s)
RUN mkdir -p /data/tdai-memory /data/config

# ── Runtime configuration ──
# 配置文件路径: 挂载 yaml 到 /data/config/tdai-gateway.yaml
# 敏感凭证通过环境变量注入 (env 优先级高于配置文件)
ENV NODE_ENV=production \
    TDAI_GATEWAY_CONFIG=/data/config/tdai-gateway.yaml \
    TDAI_GATEWAY_HOST=0.0.0.0 \
    TDAI_DATA_DIR=/data/tdai-memory \
    NODE_OPTIONS="--max-old-space-size=1536"

EXPOSE 8420

# K8s-friendly health check (liveness/readiness probes still defined separately
# in the Deployment manifest — this is the local-runtime fallback).
# 默认值只写在这里，不放进 ENV：TDAI_GATEWAY_PORT 一旦成为镜像环境变量就会盖掉
# 挂载配置里的 server.port（env 优先级高于 yaml），而 8420 与 gateway 的代码默认值一致。
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD curl -fsS http://127.0.0.1:${TDAI_GATEWAY_PORT:-8420}/health || exit 1

# Use tini as PID 1 so SIGTERM/SIGKILL propagate cleanly to Node and its
# pipeline workers, and zombie processes are reaped.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start the Gateway. tsx is a runtime dependency so we can run TypeScript directly.
CMD ["node", "--import", "tsx", "src/gateway/server.ts"]
