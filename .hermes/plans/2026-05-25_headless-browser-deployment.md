# 无头浏览器部署计划

**日期**: 2026-05-25
**状态**: 待评估
**目标**: 为 Hermes Agent 部署可用的无头浏览器（headless Chrome/Chromium）

---

## 当前环境约束

| 项目 | 状态 |
|------|------|
| 系统 | CentOS 7.9 (glibc 2.17, OpenSSL 1.0.2k) |
| Node.js | 无法运行（需 glibc 2.28+） |
| Docker Hub | 直连失败（TLS 不兼容），镜像站不稳定 |
| Chrome/Chromium | 系统未安装，yum 源无包 |
| 代理 | mihomo 127.0.0.1:7890，HTTPS 有 TLS 兼容问题 |
| 内存 | 11GB 总计，约 4.5GB 空闲 |

## Hermes 浏览器架构

Hermes 使用 `agent-browser` CLI（Node.js 工具）操控浏览器：
- **本地模式**: `agent-browser install` 下载 Chromium，通过子进程调用
- **CDP 模式**: 通过 `browser.cdp_url` 连接远程 Chrome（WebSocket 或 HTTP discovery）
- 引擎选项: `auto`（默认）, `chrome`, `lightpanda`

**关键**: CentOS 7 不能跑 Node.js → 不能本地运行 agent-browser → **必须走 Docker + CDP 远程连接**

---

## 方案评估

### 方案A: Docker 容器运行 agent-browser（推荐）⭐
**可行性: 8/10 | 风险: 低 | 耗时: 30-45 分钟**

思路: 在 `node:20-slim` 容器里安装 agent-browser + Chromium，暴露 CDP 端口，Hermes 通过 `cdp_url` 连接。

**步骤**:
1. 创建 Dockerfile（基于 node:20-slim，安装 agent-browser + Chromium + 系统依赖）
2. 构建镜像 `hermes-browser`
3. 运行容器，暴露端口 9222
4. 配置 Hermes `browser.cdp_url = http://127.0.0.1:9222`
5. 验证 Hermes browser 工具可用

**优点**:
- 完全绕过 CentOS 7 限制
- 隔离性好，不影响其他容器
- 使用官方 agent-browser，功能完整
- 本地运行，无外部服务依赖

**风险**:
- Docker 镜像拉取可能需要走镜像站（node:20-slim 约 200MB，Chromium 约 400MB）
- 内存开销约 500MB-1GB（Chrome + agent-browser）
- 首次构建耗时较长

**Dockerfile 概要**:
```dockerfile
FROM node:20-slim
RUN apt-get update && apt-get install -y \
    chromium fonts-liberation libasound2 libatk-bridge2.0-0 \
    libdrm2 libgbm1 libgtk-3-0 libnss3 libxss1 xdg-utils \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN npm install -g agent-browser
RUN agent-browser install --with-deps
EXPOSE 9222
CMD ["agent-browser", "serve", "--port", "9222"]
```

---

### 方案B: Docker 容器跑 Chrome + 直接 CDP（备选）
**可行性: 7/10 | 风险: 低 | 耗时: 20-30 分钟**

思路: 不用 agent-browser，直接在容器里跑 `chromium --headless --remote-debugging-port=9222`。

**步骤**:
1. 拉取 `zenika/alpine-chrome` 或 `chromedp/headless-shell` 镜像（需镜像站）
2. 如果镜像不可用，用 `node:20-slim` + `apt install chromium` 手动搭建
3. 运行容器暴露 9222 端口
4. 配置 Hermes `cdp_url`

**优点**: 更轻量（不需要 agent-browser）
**风险**: Hermes 的 browser 工具依赖 agent-browser CLI 做命令分发，纯 CDP 可能功能不完整。需验证 Hermes 是否支持纯 CDP 模式。

---

### 方案C: Firefox Headless（应急）
**可行性: 5/10 | 风险: 中 | 耗时: 30 分钟**

思路: CentOS 7 的 updates 源有 Firefox ESR 115，直接 `yum install firefox`。

**步骤**:
1. `yum install -y firefox`
2. `firefox --headless --remote-debugging-port=9222 &`
3. 配置 Hermes cdp_url

**优点**: 最快，不需要 Docker
**风险**: Hermes 的 agent-browser 只支持 Chrome/Chromium 引擎，不支持 Firefox。CDP 协议 Chrome 和 Firefox 不完全兼容。大概率不可用。

---

### 方案D: 系统升级到 Rocky Linux 9（长期）
**可行性: 9/10 | 风险: 高 | 耗时: 1-2 天**

彻底解决 CentOS 7 的所有兼容性问题。但需要重新部署 20+ 个容器，风险大。

---

## 推荐执行计划

### 第一阶段: 验证可行性（10 分钟）
1. 确认 `node:20-slim` 镜像能否从镜像站拉取
2. 验证 Hermes 是否支持纯 CDP 模式（不依赖 agent-browser CLI）

### 第二阶段: 构建镜像（15-20 分钟）
1. 写 Dockerfile
2. 构建 `hermes-browser` 镜像
3. 测试容器内 agent-browser 是否正常

### 第三阶段: 部署配置（10 分钟）
1. 运行容器，映射端口 9222
2. 配置 Hermes `browser.cdp_url`
3. 重启 Hermes 网关

### 第四阶段: 验证（5 分钟）
1. 用 Hermes browser 工具打开测试页面
2. 验证截图、点击、输入等功能

---

## 资源预估

| 资源 | 用量 |
|------|------|
| 磁盘 | ~1.5GB（镜像 + 运行时） |
| 内存 | 500MB-1GB |
| CPU | 空闲时微量，浏览时 1-2 核 |
| 端口 | 9222（CDP） |

## 开放问题

1. Hermes 的 agent-browser CLI 调用方式 — 是子进程还是连接已有 CDP？
2. 如果 agent-browser CLI 必须在宿主机运行，需要在容器里通过什么方式暴露？
3. `node:20-slim` 镜像能否通过现有镜像站拉取？
