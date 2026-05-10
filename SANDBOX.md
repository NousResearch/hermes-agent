# Hermes Agent Docker 沙箱运行指南

本方案将 Hermes Agent 及其依赖数据库（Redis、MySQL、PostgreSQL）运行在完全隔离的 Docker 网络中，提供资源限制、权限隔离和网络隔离等沙箱安全特性。

## 快速开始

```bash
# 1. 配置环境变量（修改数据库密码和 API keys）
cp .env.sandbox.example .env.sandbox
# 编辑 .env.sandbox，填入你的 API keys

# 2. 一键启动沙箱
source .env.sandbox
./scripts/run-sandbox.sh up

# 3. 进入交互式 CLI
./scripts/run-sandbox.sh chat

# 4. 停止沙箱
./scripts/run-sandbox.sh down
```

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    宿主机 (Host)                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │         隔离网络：hermes-sandbox                   │  │
│  │  172.25.0.0/16                                    │  │
│  │                                                   │  │
│  │  ┌─────────────┐    ┌─────────────┐              │  │
│  │  │ hermes-agent│◄──►│   Redis     │ :6379        │  │
│  │  │  (非 root)  │    │  (无端口映射)│              │  │
│  │  └─────────────┘    └─────────────┘              │  │
│  │         ▲           ┌─────────────┐              │  │
│  │         └──────────►│   MySQL     │ :3306        │  │
│  │                     │  (无端口映射)│              │  │
│  │                     └─────────────┘              │  │
│  │         ▲           ┌─────────────┐              │  │
│  │         └──────────►│  PostgreSQL │ :5432        │  │
│  │                     │  (无端口映射)│              │  │
│  │                     └─────────────┘              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 安全特性

| 层级 | 措施 | 说明 |
|------|------|------|
| **网络隔离** | 自定义 bridge 网络 | 所有服务仅在 `hermes-sandbox` 网络内通信 |
| **端口暴露** | 默认不映射数据库端口 | Redis/MySQL/PostgreSQL 无法从宿主机外部访问 |
| **权限隔离** | `no-new-privileges:true` | 禁止容器内进程通过 setuid 获取新特权 |
| **用户隔离** | entrypoint 自动 drop privileges | 以 root 启动，初始化完成后降为 `hermes` 用户运行 |
| **资源限制** | CPU / Memory limits | hermes-agent 限制为 2 CPU + 4GB 内存 |
| **容器逃逸防护** | 不挂载 Docker Socket | 默认关闭，防止容器内控制宿主机 Docker |

## 常用命令

### 启动与停止

```bash
# 构建并启动（首次或 Dockerfile 变更后）
./scripts/run-sandbox.sh up

# 仅启动（不重建镜像）
docker compose -f docker-compose.sandbox.yml up -d

# 停止并移除容器（数据卷保留）
./scripts/run-sandbox.sh down

# 完全清理（包括数据卷）
docker compose -f docker-compose.sandbox.yml down -v
```

### 交互操作

```bash
# 启动交互式对话
./scripts/run-sandbox.sh chat

# 进入容器 Shell
./scripts/run-sandbox.sh shell

# 查看实时日志
./scripts/run-sandbox.sh logs

# 查看特定服务日志
./scripts/run-sandbox.sh logs -f hermes
```

### 手动 Docker 命令

```bash
# 执行单次命令
docker exec hermes-agent hermes chat -q "写一个 Python 快排"

# 在容器内安装额外 Python 包
docker exec hermes-agent uv pip install requests-html

# 复制宿主机配置到沙箱
docker cp ~/.hermes/.env hermes-agent:/opt/data/.env
docker cp ~/.hermes/config.yaml hermes-agent:/opt/data/config.yaml
```

## 数据持久化

沙箱使用 **Docker Named Volumes** 持久化数据：

| Volume | 服务 | 路径 | 说明 |
|--------|------|------|------|
| `hermes-data` | hermes-agent | `/opt/data` | 配置、会话、日志、技能、记忆 |
| `redis-data` | Redis | `/data` | AOF 持久化数据 |
| `mysql-data` | MySQL | `/var/lib/mysql` | 数据库文件 |
| `postgres-data` | PostgreSQL | `/var/lib/postgresql/data` | 数据库文件 |

> 数据卷在 `docker compose down` 后仍然保留，下次启动可自动复用。
> 如需彻底重置，使用 `docker compose -f docker-compose.sandbox.yml down -v`。

## 数据库连接

容器内已通过环境变量配置好内部连接地址：

```bash
# Redis
redis://redis:6379/0

# MySQL
mysql://hermes:hermes_pass@mysql:3306/hermes

# PostgreSQL
postgresql://hermes:hermes_pass@postgres:5432/hermes
```

如果你需要从**宿主机**连接数据库进行调试，编辑 `docker-compose.sandbox.yml`，取消对应数据库的 `ports` 注释（仅绑定 `127.0.0.1`）。

## 连接外部已有数据库

如果你已经在宿主机上运行了 Redis / MySQL / PostgreSQL，不想在 compose 中重复创建：

1. **方案 A：将外部容器接入沙箱网络**

   ```bash
   # 假设已有 redis 容器名为 my-redis
   docker network connect hermes-sandbox my-redis
   ```

   然后修改 `docker-compose.sandbox.yml` 中 hermes 服务的环境变量：
   ```yaml
   environment:
     - REDIS_URL=redis://my-redis:6379/0
   ```

2. **方案 B：使用宿主机网络**

   修改 hermes 服务的网络模式（牺牲隔离性）：
   ```yaml
   network_mode: host
   ```

3. **方案 C：从 compose 中移除数据库服务**

   直接注释掉 `docker-compose.sandbox.yml` 中的 `redis`、`mysql`、`postgres` 服务，只保留 `hermes`，并修改 `depends_on` 和环境变量。

## 运行 Gateway 或 Dashboard

默认模式下，hermes-agent 容器运行 `sleep infinity`，方便你随时 `docker exec` 进入交互。

如需长期运行 Gateway：

```yaml
# docker-compose.sandbox.yml 中修改 hermes 服务
command: ["gateway", "run"]
ports:
  - "127.0.0.1:9119:9119"      # Dashboard
  - "127.0.0.1:8080:8080"      # API Server（如启用）
```

并确保 `.env.sandbox` 中设置了 `API_SERVER_KEY`。

## 启用 Docker Terminal Backend（高级）

Hermes 的 `terminal` 工具支持在 Docker 容器中执行命令（`backend: docker`）。如果你在沙箱内的 Hermes 也想使用此功能，需要让沙箱内的 Hermes 能访问宿主机的 Docker Daemon。

**⚠️ 安全风险**：挂载 Docker Socket 后，容器内的进程可以完全控制宿主机的 Docker，相当于获得了宿主机的 root 权限。请仅在可信代码环境下启用。

```yaml
# docker-compose.sandbox.yml 中 hermes 服务的 volumes 下添加：
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:ro
```

然后在沙箱内的 `config.yaml` 中配置：

```yaml
terminal:
  backend: "docker"
  cwd: "/workspace"
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
```

## 故障排查

### 容器启动后立即退出

```bash
# 查看日志
docker logs hermes-agent

# 常见原因：
# 1. depends_on 健康检查失败 — 检查数据库日志
# 2. entrypoint 权限问题 — 确认 volume 可写
```

### 数据库连接失败

```bash
# 进入容器测试网络连通性
docker exec -it hermes-agent bash
ping redis
ping mysql
ping postgres

# 测试数据库连接
python3 -c "import urllib.request; print(urllib.request.urlopen('http://redis:6379').read())"
```

### 权限错误（Permission denied）

```bash
# 查看 volume 的拥有者
docker run --rm -v hermes-sandbox_hermes-data:/opt/data busybox ls -la /opt/data

# 如需重置权限，可临时以 root 运行一次
docker compose -f docker-compose.sandbox.yml run --rm --user root hermes \
  chown -R 10000:10000 /opt/data
```

### 端口冲突

如果宿主机已运行独立的数据库容器，沙箱内的数据库服务端口映射（如果启用）会冲突。**默认配置下不映射端口，因此不会冲突**。如果手动启用了端口映射且发生冲突，请改为仅绑定 `127.0.0.1` 或更换宿主机映射端口。

## 与其他 Docker Compose 文件的关系

| 文件 | 用途 | 网络模式 |
|------|------|----------|
| `docker-compose.yml` | 生产部署（Gateway + Dashboard） | `host` |
| `docker-compose.sandbox.yml` | 开发/沙箱环境（Agent + 数据库） | 隔离 bridge |

两者互不干扰，可根据场景选择使用。
