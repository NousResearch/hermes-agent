# Hermes 适配参考：TencentDB Agent Memory v2

简体中文 · [English](./README.md)

该目录是 Hermes 接入 TencentDB Agent Memory v2 API 的参考实现。它实现了一个 Hermes `MemoryProvider`，通过 Python SDK 连接一个已经运行的 Memory Gateway。该 provider 不启动、不管理 Gateway 子进程。

在 standalone 本地模式下，推荐的 Gateway 地址是 `http://127.0.0.1:8420`。默认约定是 `api_key = "local"`、`service_id = "default"`。如果 Gateway 显式启用了 `TDAI_GATEWAY_API_KEY`，则 `TDAI_MEMORY_API_KEY` 应使用同一个值。

## 架构

```text
Hermes Agent
  └─ MemoryManager
       └─ memory_tencentdb_v2 provider
            ├─ sync_turn()        完整对话轮次 -> add_conversation (L0)
            ├─ prefetch()         prompt 前召回记忆/core/scenario
            ├─ tdai_memory_search 工具
            ├─ tdai_conversation_search 工具
            └─ tdai_read_scene 工具
                 │
                 ▼
            tencentdb_agent_memory.MemoryClient
                 │ HTTP v2 API
                 ▼
            TencentDB Agent Memory Gateway（standalone :8420 或远端服务）
```

## v1 与 v2 对比

| | v1 `memory_tencentdb` | v2 `memory_tencentdb_v2` |
|---|---|---|
| API | 旧版 `/recall`、`/capture`、`/search/*` | v2 `/v2/*` |
| HTTP 客户端 | 原生 `urllib.request` | `tencentdb_agent_memory` Python SDK（`httpx`） |
| Gateway 生命周期 | 可能启动本地 Gateway 子进程 | 连接外部或容器管理的 Gateway |
| standalone 约定 | localhost only | `endpoint=http://127.0.0.1:8420`、`api_key=local`、`service_id=default` |
| 工具 | memory search、conversation search | 额外支持 `tdai_read_scene` |

## 快速开始

推荐从仓库根目录执行自动安装脚本：

```bash
bash scripts/install-hermes-plugin-v2.sh
```

脚本会下载并安装 Python SDK wheel，将 `memory_tencentdb_v2` provider 软链到 Hermes 的 memory 插件目录，检查 `~/.hermes/config.yaml` 是否启用 provider，并把 `TDAI_MEMORY_ENDPOINT`、`TDAI_MEMORY_API_KEY`、`TDAI_MEMORY_SERVICE_ID` 写入 `~/.hermes/.env`。SDK 会安装到 Hermes 实际使用的 Python 环境：优先使用显式传入的 `PYTHON_BIN`，其次使用 `HERMES_VENV_DIR/bin/python`（默认 `~/.hermes/hermes-agent/venv/bin/python`），最后才回退到系统 `python3`。如需覆盖路径，可设置 `HERMES_HOME`、`HERMES_AGENT_DIR`、`HERMES_VENV_DIR`、`HERMES_MEMORY_PLUGIN_DIR`、`HERMES_ENV` 或 `PYTHON_BIN`。

如需手动安装，可按下面步骤执行。

### 1. 安装 SDK

Python SDK 暂未发布到 PyPI，请先下载 wheel 包再安装：

```bash
curl -L -o tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl \
  "https://cnb.cool/tencent/cloud/nosql/nosql-utilities/-/commit-assets/download/cc74bd6dbc931727da9ab6907b5ab1a07d7afd9d/tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl"

# 请使用 Hermes 实际运行时的 Python；常见路径如下：
~/.hermes/hermes-agent/venv/bin/python -m pip install ./tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl

# 如果你的 Hermes 不是使用该 venv，请改用对应的 python：
# PYTHON_BIN=/path/to/hermes/python
# "$PYTHON_BIN" -m pip install ./tencentdb_agent_memory_sdk_python-0.1.0-py3-none-any.whl
```

Python import 路径是 `tencentdb_agent_memory`。

### 2. 配置环境变量

standalone 本地 Gateway：

```bash
export TDAI_MEMORY_ENDPOINT="http://127.0.0.1:8420"
export TDAI_MEMORY_API_KEY="local"
export TDAI_MEMORY_SERVICE_ID="default"
```

如果 Gateway 启用了 `TDAI_GATEWAY_API_KEY`，请将 `TDAI_MEMORY_API_KEY` 设置成同一个值。

### 3. 在 Hermes 配置中启用（`~/.hermes/config.yaml`）

```yaml
memory:
  provider: memory_tencentdb_v2
```

### 4. 安装 provider 到 Hermes

开发环境推荐软链：

```bash
ln -s "$(pwd)/hermes-plugin/memory/memory_tencentdb_v2" \
  <hermes-agent>/plugins/memory/memory_tencentdb_v2
```

部署时也可以复制：

```bash
cp -r hermes-plugin/memory/memory_tencentdb_v2 \
  <hermes-agent>/plugins/memory/memory_tencentdb_v2
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TDAI_MEMORY_ENDPOINT` | `http://127.0.0.1:8420` | Memory Gateway 地址 |
| `TDAI_MEMORY_API_KEY` | standalone 示例中为 `local` | SDK 发送的 Bearer token |
| `TDAI_MEMORY_SERVICE_ID` | standalone 示例中为 `default` | Memory 空间 ID，通过 `x-tdai-service-id` 发送 |

## Provider 职责

| 方法 / 工具 | 说明 |
|---|---|
| `initialize(session_id)` | 创建 SDK client，并绑定 Hermes session ID |
| `sync_turn(user_content, assistant_content)` | 完整对话结束后调用 `add_conversation()` 写入 L0 |
| `prefetch(query)` | 下一轮 prompt 前搜索 L1，并读取 L3/L2 上下文 |
| `tdai_memory_search` | Agent 可主动调用的 L1 记忆搜索 |
| `tdai_conversation_search` | Agent 可主动调用的 L0 对话搜索 |
| `tdai_read_scene` | Agent 可主动调用的 L2 场景读取 |

## 作为其它 Agent 的适配模板

如果你要适配其它 Python Agent 框架，可以复用同样模式：

1. 初始化 `MemoryClient(endpoint, api_key, service_id)`。
2. 在每个完整对话轮次结束后调用 `add_conversation()`。
3. 在下一轮 prompt 构建前调用 `search_atomic()`、`read_core()`，必要时调用 `list_scenarios()`。
4. 将召回结果格式化成清晰标注的上下文块。
5. 暴露主动工具，让 Agent 可以搜索记忆和读取场景。
6. 保持 best-effort：Memory 失败不应该阻塞 Agent 主回答链路。

## 可靠性

- 熔断：连续 5 次失败后进入 60 秒冷却。
- 线程安全：内部状态变更使用锁保护。
- 优雅降级：prefetch 或工具调用失败时返回空结果或友好错误，不让 Agent 崩溃。
