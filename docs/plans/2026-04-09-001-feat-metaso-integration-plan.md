---
title: "feat: Integrate Metaso as web search backend and MCP toolset"
type: feat
status: active
date: 2026-04-09
origin: docs/brainstorms/2026-04-09-metaso-web-search-integration-requirements.md
---

# Metaso 双路径集成实现规划

## Overview

将 metaso 以双路径集成到 Hermes：（1）作为第 5 个 native web backend，使 `web_search`/`web_extract` 可通过 `config.yaml` 切换到 metaso；（2）作为 MCP 工具集，暴露 metaso 的全部 6 个工具（search、reader、chat、topic_list、topic_search、topic_file_content）。

## Problem Frame

Hermes 当前 4 个 web 搜索后端（Exa、Firecrawl、Parallel、Tavily）均需要海外 API Key。中国地区用户无法使用 web 搜索功能。metaso（metaso.cn）提供中文优化的搜索、阅读、AI 问答能力，已有可用 MCP server（`mcpm run metaso`）和 REST API（`/api/v1/search`、`/api/v1/reader`、`/api/v1/chat/completions`），但尚未接入 Hermes 的工具系统。

## Requirements Trace

- R1. metaso 作为第 5 个 Web 搜索后端注册到 `web_tools.py` → Unit 1, 2
- R2. 支持 `config.yaml` 的 `web.backend: metaso` 切换 + 自动回退 → Unit 1
- R3. 输出格式归一化为 `{success, data.web[]}` → Unit 1
- R4. `web_extract` 走 metaso reader API → Unit 2
- R5. MCP 工具发现与注册 → Unit 3
- R6. 独立 toolset `mcp-metaso` → Unit 3
- R7. `mcp-metaso` 注入到 hermes-cli 及 messaging toolsets → Unit 4
- R8. webpage 默认 scope，扩展 scope 走 MCP → Unit 1, 3
- R9. metaso chat 作为独立 MCP 工具 → Unit 3
- R10. metaso topic 功能作为独立 MCP 工具 → Unit 3

## Scope Boundaries

- 不修改 metaso MCP server 本身（`mcpm run metaso` 已可用）
- 不替换/删除现有 Exa/Firecrawl/Parallel/Tavily 后端
- 不改动 Hermes 工具注册表核心逻辑，仅扩展
- image/video/podcast scope 通过 MCP 工具集暴露，不塞入 native `web_search`

## Context & Research

### Relevant Code and Patterns

- **`tools/web_tools.py`** — 现有后端实现模板。`_exa_search()` 最接近 metaso（简单的 HTTP POST + JSON 归一化）
- **`tools/registry.py`** — `registry.register()` 模式，`tool_error()`/`tool_result()` 辅助函数
- **`tools/mcp_tool.py`** — `discover_mcp_tools()` 从 `~/.hermes/config.yaml` 读取 `mcp_servers` 配置，通过 stdio 连接 MCP server
- **`toolsets.py`** — `TOOLSETS` dict + `_HERMES_CORE_TOOLS` 列表
- **`hermes_cli/config.py`** — `load_config()` 读取 YAML，`OPTIONAL_ENV_VARS` 定义 API key 元数据
- **`model_tools.py`** — `_discover_tools()` 中的 `_modules` 列表自动导入工具模块

### Institutional Learnings

- 需求文档已确认双路径策略、API Key 通过环境变量、默认 webpage scope
- metaso API 使用 Bearer token 认证（`Authorization: Bearer {KEY}`）
- metaso search API 的 scope 枚举：`webpage`、`document`、`paper`、`image`、`video`、`podcast`

### External References

- metaso API: `https://metaso.cn/api/v1/` (search, reader, chat/completions)
- MCP SDK: Hermes 使用 `mcp` Python SDK 的 stdio transport

## Key Technical Decisions

- **Native backend 用 httpx 直接调用 REST API**（不通过 MCP SDK）：metaso REST API 已验证可用，httpx 调用比 MCP stdio 更轻量，且与现有 Exa/Tavily 后端风格一致。（see origin: docs/brainstorms/2026-04-09-metaso-web-search-integration-requirements.md）
- **MCP 工具集通过 `mcpm run metaso` stdio 连接**：保留 metaso MCP server 的全部功能（topic 检索、AI 问答），复用 `discover_mcp_tools()` 现有机制。（see origin: origin doc）
- **metaso 在自动回退链中优先级最低**：放在 Firecrawl > Parallel > Tavily > Exa 之后，确保不干扰现有配置
- **metaso reader 返回 markdown**：`Accept: text/plain` header 直接获取 markdown，跳过 LLM 二次摘要（metaso 本身已做内容清洗）

## Open Questions

### Resolved During Planning

- **metaso REST API 响应格式**：search API 的响应结构与现有后端不同，需要在 `_normalize_metaso_search_results()` 中做映射。从 metaso 脚本反推，搜索返回的结果包含 `url`、`title`、`snippet` 等字段，需归一化为 `{url, title, description, position}`。

### Deferred to Implementation

- **metaso search API 响应精确 schema**：脚本中用 `response.raise_for_status()` 但未记录完整响应体。实现时先用已知 API key 做一次真实请求验证，确认字段名。
- **MCP stdio 多 session 复用**：`mcp_tool.py` 的 `_ensure_mcp_loop()` 使用单个后台线程管理所有 server 连接，metaso 作为标准 MCP server 配置即可共享连接，无需额外处理。
- **metaso reader 对大页面的处理**：metaso reader 返回的 markdown 是否有长度限制？与现有 `web_extract` 的 ~5000 char 截断策略如何协调？实现时验证后决定。

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification.*

### Native Backend Path (web_search / web_extract)

```
user query
    │
    ▼
web_search_tool(query)
    │ _get_backend() → "metaso"
    ▼
_metaso_search(query, limit)
    │ httpx.post("https://metaso.cn/api/v1/search",
    │   headers={"Authorization": f"Bearer {METASO_API_KEY}"},
    │   json={"q": query, "scope": "webpage", "size": str(limit)})
    ▼
_normalize_metaso_search_results(response)
    │ maps {url, title, snippet, ...} → {url, title, description, position}
    ▼
{"success": true, "data": {"web": [...]}}
```

### MCP Toolset Path

```
Hermes startup
    │
    ▼
discover_mcp_tools()
    │ reads mcp_servers from config.yaml:
    │   metaso:
    │     command: mcpm
    │     args: ["run", "metaso"]
    │
    ▼
MCP stdio subprocess (mcpm run metaso)
    │
    ▼
6 tools registered:
  mcp_metaso_web_search
  mcp_metaso_web_reader
  mcp_metaso_chat
  mcp_metaso_topic_list
  mcp_metaso_topic_search
  mcp_metaso_topic_file_content
```

### Configuration Flow

```
~/.hermes/config.yaml
    │
    ├── web.backend: "metaso"     → _get_backend() returns "metaso"
    │                                (需将 "metaso" 加入 _get_backend() 白名单)
    ├── mcp_servers.metaso:       → discover_mcp_tools() connects
    │     command: mcpm
    │     args: ["run", "metaso"]
    │
    └── env: METASO_API_KEY       → _has_env("METASO_API_KEY")
```

## Implementation Units

- [ ] **Unit 1: Add metaso as native web_search backend**

**Goal:** metaso 作为第 5 个后端接入 `web_search_tool()`，支持搜索查询归一化。

**Requirements:** R1, R2, R3, R8

**Dependencies:** 无（独立于 MCP 路径）

**Files:**
- Modify: `tools/web_tools.py`（核心：backend 逻辑 + search/extract + dispatch）
- Test: `tests/test_web_tools.py`（如存在）或新建 `tests/test_metaso_backend.py`

**Approach:**
- 添加 `_has_metaso_env()` 检查 `METASO_API_KEY`
- 添加 `_is_metaso_available()` 验证 API key 存在
- 在 `_get_backend()` 的显式白名单中添加 `"metaso"`（line 91: `if configured in ("...", "metaso"):`）
- 在 `_get_backend()` 的 fallback 链末尾添加 `("metaso", _has_metaso_env())`
- 在 `_is_backend_available()` 中添加 `"metaso"` 分支
- **修复默认回退**：line 107 的 `return "firecrawl"` 改为动态返回首个可用后端，确保无其他 Key 时 metaso 能被自动选中
- 添加 `_metaso_search(query, limit)` 函数，调用 `POST https://metaso.cn/api/v1/search`
- 添加 `_normalize_metaso_search_results(raw_response)` 将响应映射为 `{url, title, description, position}`
- 在 `web_search_tool()` 的 dispatch 中添加 `if backend == "metaso"` 分支
- 在 `check_web_api_key()` 中添加 metaso 支持（line ~1922）
- 在 `_web_requires_env()` 中添加 `"METASO_API_KEY"`
- **防御性归一化**：`_normalize_metaso_search_results()` 中验证期望字段存在，缺失时用 `tool_error()` 返回清晰错误而非静默返回空结果
- 注意：`.env.example`、`cli-config.yaml.example`、`OPTIONAL_ENV_VARS` 由 Unit 4 负责

**Patterns to follow:**
- `_exa_search()` 的 HTTP 请求和归一化模式（最简洁的参考）
- `_tavily_request()` 的 httpx 调用模式

**Test scenarios:**
- Happy path: 给定有效 METASO_API_KEY 和查询，返回 `{success: true, data.web: [...]}` 且每项包含 url/title/description
- Edge case: limit=0 或 limit 超过 API 上限，行为正常
- Edge case: 中文查询返回非空结果
- Error path: 无效 API key 返回友好错误（不抛出未捕获异常）
- Error path: 网络超时/连接失败的错误处理
- Integration: `check_web_api_key()` 在仅设置 METASO_API_KEY 时返回 True

**Verification:**
- `python -c "from tools.web_tools import web_search_tool; print(web_search_tool('Python 教程'))"` 返回 JSON 且包含有效搜索结果

- [ ] **Unit 2: Add metaso as native web_extract backend**

**Goal:** `web_extract_tool()` 支持通过 metaso reader API 提取 URL 内容。

**Requirements:** R4

**Dependencies:** Unit 1（共享 metaso 客户端和认证逻辑）

**Files:**
- Modify: `tools/web_tools.py`
- Test: `tests/test_metaso_backend.py`

**Approach:**
- 添加 `_metaso_extract(urls)` 函数，对每个 URL 调用 `POST https://metaso.cn/api/v1/reader`（`Accept: text/plain`）
- **URL 安全验证**：在传给 metaso reader 前验证每个 URL：仅允许 http/https 协议，拒绝私有 IP 段（10.x、172.16-31.x、192.168.x、127.x、localhost）
- 返回格式归一化为 `{"results": [{url, title, content, error?}]}`，与现有 extract 一致
- 在 `web_extract_tool()` 的 dispatch 中添加 metaso 分支
- metaso reader 返回纯 markdown，跳过 LLM 二次摘要（metaso 已做内容清洗）；如果 metaso 返回内容超过现有 `DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION`（5000 chars），仍会进入辅助 LLM 管道，这是可接受的行为
- 大页面处理：如果返回内容超过现有限制，保持截断策略与现有后端一致

**Patterns to follow:**
- `_parallel_extract()` 的 async URL 批处理模式
- Firecrawl extract 的 markdown 返回格式

**Test scenarios:**
- Happy path: 给定有效 URL 列表，返回 markdown 内容
- Edge case: 空 URL 列表，返回空结果
- Error path: 无效 URL 或 404 页面，在结果中标记 error 字段
- Error path: 非 HTTP 协议 URL（file://, localhost）的拒绝处理
- Integration: web_extract 与 web_search 联用（搜索结果中的 URL 传给 extract）

**Verification:**
- `python -c "from tools.web_tools import web_extract_tool; ..."` 对已知 URL 返回可读 markdown 内容

- [ ] **Unit 3: Configure metaso MCP server and expose tools**

**Goal:** 通过 config.yaml 配置 metaso MCP server，使 6 个 metaso 工具被 `discover_mcp_tools()` 自动发现和注册。

**Requirements:** R5, R6, R9, R10

**Dependencies:** 无（与 native backend 路径并行独立）

**Files:**
- Modify: `cli-config.yaml.example`（添加 metaso MCP server 配置示例）
- Modify: `hermes_cli/config.py`（如需要添加 mcpm 命令检测）
- Test: `tests/test_mcp_tool.py`（如存在）

**Approach:**
- 在 `cli-config.yaml.example` 中添加 metaso MCP server 配置模板：
  ```yaml
  mcp_servers:
    metaso:
      command: mcpm
      args: ["run", "metaso"]
      timeout: 60
  ```
- 确认 `mcp_tool.py` 的 `_load_mcp_config()` 能正确读取此配置
- 确认 `discover_mcp_tools()` 能自动发现并注册 6 个工具
- 工具命名约定：`mcp_metaso_web_search`、`mcp_metaso_web_reader`、`mcp_metaso_chat` 等（注意：单下划线分隔，`mcp_tool.py` 使用 `f"mcp_{safe_server_name}_{safe_tool_name}"` 格式）
- 自动创建 `mcp-metaso` toolset（`mcp_tool.py` 的 `create_custom_toolset()` 已支持）
- 验证 `_sync_mcp_toolsets()` 自动将 `mcp_metaso_*` 工具注入所有 hermes-* toolset（line 1568-1574）

**Test scenarios:**
- Happy path: 配置 metaso MCP server 后，`discover_mcp_tools()` 返回 6 个工具名
- Happy path: `resolve_toolset("hermes-cli")` 返回的结果包含 `mcp_metaso_*` 工具名
- Edge case: `mcpm` 命令不在 PATH 中的错误处理
- Edge case: MCP server 启动超时（connect_timeout）的错误传播
- Integration: 通过 Hermes 对话调用 `mcp_metaso_chat` 并获取 AI 问答结果

**Verification:**
- Hermes 启动日志中显示 MCP 工具注册成功
- `python -c "from toolsets import resolve_toolset; print(resolve_toolset('hermes-cli'))"` 包含 `mcp_metaso_` 工具名

**Patterns to follow:**
- 现有 `cli-config.yaml.example` 中的 MCP server 配置格式
- `mcp_tool.py` 中其他 MCP server 的注册模式

**Test scenarios:**
- Happy path: 配置 metaso MCP server 后，`discover_mcp_tools()` 返回 6 个工具名
- Edge case: `mcpm` 命令不在 PATH 中的错误处理
- Edge case: MCP server 启动超时（connect_timeout）的错误传播
- Integration: 通过 Hermes 对话调用 `mcp__metaso__chat` 并获取 AI 问答结果

**Verification:**
- Hermes 启动日志中显示 "Registered 6 MCP tools from metaso server"
- 在 Hermes 对话中能调用 `mcp__metaso__web_search` 并获取结果

- [ ] **Unit 4: Update configuration wizard and documentation**

**Goal:** `hermes tools` 设置向导支持 metaso 选项，`.env.example` 和 `cli-config.yaml.example` 更新。

**Requirements:** R2（success criteria: hermes tools 设置中能看到 metaso 选项）

**Dependencies:** Unit 1, 3

**Files:**
- Modify: `hermes_cli/config.py`（OPTIONAL_ENV_VARS 添加 METASO_API_KEY，注意：此条目也是 Unit 1 的功能依赖，但实际添加工作由 Unit 5 完成）
- Modify: `.env.example`
- Modify: `cli-config.yaml.example`
- Modify: `hermes_cli/` 中 tools 设置相关代码（如 `hermes tools` 命令的 backend 选择菜单）

**Approach:**
- 在 `OPTIONAL_ENV_VARS` 中添加 metaso 条目：
  ```python
  "METASO_API_KEY": {
      "description": "Metaso API key for Chinese-optimized web search",
      "prompt": "Metaso API key",
      "url": "https://metaso.cn",
      "password": True,
      "category": "tool",
  }
  ```
- 在 `.env.example` 中添加 `METASO_API_KEY=mk-xxxxxxxxx` 的注释行
- 在 `cli-config.yaml.example` 的 `web:` 部分添加 `metaso` 作为 backend 选项注释
- 在 `cli-config.yaml.example` 的 `mcp_servers:` 部分添加 metaso 配置示例
- 更新 `hermes tools` 交互菜单的 backend 选项列表（搜索 `hermes tools` 相关代码找到具体文件）

**Test scenarios:**
- Happy path: `hermes tools` 命令显示 metaso 作为可选后端
- Happy path: 设置 web.backend=metaso 并配置 METASO_API_KEY 后，web_search 工作正常

**Verification:**
- `hermes tools` 交互菜单中显示 metaso 选项
- `.env.example` 和 `cli-config.yaml.example` 中包含 metaso 配置示例

## System-Wide Impact

- **Interaction graph:** `web_search` 和 `web_extract` 的行为在 metaso backend 下保持一致接口，模型端无需修改调用方式。MCP 路径新增 6 个工具，模型可选择性使用。
- **Error propagation:** metaso API 错误通过 `tool_error()` 返回 JSON，与现有模式一致。网络超时通过 httpx 的 `TimeoutException` 捕获并转换为工具错误。
- **State lifecycle risks:** 无状态变更，纯查询操作。
- **API surface parity:** `web_search` 和 `web_extract` 的输入输出 schema 不变。MCP 工具是新增表面，不影响现有工具。
- **Integration coverage:** 关键场景是 native backend 和 MCP 工具集都能独立工作，互不干扰。
- **Unchanged invariants:** Exa/Firecrawl/Parallel/Tavily 后端的行为和优先级不受影响。`check_web_api_key()` 在 metaso 未配置时仍能正确检测其他后端。

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| metaso REST API 响应格式与预期不符 | Medium | High | Unit 1 先用真实请求验证响应 schema；归一化层做字段验证，缺失时返回清晰错误 |
| `_get_backend()` 硬编码 firecrawl 默认值，metaso 无法自动回退 | Medium | Medium | Unit 1 已包含修复：将 line 107 改为动态返回首个可用后端 |
| `mcpm` CLI 不在 PATH 中导致 MCP 工具发现失败 | Medium | Medium | MCP 路径是增强功能，失败不影响 native backend；`discover_mcp_tools()` 已有错误处理；需在文档中注明 `mcpm` 安装前提 |
| metaso API 速率限制 | Medium | Medium | 通过 httpx timeout 配置控制，与现有后端一致；后续可加请求缓存 |
| MCP stdio 多 session 并发冲突 | Low | High | `mcp_tool.py` 已有单例连接管理，复用同一 stdio 通道 |
| 中文内容在 ensure_ascii=False 时编码异常 | Low | Low | 现有 `json.dumps(..., ensure_ascii=False)` 已支持 Unicode，与其他后端一致 |

## Documentation / Operational Notes

- 更新 `.env.example` 和 `cli-config.yaml.example` 中的 metaso 配置示例
- 在 README 或 docs 中添加 metaso 集成的简短说明（如果 Hermes 有集成文档）
- 无需数据库迁移或回滚计划
- 无需监控或告警配置（个人使用场景）

## Sources & References

- **Origin document:** [docs/brainstorms/2026-04-09-metaso-web-search-integration-requirements.md](docs/brainstorms/2026-04-09-metaso-web-search-integration-requirements.md)
- **metaso API 参考:** `/Users/xuxukang/.agents/skills/metaso/references/api-reference.md`
- **现有后端模式:** `tools/web_tools.py` 中的 `_exa_search()`、`_tavily_request()`
- **MCP 工具发现:** `tools/mcp_tool.py` 中的 `discover_mcp_tools()`、`register_mcp_servers()`
- **Toolset 系统:** `toolsets.py` 中的 `TOOLSETS`、`resolve_toolset()`
- **配置系统:** `hermes_cli/config.py` 中的 `load_config()`、`OPTIONAL_ENV_VARS`
- **工具注册模式:** `tools/registry.py` 中的 `registry.register()`
- **工具开发指南:** `AGENTS.md` 中的 "Adding New Tools" 章节
