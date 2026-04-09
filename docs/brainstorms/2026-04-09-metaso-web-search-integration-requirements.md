---
date: 2026-04-09
topic: metaso-web-search-integration
---

# Metaso 搜索集成到 Hermes

## Problem Frame

Hermes 当前支持 4 个 Web 搜索后端（Exa、Firecrawl、Parallel、Tavily），但都需要海外 API Key。用户已有一个 metaso MCP 服务（通过 `mcpm run metaso` 运行），提供中文优化的搜索能力，但尚未接入 Hermes 的工具系统。

**影响范围：** 所有使用 `web_search` / `web_extract` 工具的 Hermes 平台（CLI、Telegram、Discord 等），以及需要通过 MCP 使用 metaso 全部 6 个工具（含专题检索、AI 问答）的场景。

## Requirements

**Native Backend 集成**

- R1. metaso 作为第 5 个 Web 搜索后端注册到 `web_tools.py`，与 Exa/Firecrawl/Parallel/Tavily 平级
- R2. 支持通过 `~/.hermes/config.yaml` 的 `web.backend: metaso` 切换，也支持自动回退检测（无其他 API Key 时自动选中 metaso）
- R3. `web_search` 工具调用 metaso API 时，输出格式与现有后端一致（`{success, data.web[]}` 统一结构），确保模型端无感知
- R4. `web_extract` 工具走 metaso reader API，将 URL 列表转为 markdown 内容

**MCP 工具集集成**

- R5. Hermes 启动时通过 `mcpm run metaso` 发现并注册 metaso 的全部 6 个 MCP 工具（web_search、web_reader、chat、topic_list、topic_search、topic_file_content）
- R6. MCP 工具独立 toolset（`mcp-metaso`），不影响现有 `web` toolset
- R7. `mcp-metaso` toolset 自动注入到 `hermes-cli` 及所有 messaging platform toolsets 中

**能力覆盖**

- R8. metaso 的 6 种搜索 scope（webpage、document、paper、image、video、podcast）均可通过 MCP 工具访问；web_search 默认使用 webpage scope，与现有后端行为一致
- R9. metaso chat（AI 问答）作为独立工具 `metaso_chat` 暴露，支持 fast/fast_thinking/ds-r1 三种模型
- R10. metaso topic 功能（专题检索）作为独立工具暴露，用于深度领域研究

## Success Criteria

- [ ] `hermes tools` 设置中能看到 metaso 选项
- [ ] 配置 `web.backend: metaso` 后 `web_search` 工具能正常返回搜索结果
- [ ] MCP 模式下 6 个 metaso 工具可在 Hermes 对话中调用
- [ ] 其他后端 API Key 存在时，metaso 不干扰现有优先级
- [ ] 无其他 API Key 时，metaso 自动成为默认后端

## Scope Boundaries

- 不修改 metaso MCP server 本身（`mcpm run metaso` 已可用）
- 不替换/删除现有 Exa/Firecrawl/Parallel/Tavily 后端
- 不改动 Hermes 的工具注册表核心逻辑，仅扩展
- API Key 通过环境变量 `METASO_API_KEY` 管理，与现有后端一致

## Key Decisions

**双路径集成**：同时支持 native backend（R1-R4）和 MCP 工具集（R5-R7）两种路径。Native backend 让 `web_search` 走 metaso 保持统一接口；MCP 工具集暴露 metaso 独有的高级功能（专题、AI 问答）。两者互补。用户明确选择此方案。

**输出格式归一化**：metaso API 返回结构与 Hermes 现有后端不同，需要在 `web_tools.py` 中做适配层，转换为 `{success: true, data.web: [...]}` 格式。

**MCP 工具自动发现**：利用 Hermes 已有的 `mcp_tool.py` 中的 `discover_mcp_tools()` 机制，在配置中声明 `mcpm run metaso` 作为 MCP server。

**API Key 管理**：通过环境变量 `METASO_API_KEY` 读取，与 Exa/Firecrawl 等保持一致的配置风格。

**默认 scope**：`web_search` 默认使用 webpage scope，与现有后端行为一致。image/video/podcast 等扩展 scope 通过 MCP 工具集暴露，不塞入 `web_search`。

**REST API 验证**：规划阶段需先确认 metaso REST API（`/api/v1/search`、`/api/v1/reader`）的请求/响应格式，可通过 metaso 脚本反推。

## Dependencies / Assumptions

- `mcpm` CLI 已安装在系统中（当前 Claude Code 已通过 mcpm 使用 metaso）
- metaso API key 通过环境变量 `METASO_API_KEY` 提供
- metaso API 的 `/api/v1/search` 和 `/api/v1/reader` 端点稳定可用

## Outstanding Questions

### Resolve Before Planning
（无 — 所有阻塞问题已解决）

### Deferred to Planning
- [Affects R5][技术] `mcpm run metaso` 作为 MCP server 的 stdio 通道在 Hermes 进程模型中如何管理？是每个 agent 实例独立启动还是复用？
- [Affects R4][技术] metaso reader API 返回格式与 Firecrawl/Exa extract 不同，是否需要 LLM 二次摘要（现有 web_extract 用 OpenRouter 做摘要）？
- [Affects R6][Needs research] Hermes 的 `hermes-cli` 等 toolset 如何注入 `mcp-metaso` toolset？用 `includes` 还是 `tools` 列表追加？

## Next Steps

→ /ce:plan 进行结构化实现规划
