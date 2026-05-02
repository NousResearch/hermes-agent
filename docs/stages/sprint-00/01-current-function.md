# Sprint 00 — 当前功能

## 1. 会话式 AI Agent

- **交互式 CLI** (`cli.py`, 11,483 行)：基于 prompt_toolkit 的全功能终端界面，支持 / 命令、REPL、文件拖放、流式输出。
- **Gateway 多平台接入** (`gateway/`)：支持 Telegram、Discord、Slack、飞书、钉钉、微信、QQ、WhatsApp、Matrix、Signal、iMessage、短信、邮件等多平台统一接入。
- **单次批量执行** (`batch_runner.py`)：支持非交互式批量任务。

## 2. 多模型适配

- **模型适配器** (`agent/`)：anthropic_adapter、bedrock_adapter、gemini_native_adapter、gemini_cloudcode_adapter、codex_responses_adapter、lmstudio_reasoning 等。
- **模型路由**：config.yaml 支持多 provider（deepseek、anthropic、kimi、z-ai、minimax、volcano-ark 等），模型别名机制。
- **Fallback 机制**：支持 provider 级别 fallback（429/503/529 错误自动切换）。

## 3. Skill 系统

- **Skill 目录**（24 个分类目录，如 software-development、research、creative 等）。
- **SKILL.md 格式**：YAML frontmatter（name、description、version、platforms、permissions 等）+ Markdown 正文。
- **Skill 加载** (`agent/skill_utils.py`、`agent/skill_commands.py`)：frontmatter 解析、平台过滤、模板变量替换、inline shell 执行。
- **Skill 工具** (`tools/skills_tool.py`、`tools/skill_manager_tool.py`)：列表、查看、安装、同步。

## 4. 工具系统

- **工具注册表** (`tools/registry.py`)：中央注册机制，工具模块自注册。
- **60+ 内置工具**：包括文件操作、终端命令、浏览器控制、Web 搜索、代码执行、图片生成、TTS/STT、飞书文档/网盘/多维表格等。
- **MCP 工具** (`tools/mcp_tool.py`)：MCP 协议集成，config.yaml 中配置了 minimax 和 openchronicle MCP server。
- **Delegate 工具** (`tools/delegate_tool.py`)：子 Agent 生成，支持隔离上下文、受限工具集。

## 5. 记忆系统

- **文件记忆** (`tools/memory_tool.py`)：MEMORY.md（agent 笔记）+ USER.md（用户偏好），使用 § 分隔条目。
- **记忆管理器** (`agent/memory_manager.py`)：支持内置 + 外部插件 provider。
- **Hindsight provider**：自动从对话中提取记忆（config 中 `memory.provider: hindsight`）。

## 6. 会话与状态

- **SQLite 状态存储** (`hermes_state.py`)：sessions 表 + messages 表 + FTS5 全文搜索。
- **JSONL 会话文件** (`~/.hermes/sessions/`)：每会话独立 JSONL 文件（522 个历史会话）。
- **会话搜索** (`tools/session_search_tool.py`)：FTS5 搜索 + Gemini Flash 摘要。

## 7. 任务管理

- **Todo 工具** (`tools/todo_tool.py`)：内存中的任务列表，支持 pending/in_progress/completed/cancelled 状态。
- **无持久化 Task Card**：当前没有结构化的任务卡片、没有任务状态机、没有验收标准字段。

## 8. 调度与自动化

- **Cron 系统** (`cron/`)：定时任务调度器，支持 cron 表达式。
- **Checkpoint 系统**：会话快照（用于长会话恢复）。

## 9. 其他能力

- **上下文压缩** (`agent/context_compressor.py`、`trajectory_compressor.py`)：自动压缩长对话。
- **Prompt Caching**：支持 Anthropic prompt cache。
- **镜像/声音**：TTS（Edge TTS / ElevenLabs）、STT（faster-whisper）、语音模式。
- **安全**：Tirith 安全检查、文件安全策略、路径安全检查、内存威胁扫描。
