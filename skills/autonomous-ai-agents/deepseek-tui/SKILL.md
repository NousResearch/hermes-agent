---
name: deepseek-tui
description: "Delegate coding to DeepSeek TUI — terminal coding agent for DeepSeek V4 models (flash/pro) with agentic tool use, sub-agents, and 1M context. Covers both `deepseek` (CLI/binary) and `deepseek-tui` (interactive TUI)."
version: 2.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, DeepSeek, Terminal, Agentic, Sub-Agents]
    related_skills: [claude-code, codex, opencode, local-llm-servers]
---

# DeepSeek TUI — Hermes Orchestration Guide

Delegate coding tasks to [DeepSeek TUI](https://github.com/Hmbown/DeepSeek-TUI), a Rust-based terminal coding agent built around DeepSeek V4 models (flash/pro). Supports file ops, shell execution, git, web search, sub-agents, MCP servers, and 1M-token context.

## Binary Locations

```bash
/root/.hermes/node/bin/deepseek       # Full CLI (27 subcommands, agentic exec)
/root/.hermes/node/bin/deepseek-tui   # Interactive TUI
```

> **Note:** Ensure `/root/.hermes/node/bin` is in `$PATH`. Add `export PATH="/root/.hermes/node/bin:$PATH"` to `~/.bashrc` if `command not found`.

## When to Use

- User explicitly asks to use DeepSeek-TUI / deepseek CLI
- You want a DeepSeek-native coding agent (better reasoning with flash/pro models)
- Task benefits from DeepSeek's long context (1M tokens)
- Need concurrent sub-agent execution (up to 10-20 parallel)
- User has a DeepSeek API key but no Anthropic/OpenAI key

## Prerequisites

- **Install:** `npm install -g deepseek-tui` (or `cargo install deepseek-tui-cli deepseek-tui`)
- **China mirror:** `npm install -g deepseek-tui --registry=https://registry.npmmirror.com`
- **Verify:** `deepseek-tui --version` and `deepseek doctor`
- **Auth:** set `DEEPSEEK_API_KEY` env var, or run `deepseek login --api-key sk-xxx`, or configure `~/.deepseek/config.toml`
- The npm package is a wrapper that downloads prebuilt Rust binaries (`deepseek` dispatcher + `deepseek-tui` TUI)

## Two Orchestration Modes

### Mode 1: `deepseek exec` — Non-Interactive One-Shot (PREFERRED)

```bash
# Fully autonomous agent (YOLO = auto-approve all tool calls)
deepseek exec "Add error handling to all API calls" --auto --yolo

# JSON output for scripting
deepseek exec "List all Python files" --auto --json --yolo

# Continue last session
deepseek exec "Continue the refactoring work" --continue --auto

# With specific model
deepseek exec "Analyze this codebase" --auto --model deepseek-v4-flash --yolo
```

**When to use exec mode:**
- One-shot coding tasks (fix bug, add feature, refactor)
- Code review over a diff
- Structured extraction with `--json`
- Piped input: `cat error.log | deepseek exec "diagnose these errors" --auto --yolo`

### Mode 2: Interactive TUI — Multi-Turn Sessions

```bash
# Start interactive TUI
deepseek-tui

# Or with workspace
deepseek-tui -w /path/to/project
```

## Key Subcommands (27 total)

### Agent Execution (主力)

| Command | Purpose |
|---------|---------|
| `deepseek exec "prompt" --auto` | **非交互式 agent 模式，最常用** |
| `deepseek exec "prompt" --auto --yolo` | 全自动批准（YOLO 模式） |
| `deepseek exec "prompt" --auto --json` | JSON 结构化输出 |
| `deepseek exec "prompt" --continue --auto` | 继续最近会话 |
| `deepseek run` | 交互式/非交互式通用入口 |
| `deepseek review` | 对 git diff 进行 AI 代码审查 |
| `deepseek eval` | 离线 TUI 评估框架 |

### Session Management

| Command | Purpose |
|---------|---------|
| `deepseek thread list` | 列出所有 thread |
| `deepseek thread resume <id>` | 恢复指定 thread |
| `deepseek thread fork <id>` | 复刻指定 thread |
| `deepseek thread set-name <id> "名称"` | 重命名 thread |
| `deepseek thread archive <id>` | 归档 thread |
| `deepseek sessions` | 列出已保存的 TUI 会话 |
| `deepseek resume` | 恢复已保存的会话 |
| `deepseek fork` | 复刻已保存的会话 |

### Configuration & Auth

| Command | Purpose |
|---------|---------|
| `deepseek config list` | 列出所有配置 |
| `deepseek config path` | 显示配置文件路径 |
| `deepseek config set <key> <value>` | 设置配置值 |
| `deepseek config get <key>` | 获取配置值 |
| `deepseek auth status` | 认证状态 |
| `deepseek auth list` | 所有 provider |
| `deepseek auth set --api-key sk-xxx` | 保存 API key |
| `deepseek login --api-key sk-xxx` | 保存到用户配置 |
| `deepseek logout` | 删除认证状态 |

### Model Management

| Command | Purpose |
|---------|---------|
| `deepseek model list` | 列出可用模型 |
| `deepseek model resolve` | 当前使用的模型 |

### Server Modes

| Command | Purpose |
|---------|---------|
| `deepseek app-server --port 8787` | HTTP API 服务器 |
| `deepseek serve` | 本地 TUI 服务器 |
| `deepseek mcp-server` | MCP over stdio |
| `deepseek mcp` | 管理 TUI MCP 服务器 |

### Diagnostics & Monitoring

| Command | Purpose |
|---------|---------|
| `deepseek doctor` | 环境诊断 |
| `deepseek metrics --since 7d` | 用量统计（最近7天） |
| `deepseek metrics --since 30d --json` | 用量 JSON 导出 |
| `deepseek sandbox` | 评估沙箱/审批策略决策 |
| `deepseek features` | 检查功能标志 |

### Utilities

| Command | Purpose |
|---------|---------|
| `deepseek update` | 更新二进制 |
| `deepseek completion bash` | 生成 shell 补全 |
| `deepseek init` | 创建默认 AGENTS.md |
| `deepseek setup` | 引导初始化配置 |
| `deepseek apply` | 应用 patch 文件到工作树 |

## Global Options (apply to all subcommands)

| Option | Description |
|--------|-------------|
| `--provider` | `deepseek`, `nvidia-nim`, `openai`, `openrouter`, `novita`, `fireworks`, `sglang`, `vllm`, `ollama` |
| `--model` | Model name |
| `--api-key` | API key (overrides config) |
| `--base-url` | Custom API base URL |
| `--yolo` | Auto-approve all tool calls |
| `--config` | Config file path |
| `--approval-policy` | Approval policy |
| `--sandbox-mode` | Sandbox mode |
| `--output-mode` | Output mode |
| `--log-level` | Log level |
| `--telemetry` | `true` / `false` |
| `--skip-onboarding` | Skip first-run onboarding |
| `-p, --prompt` | Pass prompt via argument |

## exec Flags

| Flag | Description |
|------|-------------|
| `--auto` | Enable agentic mode (tool access) |
| `--json` | JSON summary output |
| `--resume <ID>` | Resume session by ID |
| `--continue` | Continue most recent session |
| `--output-format` | `text` or `stream-json` |

## Common Workflows

### Non-interactive agent (primary CLI mode)
```bash
deepseek exec "analyze this project" --auto --yolo
deepseek exec "refactor module X" --auto --model deepseek-v4-flash --yolo
```

### Pipe input
```bash
cat error.log | deepseek exec "diagnose these errors" --auto --yolo
git diff HEAD~1 | deepseek exec "review this diff" --auto --json --yolo
```

### Multi-provider
```bash
deepseek exec "..." --auto --provider openai --model gpt-4.1 --api-key sk-xxx
deepseek exec "..." --auto --provider ollama --model deepseek-coder:1.3b
```

### Session continuity
```bash
deepseek exec "start task A" --auto
deepseek exec "continue task A" --continue --auto
deepseek thread resume <id>
```

## Config File

- **Path:** `~/.deepseek/config.toml` (run `deepseek config path` to confirm)
- **Auth lookup order:** config → secret store → env var (`DEEPSEEK_API_KEY`)

```toml
api_key = "sk-xxx"
provider = "deepseek"
default_text_model = "deepseek-v4-pro"
reasoning_effort = "max"

[projects."/root"]
trust_level = "trusted"
```

## Sub-Agents (Concurrent Background Execution)

DeepSeek TUI can dispatch multiple sub-agents that run in parallel:

- Non-blocking launch: parent keeps working while children execute
- Default cap: 10 concurrent sub-agents (configurable to 20)
- Completion notification: structured `<deepseek:subagent.done>` event with summary
- Bounded result retrieval: large transcripts parked behind `var_handle` references

In agentic mode (`--auto`), the model can autonomously spawn sub-agents for parallel work.

## Feature Flags

| Flag | Stage | Default | Description |
|------|-------|---------|-------------|
| `shell_tool` | stable | ✅ | Shell execution in agentic mode |
| `subagents` | experimental | ✅ | Concurrent background sub-agents |
| `web_search` | experimental | ✅ | Web search/browse tools |
| `apply_patch` | experimental | ✅ | Patch application |
| `mcp` | experimental | ✅ | MCP server support |
| `exec_policy` | experimental | ✅ | Execution policy tooling |
| `vision_model` | experimental | ❌ | Vision model support |

Enable/disable via `--enable <FEATURE>` or `--disable <FEATURE>` on any command.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `DEEPSEEK_API_KEY` | API key (overrides config file) |
| `DEEPSEEK_CORS_ORIGINS` | Comma-separated CORS origins for HTTP server |
| `DEEPSEEK_RUNTIME_TOKEN` | Bearer token for runtime API auth |

## Cost & Performance Tips

1. **Use `--model deepseek-v4-flash`** for most tasks (fast, cheap). Reserve `--model deepseek-v4-pro` for complex multi-step reasoning.
2. **Use `--yolo` for one-shots** — avoids interactive approval dialogs.
3. **Use `--json`** for programmatic result parsing.
4. **Use pipe input** — `cat file | deepseek exec "..." --auto --yolo` instead of pasting content.
5. **Use `--continue`** to resume long-running sessions.
6. **Sub-agents for parallel work** — the model can spawn up to 20 concurrent children.

## Pitfalls & Gotchas

1. **npm wrapper downloads binaries on first run** — the `deepseek` dispatcher (~17MB) downloads immediately; `deepseek-tui` TUI (~47MB) downloads on first TUI/exec invocation. Total ~64MB.
2. **WSL2 GitHub download speeds can be very slow** (~100KB/s). Use `--registry=https://registry.npmmirror.com` for npm, but binary downloads still go through GitHub. The built-in downloader handles retries well — just set a long timeout (300s+).
3. **`exec` subcommand requires a prompt argument** — `deepseek exec "task"` not `deepseek "task"`.
4. **API key via env var or login** — `export DEEPSEEK_API_KEY="sk-..."` works; `deepseek login` persists it to user config.
5. **Auto mode enables tool access** — without `--auto`, exec mode is read-only analysis.
6. **Chinese locale auto-detected** — UI defaults to `zh-Hans` if system locale is Chinese.
7. **PATH issue** — If `deepseek: command not found`, add `/root/.hermes/node/bin` to `$PATH` in `~/.bashrc`.

## Advanced Commands

### HTTP Server Mode (Headless Orchestration)

```bash
# Start server in background
deepseek app-server --port 8787

# Or use deepseek-tui serve
deepseek-tui serve --http --port 7878 --insecure
```

### MCP Management

```bash
deepseek mcp list                    # List configured servers
deepseek mcp add <name> -- <cmd>     # Add server
deepseek mcp connect                 # Test connections
deepseek mcp tools                   # List discovered tools
deepseek mcp enable/disable <name>   # Toggle server
deepseek mcp add-self                # Register as MCP stdio server
```

## Rules for Hermes Agents

1. **Prefer `deepseek exec --auto --yolo` for single tasks** — clean, non-interactive, auto-approved.
2. **Always set `workdir`** — keep DeepSeek focused on the right project directory.
3. **Set long timeouts (120-300s)** — binary download on first run and model inference both take time.
4. **Use `--model deepseek-v4-flash` by default** — best price/performance ratio.
5. **Use `--json` for structured output** — easier to parse results programmatically.
6. **Check `deepseek doctor` before running** — verify config and API connectivity if issues arise.
7. **Report concrete outcomes** — files changed, tests run, remaining risks.
8. **Use `--continue` or `deepseek thread resume`** — don't start new sessions for follow-up work.

## Full Reference

For complete documentation of all 27 subcommands, detailed options, environment variables, and additional workflow examples, see `references/cli-reference.md`.
