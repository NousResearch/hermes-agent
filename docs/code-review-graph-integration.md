# code-review-graph + claude-token-efficient Integration Guide

> Install and configure code-review-graph and claude-token-efficient rules for Hermes Agent code-task pipelines.

## Overview

This guide covers integrating two complementary tools into the Hermes Agent pipeline:

| Tool | Purpose | Type |
|------|---------|------|
| **code-review-graph** | Structural code knowledge graph — reduces token usage by giving the agent precise context instead of scanning entire codebases | MCP server (Python) |
| **claude-token-efficient** | Prompt rules for concise, direct agent output | CLAUDE.md rules file |

Both are **local-first**, **no mandatory cloud dependencies**, and integrate via Hermes Agent's native MCP client.

---

## Part 1: Install code-review-graph

### Prerequisites

- Python 3.10+
- `pip` or `uv`
- Node.js (optional, for tree-sitter WASM binaries)

### Step 1: Install the package

```bash
# Standard install (includes core dependencies)
pip install code-review-graph

# With optional embeddings (for semantic search)
pip install "code-review-graph[embeddings]"

# With Google Gemini embeddings
pip install "code-review-graph[google-embeddings]"

# With community detection (igraph)
pip install "code-review-graph[communities]"
```

### Step 2: Verify installation

```bash
code-review-graph --version
# Should output: 2.3.3 (or current version)
```

### Step 3: Configure as Hermes Agent MCP server

Add the server to `~/.hermes/config.yaml` under `mcp_servers`:

```yaml
mcp_servers:
  code-review-graph:
    command: "code-review-graph"
    args: ["mcp"]
    timeout: 120
    connect_timeout: 60
```

**Restart Hermes Agent** after adding this config. On startup, Hermes will:
1. Launch the MCP server as a subprocess
2. Discover the 27 available tools
3. Register them with prefix `mcp_code_review_graph_`

### Step 4: Build the graph for a project

In any project directory:

```bash
# Full build (first time)
code-review-graph build

# Incremental build (after changes)
code-review-graph update

# Status check
code-review-graph status
```

The graph is stored in `.code-review-graph/graph.db` (SQLite) within the project root.

### Available MCP Tools (after restart)

Tools are registered as `mcp_code_review_graph_<tool_name>`. Key tools:

| Tool | Purpose |
|------|---------|
| `mcp_code_review_graph_build_or_update_graph` | Build or incrementally update the code graph |
| `mcp_code_review_graph_get_impact_radius` | Get blast radius of changed files |
| `mcp_code_review_graph_query_graph` | Query graph relationships (callers, callees, imports, tests) |
| `mcp_code_review_graph_get_review_context` | Get focused subgraph for code review |
| `mcp_code_review_graph_semantic_search_nodes` | Keyword + vector search across code nodes |
| `mcp_code_review_graph_detect_changes` | Risk-scored change impact analysis |
| `mcp_code_review_graph_get_affected_flows` | Find execution flows impacted by changes |
| `mcp_code_review_graph_list_communities` | List detected code communities |
| `mcp_code_review_graph_get_architecture_overview` | High-level architecture from community structure |
| `mcp_code_review_graph_refactor_tool` | Unified refactoring (rename preview, dead code) |

---

## Part 2: Install claude-token-efficient Rules

### What it is

A single `CLAUDE.md` file containing 7 prompt rules that make the agent respond more concisely. No code, no dependencies, no installation script.

### Step 1: Add the rules

Create or update `CLAUDE.md` in your project root:

```markdown
## Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.
```

### Step 2: (Optional) Global rules

For project-wide application, place the same rules in `~/.claude/CLAUDE.md` (or the Hermes Agent equivalent global config location).

---

## Part 3: Pipeline Integration

### How code-review-graph reduces token usage

When Hermes Agent works on code tasks in a project with an active code-review-graph:

1. **Before any code exploration**: The agent uses `mcp_code_review_graph_query_graph` or `mcp_code_review_graph_semantic_search_nodes` instead of raw file scanning
2. **For code review**: The agent uses `mcp_code_review_graph_get_review_context` to get only the relevant files and blast radius, not the entire codebase
3. **For change impact**: The agent uses `mcp_code_review_graph_get_impact_radius` to find affected files instead of manual grep
4. **For refactoring**: The agent uses `mcp_code_review_graph_refactor_tool` for safe rename/dead-code operations

### How claude-token-efficient rules reduce token usage

The 6 rules in CLAUDE.md directly reduce output tokens:

| Rule | Token savings mechanism |
|------|------------------------|
| "Read existing files before writing" | Prevents redundant reads that waste context window |
| "Thorough in reasoning, concise in output" | Cuts verbose explanations (60-75% reduction in output) |
| "Skip files over 100KB unless required" | Prevents loading massive files into context |
| "No sycophantic openers or closing fluff" | Eliminates filler text ("Great question!", "Hope this helps!") |
| "No emojis or em-dashes" | Removes formatting noise |
| "Do not guess... Verify by reading" | Prevents hallucinated code that requires correction cycles |

### Combined effect

In a typical code review task on a medium project (~500 files):

- **Without integration**: Agent reads entire codebase → ~45K tokens input
- **With code-review-graph**: Agent reads only blast-radius files → ~4K tokens input (9x reduction)
- **With claude-token-efficient rules**: Agent output reduced by ~65% → ~30 tokens output per task

---

## Part 4: Embedding Configuration (Optional)

code-review-graph supports semantic search with optional embedding providers. Choose one:

### Option A: Local embeddings (no cloud)

```bash
pip install "code-review-graph[embeddings]"
```

Uses `sentence-transformers` (downloads models on first use). No API keys needed.

### Option B: OpenAI embeddings

```bash
pip install "code-review-graph[embeddings]"
```

Set in MCP config:
```yaml
mcp_servers:
  code-review-graph:
    command: "code-review-graph"
    args: ["mcp"]
    env:
      OPENAI_API_KEY: "sk-..."
      EMBEDDING_PROVIDER: "openai"
      EMBEDDING_MODEL: "text-embedding-3-small"
```

### Option C: Google Gemini embeddings

```bash
pip install "code-review-graph[google-embeddings]"
```

Set in MCP config:
```yaml
mcp_servers:
  code-review-graph:
    command: "code-review-graph"
    args: ["mcp"]
    env:
      GOOGLE_API_KEY: "..."
      EMBEDDING_PROVIDER: "google"
      EMBEDDING_MODEL: "gemini-embedding-001"
```

### Option D: Ollama (self-hosted)

```yaml
mcp_servers:
  code-review-graph:
    command: "code-review-graph"
    args: ["mcp"]
    env:
      EMBEDDING_PROVIDER: "openai"  # OpenAI-compatible API mode
      OPENAI_BASE_URL: "http://localhost:11434/v1"
      OPENAI_API_KEY: "ollama"
      EMBEDDING_MODEL: "nomic-embed-text"
```

---

## Part 5: Verification

### Verify MCP server is connected

Check Hermes Agent startup logs:
```
[INFO] MCP server 'code-review-graph' connected successfully
[INFO] Discovered 27 tools from code-review-graph
```

### Verify tools are available

The agent should see tools prefixed with `mcp_code_review_graph_` in its toolset.

### Verify graph is working

```bash
# Check graph stats
code-review-graph stats

# Should show file count, node count, edge count
```

### Verify token savings

Compare token usage before and after:
- Without graph: agent scans all files with `terminal(grep)` or `read_file`
- With graph: agent uses `mcp_code_review_graph_get_review_context` or `mcp_code_review_graph_query_graph`

---

## Part 6: Maintenance

### Updating code-review-graph

```bash
pip install --upgrade code-review-graph
# After update, rebuild graphs in affected projects
code-review-graph build
```

### Rebuilding a project graph

```bash
cd /path/to/project
code-review-graph build --force  # full rebuild
```

### Clearing a project graph

```bash
cd /path/to/project
rm -rf .code-review-graph/
```

### Monitoring MCP server health

Check Hermes Agent logs for connection errors. If the server fails to start:
1. Verify `code-review-graph` is on PATH: `which code-review-graph`
2. Check Python version: `python3 --version` (needs 3.10+)
3. Test manually: `code-review-graph mcp` (should output MCP JSON-RPC messages)

---

## Troubleshooting

### "Failed to connect to MCP server 'code-review-graph'"

- Verify installation: `code-review-graph --version`
- Check PATH: `which code-review-graph`
- Test MCP mode: `code-review-graph mcp` (should not error)
- Verify Python 3.10+: `python3 --version`

### "No tools discovered from code-review-graph"

- Server may have crashed during startup. Check logs.
- Try manual start: `code-review-graph mcp` — if it errors, fix the error first.
- Restart Hermes Agent after fixing.

### Graph not updating after file changes

- Run `code-review-graph update` manually
- Check that the project has a `.git` directory (required for change detection)
- Verify the graph exists: `ls .code-review-graph/graph.db`

### Embedding errors

- Check API key is set correctly in MCP config env
- For local embeddings, ensure `sentence-transformers` is installed: `pip install sentence-transformers`
- Check network connectivity if using cloud embeddings

---

## Security Notes

- **code-review-graph** is local-first. Embedding providers are **optional** — the core graph works without any external service.
- Embedding calls go to the configured provider (OpenAI, Google, MiniMax, or local). No data is sent to code-review-graph's authors.
- **claude-token-efficient** has no code — it is purely a text file of prompt rules. Zero attack surface.
- Neither tool collects telemetry or sends data externally by default.
