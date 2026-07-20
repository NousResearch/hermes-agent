# Pixeltable MCP Integration for Hermes

## Hermes config.yaml Entry

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  pixeltable:
    command: uvx
    args: [mcp-server-pixeltable-developer]
```

If `uvx` is not available, install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional environment variables:

```yaml
mcp_servers:
  pixeltable:
    command: uvx
    args: [mcp-server-pixeltable-developer]
    env:
      PIXELTABLE_HOME: /path/to/custom/home
      OPENAI_API_KEY: ${OPENAI_API_KEY}
```

After adding, restart Hermes. Tools appear under `native-mcp` with the `mcp_pixeltable_` prefix
(Hermes names MCP tools `mcp_<server>_<tool>`).

## Tool Catalog (32 tools)

### Table Operations

| Tool | What it does |
|---|---|
| `mcp_pixeltable_create_table` | Create a table with typed columns |
| `mcp_pixeltable_drop_table` | Delete a table and its data |
| `mcp_pixeltable_create_view` | Create a view with an iterator (e.g., DocumentSplitter, FrameIterator) |
| `mcp_pixeltable_create_snapshot` | Create an immutable snapshot of a table |

### Data Operations

| Tool | What it does |
|---|---|
| `mcp_pixeltable_insert_data` | Insert rows into a table |
| `mcp_pixeltable_query_table` | Query a table with optional filters |
| `mcp_pixeltable_query` | Run a general query expression |
| `mcp_pixeltable_create_replica` | Replicate a table from a shared source |

### Column Operations

| Tool | What it does |
|---|---|
| `mcp_pixeltable_add_computed_column` | Add a column that auto-computes on insert |
| `mcp_pixeltable_create_udf` | Create a user-defined function |
| `mcp_pixeltable_create_array` | Create an array value for array-typed columns |
| `mcp_pixeltable_create_type` | Create a column type (Image, Video, Audio, etc.) |
| `mcp_pixeltable_create_tools` | Register LLM tool-calling functions |
| `mcp_pixeltable_connect_mcp` | Import tools from another MCP server |

### Directory Operations

| Tool | What it does |
|---|---|
| `mcp_pixeltable_create_dir` | Create a namespace directory |
| `mcp_pixeltable_drop_dir` | Delete a directory and its contents |
| `mcp_pixeltable_move` | Move/rename a table or directory |

### Dependency Management

| Tool | What it does |
|---|---|
| `mcp_pixeltable_check_dependencies` | Check if packages needed for a function are installed |
| `mcp_pixeltable_install_dependency` | Install a missing Python dependency |
| `mcp_pixeltable_install_package` | Install an arbitrary pip package |

### REPL and Introspection

| Tool | What it does |
|---|---|
| `mcp_pixeltable_execute_python` | Run arbitrary Python in a persistent Pixeltable REPL |
| `mcp_pixeltable_introspect_function` | Get signature and docs for a Pixeltable function |
| `mcp_pixeltable_list_available_functions` | List all registered functions by module |

### Configuration

| Tool | What it does |
|---|---|
| `mcp_pixeltable_configure_logging` | Set logging verbosity |
| `mcp_pixeltable_set_datastore` | Configure the backing datastore |
| `mcp_pixeltable_search_docs` | Search Pixeltable documentation |

### Display and Logging

| Tool | What it does |
|---|---|
| `mcp_pixeltable_display_in_browser` | Render table data in a browser canvas |
| `mcp_pixeltable_log_bug` | Log a bug for the current session |
| `mcp_pixeltable_log_missing_feature` | Log a feature request |
| `mcp_pixeltable_log_success` | Log a successful operation |
| `mcp_pixeltable_generate_bug_report` | Generate a structured bug report |
| `mcp_pixeltable_get_session_summary` | Get a summary of the current session |

## Example: Using MCP Tools in Hermes

Once configured, the agent can call tools directly via `native-mcp`:

```
User: Create a table for my research papers with title, abstract, and PDF columns

Agent uses: mcp_pixeltable_create_dir(path="research")
Agent uses: mcp_pixeltable_create_table(
    path="research.papers",
    schema={"title": "String", "abstract": "String", "pdf": "Document"}
)

User: Add my paper about transformers

Agent uses: mcp_pixeltable_insert_data(
    table_path="research.papers",
    data=[{"title": "Attention Is All You Need", "abstract": "...", "pdf": "/path/to/paper.pdf"}]
)

User: Search for papers about attention mechanisms

Agent uses: mcp_pixeltable_query_table(
    table_path="research.papers",
    filter="title.contains('attention')"
)
```

For operations not covered by MCP tools, use `mcp_pixeltable_execute_python` to run
arbitrary Pixeltable Python code in the persistent REPL.
