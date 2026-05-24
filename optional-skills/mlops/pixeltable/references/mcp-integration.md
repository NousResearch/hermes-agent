# Pixeltable MCP Integration for Hermes

## Hermes config.yaml Entry

Add to `~/.hermes/config.yaml`:

```yaml
mcpServers:
  pixeltable:
    command: uvx
    args: [mcp-server-pixeltable-developer]
```

If `uvx` is not available, install with:

```bash
pip install mcp-server-pixeltable-developer
```

Then use:

```yaml
mcpServers:
  pixeltable:
    command: mcp-server-pixeltable-developer
```

Optional environment variables:

```yaml
mcpServers:
  pixeltable:
    command: uvx
    args: [mcp-server-pixeltable-developer]
    env:
      PIXELTABLE_HOME: /path/to/custom/home
      OPENAI_API_KEY: ${OPENAI_API_KEY}
```

After adding, restart Hermes. Tools appear under `native-mcp` with the `pixeltable_` prefix.

## Tool Catalog (32 tools)

### Table Operations

| Tool | What it does |
|---|---|
| `pixeltable_create_table` | Create a table with typed columns |
| `pixeltable_drop_table` | Delete a table and its data |
| `pixeltable_create_view` | Create a view with an iterator (e.g., DocumentSplitter, FrameIterator) |
| `pixeltable_create_snapshot` | Create an immutable snapshot of a table |

### Data Operations

| Tool | What it does |
|---|---|
| `pixeltable_insert_data` | Insert rows into a table |
| `pixeltable_query_table` | Query a table with optional filters |
| `pixeltable_query` | Run a general query expression |
| `pixeltable_create_replica` | Replicate a table from a shared source |

### Column Operations

| Tool | What it does |
|---|---|
| `pixeltable_add_computed_column` | Add a column that auto-computes on insert |
| `pixeltable_create_udf` | Create a user-defined function |
| `pixeltable_create_array` | Create an array value for array-typed columns |
| `pixeltable_create_type` | Create a column type (Image, Video, Audio, etc.) |
| `pixeltable_create_tools` | Register LLM tool-calling functions |
| `pixeltable_connect_mcp` | Import tools from another MCP server |

### Directory Operations

| Tool | What it does |
|---|---|
| `pixeltable_create_dir` | Create a namespace directory |
| `pixeltable_drop_dir` | Delete a directory and its contents |
| `pixeltable_move` | Move/rename a table or directory |

### Dependency Management

| Tool | What it does |
|---|---|
| `pixeltable_check_dependencies` | Check if packages needed for a function are installed |
| `pixeltable_install_dependency` | Install a missing Python dependency |
| `pixeltable_install_package` | Install an arbitrary pip package |

### REPL and Introspection

| Tool | What it does |
|---|---|
| `pixeltable_execute_python` | Run arbitrary Python in a persistent Pixeltable REPL |
| `pixeltable_introspect_function` | Get signature and docs for a Pixeltable function |
| `pixeltable_list_available_functions` | List all registered functions by module |

### Configuration

| Tool | What it does |
|---|---|
| `pixeltable_configure_logging` | Set logging verbosity |
| `pixeltable_set_datastore` | Configure the backing datastore |
| `pixeltable_search_docs` | Search Pixeltable documentation |

### Display and Logging

| Tool | What it does |
|---|---|
| `pixeltable_display_in_browser` | Render table data in a browser canvas |
| `pixeltable_log_bug` | Log a bug for the current session |
| `pixeltable_log_missing_feature` | Log a feature request |
| `pixeltable_log_success` | Log a successful operation |
| `pixeltable_generate_bug_report` | Generate a structured bug report |
| `pixeltable_get_session_summary` | Get a summary of the current session |

## Example: Using MCP Tools in Hermes

Once configured, the agent can call tools directly via `native-mcp`:

```
User: Create a table for my research papers with title, abstract, and PDF columns

Agent uses: pixeltable_create_dir(path="research")
Agent uses: pixeltable_create_table(
    path="research.papers",
    schema={"title": "String", "abstract": "String", "pdf": "Document"}
)

User: Add my paper about transformers

Agent uses: pixeltable_insert_data(
    table_path="research.papers",
    data=[{"title": "Attention Is All You Need", "abstract": "...", "pdf": "/path/to/paper.pdf"}]
)

User: Search for papers about attention mechanisms

Agent uses: pixeltable_query_table(
    table_path="research.papers",
    filter="title.contains('attention')"
)
```

For operations not covered by MCP tools, use `pixeltable_execute_python` to run
arbitrary Pixeltable Python code in the persistent REPL.
