# MCP Server & Client

## When to Use
- User wants to expose RAG APIs as MCP tools for agentic workflows
- User asks about MCP transport modes, NeMo Agent Toolkit integration, or ReAct agents

## Process
1. Read `docs/mcp.md` for full MCP server/client setup and configuration
2. Choose transport mode: `sse`, `streamable_http`, or `stdio`
3. Run MCP server from `examples/nvidia_rag_mcp/mcp_server.py`
4. For agentic RAG, see ReAct agent example in `examples/rag_react_agent/`

## Agent-Specific Notes
- MCP wraps both RAG tools (`generate`, `search`, `get_summary`) and Ingestor tools (`create_collection`, `upload_documents`, etc.) via FastMCP
- `stdio` transport does not require a running server — client spawns it directly
- ReAct agent requires: Python 3.11+, `NVIDIA_API_KEY`, and data already ingested into Milvus
- Configure Milvus endpoint in `examples/rag_react_agent/src/rag_react_agent/configs/config.yml` or via `APP_VECTORSTORE_URL`

## Notebooks
| Notebook | Description |
|----------|-------------|
| `notebooks/mcp_server_usage.ipynb` | End-to-end MCP workflow: collection creation, upload, RAG queries |
| `notebooks/nat_mcp_integration.ipynb` | NeMo Agent Toolkit integration with RAG MCP server |

## Source Documentation
- `docs/mcp.md` -- full MCP server/client documentation and transport configuration
