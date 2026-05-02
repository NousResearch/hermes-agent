# Hermes CocoIndex Documents

This plugin indexes `~/Dokumente` with CocoIndex and exposes semantic search to
Hermes through an MCP stdio server.

For reproducible setup on another machine, read:

```text
installation.md
```

Runtime locations:

- CocoIndex state: `~/.local/state/hermes/cocoindex/cocoindex.db`
- SQLite vector store: `~/.local/share/hermes/cocoindex/documents.sqlite3`
- Source directory: `~/Dokumente`

Build or update the index once:

```bash
cd /home/tobi/skripte/hermes-agent
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db \
  /home/tobi/skripte/hermes-agent/venv/bin/cocoindex update \
  plugins/cocoindex_docs/pipeline.py -f
```

Run the live indexer:

```bash
cd /home/tobi/skripte/hermes-agent
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db \
  /home/tobi/skripte/hermes-agent/venv/bin/cocoindex update \
  plugins/cocoindex_docs/pipeline.py -f -L
```

Register the MCP server:

```bash
/home/tobi/.local/bin/hermes mcp add cocoindex_docs \
  --command /home/tobi/skripte/hermes-agent/venv/bin/python \
  --args /home/tobi/skripte/hermes-agent/plugins/cocoindex_docs/mcp_server.py
```

Restart Hermes Gateway after registering the MCP server:

```bash
systemctl --user restart hermes-gateway.service
```
