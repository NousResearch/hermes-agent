---
name: graphify
description: "any input (code, docs, papers, images) - knowledge graph - clustered communities - HTML + JSON + audit report"
version: 1.0.0
author: Jason Colapietro
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [knowledge-graph, research, codebase, graphrag, visualization, neo4j, obsidian]
    related_skills: [gitnexus-explorer, domain-intel, code-wiki]
    category: research
    homepage: https://github.com/safishamsi/graphifyy
---

# /graphify

Turn any folder of files into a navigable knowledge graph with community detection, an honest audit trail, and three outputs: interactive HTML, GraphRAG-ready JSON, and a plain-language GRAPH_REPORT.md.

## When to Use

graphify is built around Andrej Karpathy's /raw folder workflow: drop anything into a folder — papers, tweets, screenshots, code, notes — and get a structured knowledge graph that shows you what you didn't know was connected.

Three things it does that Claude alone cannot:
1. **Persistent graph** — relationships stored in `graphify-out/graph.json` survive across sessions.
2. **Honest audit trail** — every edge is tagged EXTRACTED, INFERRED, or AMBIGUOUS.
3. **Cross-document surprise** — community detection finds connections you would never think to ask about.

Use it for:
- A codebase you're new to (understand architecture before touching anything)
- A reading list (papers + tweets + notes → one navigable graph)
- A research corpus (citation graph + concept graph in one)
- Your personal /raw folder (drop everything in, let it grow, query it)

## Usage

```
/graphify                                             # full pipeline on current directory
/graphify <path>                                      # full pipeline on specific path
/graphify https://github.com/<owner>/<repo>           # clone repo then run full pipeline
/graphify https://github.com/<owner>/<repo> --branch <branch>
/graphify <url1> <url2> ...                           # multi-repo cross-graph
/graphify <path> --mode deep                          # thorough extraction, richer INFERRED edges
/graphify <path> --update                             # incremental - re-extract only new/changed files
/graphify <path> --directed                           # directed graph (preserves source→target)
/graphify <path> --whisper-model medium               # larger Whisper model for transcription
/graphify <path> --cluster-only                       # rerun clustering on existing graph
/graphify <path> --no-viz                             # skip visualization, just report + JSON
/graphify <path> --svg                                # also export graph.svg
/graphify <path> --graphml                            # export graph.graphml (Gephi, yEd)
/graphify <path> --neo4j                              # generate graphify-out/cypher.txt
/graphify <path> --neo4j-push bolt://localhost:7687   # push directly to Neo4j
/graphify <path> --mcp                                # start MCP stdio server for agent access
/graphify <path> --watch                              # watch folder, auto-rebuild on code changes
/graphify <path> --wiki                               # agent-crawlable wiki (index.md + articles)
/graphify <path> --obsidian --obsidian-dir ~/vaults/my-project
/graphify add <url>                                   # fetch URL, save to ./raw, update graph
/graphify add <url> --author "Name"
/graphify query "<question>"                          # BFS traversal - broad context
/graphify query "<question>" --dfs                    # DFS - trace a specific path
/graphify query "<question>" --budget 1500            # cap answer at N tokens
/graphify path "AuthModule" "Database"                # shortest path between two concepts
/graphify explain "SwinTransformer"                   # plain-language explanation of a node
```

## Pipeline

When invoked, run these steps in order. Full implementation (all bash/Python blocks) in [`references/pipeline.md`](references/pipeline.md).

| Step | What it does |
|------|-------------|
| **0 — Clone** | GitHub URL only — clone to `~/.graphify/repos/<owner>/<repo>` |
| **1 — Install** | Detect or install graphify; write interpreter to `graphify-out/.graphify_python` |
| **2 — Detect** | Scan files; print corpus summary; warn if >200 files or >2M words |
| **2.5 — Transcribe** | Video/audio only — Whisper transcription before extraction |
| **3 — Extract** | AST (code) + parallel semantic subagents (docs/papers/images) dispatched in one message |
| **4 — Build + cluster** | Community detection, cohesion scoring, god nodes, surprises |
| **5 — Label** | Write 2-5 word names per community; regenerate report |
| **6 — Visualize** | HTML always (unless `--no-viz`); Obsidian only if `--obsidian` |
| **6b — Wiki** | Only if `--wiki` — index.md + one article per community |
| **7/7b/7c/7d** | Neo4j, SVG, GraphML, MCP — only if those flags were given |
| **8 — Benchmark** | Token reduction stats — only if total_words > 5,000 |
| **9 — Finalize** | Save manifest, update cost tracker, clean temp files, report outputs |

After the pipeline, paste the God Nodes, Surprising Connections, and Suggested Questions sections from GRAPH_REPORT.md, then offer to trace the single most interesting suggested question.

## Subcommands

Full implementation for each in [`references/subcommands.md`](references/subcommands.md).

| Subcommand | What it does |
|---|---|
| `--update` | Incremental re-extraction; code-only changes skip LLM entirely |
| `--cluster-only` | Re-run clustering on existing graph; skip extraction |
| `query` | BFS/DFS traversal; answer from graph only; save result back into graph |
| `path` | Shortest path between two named concepts |
| `explain` | Plain-language explanation of a single node and its connections |
| `add` | Fetch URL → save to `./raw` → auto-run `--update` |
| `--watch` | Background watcher; code changes → instant rebuild; doc changes → flag |
| git hook | `graphify hook install` — post-commit AST rebuild, append-safe |
| CLAUDE.md | `graphify claude install` — wires graphify into every future session |

Before running any subcommand, check that `graphify-out/.graphify_python` exists. If missing, re-resolve the interpreter using the detection block in [`references/pipeline.md`](references/pipeline.md) Step 1.

## Honesty Rules

- Never invent an edge. If unsure, use AMBIGUOUS.
- Never skip the corpus check warning.
- Always show token cost in the report.
- Never hide cohesion scores behind symbols — show the raw number.
- Never run HTML viz on a graph with more than 5,000 nodes without warning the user.
