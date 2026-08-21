---
id: mcp.graphify
kind: object
universe: mcp
name: graphify
summary: MCP server exposing repo knowledge-graph tools.
aliases: [graphify]
tags: [mcp, graphify]
shape: object
path: optional-mcps/graphify/manifest.yaml
interface: [query_graph, get_node, get_neighbors, shortest_path, get_community, god_nodes, graph_stats]
depends_on: []
---

# graphify

MCP server exposing repo knowledge-graph tools.

## Location

`optional-mcps/graphify/manifest.yaml`

## Interface

- `query_graph`
- `get_node`
- `get_neighbors`
- `shortest_path`
- `get_community`
- `god_nodes`
- `graph_stats`

## Dependencies

None.
