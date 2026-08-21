---
id: runtime.graph.build
kind: process
universe: runtime
name: Graphify Build
summary: Build the repo knowledge graph from source.
aliases: [graphify build]
tags: [runtime, graphify]
shape: process
steps:
  - id: step.index
    summary: Index repo symbols and edges.
  - id: step.persist
    summary: Write graph.json via GraphJsonRepository.
entrypoints: [step.index]
produces: [mcp:graphify]
consumes: []
---

# Graphify Build

Build the repo knowledge graph from source.

## Steps

1. **step.index**: Index repo symbols and edges.
2. **step.persist**: Write graph.json via GraphJsonRepository.

## Entrypoints

- `step.index`

## Artifacts

- `mcp:graphify`
