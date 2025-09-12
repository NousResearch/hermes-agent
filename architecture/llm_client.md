# LLM Client

A quick wrapper over openai apis

## Responsibilities

- Transform "normal" chat/completions requests into graphs
- Translate graphs into LLM requests
- Keep a history of graphs parsed by it
  - On Policy Data
  - Deduplicating graphs, so we don't keep previous history as separate graphs

## How to use
Exactly the same as the openai api! Just with the additional support of graph inputs and outputs.