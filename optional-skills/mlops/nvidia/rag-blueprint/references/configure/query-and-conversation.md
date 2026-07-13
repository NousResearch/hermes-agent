```markdown
# Query Rewriting, Query Decomposition, and Multi-Turn

Use these features when you want the system to understand follow-up questions, rewrite queries for better retrieval, or break complex questions into smaller parts.

## When to use

Use these settings when:

- You want to enable multi-turn conversations or support follow-up questions.
- You want query rewriting to improve retrieval accuracy.
- You need complex multi-hop query decomposition.
- You are configuring or debugging conversation history behavior.[file:1]

## Restrictions

- Query rewriting and multi-turn both require `CONVERSATION_HISTORY > 0`. If it is set to 0, query rewriting has no effect.[file:1]
- Query decomposition works only when `use_knowledge_base=true` and with a single collection.[file:1]
- On Helm, query rewriting is supported only with an on-prem LLM, not with cloud-hosted models.[file:1]

## Dependencies

`CONVERSATION_HISTORY` is shared by query rewriting and multi-turn, so changing it affects both behaviors.

| Setting                 | Depends on                 | Side effect when changed                                  |
|-------------------------|----------------------------|-----------------------------------------------------------|
| `ENABLE_QUERYREWRITER`  | `CONVERSATION_HISTORY > 0` | Enabling requires conversation history; disabling has no side effects |
| `CONVERSATION_HISTORY`  | —                          | Setting to `0` also effectively disables query rewriting  |[file:1]

## Process

First detect the deployment mode.  
- Docker: edit the active environment file.  
- Helm: edit `values.yaml`.  
- Library: edit `notebooks/config.yaml`.[file:1]

### Query rewriting

1. Review `docs/multiturn.md` for full configuration details.[file:1]
2. To enable, set `ENABLE_QUERYREWRITER=True`. If `CONVERSATION_HISTORY` is `0`, set it to `5` or another positive value.[file:1]
3. To disable, unset or comment out `ENABLE_QUERYREWRITER`.[file:1]
4. Restart the RAG server.[file:1]

### Multi-turn

1. Review `docs/multiturn.md` for configuration, retrieval strategies, and API usage.[file:1]
2. To enable, set `CONVERSATION_HISTORY > 0` and choose the retrieval strategy you want to use.[file:1]
3. To disable, set `CONVERSATION_HISTORY=0`.[file:1]
4. Restart the RAG server.[file:1]

### Query decomposition

1. Review `docs/query_decomposition.md` for the decomposition algorithm, limitations, and examples.[file:1]
2. Set `ENABLE_QUERY_DECOMPOSITION=true` and `MAX_RECURSION_DEPTH=3` (or a different depth that fits your use case).[file:1]
3. Restart the RAG server.[file:1]

## Decision table

| Goal                          | Source doc                 | Key settings                                              |
|-------------------------------|----------------------------|-----------------------------------------------------------|
| Multi-turn with best accuracy | `docs/multiturn.md`        | `CONVERSATION_HISTORY=5`, `ENABLE_QUERYREWRITER=True`    |
| Multi-turn with low latency   | `docs/multiturn.md`        | `CONVERSATION_HISTORY=5`, `MULTITURN_RETRIEVER_SIMPLE=True` |
| Complex multi-hop queries     | `docs/query_decomposition.md` | `ENABLE_QUERY_DECOMPOSITION=true`, `MAX_RECURSION_DEPTH=3` |
| Disable multi-turn (default)  | —                          | `CONVERSATION_HISTORY=0`                                 |[file:1]

## Agent-specific notes

- `MULTITURN_RETRIEVER_SIMPLE` only applies when query rewriting is disabled. If both are configured, query rewriting takes precedence.[file:1]
- You can toggle query rewriting per request by setting `enable_query_rewriting: true` in `POST /generate`, but `CONVERSATION_HISTORY` must still be greater than 0.[file:1]
- By default, multi-turn is disabled with `CONVERSATION_HISTORY=0`.[file:1]
- Query decomposition adds latency and is most useful for multi-hop queries that involve multiple entities or steps.[file:1]
- In library mode, configure these settings in `notebooks/config.yaml` instead of using environment variables.[file:1]

## Notebooks

- `notebooks/retriever_api_usage.ipynb`: RAG retriever API usage with search and end-to-end query examples.[file:1]

## Source documentation

- `docs/query_decomposition.md`: Decomposition algorithm details, when to use it, and recursion depth guidance.[file:1]
- `docs/multiturn.md`: Conversation history behavior, retrieval strategies, API usage, and Helm configuration.[file:1]
```
