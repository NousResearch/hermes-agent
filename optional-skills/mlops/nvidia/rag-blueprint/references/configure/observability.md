# Observability

## When to Use
- User wants tracing, metrics, or monitoring for the RAG pipeline
- User asks about latency debugging, Zipkin, Grafana, or Prometheus

## Process
1. Detect the deployment mode. Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read `docs/observability.md` for full setup (Docker and Helm)
3. Set `OPENTELEMETRY_CONFIG_FILE` and `APP_TRACING_ENABLED=True` in the active config
4. Start observability stack and restart RAG server
5. Import Grafana dashboard from `deploy/config/rag-metrics-dashboard.json`

## Agent-Specific Notes
- Library mode: set `OPENTELEMETRY_CONFIG_FILE` in the environment for tracing; the Docker-based Prometheus/Grafana stack is independent
- Helm: Prometheus Operator CRDs must be installed before deploying with observability enabled
- Default Grafana credentials: `admin` / `admin`
- Zipkin spans cover: `query-rewriter`, `retriever`, `context-reranker`, `llm-stream`
- Span I/O visible via `traceloop.entity.input` / `traceloop.entity.output` fields

### Quick Latency Triage
| Symptom | Check |
|---------|-------|
| Slow first token | `rag_ttft_ms` — compare retriever and reranker spans |
| Slow full response | `llm_generation_time_ms` / `llm-stream` span |
| Retrieval heavy | Compare `retrieval_time_ms` vs `context_reranker_time_ms` |

## Source Documentation
- `docs/observability.md` -- full Docker/Helm setup, env vars, metrics reference, and dashboard import
