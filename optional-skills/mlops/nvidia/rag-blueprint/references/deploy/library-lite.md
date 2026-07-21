# Library Mode (Lite / Containerless)

## When to Use
- Quick prototyping with zero infrastructure (no Docker, no GPU)
- User wants the fastest path to try RAG
- CI/CD pipelines needing lightweight RAG testing

## Restrictions
- No image/table/chart citations
- No document summarization
- Subject to NVIDIA API rate limits (cloud-hosted inference)
- Requires Python 3.11+ (< 3.14), internet access, and `NGC_API_KEY`

## Process
1. Read `docs/python-client.md` for full library mode documentation
2. Create virtualenv and install: `pip install nvidia-rag[all]`
3. Ensure `NGC_API_KEY` is exported — maps to `NVIDIA_API_KEY` internally
4. Run the lite notebook: `jupyter lab notebooks/rag_library_lite_usage.ipynb`

## Agent-Specific Notes
- `NVIDIA_API_KEY` (used by `nvidia_rag` package) must be set from `NGC_API_KEY`: `os.environ["NVIDIA_API_KEY"] = os.environ.get("NGC_API_KEY", "")`
- Lite config lives in `notebooks/config.yaml`; override `server_url` for embeddings to the NVIDIA API Catalog endpoint (see `docs/python-client.md` for current URL), and set LLM/ranking URLs to empty string for cloud defaults
- Milvus Lite runs embedded (no container), NV-Ingest runs as subprocess (no container)
- Also install `python-dotenv jupyterlab` for notebook support

## When Not to Use
- Production workloads — use Docker or Kubernetes
- Large-scale ingestion — rate limits apply
- Need citations from images/tables/charts or document summarization

## Notebooks
| Notebook | Description |
|----------|-------------|
| `notebooks/rag_library_lite_usage.ipynb` | End-to-end lite mode: collection creation, ingestion, querying, search |

## Source Documentation
- `docs/python-client.md` -- full library mode documentation (lite and full)
