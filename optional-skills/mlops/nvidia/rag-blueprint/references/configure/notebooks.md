# Notebooks

## When to Use
- Hands-on examples of NVIDIA RAG Blueprint features are needed
- There are questions about Jupyter notebooks, tutorials, or code samples

## Process
1. Read `docs/notebooks.md` for full notebook descriptions and prerequisites.
2. Set up the environment: virtualenv, `jupyterlab`, and `git lfs pull` for test data.
3. Open JupyterLab at `http://<server-ip>:8889`.

## Agent-Specific Notes
- Git LFS is required because several notebooks rely on large data files (`git lfs install && git lfs pull`).
- In Docker mode, deploy NVIDIA RAG Blueprint first, then run notebooks against the running services.
- In library mode, use `rag_library_usage.ipynb` (full) or `rag_library_lite_usage.ipynb` (containerless).
- The custom VDB operator notebook requires Docker for OpenSearch services.

## Notebook Catalog

### Beginner
| Notebook                    | Topic                               |
|-----------------------------|-------------------------------------|
| `ingestion_api_usage.ipynb` | Document ingestion through the API  |
| `retriever_api_usage.ipynb` | Search and retrieval API            |
| `image_input.ipynb`         | Image upload and multimodal queries |

### Intermediate
| Notebook                       | Topic                                  |
|--------------------------------|----------------------------------------|
| `summarization.ipynb`          | Document summarization strategies      |
| `evaluation_01_ragas.ipynb`    | RAGAS accuracy, relevancy, groundedness|
| `evaluation_02_recall.ipynb`   | Recall at top-k cutoffs                |
| `nb_metadata.ipynb`            | Custom metadata and filtered retrieval |
| `rag_library_usage.ipynb`      | Full library mode end-to-end           |
| `rag_library_lite_usage.ipynb` | Lite, containerless library mode       |

### Advanced
| Notebook                          | Topic                               |
|-----------------------------------|-------------------------------------|
| `building_rag_vdb_operator.ipynb` | Custom OpenSearch VDB operator      |
| `mcp_server_usage.ipynb`          | MCP server with transport modes     |
| `nat_mcp_integration.ipynb`       | NeMo Agent Toolkit plus MCP         |

### Deployment
| Notebook           | Topic                 |
|--------------------|-----------------------|
| `launchable.ipynb` | Brev cloud deployment |

## Source Documentation
- `docs/notebooks.md` — full notebook descriptions, setup, and prerequisites.
