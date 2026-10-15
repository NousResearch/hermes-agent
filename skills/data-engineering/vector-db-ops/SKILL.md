---
name: vector-db-ops
description: "Unified vector database operations for Qdrant, Pinecone, Weaviate, Milvus, Chroma — collection CRUD, vector upsert, ANN search, reranking, backup/restore."
version: "0.1.0"
author: "kenya"
license: "MIT"
platforms: [linux, macos, windows]
category: "data-engineering"
tags: [vector, database, qdrant, pinecone, weaviate, milvus, chroma, rag, embedding]
depends_on: []
compatibility:
  hermes: ">=0.18.0"
  claude-code: ">=1.0.0"
  codex: ">=1.0.0"
  opencode: ">=0.5.0"
maturity: "beta"
homepage: "https://github.com/NousResearch/hermes-agent"
repository: "https://github.com/NousResearch/hermes-agent"
---

# Vector DB Operations

> Unified vector database operations for Qdrant, Pinecone, Weaviate, Milvus, Chroma — collection CRUD, vector upsert, ANN search, reranking, backup/restore.

## Prerequisites
- Python 3.10+ with appropriate client packages
- Access to target vector database
- API keys/credentials for managed services

## Installation
```bash
hermes skill install vector-db-ops
# Or manual
pip install qdrant-client pinecone-client weaviate-client pymilvus chromadb
```

## Configuration
| Environment Variable | Required | Description | Example |
|----------------------|----------|-------------|---------|
| `VECTOR_DB_TYPE` | Yes | Database type: qdrant, pinecone, weaviate, milvus, chroma | `qdrant` |
| `VECTOR_DB_URL` | Yes | Connection URL/host | `http://localhost:6333` |
| `VECTOR_DB_API_KEY` | For managed | API key for managed services | `pclocal-xxx` |
| `VECTOR_DB_COLLECTION` | No | Default collection name | `documents` |

## Usage
### vector_collection
Collection management - create, list, describe, delete.

```bash
# Create collection
hermes skill run vector-db-ops vector_collection --action create --name my_docs --dimension 1024 --distance cosine

# List collections
hermes skill run vector-db-ops vector_collection --action list

# Describe collection
hermes skill run vector-db-ops vector_collection --action describe --name my_docs

# Delete collection
hermes skill run vector-db-ops vector_collection --action delete --name my_docs
```

### vector_upsert
Insert or update vectors with payloads.

```bash
# Upsert from JSONL file (each line: {"id": "...", "vector": [...], "payload": {...}})
hermes skill run vector-db-ops vector_upsert --collection my_docs --file vectors.jsonl

# Upsert single vector
hermes skill run vector-db-ops vector_upsert --collection my_docs --id doc1 --vector '[0.1,0.2,...]' --payload '{"title":"Test"}'
```

### vector_search
ANN search with optional filtering and reranking.

```bash
# Search by vector
hermes skill run vector-db-ops vector_search --collection my_docs --vector '[0.1,0.2,...]' --limit 10

# Search with filter
hermes skill run vector-db-ops vector_search --collection my_docs --vector '[0.1,0.2,...]' --filter '{"category": "tech"}' --limit 5
```

### vector_backup / vector_restore
Backup and restore collections.

```bash
# Backup to directory
hermes skill run vector-db-ops vector_backup --collection my_docs --output ./backup

# Restore from directory
hermes skill run vector-db-ops vector_restore --collection my_docs --input ./backup
```

## API / Tools
| Tool | Description | Parameters |
|------|-------------|------------|
| `vector_collection` | Collection CRUD | `action: enum[create,list,describe,delete], name: str, dimension: int, distance: enum[cosine,euclid,dot]` |
| `vector_upsert` | Upsert vectors | `collection: str, file: str, id: str, vector: list, payload: dict` |
| `vector_search` | ANN search | `collection: str, vector: list, limit: int, filter: dict, with_payload: bool, with_vectors: bool` |
| `vector_backup` | Backup collection | `collection: str, output: str` |
| `vector_restore` | Restore collection | `collection: str, input: str` |

## Examples
```bash
# Create Qdrant collection for 1024-dim embeddings
export VECTOR_DB_TYPE=qdrant
export VECTOR_DB_URL=http://localhost:6333
hermes skill run vector-db-ops vector_collection --action create --name knowledge_base --dimension 1024 --distance cosine

# Bulk upsert from embedding pipeline output
hermes skill run vector-db-ops vector_upsert --collection knowledge_base --file embeddings.jsonl

# Semantic search
hermes skill run vector-db-ops vector_search --collection knowledge_base --vector '[0.12,-0.34,...]' --limit 5 --with-payload
```

## Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| `Connection refused` | DB not running | Start service or check URL/port |
| `API key invalid` | Wrong credentials | Check VECTOR_DB_API_KEY |
| `Collection not found` | Typo in name | List collections first |
| `Dimension mismatch` | Wrong vector size | Ensure embedding model matches collection dimension |

## Changelog
### v0.1.0 (2026-08-15)
- Initial release with Qdrant, Pinecone, Weaviate, Milvus, Chroma support
- Collection CRUD, upsert, search, backup/restore
- JSON/Table output formats