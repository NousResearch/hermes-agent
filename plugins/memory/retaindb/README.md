# RetainDB Memory Provider

Cloud memory API with hybrid search (Vector + BM25 + Reranking) and 7 memory types.

## Requirements

- RetainDB account ($20/month) from [retaindb.com](https://www.retaindb.com)
- `pip install requests`

## Setup

```bash
hermes memory setup    # select "retaindb"
```

Or manually:
```bash
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

All config via environment variables in `.env`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `RETAINDB_API_KEY` | (required) | API key |
| `RETAINDB_BASE_URL` | `https://api.retaindb.com` | API endpoint |
| `RETAINDB_PROJECT` | auto (profile-scoped) | Project identifier |
| `RETAINDB_ENABLE_LOCAL_FILE_UPLOADS` | disabled | Set to `true` to expose local file upload tools. |
| `RETAINDB_FILE_UPLOAD_ROOTS` | current working directory | `:`-separated trusted directories allowed for local uploads. |

## Tools

| Tool | Description |
|------|-------------|
| `retaindb_profile` | User's stable profile |
| `retaindb_search` | Semantic search |
| `retaindb_context` | Task-relevant context |
| `retaindb_remember` | Store a fact with type + importance |
| `retaindb_forget` | Delete a memory by ID |
| `retaindb_list_files` | List RetainDB-stored files |
| `retaindb_read_file` | Read text content from a RetainDB-stored file |
| `retaindb_ingest_file` | Ingest a RetainDB-stored file into memory |
| `retaindb_delete_file` | Delete a RetainDB-stored file |
| `retaindb_upload_file` | Upload a local file; hidden unless `RETAINDB_ENABLE_LOCAL_FILE_UPLOADS=true` |

Local uploads reject symlinks, sensitive-looking paths, files outside `RETAINDB_FILE_UPLOAD_ROOTS`, and files over 10 MiB.
