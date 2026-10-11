# Financial Business Data: LanceDB → Zvec Migration

Migration plan for replacing `lancedb_news/` and `lancedb_analysis/` with Zvec collections
in the <profile> profile.

> **Scope**: News articles, telegraph flashes, and analysis reports — NOT session memory
> (that's covered by the main migration playbook).

## Source Data Inventory

| Collection | Path | Table(s) | Rows | Purpose |
|-----------|------|----------|------|---------|
| `lancedb_news` | `金融数据库/lancedb_news/` | `financial_news` | ~174 | Deep articles (sentiment, industry tags, market impact) |
| | | `financial_telegraph` | ~104 | Short telegraph flashes |
| `lancedb_analysis` | `金融数据库/lancedb_analysis/` | `analysis_reports` | ~10 | Analysis report semantic retrieval |

**Embedding model**: bge-m3:latest (1024-dim) — vectors can be **reused directly** (same model).

## Target Layout

```
金融数据库/zvec_news/                    ← replaces lancedb_news/
  financial_news/                        ← Zvec collection
  financial_telegraph/                    ← Zvec collection

金融数据库/zvec_research/                ← replaces lancedb_analysis/
  analysis_reports/                      ← Zvec collection
```

## Call Graph (All lance_client Consumers)

### Via lance_client.py (unified import)

| File | Path (relative to skills/) | Key Imports |
|------|---------------------------|-------------|
| pipeline_core.py | news-emotion-pipeline/scripts/ | `connect, upsert_rows, read_table` |
| pipeline_briefing.py | news-emotion-pipeline/scripts/ | `connect, read_table` |
| analysis_db.py | strategy-analysis/scripts/ | `connect, embed_text, search_similar, create_table, open_table, count_rows, read_all_tables` |
| sentiment_strategy.py | strategy-analysis/scripts/ | `connect, read_table, read_all_tables, count_rows` |

### Direct `import lancedb` (bypassing lance_client)

| File | Path (relative to skills/) | Pattern |
|------|---------------------------|---------|
| strategy_engine.py | auto-trading-robot/scripts/ | `lancedb.connect()` → `open_table().to_pandas()` |
| report_generator.py | auto-trading-robot/scripts/ | same |
| dashboard.py | auto-trading-robot/scripts/ | same |

### Config / docs

| File | Path | Content |
|------|------|---------|
| config.py | auto-trading-robot/scripts/ | `LANCEDB_MANAGER_DIR`, `LANCEDB_DIR` constants |
| financial-data-hub SKILL.md | finance/financial-data-hub/ | Architecture diagram references |

## lance_client.py → zvec_client.py Function Mapping

| lance_client Function | Zvec Equivalent | Notes |
|----------------------|----------------|-------|
| `connect(path)` | `zvec.open(collection_path)` | One collection per call, not a DB handle |
| `open_table(db, name)` | N/A (collection = table) | Connect directly to collection |
| `create_table(db, name, schema)` | `zvec.create_and_open(path, schema)` | Path must NOT exist |
| `ensure_table(db, name, schema)` | open → fallback create_and_open | 4-layer fallback pattern |
| `drop_table(db, name)` | `collection.destroy()` | Permanent, no undo |
| `list_tables(db)` | `os.listdir(parent_dir)` | One dir per collection |
| `count_rows(db, name)` | `collection.stats.doc_count` | Built-in stat |
| `upsert_rows(db, name, df, ...)` | `collection.upsert(docs)` | **50 lines → 1 call**. Zvec native upsert by id |
| `read_table(db, name)` | Query all → assemble DataFrame | Need to iterate results |
| `read_all_tables(db, names)` | Loop per collection | Zvec has no cross-collection query |
| `query_recent(db, name, hours)` | `filter="publish_time >= X"` | **InvertIndexParam enables index-level pre-filter** |
| `search_similar(db, name, ...)` | `collection.query(Query(field="vector", vector=...), topk=k)` | Direct ANN search |
| `search_keyword(db, name, ...)` | `collection.query(Query(field="content", fts=Fts(match_string=...)), topk=k)` | RocksDB-native FTS with jieba tokenizer |
| `search_hybrid(db, name, ...)` | `collection.query(queries=[vec_q, fts_q], reranker=RrfReRanker())` | **50 lines hand-written RRF → 3 lines native** |
| `embed_text(text)` | Unchanged | Same Ollama bge-m3:latest call |
| `embed_batch(texts)` | Unchanged | Same Ollama batch embed |
| `create_fts_index(db, name, col)` | Declared in schema `FtsIndexParam(tokenizer_name="jieba")` | Schema-level, not post-hoc |
| `health_check(db, tables)` | `collection.stats` | Built-in |

## New Functions (No LanceDB Equivalent)

```python
def fetch_by_ids(db, table_name, ids):
    """Exact ID lookup — no search/scoring needed."""

def update_doc(db, table_name, doc_id, fields_dict):
    """Partial field update without rebuilding."""

def delete_by_filter(db, table_name, filter_expr):
    """Condition-based bulk delete."""

def stats(db, table_name):
    """collection.stats: doc_count + index_completeness."""

def optimize(db, table_name):
    """Merge flat buffer → HNSW graph."""

def add_field(db, table_name, field_schema, default_value=None):
    """Dynamic schema evolution without migration."""
```

## Schema Designs

### financial_news / financial_telegraph (shared schema)

```python
zvec.CollectionSchema(
    name="financial_news",  # or "financial_telegraph"
    fields=[
        zvec.FieldSchema("id", zvec.DataType.STRING),
        zvec.FieldSchema("publish_time", zvec.DataType.INT64,
            index_param=zvec.InvertIndexParam(enable_range_optimization=True)),
        zvec.FieldSchema("source", zvec.DataType.STRING),
        zvec.FieldSchema("title", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("summary", zvec.DataType.STRING),
        zvec.FieldSchema("url", zvec.DataType.STRING),
        zvec.FieldSchema("sentiment_label", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("language", zvec.DataType.STRING),
        zvec.FieldSchema("type", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("is_important", zvec.DataType.BOOL,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("market_impact", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("market_relevance", zvec.DataType.FLOAT,
            index_param=zvec.InvertIndexParam(enable_range_optimization=True)),
        zvec.FieldSchema("content", zvec.DataType.STRING,
            index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"])),
        # Native array types — NO JSON serialization needed
        zvec.FieldSchema("industry_tags", zvec.DataType.ARRAY_STRING),
        zvec.FieldSchema("emotion_scores", zvec.DataType.ARRAY_FLOAT),
        zvec.FieldSchema("related_symbols", zvec.DataType.ARRAY_STRING),
    ],
    vectors=[
        zvec.VectorSchema("vector", zvec.DataType.VECTOR_FP32, dimension=1024,
            index_param=zvec.HnswIndexParam(
                metric_type=zvec.MetricType.COSINE, m=16, ef_construction=100)),
    ],
)
```

### analysis_reports

```python
zvec.CollectionSchema(
    name="analysis_reports",
    fields=[
        zvec.FieldSchema("id", zvec.DataType.STRING),
        zvec.FieldSchema("report_date", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("title", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("content", zvec.DataType.STRING,
            index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"])),
    ],
    vectors=[
        zvec.VectorSchema("vector", zvec.DataType.VECTOR_FP32, dimension=1024,
            index_param=zvec.HnswIndexParam(
                metric_type=zvec.MetricType.COSINE, m=16)),
    ],
)
```

## Zvec Capabilities Beyond LanceDB

### Completely New (LanceDB has nothing)

| # | Feature | Financial Use Case |
|---|----------|-------------------|
| 1 | `upsert()` native by-id | Replace 50-line drop_recreate merge |
| 2 | `update()` partial fields | Change sentiment_label without full rebuild |
| 3 | `delete_by_filter()` | Purge old news by date |
| 4 | `fetch(ids=...)` exact lookup | Get specific article by ID |
| 5 | `InvertIndexParam` scalar index | Index-level pre-filter vs post-ANN scan |
| 6 | `enable_range_optimization=True` | Fast date range queries |
| 7 | Native `RrfReRanker` | Hybrid in 3 lines vs 50 |
| 8 | `collection.optimize()` | HNSW index quality control |
| 9 | `collection.flush()` | Explicit durability |
| 10 | `collection.destroy()` | Clean removal |
| 11 | Dynamic schema (`add_column`) | Add fields without migration |
| 12 | `ARRAY_STRING` / `ARRAY_FLOAT` | Native arrays, no JSON serialization |
| 13 | `collection.stats` | Built-in health metrics |
| 14 | `CollectionOption(read_only=True)` | Safe cross-process reads |

### Zvec Does Better

| Aspect | LanceDB | Zvec |
|--------|---------|------|
| FTS Chinese | Tantivy bigram CJK | RocksDB FTS + jieba (word-level) |
| Hybrid search | App-layer 2-pass RRF (~50 LOC) | Native MultiQuery + RrfReRanker (3 LOC) |
| Scalar filter | Post-ANN scan | Index-level pre-filter |
| Write idempotency | No guarantee | `upsert()` native |

## Migration Phases

### Phase 1: Build Foundation
- Create `zvec-manager` skill (replaces `lancedb-manager`)
- Implement `zvec_client.py` with all lance_client signatures + new functions
- Create Zvec collection directories

### Phase 2: Migrate Data
- Write migration script (reuse `lancedb` python package to read, `zvec` to write)
- Vectors reuse directly (same bge-m3:latest model, 1024-dim)
- `ARRAY_STRING`/`ARRAY_FLOAT` fields: deserialize from pandas readback
- Verify row counts match
- Run `collection.optimize()` for HNSW index
- Verify FTS with Chinese keyword test

### Phase 3: Switch Callers
- Change `from lance_client import ...` → `from zvec_client import ...` in 4 pipeline files
- Change direct `import lancedb` → `from zvec_client import read_table` in 3 auto-trading files
- Update config.py constants
- Update financial-data-hub SKILL.md

### Phase 4: Verify & Cleanup
- Run cron jobs (news-emotion-pipeline, strategy-analysis)
- Compare search results (hybrid/keyword/similar)
- Backup old lancedb data
- Delete lancedb-manager skill

## Risk Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| Zvec lock conflict during migration | Script can't write | Stop gateway or migrate during idle |
| `query_recent` full-scan pattern | Slower than LanceDB equivalent | Use InvertIndexParam on publish_time + filter= |
| List type field conversion | Data loss if deserialize fails | Validate before insert, log skips |
| auto-trading-robot bypasses abstraction | Harder to switch | Unify through zvec_client |
