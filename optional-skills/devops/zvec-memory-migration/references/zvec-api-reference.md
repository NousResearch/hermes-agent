# Zvec Python API Quick Reference

## Collection Operations

```python
import zvec

# Open existing
coll = zvec.open("./my_data")

# Create + open
coll = zvec.create_and_open("./my_data", schema)

# Schema
schema = zvec.CollectionSchema(
    name="my_collection",
    fields=[
        zvec.FieldSchema("text", zvec.DataType.STRING,
            index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"])),
        zvec.FieldSchema("category", zvec.DataType.STRING,
            index_param=zvec.InvertIndexParam()),
    ],
    vectors=[
        zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 1024,
            index_param=zvec.HnswIndexParam(
                metric_type=zvec.MetricType.COSINE, m=16, ef_construction=100)),
    ],
)
```

## CRUD

```python
# Insert
doc = zvec.Doc(id="1", vectors={"embedding": vec}, fields={"text": "hello", "category": "test"})
coll.insert(doc)
coll.flush()

# Query (vector)
results = coll.query(zvec.Query(field_name="embedding", vector=vec), topk=10)

# Query (FTS)
results = coll.query(zvec.Query(field_name="text", fts=zvec.Fts(match_string="keyword")), topk=10)

# Hybrid (MultiQuery + RRF)
results = coll.query(
    queries=[
        zvec.Query(field_name="embedding", vector=vec),
        zvec.Query(field_name="text", fts=zvec.Fts(match_string="keyword")),
    ],
    reranker=zvec.RrfReRanker(rank_constant=60),
    topk=10,
    filter="category = 'test'",
)

# Delete
coll.delete(ids=["1"])

# Optimize
coll.optimize()
```

## Index Management

```python
# Create index on existing field
coll.create_index("category", zvec.InvertIndexParam(enable_range_optimization=True))
coll.create_index("text", zvec.FtsIndexParam(tokenizer_name="jieba"))

# Drop index
coll.drop_index("category")

# Stats
stats = coll.stats  # {"doc_count": N, "index_completeness": {"vector": 0.0-1.0}}
```
