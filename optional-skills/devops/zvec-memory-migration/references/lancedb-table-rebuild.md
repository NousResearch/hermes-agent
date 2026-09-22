# LanceDB Table Rebuild — Fixing "Table not found" After an Upgrade

**Date:** 2026-06-01
**Trigger:** after a lancedb library upgrade, an old profile's `memories.lance` reports `ValueError: Table 'memories' was not found`.
**Affects:** `.lance` single files written by an old lancedb (<0.10, estimated 0.3.x) cannot be opened via `open_table` after upgrading to `0.30.x`.

## Failure Symptoms

```
ValueError: Table 'memories' was not found
```

**Actual cause**: the data is not lost; the table's manifest metadata directories (`_versions/`, `_transactions/`, `_deletions/`, `_indices/`) are missing or malformed, so the new lancedb cannot find the manifest and refuses to open the table.

**Diagnostic steps**:
```bash
# 1. List the subdirectories inside memories.lance/
ls ~/.hermes/profiles/<name>/lance_memory/memories.lance/

# Normal (default profile, after upgrade):
#   _deletions  _indices  _transactions  _versions  data

# Broken (fe/<profile>):
#   data   ← only data

# 2. Check whether data/ contains .lance files
ls ~/.hermes/profiles/<name>/lance_memory/memories.lance/data/ | head -5
# 000000000001000001100100a773bf44949948be9edaa4ee72.lance
# 000000011101001010111110968c1d48e58cf2605612ae50fd.lance
# ...

# 3. Each .lance file = 1 memory record
# 4. First 4 bytes of each file = 0x24 (LANCEDB magic) → the file format itself is fine
```

## Repair Procedure

### Step 1: Backup

```bash
TS=$(date +%Y%m%d_%H%M%S)
cp -a ~/.hermes/profiles/<prof>/lance_memory \
      ~/.hermes/backups/lance_memory_<prof>_pre-fix_$TS
```

### Step 2: Read out all data with lance.file.LanceFileReader

**Key API**: `lance.file.LanceFileReader(path).read_all().to_table()` reads a single .lance file without depending on the `_versions/` directory.

```python
import lance.file as lf
import pyarrow as pa
from pathlib import Path

data_dir = Path.home() / ".hermes/profiles/<prof>/lance_memory/memories.lance/data"

all_tables = []
for f in sorted(data_dir.iterdir()):
    if not f.name.endswith(".lance"):
        continue
    try:
        tbl = lf.LanceFileReader(str(f)).read_all().to_table()
        if tbl.num_rows > 0:
            all_tables.append(tbl)
    except Exception as e:
        print(f"ERR {f.name}: {e}")

combined = pa.concat_tables(all_tables)
# combined is a pyarrow Table; schema matches default
```

### Step 3: Rebuild the table

```python
import lancedb
import pyarrow as pa
import shutil

profile_dir = Path.home() / ".hermes/profiles/<prof>"
lance_dir = profile_dir / "lance_memory"
memories_dir = lance_dir / "memories.lance"

# 1. Delete the old table directory
shutil.rmtree(memories_dir)

# 2. Rebuild the table
db = lancedb.connect(str(lance_dir))
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("content", pa.string()),
    pa.field("role", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    pa.field("created_at", pa.float64()),
    pa.field("metadata", pa.string()),
])
rows = combined.to_pylist()
tbl = db.create_table("memories", data=rows, schema=schema, mode="overwrite")

# 3. Re-add the FTS index (so hybrid search works)
tbl.create_fts_index("content", replace=True)
```

### Step 4: Verify

```python
db = lancedb.connect(str(lance_dir))
tbl = db.open_table("memories")
assert tbl.count_rows() == expected_count
assert tbl.list_indices()  # FTS index present
```

## Verified Repair Cases (2026-06-01)

| Profile | rows before fix | rows after fix | time |
|---|---|---|---|
| <profile> | 0 (Table not found) | 325 | 0.42s |
| <profile> | 0 (Table not found) | 237 | 0.10s |

## Pitfalls

### 1. Do not use `lance.dataset(path)` to read a single file

```python
# ❌ Wrong: lance.dataset() expects a dataset directory, not a single file
ds = lance.dataset(str(file))  
# → ValueError: LanceError(IO): Generic LocalFileSystem error: Unable to walk dir

# ✅ Correct: use lance.file.LanceFileReader to read a single file
reader = lance.file.LanceFileReader(str(file))
tbl = reader.read_all().to_table()
```

### 2. `read_all()` does not return a pyarrow Table

`LanceFileReader.read_all()` returns `ReaderResults`; convert with `.to_table()` first:

```python
# ❌ reader_results.num_rows errors
# ✅ reader_results.to_table().num_rows
```

### 3. Do not pass a `num_rows` argument to `read_all()`

API design: `read_all()` reads everything; `read_range(start=, num_rows=)` reads a range.

### 4. Install pylance

`lance.file` lives in the `pylance` package, not lancedb:

```bash
uv pip install --python ~/.hermes/venv/bin/python pylance
```

### 5. Vector search verification

```python
import requests, numpy as np
emb = np.array(
    requests.post("http://localhost:11434/api/embed",
                  json={"model": "bge-m3:latest", "input": ["test"]}).json()["embeddings"][0],
    dtype=np.float32)
results = tbl.search(emb).limit(3).to_list()
```

## Automation Script

See `scripts/lancedb_rebuild_table.py` (same directory) — one command to scan all profiles, detect "Table not found" failures, and rebuild automatically.
