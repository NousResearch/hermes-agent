---
name: pixeltable
description: Multimodal data tables with auto-computed AI columns.
version: 1.0.0
author: Pixeltable (pixeltable.com)
license: Apache-2.0
prerequisites:
  commands: [python3, pip]
metadata:
  hermes:
    tags: [mlops, multimodal, rag, vector-search, data-pipeline]
    related_skills: [chroma, qdrant-vector-search, qmd]
    requires_toolsets: [terminal]
---

# Pixeltable

Pixeltable is a Python library for multimodal data infrastructure. Tables store images, video, audio, and documents as native column types. Computed columns declare AI transformations (embedding, transcription, classification) that execute automatically when data is inserted. It is not a vector database — it is a declarative data layer that includes vector search.

## When to Use

- User wants to process video frames, audio segments, images, or documents through AI models
- User needs a RAG pipeline over multimodal content (not just text)
- User wants computed columns that auto-run embeddings, transcriptions, or LLM calls on insert
- User needs similarity search across images, text, or audio
- User wants to chain multiple AI transformations declaratively (e.g., video → frames → captions → embeddings)
- User asks about incremental data pipelines that skip already-processed rows

Do NOT use for plain text vector search — `chroma`, `qdrant-vector-search`, or `qmd` are simpler for that.

## Prerequisites

Install Pixeltable via `terminal`:

```bash
pip install pixeltable
```

Verify the install:

```bash
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"
```

Pixeltable auto-starts an embedded PostgreSQL instance on first import. No external database setup is needed.

**Optional — AI provider keys** (set in `~/.hermes/.env`):

- `OPENAI_API_KEY` — for OpenAI embeddings, chat completions
- `ANTHROPIC_API_KEY` — for Claude models
- `TOGETHER_API_KEY` — for open-source models via Together

## How to Run

Before running, verify the correct Python has Pixeltable installed:

```bash
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"
```

If this fails (ImportError, numpy incompatibility, or wrong version), find the right interpreter:

```bash
which python3
pip show pixeltable
```

If Pixeltable is in a conda or virtualenv (e.g., `/opt/miniconda3/envs/pxt/bin/python3`), use the full path for all commands.

**Always check existing state first** before creating tables:

```bash
python3 -c "import pixeltable as pxt; print(pxt.list_tables())"
```

Run Pixeltable operations via `terminal`:

```bash
python3 -c "
import pixeltable as pxt
pxt.create_dir('demo', if_exists='ignore')
t = pxt.create_table('demo.docs', {'text': pxt.String}, if_exists='ignore')
t.insert([{'text': 'hello world'}])
print(t.collect())
"
```

For multi-step workflows, write a Python script with `write_file` and execute with `terminal`.

## Quick Reference

| Operation | Code |
|---|---|
| Create directory | `pxt.create_dir('mydir', if_exists='ignore')` |
| Create table | `pxt.create_table('mydir.t', {'text': pxt.String, 'img': pxt.Image})` |
| Insert rows | `t.insert([{'text': 'hello', 'img': 'path/to/img.jpg'}])` |
| Add computed column | `t.add_computed_column(emb=embed_fn(t.text), if_exists='ignore')` |
| Add embedding index | `t.add_embedding_index('text', embedding=embed_fn, if_exists='ignore')` |
| Similarity search | `t.order_by(t.text.similarity(string='query'), asc=False).limit(5).collect()` |
| Create view | `pxt.create_view('mydir.chunks', t, iterator=document_splitter(t.doc, separators='token_limit', limit=300))` |
| Query with filter | `t.where(t.category == 'science').select(t.text, t.score).collect()` |
| Define a UDF | `@pxt.udf` decorator on a typed Python function |
| Define a query function | `@pxt.query` decorator — returns a query expression, no `.collect()` inside |
| Drop table | `pxt.drop_table('mydir.t')` |
| List tables | `pxt.list_tables()` |
| Get existing table | `t = pxt.get_table('mydir.t')` |
| Row count | `t.count()` |

Column types: `pxt.String`, `pxt.Int`, `pxt.Float`, `pxt.Bool`, `pxt.Timestamp`, `pxt.Json`, `pxt.Array`, `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`.

**ResultSet**: `.collect()` returns a `ResultSet`. Print it directly with `print(results)` for a table view. Iterate with `for row in results: print(row)` (each row is a dict). Do NOT use `.iterrows()`, `.to_string()`, or index with `results[i][1]`.

## Procedure

### 0. Discover existing tables (always run first)

Before creating anything, check what already exists:

```python
import pixeltable as pxt
tables = pxt.list_tables()
print(tables)
```

If the table you need already exists, reuse it:

```python
t = pxt.get_table('images.local_gallery')
print(t.count())
print(t.select(t.image).limit(3).collect())
```

Only create new tables if `list_tables()` shows nothing relevant. Use `if_exists='ignore'` so re-runs are safe.

### 1. Image cross-modal search with CLIP (no API keys needed)

Search your own local images using natural language and CLIP. Tested patterns and pitfall notes included.

#### To search your local images for a phrase (e.g., "a sunset over mountains"):

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip

# Adjust these as needed
IMAGES_DIR = 'images'  # Directory to organize images
TABLE_NAME = 'images.local_gallery'

pxt.create_dir(IMAGES_DIR, if_exists='ignore')
t = pxt.create_table(TABLE_NAME, {'image': pxt.Image}, if_exists='ignore')

# (Optional) Insert images if table is empty:
# t.insert([{'image': '/absolute/path/to/your/image1.jpg'}, ...])

embed_fn = clip.using(model_id='openai/clip-vit-base-patch32')
t.add_embedding_index('image', embedding=embed_fn, if_exists='ignore')

# Cross-modal search: 
t_query = 'a sunset over mountains'
sim = t.image.similarity(string=t_query)
results = t.order_by(sim, asc=False).limit(5).select(t.image, sim).collect()
print(f"Top matches for: '{t_query}'")
print(results)
```

- **Binary incompatibility warning**: If you see an error like `numpy.dtype size changed, may indicate binary incompatibility`, you need to downgrade numpy:
  ```bash
  pip install 'numpy>=1.22.4,<2.3.0'
  ```
- **HEIF support error**: If `ModuleNotFoundError: No module named 'pillow_heif'`, install it:
  ```bash
  pip install pillow-heif
  ```
- Always use the python/pip inside your venv: e.g., `.venv/bin/python3` and `.venv/bin/pip` for consistent environments.

You can also use public web images for testing by inserting URLs instead of file paths.

---

### 2. Video analysis pipeline (requires OPENAI_API_KEY)

```python
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.openai import chat_completions

pxt.create_dir('video', if_exists='ignore')

videos = pxt.create_table('video.raw', {'video': pxt.Video}, if_exists='ignore')

frames = pxt.create_view(
    'video.frames', videos,
    iterator=frame_iterator(videos.video, fps=1),
    if_exists='ignore'
)

frames.add_computed_column(
    caption=chat_completions(
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': frames.frame}},
            {'type': 'text', 'text': 'Describe this frame in one sentence.'}
        ]}],
        model='gpt-4.1-mini'
    ).choices[0].message.content,
    if_exists='ignore'
)

videos.insert([{'video': '/path/to/video.mp4'}])
print(frames.select(frames.frame, frames.caption).limit(5).collect())
```

### 3. Document RAG pipeline

```python
import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer

pxt.create_dir('rag', if_exists='ignore')

docs = pxt.create_table('rag.docs', {'doc': pxt.Document}, if_exists='ignore')

chunks = pxt.create_view(
    'rag.chunks', docs,
    iterator=document_splitter(docs.doc, separators='token_limit', limit=300),
    if_exists='ignore'
)

embed_fn = sentence_transformer.using(model_id='intfloat/e5-large-v2')
chunks.add_embedding_index('text', embedding=embed_fn, if_exists='ignore')

docs.insert([{'doc': '/path/to/document.pdf'}])

sim = chunks.text.similarity(string='What is the main conclusion?')
results = chunks.order_by(sim, asc=False).limit(5).select(chunks.text, sim).collect()
print(results)
```

## Pitfalls

- **Embedded PostgreSQL**: Pixeltable starts an embedded PostgreSQL process on first import. If port 5432 is already in use, set `PXT_HOME` to an alternate directory or configure a different port in `~/.pixeltable/config.toml`.
- **Media storage**: Images, video, and audio files are copied to `~/.pixeltable/media/`. For large datasets, ensure sufficient disk space or configure `PXT_HOME` to point to a volume with more room.
- **Computed column re-execution**: Changing a computed column expression re-processes all existing rows. For large tables, add computed columns before bulk inserts.
- **`if_exists='ignore'` won't fix bugs**: If a computed column has wrong logic, `if_exists='ignore'` silently skips it. You must `t.drop_column('col')` then recreate.
- **API keys**: Functions from `pixeltable.functions.openai`, `.anthropic`, etc. require the corresponding API keys set as environment variables.
- **First import is slow**: The initial `import pixeltable` downloads and starts PostgreSQL (~10 seconds on first run, ~2 seconds thereafter).
- **Wrong Python interpreter**: If `python3` resolves to a system or virtualenv Python without Pixeltable, commands will fail. Run `python3 -c "import pixeltable"` first. If it fails, use the full path to the correct interpreter (e.g., `/opt/miniconda3/envs/pxt/bin/python3`) for all subsequent commands.

- **Rebuilding existing tables**: Always call `pxt.list_tables()` before creating. Use `pxt.get_table('name')` to reuse existing tables. Never use `if_exists='replace'` unless the user explicitly asks to start fresh — it destroys all data and computed columns.

**Common mistakes — do NOT use these patterns:**

| Wrong | Correct |
|---|---|
| `openai.vision(...)` | Does not exist. Use `openai.chat_completions(messages=[...])` with `image_url` blocks |
| `from pixeltable.iterators import FrameIterator` | Deprecated. Use `from pixeltable.functions.video import frame_iterator` |
| `t.col.similarity('query')` | Positional arg deprecated. Use `t.col.similarity(string='query')` |
| `pxt.Table(...)` or `pxt.connect(...)` | Do not exist. Use `pxt.create_table(...)` and `pxt.get_table(...)` |
| `for row in data: model.predict(row)` | Use computed columns instead — they auto-execute on insert |

## Verification

Run this single command to confirm Pixeltable is working:

```bash
python3 -c "
import pixeltable as pxt
pxt.create_dir('pxtverify', if_exists='ignore')
t = pxt.create_table('pxtverify.test', {'text': pxt.String}, if_exists='replace')
t.insert([{'text': 'hello'}, {'text': 'world'}])
assert len(t.collect()) == 2
pxt.drop_table('pxtverify.test')
pxt.drop_dir('pxtverify')
print('Pixeltable verification passed')
"
```
