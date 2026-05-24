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

**Optional — MCP server for native tool access:**

```bash
pip install mcp-server-pixeltable-developer
```

Then add to `~/.hermes/config.yaml`:

```yaml
mcpServers:
  pixeltable:
    command: uvx
    args: [mcp-server-pixeltable-developer]
```

**Optional — AI provider keys** (set in `~/.hermes/.env`):

- `OPENAI_API_KEY` — for OpenAI embeddings, chat completions
- `ANTHROPIC_API_KEY` — for Claude models
- `TOGETHER_API_KEY` — for open-source models via Together

## How to Run

Run all Pixeltable operations via `terminal` using Python:

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
| Add computed column | `t.add_computed_column(emb=embed_fn(t.text))` |
| Add embedding index | `t.add_embedding_index('text', embedding=embed_fn)` |
| Similarity search | `t.order_by(t.text.similarity('query'), asc=False).limit(5).collect()` |
| Create view | `pxt.create_view('mydir.chunks', t, iterator=document_splitter(t.doc, separators='paragraph'))` |
| Query with filter | `t.where(t.category == 'science').select(t.text, t.score).collect()` |
| Drop table | `pxt.drop_table('mydir.t')` |
| List tables | `pxt.list_tables()` |

Column types: `pxt.String`, `pxt.Int`, `pxt.Float`, `pxt.Bool`, `pxt.Timestamp`, `pxt.Json`, `pxt.Array`, `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`.

## Procedure

### 1. Document RAG pipeline

```python
import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer

pxt.create_dir('rag', if_exists='ignore')

docs = pxt.create_table('rag.docs', {'doc': pxt.Document}, if_exists='ignore')

chunks = pxt.create_view(
    'rag.chunks', docs,
    iterator=document_splitter(docs.doc, separators='paragraph'),
    if_exists='ignore'
)

embed_fn = sentence_transformer.using(model_id='intfloat/e5-large-v2')
chunks.add_embedding_index('text', embedding=embed_fn, if_not_exists=True)

docs.insert([{'doc': '/path/to/document.pdf'}])

sim = chunks.text.similarity('What is the main conclusion?')
results = chunks.order_by(sim, asc=False).limit(5).select(chunks.text, sim).collect()
print(results)
```

### 2. Video analysis pipeline

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

### 3. Image similarity search

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip

pxt.create_dir('images', if_exists='ignore')

imgs = pxt.create_table('images.gallery', {'image': pxt.Image}, if_exists='ignore')

embed_fn = clip.using(model_id='openai/clip-vit-base-patch32')
imgs.add_embedding_index('image', embedding=embed_fn, if_not_exists=True)

imgs.insert([{'image': '/path/to/photo.jpg'}])

sim = imgs.image.similarity('a sunset over the ocean')
results = imgs.order_by(sim, asc=False).limit(5).select(imgs.image, sim).collect()
print(results)
```

## Pitfalls

- **Embedded PostgreSQL**: Pixeltable starts an embedded PostgreSQL process on first import. If port 5432 is already in use, set `PXT_HOME` to an alternate directory or configure a different port in `~/.pixeltable/config.toml`.
- **Media storage**: Images, video, and audio files are copied to `~/.pixeltable/media/`. For large datasets, ensure sufficient disk space or configure `PXT_HOME` to point to a volume with more room.
- **Computed column re-execution**: Changing a computed column expression re-processes all existing rows. For large tables, add computed columns before bulk inserts.
- **API keys**: Functions from `pixeltable.functions.openai`, `.anthropic`, etc. require the corresponding API keys set as environment variables.
- **First import is slow**: The initial `import pixeltable` downloads and starts PostgreSQL (~10 seconds on first run, ~2 seconds thereafter).

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
