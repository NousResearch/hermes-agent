---
name: pixeltable
description: Multimodal data tables with auto-computed AI columns.
version: 1.1.0
author: Pierre Brunelle (@pierrebrunelle), Pixeltable (pixeltable.com)
license: Apache-2.0
platforms: [linux, macos]
prerequisites:
  commands: [python3, pip]
metadata:
  hermes:
    tags: [mlops, multimodal, rag, vector-search, data-pipeline]
    related_skills: [chroma, qdrant-vector-search, qmd]
    requires_toolsets: [terminal]
---

# Pixeltable Skill

Pixeltable stores images, video, audio, and documents in tables and runs declared AI steps when rows are inserted. It is a declarative data layer that includes vector search, not a separate vector database.

Two ways to use it:

- **Application.** One `app.py` with `TableModel` classes and optional `FastAPIRouter` routes. `pxt schema update` creates tables and does not start HTTP. `pxt service update` starts HTTP. Do not run `python app.py` for a file that only declares models and routers.
- **Notebook or REPL.** `pxt.create_table()`, `add_computed_column()`, and `add_embedding_index()`, run through `terminal`. Use this for one-off pipelines. The procedures below use this form.

Do not import LangChain, LlamaIndex, Haystack, or LangGraph for chunking, search, or tool calling. Do not keep the working set in pandas (`.collect().to_pandas()` is export only). Do not loop over rows calling a model. Insert a row and let the computed columns run.

## When to Use

- User wants to process video frames, audio, images, or documents through models
- User needs retrieval over multimodal content, not only text
- User wants computed columns that embed, transcribe, or call an LLM on insert
- User needs similarity search across images, text, or audio
- User wants a declared chain (video to frames to captions to embeddings)
- User wants an HTTP API over those tables (`FastAPIRouter` plus `pxt service update`)
- User asks about incremental pipelines that skip rows already computed

For a plain-text-only search job, `chroma`, `qdrant-vector-search`, or `qmd` are simpler. Once the user is building in Pixeltable, search with its embedding index. Do not add Pinecone, Chroma, or Qdrant beside it.

## Prerequisites

Pixeltable requires Python 3.11 or newer. Install with `terminal`:

```bash
python3 -m pip install pixeltable
```

HTTP serving needs the extra:

```bash
python3 -m pip install 'pixeltable[serve]'
```

Confirm the same interpreter:

```bash
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"
```

The first import starts an embedded PostgreSQL. On Linux and macOS it listens on a local socket, so a busy TCP port 5432 is not the failure mode. No external database setup is required.

**Provider keys.** Pixeltable reads them from the process environment or `~/.pixeltable/config.toml` (override the file with `PIXELTABLE_CONFIG`). Putting a key only in `~/.hermes/.env` is not enough unless that process exports it. The `pxt` daemon reads config and the environment once, at startup. After either changes, `pxt daemon restart`, then `pxt service restart NAME` for a running service. Until the restart, commands answer 409.

Common names: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`. Gemini accepts `GOOGLE_API_KEY` or `GEMINI_API_KEY` (or `api_key` under `[gemini]`). An OpenAI-compatible proxy uses `OPENAI_BASE_URL` or `[openai] base_url`. Ollama uses `OLLAMA_HOST` or `[ollama] host`, not `base_url`. Only the OpenAI client reads a base URL from Pixeltable config. Never pass `api_key=` into the call.

## How to Run

Check the interpreter before any other command:

```bash
python3 -c "import pixeltable as pxt; print(f'Pixeltable {pxt.__version__} ready')"
python3 -m pip show pixeltable
```

If import fails, use the full path of the interpreter that has Pixeltable (conda or a virtualenv) for every later command, including `python3 -m pip`.

**Notebook path.** List tables first, then run short scripts with `terminal`. For more than a few lines, `write_file` a script and execute that file.

```bash
python3 -c "import pixeltable as pxt; print(pxt.list_tables())"
```

```bash
python3 -c "
import pixeltable as pxt
pxt.create_dir('demo', if_exists='ignore')
t = pxt.create_table('demo.docs', {'text': pxt.String}, if_exists='ignore')
t.insert([{'text': 'hello world'}])
print(t.collect())
"
```

**Application path.** From the project directory, with `terminal`:

```bash
python3 -m pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list
```

`my_app` is a catalog directory, not a folder on disk. Without a TTY, a pending `pxt service update` exits 3 unless given `-f`. `pxt service list` prints the URL; do not assume port 8000. `pxt service run` is a foreground container entrypoint, not the command to start a service. After a later schema change, routes answer 409 until `pxt service update ... -f` again. Full flags, Cloud, secrets, and retries: `references/apps-and-cloud.md`.

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
| Define a UDF | `@pxt.udf` on a typed function. Optional input is `T \| None` |
| Define a query function | `@pxt.query` returns a query. No `.collect()` inside it |
| Drop table | `pxt.drop_table('mydir.t')` |
| List tables | `pxt.list_tables()` |
| Get existing table | `t = pxt.get_table('mydir.t')` |
| Row count | `t.count()` |
| Apply an app | `pxt schema update app.py my_app` then `pxt service update app.py my_app -f` |
| Bind models in a process | `app.TableModel.bind_all('my_app')` |

Column types: `pxt.String`, `pxt.Int`, `pxt.Float`, `pxt.Bool`, `pxt.Timestamp`, `pxt.Json`, `pxt.Array`, `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`. Types are non-nullable by default. Optional is `T | None`. `pxt.Required` is deprecated.

**ResultSet.** `.collect()` returns a `ResultSet`. `print(results)` shows a table. `for row in results` yields a dict per row. `results[i]` is that dict, `results['col']` is a list, `results[i, 'col']` is one value. `.to_pandas()` exports. Do not call `.iterrows()`, `.to_string()`, or `results[i][1]`.

Call `.where()` once. A second `.where()` raises. Combine predicates with `&` and `|`.

## Procedure

### 0. Discover existing tables (always run first)

```python
import pixeltable as pxt
print(pxt.list_tables())
```

Reuse a table that already exists:

```python
t = pxt.get_table('images.local_gallery')
print(t.count())
print(t.select(t.image).limit(3).collect())
```

Create a table only when `list_tables()` shows nothing relevant. Pass `if_exists='ignore'` so a re-run is safe. Do not pass `if_exists='replace'` unless the user asked to destroy the table.

### 1. Image search with CLIP (no API key)

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip

pxt.create_dir('images', if_exists='ignore')
t = pxt.create_table('images.local_gallery', {'image': pxt.Image}, if_exists='ignore')

embed_fn = clip.using(model_id='openai/clip-vit-base-patch32')
t.add_embedding_index('image', embedding=embed_fn, if_exists='ignore')

sim = t.image.similarity(string='a sunset over mountains')
results = t.order_by(sim, asc=False).limit(5).select(t.image, sim).collect()
print(results)
```

`embedding=` covers text and image when the function's signature matches both. `image_embed=` alone cannot answer `similarity(string=...)`.

If import reports `numpy.dtype size changed`, reinstall numpy in that same interpreter:

```bash
python3 -m pip install 'numpy>=1.22.4,<2.3.0'
```

If import reports `No module named 'pillow_heif'`:

```bash
python3 -m pip install pillow-heif
```

Public image URLs can be inserted in place of file paths.

### 2. Video captions (requires OPENAI_API_KEY)

`frame_iterator` yields `frame` and `frame_attrs` (`index`, `time`, `key_frame`, and the other pyav fields). It does not yield `frame_idx` or `pos_msec`. Pass the image column itself as `image_url`. Pixeltable serializes it. The HTTP shape `{'url': ...}` is wrong here.

```python
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.openai import chat_completions

pxt.create_dir('video', if_exists='ignore')
videos = pxt.create_table('video.raw', {'video': pxt.Video}, if_exists='ignore')
frames = pxt.create_view(
    'video.frames', videos,
    iterator=frame_iterator(videos.video, fps=1.0),
    if_exists='ignore',
)
frames.add_computed_column(
    caption=chat_completions(
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': frames.frame},
            {'type': 'text', 'text': 'Describe this frame in one sentence.'},
        ]}],
        model='gpt-4.1-mini',
    ).choices[0].message.content,
    if_exists='ignore',
)
videos.insert([{'video': '/path/to/video.mp4'}])
print(frames.select(frames.frame, frames.caption).limit(5).collect())
```

Model ids in examples go stale. Check the provider's current list. Extra request fields (`temperature`, `max_tokens`, OpenAI `language` on transcriptions) go in `model_kwargs={}`. An unknown top-level keyword fails. Anthropic `messages` is the exception: it requires top-level `max_tokens=`.

### 3. Document retrieval

`separators='token_limit'` needs `tiktoken`. `separators='sentence'` needs spaCy. The `image` element is supported only with `separators='page'` on PDFs. Default elements are text only.

```python
import pixeltable as pxt
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import sentence_transformer

pxt.create_dir('rag', if_exists='ignore')
docs = pxt.create_table('rag.docs', {'doc': pxt.Document}, if_exists='ignore')
chunks = pxt.create_view(
    'rag.chunks', docs,
    iterator=document_splitter(docs.doc, separators='token_limit', limit=300),
    if_exists='ignore',
)
embed_fn = sentence_transformer.using(model_id='intfloat/e5-large-v2')
chunks.add_embedding_index('text', embedding=embed_fn, if_exists='ignore')
docs.insert([{'doc': '/path/to/document.pdf'}])
sim = chunks.text.similarity(string='What is the main conclusion?')
print(chunks.order_by(sim, asc=False).limit(5).select(chunks.text, sim).collect())
```

More notebook recipes: `references/multimodal-pipeline-examples.md`.

### 4. Application file

`pxt service example --out app.py` writes this shape. Edit it, then `pxt schema update app.py my_app`. An annotation is a stored column. An assignment is a computed column. `add_update_route` matches rows by the primary key, so the request body still carries `id`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()

class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    title: pxt.String
    body: pxt.String | None

ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body],
    outputs=[Docs.id, Docs.title],
)
```

There is no `pxt.bind_all()` function. After schema update, bind with `app.TableModel.bind_all('my_app')`, then `app.Docs.insert(...)`. Indexes on the model use `__indexes__ = [pxt.EmbeddingIndex(..., name=...)]`. The notebook method spells the same idea `idx_name=`.

Directory names from the project root down to `app.py` must be Python identifiers. `pxt schema check app.py` before the first update. If a warning says an installed package named `app` shadows the project, rename the file. `check` warns and still exits 0. `schema update` does not warn and imports the wrong module.

Cloud sign-in, secrets, and the hosted wait: `references/apps-and-cloud.md`. Ask the user to run `pxt login` in their own terminal. Never ask them to paste a key into the chat.

## Pitfalls

- **Home directory.** Catalog and media live under `PIXELTABLE_HOME` (default `~/.pixeltable`). Media files are in `PIXELTABLE_HOME/media`. Postgres data is `PIXELTABLE_PGDATA` or `PIXELTABLE_HOME/pgdata`. `PXT_HOME` is not a Pixeltable setting.
- **Daemon.** The CLI daemon listens on `127.0.0.1:22089`. Override with `PXT_PORT`. One daemon serves one project per port. A `pxt` command from another project on that port restarts it.
- **`if_exists='ignore'` does not fix logic.** It skips a column that already exists. In a notebook, change the expression with `t.alter_computed_column(summary=new_expr)`, which recomputes that column and its dependents. In an app, edit the expression, run `pxt schema update` (existing values stay), then `pxt recompute my_app/docs summary -f`.
- **Type changes.** Changing a column's type, nullability, or `stored` flag is unsupported. `--allow-destructive` does not apply it. Rename a computed column in one `--allow-destructive -f` pass. Declare `T | None` before data exists when the column must accept nulls.
- **Recompute.** `t.recompute_columns('summary', errors_only=True)`. The column is a positional argument, not `columns=[...]`.
- **Nullable UDF parameters.** A non-nullable parameter that receives `None` skips the call. The cell is `None` and `errormsg` is empty. Annotate `x: str | None` and handle `None`.
- **Provider retries.** OpenAI, Anthropic, and Gemini retry rate-limit failures up to 10 times. Other providers retry up to 3 times, paced by `rate_limit` (600 requests a minute when unset). HTTP 429, 503, 408, and 504, plus connection errors and timeouts, are retried, and `Retry-After` is honored. Do not wrap the call in `tenacity` or another client. A cell that still fails is listed by `pxt errors PATH`.
- **First import is slow.** The first `import pixeltable` prepares PostgreSQL. Later imports are faster.
- **Wrong interpreter.** `python3 -c "import pixeltable"` must succeed in the interpreter you will keep using.

| Wrong | Correct |
|---|---|
| `openai.vision(...)` | Deprecated since 0.5.18. Use `chat_completions` with `image_url` set to the image column, or `responses` |
| `{'image_url': {'url': frames.frame}}` | `{'type': 'image_url', 'image_url': frames.frame}` |
| `from pixeltable.iterators import FrameIterator` | `from pixeltable.functions.video import frame_iterator` |
| `t.col.similarity('query')` | `t.col.similarity(string='query')`. Also `image=`, `audio=`, `video=`, `document=`, `vector=`. `idx=` picks among indexes |
| `t.col.select()` / `t.col.count()` | Those ColumnRef helpers do not exist. Use `t.select(t.col)...` and `t.count()`. `t.show()` and `t.head()` still exist on the table |
| `t.where(a).where(b)` | One `.where()`. Combine with `&` and `\|` |
| `chat_completions(..., temperature=0)` | `model_kwargs={'temperature': 0}` |
| `pxt.Table(...)` or `pxt.connect(...)` | `pxt.create_table(...)` and `pxt.get_table(...)` |
| `for row in data: model.predict(row)` | A computed column, which runs on insert |

## Verification

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

## Where to look

| Need | File |
|---|---|
| CLI, Cloud sign-in, secrets, hosted wait, retries, API traps | `references/apps-and-cloud.md` |
| Hermes MCP config and tool names | `references/mcp-integration.md` |
| Notebook recipes (audio, chained summary) | `references/multimodal-pipeline-examples.md` |
