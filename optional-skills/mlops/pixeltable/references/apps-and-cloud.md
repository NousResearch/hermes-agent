# Apps, CLI, Cloud, and retries

Hermes runs these commands with `terminal`. Install with `python3 -m pip`, not a bare `pip`, so the interpreter matches the one that imports Pixeltable. Python 3.11 or newer. Official CLI reference: https://docs.pixeltable.com/platform/cli

There is no `pxt serve`, `pxt deploy`, `pxt app`, or `pxt db create`. `pxt db update` creates a hosted database. Do not write a `[tool.pixeltable.service]` table.

## Application contract

```bash
python3 -m pip install 'pixeltable[serve]'
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list
```

- `pxt init` marks the project root. A fresh directory gets `pixeltable.toml`. An existing `pyproject.toml` gets `[[tool.pixeltable.database]]` appended. Nested under another root, it exits 3.
- `pxt schema update` creates or migrates tables. It does not start HTTP. An additive or no-op schema update does not prompt.
- `pxt service update` starts or restarts HTTP. It does not create tables. Without a TTY, a pending update exits 3 unless given `-f`.
- `pxt schema check APP` before the first update. It reads no catalog. It warns when an installed package shadows a project module (`app` is the usual collision). `check` still exits 0. `schema update` and `service update` do not warn, and they import the installed package.
- Names of directories from the project root to `app.py` must be Python identifiers.
- `my_app` is a catalog directory. Local handle: `pxt.get_table('my_app.docs')`. After import, `app.TableModel.bind_all('my_app')`. There is no module-level `pxt.bind_all()` or `pxt.create_all()`.
- Do not call `pxt.create_table()` or `pxt.get_table()` at import time in `app.py`. Import must not mutate the catalog.
- `pxt service list` prints the assigned URL. Do not hard-code port 8000. `pxt service run` stays in the foreground for a container entrypoint and is not recorded by `list` or `stop`.
- After `pxt schema update` changes a table, its routes answer 409 until `pxt service update ... -f`. The restarted service keeps its port.
- A safe alter of a computed expression records the new expression and does not recompute existing rows. Follow it with `pxt recompute my_app/docs summary -f`.
- Dropping a column or index is DESTRUCTIVE and needs `--allow-destructive`. Without that flag the whole plan applies nothing.
- A type change, a nullability change, or a `stored` change is UNSUPPORTED. No flag applies it, including `--allow-destructive`. Rename the computed column instead (one `--allow-destructive -f` pass). A stored-column rename drops its data, so declare `T | None` up front when nulls are required.
- UDFs are stored as a module path relative to the project root (`app.excerpt`).

Notebook equivalent of a logic fix: `t.alter_computed_column(summary=new_expr)`. That recomputes the column and its dependents. `if_exists='ignore'` does not.

## Daemon and config

The first catalog command starts a daemon at `127.0.0.1:22089` unless `PXT_PORT` is set. `pxt daemon status`, `stop`, `start`, `restart`.

The daemon reads `config.toml` and the environment once. Local services inherit that copy. After a new key is exported or `config.toml` changes, every command answers 409 until `pxt daemon restart`, then `pxt service restart NAME`.

Home is `PIXELTABLE_HOME` (default `~/.pixeltable`). Config file is `PIXELTABLE_CONFIG` or `PIXELTABLE_HOME/config.toml`. Postgres data is `PIXELTABLE_PGDATA` or `PIXELTABLE_HOME/pgdata`. Media is `PIXELTABLE_HOME/media`. `PXT_HOME` is not read.

On Linux and macOS the embedded Postgres uses a Unix socket, chosen so it does not collide with TCP 5432. Windows picks a free TCP port. Do not aim users at port 5432.

Provider keys come from the environment or the config file, never from `api_key=` on the call. Gemini accepts `GOOGLE_API_KEY` or `GEMINI_API_KEY`, or `[gemini] api_key`. An OpenAI-compatible proxy is `OPENAI_BASE_URL` or `[openai] base_url`. Ollama is `OLLAMA_HOST` or `[ollama] host`. Other providers do not read a Pixeltable `base_url`.

## Cloud sign-in, databases, and secrets

`pxt login` uses an OAuth device grant and waits up to 10 minutes. That is longer than most agent tool calls. Ask the user to run `pxt login` in their own terminal, then check with `pxt whoami`. `PIXELTABLE_API_KEY`, or `api_key` under `[pixeltable]`, wins over the cached session and does not expire. Never ask the user to paste a secret into the chat.

A new account has no organization. `pxt org create NAME` creates one and its first database. URIs are `pxt://org` or `pxt://org:db`.

Hosted order:

```bash
pxt db update pxt://org:db -f
pxt schema update app.py pxt://org:db -f
pxt service update app.py pxt://org:db -f
```

`pxt db update` uploads the project and sets the image and workers. It does not load secrets, rows, or HTTP. The URI must match a `[[pixeltable.database]]` entry `name = 'pxt://org:db'`. If `pxt db diff` says the hosted project is behind the working copy, run `pxt db update` before schema or service. Taking capacity away needs `--allow-destructive`. `pxt db delete` removes the database and its storage. Without a TTY it exits 3 unless given `-f`.

These `pxt db` verbs wait until Cloud finishes the change: `update`, `start`, `stop`, `restart`, `retry`, `build-image`, `delete`. `--no-wait` exists only on those verbs. It returns once Cloud accepts the change. Ctrl-C also stops only the waiting. The change keeps running. After three hours the command exits 2 and the change is still in progress. Follow a timed-out or `--no-wait` call with `pxt db status pxt://org:db`.

`pxt service update` has no `--no-wait`. On a hosted target it waits for the receipt. `pxt schema update` is separate from that wait flag. A failed hosted update can be retried unchanged with `pxt db retry pxt://org:db` or `pxt service retry pxt://org:db/NAME`, or by running the same `pxt db update` again.

Secrets:

```bash
pxt secret set pxt://org OPENAI_API_KEY=<value>
```

Also `pxt://org:db` for one database. An org secret applies to every database in the org. A database secret wins on the same name. The `PIXELTABLE_` prefix is reserved and refused. A `secrets` mapping inside `pixeltable.toml` fails to load. Give the user the `pxt secret set` command to run themselves. Do not put the value in the command you run.

A process reads secrets once, at startup. After `pxt secret set` or `pxt secret delete`, run `pxt db restart pxt://org:db` and `pxt service restart pxt://org:db/NAME`. On Cloud, a proxy is `pxt secret set pxt://org OPENAI_BASE_URL=https://<proxy>/v1`, then those restarts.

Hosted media defaults to the database home bucket unless a column sets `destination=`, or the database entry sets `db_input_media_dest` / `db_output_media_dest`. The bucket is private. Sharing uses a signed URL from the storage API, not a public object link. See https://docs.pixeltable.com/cloud

Some older notes say a hosted command can fail with `CLIENT_OUTDATED` when the local install predates the control-plane protocol. That string is not raised by current Pixeltable. If a hosted command fails, read the error it actually printed and run `pxt <command> --help`.

## Retries and proxies

A computed column that calls a provider retries on its own.

- OpenAI, Anthropic, and Gemini use the rate-limit scheduler and retry up to 10 times.
- Other providers use the request-rate scheduler and retry up to 3 times. The pace is `rate_limit` in that provider's config section, 600 requests a minute when unset.
- Retried failures: HTTP 429, 503, 408, and 504, connection errors, and timeouts. A `Retry-After` value is honored, capped at 60 seconds.
- A cell that still fails keeps the error. `pxt errors PATH` lists it (the table needs a primary key). `pxt recompute PATH COLUMN --errors-only -f` runs that column again. Notebook form: `t.recompute_columns('summary', errors_only=True)`.

Do not add a retry loop, `tenacity`, or a second client around the call.

Only the OpenAI client reads `OPENAI_BASE_URL` / `[openai] base_url`. Pointing `base_url` at Ollama does nothing. Set `OLLAMA_HOST` instead.

## Call shapes that differ from the HTTP APIs

| Wrong | Correct |
|---|---|
| `{'image_url': {'url': t.image}}` | `{'type': 'image_url', 'image_url': t.image}` inside `chat_completions`. Pixeltable serializes the image column |
| `openai.vision(...)` | Deprecated since 0.5.18. `chat_completions` or `responses` |
| `transcriptions(..., language='en')` | The signature is `(audio, *, model, model_kwargs=None)`. Put `language` in `model_kwargs`. The docstring example that passes `language=` is stale |
| `messages(..., max_tokens=...)` omitted | Anthropic `messages` requires top-level `max_tokens=` |
| `chat_completions(..., temperature=0)` | `model_kwargs={'temperature': 0}` on OpenAI and the other OpenAI-compatible providers. Gemini uses `config=`. Bedrock `converse` uses `inference_config=`. Ollama uses `options=`. vLLM uses `sampling_params=` |
| `similarity('query')` | `similarity(string='query')`. Also `image=`, `audio=`, `video=`, `document=`, `vector=`. `idx=` selects among indexes on one column |
| `EmbeddingIndex(col, image_embed=clip)` for text queries | `embedding=` when the function covers text and image, or both `string_embed=` and `image_embed=` |
| `add_embedding_index(..., name=)` | Notebook: `idx_name=`. Model DSL: `EmbeddingIndex(..., name=)` |
| `make_video(order_by=t.pos)` | `make_video` is `requires_order_by`. The order key is the first positional argument: `make_video(t.pos, t.frame, fps=25)`. `order_by=` raises |
| `uuid_col.astype(pxt.String)` | `uuid.to_string()` from `pixeltable.functions.uuid` |
| `json_col.astype(pxt.String)` on a dict or list | `pxtf.json.dumps(json_col)`. Insert fails with `Expected string, got dict` |
| `recompute_columns(columns=['summary'])` | `t.recompute_columns('summary', errors_only=True)` |
| `t.where(a).where(b)` | One `where()`. Compose with `&` and `\|` |
| `pxt.Required[pxt.String]` | Non-nullable is the default. Optional is `T \| None` |
| `@pxt.udf def f(x: str)` over a nullable column | Annotate `x: str \| None`. Otherwise a `None` input skips the call and leaves `errormsg` empty |
| `whisper.load_model(...)` inside the UDF body | Weights reload every row. Use `pixeltable.functions.whisper.transcribe`, or a module-scope loader |
| `from pixeltable.iterators import ...` | Import the function from `pixeltable.functions.*` |

Extract the text field before embedding or concatenating: `.choices[0].message.content` for OpenAI-compatible chat, `.output_text` for OpenAI `responses`, `.content[0].text` for Anthropic. A Json field that holds a string needs `.astype(pxt.String)` before string operations. That cast is for a string value, not for a dict or list.

`frame_iterator` output columns are `frame` and `frame_attrs` (`index`, `time`, `key_frame`, and the other pyav fields). `fps` is a float (`fps=1.0`). `document_splitter(..., separators='token_limit')` needs `tiktoken`. `separators='sentence'` needs spaCy.

Model ids in any example are illustrations. Confirm the provider's current id. `sentence_transformer` takes `model_id=`. Most chat functions take `model=`.
