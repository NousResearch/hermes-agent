# Doubleword CLI Recipes

Use this reference when executing Doubleword commands, validating payloads,
submitting jobs, retrieving results, resuming interrupted downloads, or using
the OpenAI-compatible SDK.

Official command reference:
`https://doublewordai.github.io/dw/commands.html`

Run `dw --help` or `dw <group> --help` for exact options in the installed CLI
version.

## Readiness Checks

Confirm the API key is present without printing it:

```bash
test -n "$DOUBLEWORD_API_KEY"
```

Use the strongest readiness check available for the login mode.

For browser login, verify the authenticated CLI identity and active
organization before uploading data:

```bash
dw whoami
```

If `dw whoami` fails after browser login, do not upload data. Check whether the
Doubleword CLI is installed, whether `DOUBLEWORD_API_KEY` is available in the
shell context, and whether the active account or organization is correct.

For headless/API-key login, `dw whoami` may fail because API-key login stores
only the inference key and does not enable admin API commands. In that mode,
validate the payload locally and run a cheap, non-interactive realtime
inference probe before uploading or submitting jobs:

The readiness probe defaults to `openai/gpt-oss-20b`, the lowest-cost available
realtime model; set `MODEL` first to override it.

```bash
MODEL="${MODEL:-openai/gpt-oss-20b}"
dw files validate path/to/dataset.jsonl
dw files stats path/to/dataset.jsonl
dw realtime "$MODEL" "Reply with OK." --temperature 0 --max-tokens 2 --no-stream
```

Treat the local file commands as payload checks, not authentication checks. The
minimal realtime request is the authentication probe because it proves the
inference key works against Doubleword. Always pass a prompt argument or pipe a
tiny prompt so the probe cannot block waiting for input:

```bash
MODEL="${MODEL:-openai/gpt-oss-20b}"
printf 'Reply with OK.\n' | dw realtime "$MODEL" --temperature 0 --max-tokens 2 --no-stream
```

Useful auth/account commands:

```bash
dw login --api-key <key>
dw account current
dw account list
dw account switch <account>
```

Do not run key-management commands such as `dw keys create` or
`dw keys delete` unless the user explicitly asks.

## Model Discovery

Use the installed CLI to verify available model names and details:

```bash
dw models list
dw models list --type chat
dw models list --type embeddings
: "${MODEL:?set MODEL to the model to inspect}"
dw models get "$MODEL"
```

## JSONL Validation

Validate every generated batch payload before upload:

```bash
dw files validate path/to/dataset.jsonl
dw files stats path/to/dataset.jsonl
```

Fix validation errors locally and rerun validation before submitting.

Other local file helpers:

```bash
dw files prepare path/to/input.jsonl
dw files sample path/to/dataset.jsonl -n <count>
dw files split path/to/dataset.jsonl
dw files merge <FILES...>
dw files diff expected.jsonl actual.jsonl
```

Use `dw files cost-estimate <file_id>` after upload when the user needs a cost
estimate before creating a batch.

## Mode Commands

Realtime, for immediate single-request use:

```bash
: "${MODEL:?set MODEL to the selected chat model}"
dw realtime "$MODEL"
```

Async, for same-session background jobs:

```bash
dw stream path/to/dataset.jsonl --completion-window 1h
```

Batch, for lowest-cost large jobs:

```bash
dw stream path/to/dataset.jsonl --completion-window 24h
```

`dw stream` uploads, creates a batch, watches progress, and pipes results. Use
it only when active streaming is appropriate. For Hermes background work, prefer
explicit upload and batch creation so the agent can stop waiting after it has
reported the batch ID.

## Explicit Batch Creation

Use explicit file upload and batch creation when you need to capture file IDs,
control submission, or shard large workloads.

Upload:

```bash
dw files upload batch.jsonl
```

Create an Async job:

```bash
dw batches create --file <file_id> --completion-window 1h
```

Create a 24h Batch job:

```bash
dw batches create --file <file_id> --completion-window 24h
```

One-step upload and create is also available:

```bash
dw batches run path/to/dataset.jsonl
```

Avoid `dw batches run --watch` for long background jobs in Hermes because it
keeps the agent actively waiting.

## Status and Results

Check status with a discrete command:

```bash
dw batches get <batch_id>
```

Download results:

```bash
dw batches results <batch_id> -o results.jsonl
```

For multiple completed batches:

```bash
dw batches results <IDS...> -o results.jsonl
dw batches analytics <IDS...>
```

Use `dw batches retry <batch_id>` only when failed requests should be retried.
Use `dw batches cancel <batch_id>` only when the user explicitly asks to cancel
or the submitted job is clearly wrong and cancellation is safe.

Check the downloaded output shape:

```bash
dw files stats results.jsonl
```

## Hermes Non-Blocking Rule

For Async and Batch jobs:

1. Submit the job.
2. Capture the batch ID.
3. Report the batch ID and status command to the user.
4. Stop active waiting.
5. Run `dw batches get <batch_id>` later only when the workflow resumes or the
   user asks for status.

Do not use loops such as:

```bash
while true; do
  dw batches get <batch_id>
  sleep 60
done
```

Also avoid long-running watch commands for background jobs:

```bash
dw batches watch <batch_id>
dw batches run path/to/dataset.jsonl --watch
```

## Resuming Interrupted Downloads

If a large result download breaks mid-stream, inspect the partial file and
resume from the last fully processed line. When the CLI cannot resume directly,
use the files content endpoint with an offset:

```bash
curl -G "https://api.doubleword.ai/v1/files/<output_file_id>/content" \
  -H "Authorization: Bearer $DOUBLEWORD_API_KEY" \
  --data-urlencode "offset=<last_processed_line>"
```

Do not run commands in a way that expands and logs the live token.

## OpenAI SDK Fallback

Use the SDK only when it is clearer than CLI commands for the pipeline. For bulk
work, prefer validated JSONL plus `dw` batch commands.

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DOUBLEWORD_API_KEY"],
    base_url="https://api.doubleword.ai/v1",
)
```

Keep SDK credentials in environment variables. Do not hard-code or print them.
