---
name: claude-import
description: Import Claude conversations into any Hermes memory provider.
version: 1.0.0
author: magnus919 (Magnus Hedemark), Hermes Agent (Nous Research)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Migration, Claude, Memory, Import, Data]
    related_skills: [hermes-agent]
---

# Claude Data Export Import

Import your Claude.ai conversation history and memories into Hermes Agent
through the configured memory provider. Uses the built-in `memory` tool so
the import works with any provider (cashew, honcho, mem0, supermemory, etc.)
without substrate-specific code — every `memory` call fires
`on_memory_write()` on the active provider.

This is a **1-way** import. Claude conversations are parsed and written as
memory entries. No changes are made to your Claude export.

## When to Use

- You're migrating from Claude.ai to Hermes Agent
- You have a Claude data export and want to bring conversation context into Hermes
- You have a memory provider configured beyond the stock filesystem provider

## Prerequisites

- **Claude data export** downloaded from [claude.ai](https://claude.ai) settings → Export Data
- **Hermes Agent** with an external memory provider configured:
  ```bash
  hermes memory setup
  ```
  The builtin filesystem provider is not supported — it lacks capacity for
  the hundreds or thousands of memory entries a typical Claude export produces.
- **Python 3.10+** in the Hermes venv

## How to Run

```
/claude-import /path/to/claude-export/
```

The skill streams `conversations.json` from the export directory, extracts
memory units, and injects them via the `memory` tool.

## Procedure

### 1. Verify the export path

Confirm the path exists and contains `conversations.json`. The export may be
a single directory or contain a `data-*` subdirectory.

### 2. Check the active memory provider

Read the configured memory provider:
```python
from hermes_cli.config import load_config, cfg_get
config = load_config()
provider = cfg_get(config, "memory", "provider")
```

- If `provider` is `None`, `""`, or not set → **builtin filesystem provider is active**.
  Do not proceed. Tell the user:
  > "Your Claude export contains hundreds of conversations that would generate
  > thousands of memory entries, exceeding the builtin filesystem provider's
  > capacity. Configure an external memory provider first:
  >   `hermes memory setup`
  > Then re-run this command."

- If `provider` is set (e.g. `"cashew"`, `"honcho"`, `"mem0"`, `"supermemory"`) → proceed.

### 3. Run the parser

Run the parsing script from the skill directory to extract memory units:

```bash
python scripts/claude_parse.py /path/to/claude-export/ \
  --output /tmp/claude-memory-units.json
```

The script is at `optional-skills/migration/claude-import/scripts/claude_parse.py`
relative to the Hermes Agent repository root. Resolve the absolute path
using `SkillManager` or the installed skill path at
`~/.hermes/skills/migration/claude-import/scripts/claude_parse.py`.

The script `stream_json_array()` function handles 175 MB+ files without
loading the entire file into memory.

### 4. Read the output

```python
import json
with open("/tmp/claude-memory-units.json") as f:
    units = json.load(f)
```

Each unit has:
- `content` — the memory text (conversation name + summary + key exchange)
- `source_type` — `"conversation"` or `"project_memory"`
- `conversation_uuid` — for deduplication
- `timestamp` — the conversation start time

### 5. Batch-and-import loop

Batch conversations to stay within the agent's `max_iterations` budget
(default 90). The recommended ratio is **10 conversations per memory call**:

```python
batch_size = 10
batches = [units[i:i + batch_size] for i in range(0, len(units), batch_size)]

for i, batch in enumerate(batches):
    # Embed structured metadata for each entry (UUID, source, timestamp)
    # so the provider can surface traceable origin and enable dedup
    combined = "\n\n---\n\n".join(
        "\n".join([
            f"[UUID: {u['conversation_uuid']}]",
            f"[Source: claude]",
            f"[Type: {u.get('source_type', 'conversation')}]",
            f"[Timestamp: {u.get('timestamp', 'unknown')}]",
            u.get("content", ""),
        ])
        for u in batch
    )
    memory(action='add', target='memory', content=combined)
    # Each call fires on_memory_write() on the configured provider
```

- **Dedup by UUID**: Before importing, compare against existing memory or
  skip UUIDs already seen. The `conversation_uuid` field on each unit can
  be used to detect re-imports.
- **Project memories** (`source_type: "project_memory"`) are higher-signal
  than raw conversations. Import these first, or at a smaller batch size
  (5 per call) to preserve their structure.

### 6. Report

After the loop completes, report:

```
Imported {N} memory units from {conversation_count} conversations
  + {project_count} project memories
  across {batch_count} memory tool calls.
Source: {export_dir}
```

**Important caveat:** `on_memory_write()` returns `None` — there is
no per-write acknowledgment from the provider. This reports what was
dispatched to the provider, not what was accepted. Verify receipt by
checking your provider's storage.

## Batching Reference

| Conversations | Batch size | Memory calls | Fits 90-iteration limit? |
|---------------|------------|-------------|--------------------------|
| 100           | 10         | 10          | ✅ Yes                   |
| 500           | 10         | 50          | ✅ Yes                   |
| 856           | 10         | 86          | ✅ Yes                   |
| 856           | 20         | 43          | ✅ Yes (but coarser)     |
| 2000          | 30         | 67          | ✅ Yes                   |

If the number of batches exceeds the iteration limit, use `/goal` to
drive the import across multiple turns:
```
/goal Import the remaining {N} batches of Claude memory units from /tmp/claude-memory-units.json
```
The goal judge will verify batch-by-batch completion.

## Pitfalls

- **Builtin provider guard is load-bearing.** Do not skip the provider check.
  The filesystem provider will silently drop or corrupt entries beyond its
  capacity. There is no runtime signal — `on_memory_write()` returns `None`.
- **Memory provider must be configured BEFORE running this command.**
  The preflight check is static — if no external provider is configured,
  the command refuses to run.
- **Streaming parser vs full load.** The `claude_parse.py` script streams
  large files using `json.JSONDecoder.raw_decode` for files over 50 MB.
  Under 50 MB it uses `json.loads` for speed.
- **No dedup by default.** Importing the same export twice will create
  duplicate memory entries. The skill relies on the user's memory provider
  for dedup — if the provider doesn't deduplicate, the user will see
  duplicates. Consider checking `conversation_uuid` before writing if
  re-import is expected.
- **memories.json is richer than conversations.json.** Project memories
  contain curated knowledge summaries. If present, they produce higher-
  quality memory entries than raw conversation dumps.
- **File attachments are not imported.** The parser only extracts message
  text. Uploaded files, images, and code artifacts in conversations
  are skipped.

## Verification

After import, confirm the data landed:

```bash
# Check the provider's status
hermes memory status
# or: hermes doctor
```
