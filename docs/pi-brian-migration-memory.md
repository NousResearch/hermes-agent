# pi-brian -> Hermes memory migration

This migration keeps memory split into three layers.

## Layer 1 - hot memory

Always injected into the prompt:

- `~/.hermes/memories/USER.md`
- `~/.hermes/memories/MEMORY.md`

Use this only for stable, high-signal facts.

Examples:

- durable preferences
- recurring digest expectations
- important homelab/runtime facts
- canonical assistant rules

Do **not** dump full Mem0 here.

## Layer 2 - warm memory

Use `plugins/memory/pi_brian_mem0/` as the active Hermes external memory provider.

Purpose:

- preserve long-tail semantic recall from existing self-hosted Mem0
- automatically prefetch relevant memories each turn
- write durable new turns back into Mem0
- mirror built-in hot-memory writes back into Mem0

Activation:

```yaml
memory:
  provider: pi_brian_mem0
```

Required config:

```bash
export MEM0_BASE_URL=http://127.0.0.1:8000
export MEM0_USER_ID=1176823362
export MEM0_AGENT_ID=hermes-brian
```

## Layer 3 - cold memory

Archive and search outside the prompt:

- raw Mem0 export JSON/Markdown
- session history
- research artifacts
- task logs

## Migration workflow

### 1. Export current self-hosted Mem0

```bash
python scripts/pi_brian_migration/export_selfhosted_mem0.py \
  --base-url http://127.0.0.1:8000 \
  --user-id 1176823362 \
  --output-dir /tmp/hermes-brian-memory
```

### 2. Build hot-memory drafts

```bash
python scripts/pi_brian_migration/build_hot_memory.py \
  --mem0-export /tmp/hermes-brian-memory/mem0-export.json \
  --flat-memory /path/to/pi-brian/data/memories.json \
  --output-dir /tmp/hermes-brian-memory/drafts
```

Outputs:

- `USER.draft.md`
- `MEMORY.draft.md`
- `MANUAL_REVIEW.md`

### 3. Review drafts manually

Manual review is expected. Sensitive entries like addresses should not be auto-promoted.

### 4. Install reviewed hot memory

```bash
python scripts/pi_brian_migration/install_hot_memory.py \
  --hermes-home ~/.hermes \
  --user-file /tmp/hermes-brian-memory/drafts/USER.draft.md \
  --memory-file /tmp/hermes-brian-memory/drafts/MEMORY.draft.md
```

## Operating policy

### Built-in memory

- tiny
- curated
- durable only

### Mem0

- primary long-tail recall
- use for prior conversations, ongoing projects, people, workflows, research continuity
- do not inject wholesale; rely on bounded retrieval

### Consolidation

Recommended:

- nightly review of new Mem0 facts
- weekly compaction of `USER.md` and `MEMORY.md`
- keep hot memory below ~80% of Hermes limits

## Why this split

`pi-brian` relies on broad semantic memory. Hermes built-in memory is intentionally small and should stay small. Tight integration means:

- hot memory shapes every turn
- Mem0 stays available automatically, not just as an optional tool
- token cost stays bounded because only relevant long-tail recall is injected
