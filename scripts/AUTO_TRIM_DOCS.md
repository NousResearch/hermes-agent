# AUTO_TRIM_DOCS.md — Hermes Context Auto-Trimmer Documentation

## Overview

The Hermes Agent includes an automatic context trimming system (`auto_trim.py`) that
monitors the token budget of in-flight conversation contexts and trims them to fit
within configurable limits. This prevents context overflow errors, reduces latency,
and manages API costs.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Hermes Gateway                      │
│  (telegram_bridge.py / gateway_integration.py)      │
└──────────┬──────────────────────────────────────────┘
           │ signals context-status.json
           ▼
┌─────────────────────────────────────────────────────┐
│              auto_trim.py                            │
│                                                     │
│  1. Read context-status.json from bridge/signals/   │
│  2. Count tokens across all blocks                   │
│  3. If over threshold → two-phase trim:             │
│     Phase 1: Evict T5/T6 blocks (deletion)          │
│     Phase 2: Compress T3/T4 blocks (Ollama)         │
│  4. Archive deleted blocks to logs/archive/          │
│  5. Write trimmed result back to signals/responses/  │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│          Ollama (local inference)                    │
│  - Default model: qwen3:8b                          │
│  - Used for text compression only                    │
└─────────────────────────────────────────────────────┘
```

## Priority Tiers

Blocks in the context are assigned priority levels 0–6:

| Priority | Tier         | Treatment               |
|----------|--------------|------------------------|
| 0        | Identity     | Never trimmed           |
| 1        | Task         | Never trimmed           |
| 2        | High-import  | Never trimmed           |
| 3        | Semantic     | Compress (Phase 2)      |
| 4        | Background   | Compress (Phase 2)      |
| 5        | Tool output  | Delete first (Phase 1)  |
| 6        | Conversation | Delete first (Phase 1)  |

### Type Coercion

Non-integer priority values (strings, None, lists) default to priority **6**
(lowest) via `_block_priority()`. This prevents crashes from malformed block data.

## Configuration

All settings come from environment variables (typically set in `.env`):

| Variable             | Default             | Description                          |
|----------------------|---------------------|--------------------------------------|
| `OLLAMA_HOST`        | `http://localhost:11434` | Ollama API endpoint              |
| `TRIM_MODEL`         | `qwen3:8b`          | Model used for compression          |
| `TRIM_THRESHOLD`     | `100000`            | Token count that triggers trimming   |
| `TARGET_TOKENS`      | `60000`             | Target after trimming                |
| `WORKSPACE`          | auto-detected       | Pipeline root directory             |
| `DRY_RUN`            | `0`                 | Set to `1` for dry-run mode         |
| `MAX_PAUSE_SECONDS`  | `3600`              | Auto-resume after N seconds (0=off) |

### Safety Floor: `MIN_BLOCKS_KEPT`

Hard-coded to **3**. During Phase 1 eviction, the system will never delete
blocks if doing so would leave fewer than 3 blocks remaining, regardless of
budget status.

## Two-Phase Trimming Strategy

### Phase 1 — Eviction
- Targets priority 5 (tool output) and 6 (conversation) blocks
- Deletes blocks in priority order (6 first, then 5)
- **Archives before deletion** — copies to `logs/archive/` with timestamp
- Respects `MIN_BLOCKS_KEPT=3`
- Stops when budget is met or no more evictable blocks

### Phase 2 — Compression
- Targets priority 3 (semantic) and 4 (background) blocks
- Sends block content to Ollama for summarization
- Replaces original content with compressed summary
- Tracks compression ratio for logging
- Skips blocks listed in `protected-blocks.json`

## Signal Files

The trimmer supports external control via signal files in `bridge/signals/`:

| Signal              | Effect                              |
|---------------------|-------------------------------------|
| `pause-trim`        | Suspends all trimming               |
| `protected-blocks.json` | JSON array of block IDs to skip |
| `trigger-trim.json` | Manual trigger with mode/target     |

### Pause/Resume

```bash
# Pause trimming
python3 auto_trim.py --pause "Maintenance window"

# Resume trimming
python3 auto_trim.py --resume

# Check status
python3 auto_trim.py --pause-status
```

### Block Protection

```bash
# Protect a block from trimming
python3 auto_trim.py --protect block_id_123

# Remove protection
python3 auto_trim.py --unprotect block_id_123
```

Auto-resume: If `MAX_PAUSE_SECONDS` is set (>0), the pause signal is
automatically removed after the specified duration.

## Usage

```bash
# Automatic mode (reads trigger signals, respects pause)
python3 auto_trim.py

# Dry run (analyse only, don't modify)
python3 auto_trim.py --dry-run

# Custom budget
python3 auto_trim.py --target 40000

# Custom model
python3 auto_trim.py --model deepseek-coder:6.7b

# Validate inputs
python3 auto_trim.py --validate
```

## Integration with Gateway

The `gateway_integration.py` module wraps `auto_trim.py` functions for use
by the Hermes Agent gateway:

| Gateway Function                | Auto-Trim Equivalent        |
|---------------------------------|-----------------------------|
| `gateway_pause_trimming()`      | `pause_trimming()`          |
| `gateway_resume_trimming()`     | `resume_trimming()`         |
| `gateway_set_block_protected()` | `set_block_protected()`     |
| `gateway_status()`              | Internal status check       |
| `gateway_get_pause_info()`      | Pause signal metadata       |

These are imported by `gateway/platforms/base.py` and called during the
message lifecycle hooks in `gateway/run.py`.

## File Inventory

| File                 | Purpose                           | Location                      |
|----------------------|-----------------------------------|-------------------------------|
| `auto_trim.py`       | Core engine (784 lines)          | `scripts/auto_trim.py`       |
| `test_auto_trim.py`  | 29 test cases                     | `scripts/test_auto_trim.py`  |
| `auto-trim.sh`       | Thin bash wrapper                 | `scripts/auto-trim.sh`       |
| `AUTO_TRIM_DOCS.md`  | This documentation                | `scripts/AUTO_TRIM_DOCS.md`  |
| `context_orchestrator.py` | Orchestrator with pause/protect | `scripts/context_orchestrator.py` |
| `gateway_integration.py` | Gateway wrapper functions       | `scripts/gateway_integration.py` |
| `test_pause_protection.py` | Pause/protect tests           | `scripts/test_pause_protection.py` |

## Testing

```bash
# Full test suite (29 cases, requires Ollama)
python3 scripts/test_auto_trim.py -v

# Without Ollama (skips compression tests)
python3 scripts/test_auto_trim.py -v --no-ollama

# Specific test class
python3 -m pytest scripts/test_auto_trim.py::TestPhase1Eviction -v
```

## Troubleshooting

- **"Cannot resolve pipeline root"**: Set `WORKSPACE` env var to the pipeline directory
- **"Ollama returned empty response"**: Check `Ollama_HOST` and ensure model is loaded
- **"Missing directory: bridge/signals/"**: Run `auto-trim.sh --validate` to check setup
- **Trim not triggering**: Verify `TRIM_THRESHOLD_TOKENS` is less than actual token count
- **Blocks not being archived**: Check `logs/archive/` directory permissions