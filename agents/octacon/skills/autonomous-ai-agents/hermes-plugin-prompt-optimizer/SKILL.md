---
name: hermes-plugin-prompt-optimizer
version: 1.0.0
description: Build a Hermes Agent plugin that intercepts user messages before the agent sees them, rewrites them for model-aware token efficiency, and tracks metrics.
platforms: [linux]
metadata:
  hermes:
    tags: [hermes-agent, plugin, prompt-engineering, metrics]
---

# Building a Hermes Prompt Optimizer Plugin

## Goal
Build a plugin at `~/.hermes/plugins/prompt-optimizer/` that:
1. Intercepts user messages BEFORE the agent (gateway + CLI)
2. Rewrites them for model-aware token efficiency and terminology
3. Scores before/after quality and stores metrics
4. Provides coaching via slash commands and weekly digest

## Hermes Hook Architecture

### Gateway path (already exists)
- Hook: `pre_gateway_dispatch` in `gateway/run.py` (~line 5959)
- Plugins return `{"action": "rewrite", "text": "..."}` to replace message

### CLI path (requires ~10-line core addition)
- Add `pre_user_message` to `VALID_HOOKS` in `hermes_cli/plugins.py`
- Fire hook in `cli.py` `chat()` method before `conversation_history.append()`
- Both hooks route to the same optimizer engine in the plugin

## Plugin File Layout

```
~/.hermes/plugins/prompt-optimizer/
├── plugin.yaml          # Manifest: name, version, hooks list
├── __init__.py          # register(ctx), hooks, slash commands, DB
├── model-profiles.yaml  # Per-model optimization strategies
├── weekly_digest.py     # Standalone script for cron job
└── metrics.db           # Auto-created SQLite store
```

## Key Components

### 1. Model Profile Resolution
```python
def _resolve_profile(model_name: str, provider: str) -> str:
    model_lower = (model_name or "").lower()
    if "claude" in model_lower or "anthropic" in provider.lower():
        return "claude"
    if "deepseek" in model_lower:
        return "deepseek"
    if "gpt" in model_lower or "openai" in provider.lower():
        return "openai"
    if "gemini" in model_lower:
        return "gemini"
    return "openai"  # default
```

### 2. Scoring Heuristic
- Clarity (action verb in first 10 words): +20
- Structure (markdown/lists/headers/XML): +20
- Specificity (numbers, paths, concrete terms): +20
- Conciseness (8-15 tokens/sentence ideal): +20
- Context (references to prior work): +20

### 3. Optimizer LLM Call
Use `ctx.llm` / `PluginLlm` with:
- Model: `deepseek-v4-flash`
- Provider: `nous`
- Timeout: 500ms
- Max tokens: 512

### 4. Metrics DB Schema
```sql
CREATE TABLE rewrites (
    id INTEGER PRIMARY KEY,
    ts REAL, session_id TEXT, platform TEXT,
    original TEXT, rewritten TEXT,
    quality_before REAL, quality_after REAL,
    token_delta_pct REAL, model_profile TEXT,
    model_used TEXT, mode TEXT, approved INTEGER, bypassed INTEGER
);
CREATE TABLE daily_stats (
    day TEXT PRIMARY KEY, rewrites INTEGER, approved INTEGER, bypassed INTEGER,
    avg_quality_before REAL, avg_quality_after REAL,
    avg_token_delta_pct REAL, top_profile TEXT
);
```

### 5. Slash Commands
- `/prompt-optimizer [auto|interactive|off|status]`
- `/prompt-stats [--week|--today|--best|--raw] [--model <m>] [--profile <p>]`

### 6. Badge Hook
Register `transform_llm_output` to append after each response:
```
📝 Optimized · +12% tokens · quality 45→72 · /prompt-stats
```

## Core Patches Required

### hermes_cli/plugins.py — add hook to VALID_HOOKS
```python
"pre_approval_request",
"post_approval_response",
# Fired in CLI/TUI before the user message is added to conversation history.
# Plugins receive the message string and may return a dict:
#   {"action": "rewrite", "text": "..."}  -> replace message, continue
#   {"action": "allow"}  /  None          -> normal dispatch
# Kwargs: message: str, session_id: str, platform: str
"pre_user_message",
```

### cli.py — fire hook before adding to history
```python
# Fire pre_user_message plugin hook
from hermes_cli.plugins import invoke_hook as _invoke_hook
_hook_results = _invoke_hook(
    "pre_user_message",
    message=message,
    session_id=self.session_id,
    platform="cli",
)
for _result in _hook_results:
    if isinstance(_result, dict) and _result.get("action") == "rewrite":
        _new_text = _result.get("text")
        if isinstance(_new_text, str):
            message = _new_text
        break

self.conversation_history.append({"role": "user", "content": message})
```

## Weekly Digest Cron
- Script: `~/.hermes/scripts/prompt-optimizer-weekly.py`
- Schedule: `0 10 * * 1` (Monday 10AM)
- Mode: `no_agent=True` — script produces output directly

## Enable Plugin
Add to `~/.hermes/config.yaml`:
```yaml
plugins:
  enabled:
    - disk-cleanup
    - prompt-optimizer
```

## Bypass Prefixes
Messages starting with these skip optimization:
- `/quick`
- `*simple`
- `#basic`

## Verification Checklist
- [ ] `python3 -m py_compile __init__.py` passes
- [ ] `python3 -m py_compile cli.py` passes (after patch)
- [ ] `python3 -m py_compile plugins.py` passes (after patch)
- [ ] `pre_user_message` in `VALID_HOOKS`
- [ ] Plugin in `config.yaml` `plugins.enabled`
- [ ] `metrics.db` auto-creates on first load
- [ ] Slash commands register without conflict
- [ ] Weekly digest cron created and scheduled
