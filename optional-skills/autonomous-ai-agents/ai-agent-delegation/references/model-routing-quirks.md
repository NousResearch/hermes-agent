# OpenCode Zen Model Routing — Confirmed Quirks

## M2.7 vs M2.5 Bug (May 2026)

**What:** `minimax-m2.5-free` on OpenCode Zen resolves to `minimax/minimax-m2.7-20260318`, NOT `minimax-m2.5`.

**API response when resolved to M2.7:**
```json
{
  "model": "minimax/minimax-m2.7-20260318",
  "choices": [{
    "message": {
      "content": null,
      "reasoning": "user wants Greeting function. S  content S",
    },
    "finish_reason": "length"
  }]
}
```

**API response when resolved to M2.5:**
```json
{
  "model": "minimax/minimax-m2.5",
  "choices": [{
    "message": {
      "content": "Hi!",
      "reasoning": null
    },
    "finish_reason": "stop"
  }]
}
```

**Why it matters:** M2.7 uses extended reasoning. With `max_tokens` ≤ 10, all tokens go to `reasoning`, leaving `content: null`. Claude Code sees `content: null` → thinks call failed → retries → `Retrying in Xs...` loop.

**Fix applied:**
- Claude Code `settings.json`: `ANTHROPIC_MODEL: "minimax-m2.5"` (not `-free`)
- Claude Code `settings.json`: `maxTokens: 2048`
- Hermes can still use `minimax-m2.5-free` (doesn't hit the same M2.7 routing in Hermes context)

## OpenCode Zen Model Catalog (May 2026)

Available models via OpenCode Zen:
- `minimax-m2.7`, `minimax-m2.5`, `minimax-m2.5-free`
- `kimi-k2.6`, `kimi-k2.5`
- `qwen3.6-plus`, `qwen3.5-plus`
- `deepseek-v4-flash-free`, `ring-2.6-1t-free`, `trinity-large-preview-free`
- `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`
- `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-pro`
- `gemini-3.1-pro`, `gemini-3-flash`

## Claude Code Hook Overhead Benchmark (May 2026)

Before optimization: 9 hook events, ~0.591s per-task overhead, M2.7 routing bug.
After optimization: 4 hook events, ~0.655s per-task (post-edit heavier), model fixed.

Key insight: hook overhead was NOT the main problem. The M2.7 routing bug was.

Removed hooks: UserPromptSubmit (route), auto-memory import/sync, SubagentStart/Stop, PreCompact, Notification. Kept: SessionStart, SessionEnd, PreToolUse (Bash), PostToolUse (Bash + Edit).

## Test Command
```
curl -s -X POST https://opencode.ai/zen/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model":"minimax-m2.5","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
```
Check `model` field in response — if it says `minimax-m2.7`, the routing bug is active.

## WSL-to-Windows Invocation (May 2026)

From Hermes (WSL), invoking the Windows `claude` binary requires:
```bash
/mnt/c/Windows/System32/cmd.exe /c "cd /d <WINDOWS_PATH> && claude -p 'task' --output-format json"
```

**Stdin pipes fail:** `echo "task" | claude -p` drops the prompt. Use direct `-p 'task'` argument.
**`--verbose` requirement:** `--output-format stream-json` needs `--verbose` flag. Plain `--output-format json` does not.