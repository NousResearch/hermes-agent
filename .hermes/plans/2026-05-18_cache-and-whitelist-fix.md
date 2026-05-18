# Plan: Fix Background Review Crash + Enable Prompt Caching for DeepSeek/Kimi

## Goal
Fix two bugs in `run_agent.py` that are burning money and breaking functionality:
1. Background review crashes because `set_thread_tool_whitelist` is called but never defined
2. DeepSeek and Kimi providers don't get prompt cache markers, costing 2-3x on multi-turn sessions

## Current Context

**Bug 1: Missing function definitions**
- `_spawn_background_review()` (line ~3868) calls `set_thread_tool_whitelist()` at line 3963 and `clear_thread_tool_whitelist()` at line 3981
- Neither function is defined anywhere in our fork
- Upstream has these functions (referenced at line 4357 in upstream's run_agent.py per issue #26352)
- The background review is Hermes's post-conversation reflection mechanism — it reviews the conversation and saves memory/skill improvements
- Every session that completes triggers this crash. The error is caught silently, so you only see it in logs
- Impact: The self-improvement loop is partially broken — background review for memory and skill saves fails every time

**Bug 2: Missing cache markers for DeepSeek/Kimi**
- `_anthropic_prompt_cache_policy()` (line 3140) decides which providers get Anthropic-style `cache_control` breakpoints
- Current coverage: Native Anthropic, OpenRouter Claude, Anthropic-wire gateways with Claude, MiniMax, Qwen/Alibaba
- NOT covered: DeepSeek (our primary provider), Kimi/Moonshot
- DeepSeek's API supports `cache_control` markers (documented)
- Kimi/Moonshot reports `prompt_cache_hit_tokens` in usage responses — the logging exists, just no markers
- Without markers: 0% cache hits, full price on every turn
- With markers: ~75% cache read discount on cached tokens
- The `_log_cache_stats()` function already monitors Kimi cache stats — it's ready to validate

## Proposed Approach

Both fixes are isolated edits to `run_agent.py`. No new files, no dependency changes, no config changes.

### Fix 1: Add missing thread whitelist functions

Find the upstream commit that introduced `set_thread_tool_whitelist` and `clear_thread_tool_whitelist`, cherry-pick it. 

Fallback if cherry-pick conflicts: write the functions manually based on the pattern from the background review code. The functions likely:
- Use `threading.local()` to store a per-thread whitelist set + deny message format
- Check the whitelist in the tool dispatch path (in `handle_function_call` or similar)
- The deny message format is a template string for the error when a non-whitelisted tool is called

### Fix 2: Extend caching policy

Add two new branches to `_anthropic_prompt_cache_policy()`:

**DeepSeek**: DeepSeek uses OpenAI-wire chat completions, not Anthropic messages. They support `cache_control` markers on the message content blocks (envelope layout, same as OpenRouter Claude). Condition: provider is "deepseek" AND model contains "deepseek".

**Kimi/Moonshot**: Moonshot uses OpenAI-wire. They have `prompt_cache_hit_tokens` in their usage response. Condition: provider is "moonshot" AND model contains "kimi".

Both use `native_anthropic=False` (envelope layout — markers on inner content blocks, not top-level messages).

## Step-by-Step Plan

### Step 1: Locate the upstream commit for thread whitelist
```bash
cd /Users/wills_mac_mini/.hermes/hermes-agent
git log upstream/main --oneline -- run_agent.py | head -20
# Look for commits mentioning "background-review", "thread whitelist", "tool whitelist"
```

### Step 2: Cherry-pick the thread whitelist commit
```bash
git cherry-pick <commit-sha>
```
If clean: commit with P-number, update PATCHES.md.
If conflicts: resolve manually, commit with P-number.

### Step 3: Verify Fix 1
```bash
grep -n "def set_thread_tool_whitelist\|def clear_thread_tool_whitelist" run_agent.py
# Should find both function definitions
```

### Step 4: Add DeepSeek caching branch
Insert after the Qwen/Alibaba block (around line 3227 in our version), before `return False, False`:

```python
# DeepSeek: OpenAI-wire transport, documented cache_control support.
# Without markers: 0% cache hits on multi-turn.
model_is_deepseek = "deepseek" in model_lower
if provider_lower == "deepseek" and model_is_deepseek:
    return True, False  # envelope layout

# Moonshot/Kimi: OpenAI-wire transport, prompt_cache_hit_tokens in usage.
model_is_kimi = "kimi" in model_lower
if provider_lower == "moonshot" and model_is_kimi:
    return True, False  # envelope layout
```

### Step 5: Verify Fix 2
```bash
# No syntax errors
python -m py_compile run_agent.py
```

### Step 6: Git workflow
```bash
git add run_agent.py scripts/hermes-patches/PATCHES.md
git commit -m "fix: add thread tool whitelist functions + extend prompt caching to DeepSeek/Kimi"
```

### Step 7: Functional verification
1. Restart Hermes and run a session
2. Check `~/.hermes/logs/errors.log` — no more "set_thread_tool_whitelist is not defined"
3. Check `~/.hermes/logs/agent.log` for "cache_stats" lines showing hits for DeepSeek/Kimi

## Files to Change

| File | Change | Complexity |
|------|--------|------------|
| `run_agent.py` | Cherry-pick or manually add `set_thread_tool_whitelist` + `clear_thread_tool_whitelist` definitions | Small |
| `run_agent.py` | Add DeepSeek + Kimi branches to `_anthropic_prompt_cache_policy()` | Trivial (~8 lines) |
| `scripts/hermes-patches/PATCHES.md` | Register new P-number(s) | Trivial |

## Risks

1. **Cherry-pick conflicts on Fix 1**: The thread whitelist commit may touch lines that our patches have modified. Manual resolution needed. Low risk of regression since the functions are net-new additions.
2. **DeepSeek cache markers may cause errors**: If DeepSeek's API doesn't accept the exact Anthropic-format `cache_control`, we might get 400 errors. Mitigation: if errors occur, revert this single branch — it's isolated code.
3. **Kimi cache marker format unknown**: We're assuming Moonshot accepts the same envelope layout as OpenRouter. If not, Kimi will ignore the markers silently (no errors, just no cache hits). Harmless fallback.

## Verification Checklist
- [ ] `set_thread_tool_whitelist` and `clear_thread_tool_whitelist` have definitions in run_agent.py
- [ ] `python -m py_compile run_agent.py` passes
- [ ] `_anthropic_prompt_cache_policy()` returns `True, False` for provider="deepseek" + model containing "deepseek"
- [ ] `_anthropic_prompt_cache_policy()` returns `True, False` for provider="moonshot" + model containing "kimi"
- [ ] PATCHES.md updated with new P-number
- [ ] errors.log shows no more "set_thread_tool_whitelist is not defined" after restart
