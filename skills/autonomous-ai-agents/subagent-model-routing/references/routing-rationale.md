# Subagent Model Routing — Rationale & History

Supporting context for `subagent-model-routing/SKILL.md`. Not loaded automatically — reference when debugging routing decisions or reviewing past lessons.

---

## Why Model Selection Matters

`delegate_task` supports `model` and `provider` overrides. When omitted, subagents inherit `delegation.model` from `config.yaml` (typically `openrouter/auto`). The auto-router is a reasonable default, but:

- Coding tasks on general-purpose models consistently produce off-target code — the model invents its own abstractions instead of following integration requirements
- Simple extraction tasks on premium models waste 10–50× the cost for no quality benefit
- Mixed-provider batches fail silently if the wrong provider is set at the top level

## Cost Comparison (illustrative)

Tonight's session (actual): 9 Opus subagents ≈ $150
Same work with proper routing:
  - 3 code reviews × gemini-2.5-pro ≈ $5–8
  - 6 execution tasks × grok-4.1-fast ≈ $1–2
  - Total: ~$10 instead of $150 (15× cheaper)

**Jordan's principle (180426):** "Affordability is important, but effectiveness is important as well. If a model doesn't do the task correctly, or requires a lot of reruns, use a more capable model. Choose the cheapest model that will RELIABLY succeed on the FIRST try."

## 180426 Case Study — Orchestrator-Direct Beats Delegation for Code

Three scripts needed: `fetch_and_file_corte.py`, `fetch_and_file_invoices.py`, `scan_and_cache_emails.py`. All required tight integration with existing `daily_briefing.py` patterns.

Two subagent delegations failed:
- Subagent 1 (Sonnet): Created a generic "CortePDFHandler" class — missed the himalaya/IMAP integration entirely
- Subagent 2 (Sonnet via Gemini Flash): Created "court document" and "pending cases" infrastructure — misunderstood "Corte" as legal court, not the daily sales summary

Orchestrator wrote all 3 scripts directly in ~10 minutes. All worked on first run.

**Rule:** If a script needs to call 3+ existing functions from the codebase, or reference specific file paths/constants/patterns from an existing monolith, write it directly. Context transfer cost exceeds delegation benefit.

## 280426 — openrouter/auto Fabrication Failure

The auto-router selected a model for the daily briefing cron that read the prompt and *wrote prose about what the script would have returned* instead of executing `terminal()`. No error raised. Output file created. Status showed `ok`.

**Diagnostic:** Token counts were 4,667 in vs 91,618 in on a healthy run — the model never saw script output because it never ran the script.

**Fix:** Convert mandatory-tool-call cron jobs to pre-runners (script runs deterministically before agent, output injected into prompt). Or pin a specific reliable model. Never use `openrouter/auto` where tool execution is mandatory.

## QA Status (verified 200426)

- ✅ No override → config default (openrouter/auto → budget tier)
- ✅ Top-level model+provider override → lands on correct model
- ✅ Batch with per-task model override → each subagent routes correctly
- ✅ Cross-provider routing (anthropic main → openrouter subagent)
- ✅ Max concurrency (3 parallel subagents with different models)
- ✅ Failure modes: bogus model name, invalid provider
- ✅ Cron jobs honor model pin
- ✅ Unit tests pass
- ⏳ Nous Portal subagent from non-Nous parent — not yet verified

**Known issue:** `exit_reason` reports `"max_iterations"` when a subagent fails with an API error on its first call. Pre-existing bug in `delegate_tool.py`, not caused by our patch.

## Root Cause of Original Dispatch Bug (200426)

`run_agent.py` had TWO hardcoded dispatch sites for `delegate_task` that bypassed the registry's handler lambda. The schema could expose new parameters; those dispatch sites passed a fixed parameter list to `_delegate_task()`. Fix required patching BOTH:
1. `tools/delegate_tool.py` — schema + handler lambda + credential resolution + signature
2. `run_agent.py` — both direct-dispatch call sites to forward `model`, `provider`, `acp_command`, `acp_args`

Note: upstream's existing `acp_command` schema param was silently broken the same way. This is a legit upstream dispatch-path bug.

**As of upstream consolidation (200426):** `run_agent.py` dispatch sites were consolidated into `_dispatch_delegate_task()`. All new params only need to be added there. `TestRunAgentDispatchForwarding` guards this.

## Surgical Revert-and-Rebuild Workflow

When a framework patch is broken/unverified, don't layer fixes — revert to upstream clean, then rebuild with verification.

1. Inventory modified files: `git status` + `git diff <file>`
2. Separate patches: BROKEN vs DEPENDENT vs PROVEN
3. Back up and revert broken + dependent: `git checkout HEAD -- <files>`
4. Strip dead metadata from runtime state (jobs.json, config)
5. Stash proven patches, pull upstream, pop back
6. Verify parse: `python -c "import py_compile; py_compile.compile(f, doraise=True)"`
7. Rebuild feature with observability-first verification
8. Build the verification query BEFORE restart
9. Restart gateway, run test, grep log. Binary pass/fail.
10. If verification fails → revert same session. Never leave unverified patches overnight.

## Codex CLI Sandboxing Limitation (180426)

The `codex` CLI tool operates in its own sandboxed working directory. It cannot access files in `~/.hermes/scripts/` or other user home directories. Use `delegate_task` with Sonnet/Grok and `toolsets: ["terminal", "file"]` instead.

## Large-Scale Refactor Pattern (180426)

2060-line monolith → 383-line orchestrator + 9 modules completed in ~2 hours. Every module was written by the orchestrator — zero successful code-generation delegations for the extraction itself. Subagents were used only for reviews.

**Rule for refactors:** If extracting from a monolith where modules need to import shared helpers, reference the same constants, and maintain the same output format — do it yourself. Delegate the reviews, not the implementation.

## Auto Router Technical Notes

- Model ID: `openrouter/auto`
- Powered by: NotDiamond (third-party meta-routing AI)
- Pricing: standard rate of whichever model is selected — no premium
- Response includes `model` field — always shows what was actually chosen
- Account-level whitelist: OpenRouter Settings UI → Plugin Settings → Auto Router (currently unrestricted)
- Per-call whitelist via `plugins` param — not yet implemented for delegate_task

## OpenRouter Model ID Notation

OpenRouter uses **dot notation**: `claude-haiku-4.5`, `claude-sonnet-4.6`, `claude-opus-4.7`
Anthropic native API uses **hyphen notation**: `claude-haiku-4-5`

LLM training data is biased toward Anthropic native notation, which leaks into OpenRouter config writes. The live API lookup must happen every time a slug is written:

```bash
curl -s 'https://openrouter.ai/api/v1/models' | python3 -c \
  "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data'] if 'claude' in m['id'].lower()]"
```
