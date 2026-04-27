# multimodal-recall memory provider

Optional external memory provider that adds compact multimodal prefetch over the existing Hermes recall wrappers.

## Purpose
This provider is the phase-1 entry point for multimodal memory-provider integration.
It does not replace built-in memory or transcript recall. Instead, it adds bounded prefetch for turns that likely need prior screenshots, PDFs, OCR evidence, attachments, or other artifacts.

Current design:
- built-in `memory` remains the durable compact fact store
- `session_search` remains transcript-first recall
- `multimodal_recall` remains the low-level artifact wrapper
- `recall_with_artifacts` remains the main hybrid recall tool
- this provider adds optional compact prefetch on top of that stack

## Current phase-1 behavior
Implemented now:
- plugin discovery through `plugins.memory`
- config-driven activation through `memory.provider`
- conservative `is_available()` based on local multimodal MCP connectivity
- compact `system_prompt_block()` contribution
- bounded `prefetch(query)` for multimodal-style queries
- background/non-blocking `queue_prefetch(query)` for next-turn warmup using a compact per-query cache
- minimal `sync_turn(user_content, assistant_content)` lightweight signal capture
- minimal `on_session_end(messages)` signal capture for multimodal-relevant sessions
- minimal `on_memory_write(action, target, content)` lightweight signal capture
- `shutdown()` cleanup for background prefetch thread
- delegation to `recall_with_artifacts(...)` instead of reimplementing local-mmrag internals
- no provider-specific tool schemas yet (`get_tool_schemas() == []`)
- runtime drop protection: if multimodal MCP connectivity disappears, prefetch returns empty and does not attempt recall

Not implemented yet:
- provider-owned tool surface
- persistent rich `sync_turn(...)` writes beyond lightweight signal capture
- richer `on_session_end(...)` extraction beyond lightweight signal capture
- richer `on_memory_write(...)` mirroring beyond lightweight signal capture
- richer long-horizon scoring beyond the current lightweight scored trigger and cooldown

## What it does
- adds bounded multimodal prefetch for turns that likely reference:
  - screenshots
  - PDFs
  - OCR
  - attachments
  - evidence
  - artifacts
  - reports
- returns compact context only
- includes one small evidence hint when available
- stays additive to the existing Hermes memory stack

## What it does not do
- replace built-in memory
- replace `session_search`
- expose a duplicate large recall tool surface
- inject raw OCR/PDF dumps into the prompt
- reimplement local-mmrag retrieval internals inside the provider

## Availability model
This provider should only activate when the local multimodal MCP layer is actually ready.

Current rule:
- `is_available()` returns true only when the local multimodal recall wrapper reports connected MCP state

Implication:
- loading the provider module alone is not enough
- bare scripts that do not initialize MCP connections will usually see `available = False`
- this is intentional and prevents false-positive activation

## Prefetch trigger heuristic
Current heuristic is still conservative, but no longer pure substring matching.

The provider now uses a tiny scored trigger:
- strong multimodal/artifact terms such as `screenshot`, `pdf`, `attachment`, `image`, `ocr`, `evidence`, `artifact` and common Chinese equivalents can trigger prefetch immediately
- weaker artifact terms such as `report`, `document`, `file`, `note`, or `scan` are not enough on their own
- weak artifact queries can become eligible when recent lightweight provider signals (`_turn_signal`, `_session_end_signal`, `_memory_write_signal`) indicate multimodal context

This keeps prefetch selective while still helping follow-up turns like “what was the ETA from that file?” after Hermes has just discussed screenshot/PDF evidence.

## Cadence / cooldown behavior
The provider also applies a small in-memory cooldown to repeated fresh recalls:
- repeated identical fresh `prefetch(query)` calls are suppressed for a short window
- cached `queue_prefetch(...)` results still return normally
- this reduces low-value duplicate recall attempts without adding persistence

## Runtime behavior
When active and triggered:
1. provider checks runtime multimodal connectivity again
2. provider calls `recall_with_artifacts(query=..., session_limit=2, artifact_top_k=2)`
3. provider extracts:
   - `combined_summary`
   - at most one evidence source path
4. provider returns a compact block under ~800 chars

This context is later fenced and injected by Hermes into the current user message using `<memory-context>...</memory-context>`.

## Integration points
Relevant files:
- provider implementation:
  - `plugins/memory/multimodal-recall/__init__.py`
  - `plugins/memory/multimodal-recall/plugin.yaml`
- Hermes wrappers it depends on:
  - `tools/multimodal_recall_tool.py`
  - `tools/recall_with_artifacts_tool.py`
- manager/runtime integration:
  - `agent/memory_manager.py`
  - `run_agent.py`

## How to enable
Set Hermes config so the active external memory provider is:
- `multimodal-recall`

Because Hermes only allows one external memory provider at a time, enabling this provider means choosing it instead of another external provider.

## Test coverage
Main focused tests currently cover:
- plugin loadability
- conservative availability checks
- no tool schemas initially
- compact prefetch behavior
- runtime connectivity-drop protection
- MemoryManager prefetch integration
- config-driven provider activation in `AIAgent`
- system prompt block inclusion
- fenced prefetch injection into the current user message

Representative test files:
- `tests/plugins/memory/test_multimodal_recall_provider.py`
- `tests/agent/test_memory_provider.py`
- `tests/run_agent/test_run_agent.py`

## Recommended next steps
After this checkpoint, the most natural follow-on work is:
- richer long-horizon scoring if lightweight scoring proves insufficient
- optional `on_session_end(...)` enrichment beyond lightweight signal capture
- stronger confidence/cadence control for prefetch if real usage justifies it
- only later, consider provider-specific tools if wrappers prove insufficient

## Design rule to preserve
Keep this provider small, bounded, and additive.
If behavior starts duplicating wrappers or replacing transcript recall, the design is drifting in the wrong direction.
