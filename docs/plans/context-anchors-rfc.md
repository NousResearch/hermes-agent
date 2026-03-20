# RFC: Context Anchors - Persistent Project Memory That Survives Compression

**Author:** @DarkPancakes (Louis) + Hermes Agent  
**Date:** 2026-03-19  
**Status:** Draft  
**Related issues:** #2046 (observation masking), #1136 (input compression), #886 (silent history loss)

## Problem

When context compression fires, the LLM summary is lossy. Critical project-specific rules, states, and constraints get lost. Users who manage multiple long-running projects (trading bots, websites, deployments) lose track of what was fixed, what's running, and what NOT to touch.

Real-world failure modes observed:
1. Agent re-debugs a bug that was fixed hours ago, wasting time and frustrating the user
2. Agent forgets project-specific rules ("reservations work, stop re-debugging")  
3. Agent loses track of which services are running on which ports
4. Agent repeats destructive operations that were already rolled back

The existing `MEMORY.md` (2200 chars) is too small for multi-project state. Users work around this with manual context files (`~/.hermes/context/*.md`) that the agent reads on request, but after compression these files are NOT re-read automatically.

## Solution: Context Anchors

Context anchors are user-defined markdown files that:
1. **Auto-inject** into the conversation after every compression (read path)
2. **Auto-update** when the agent completes work on the associated project (write path)

This creates a bidirectional, persistent project memory that survives compression indefinitely.

## Design

### Configuration

New section in `config.yaml`:

```yaml
context_anchors:
  - path: ~/.hermes/context/eclatauto.md
    keywords: [eclatauto, eclatauto13.fr, /var/www/eclatauto]
    max_chars: 5000
    
  - path: ~/.hermes/context/sirius.md
    keywords: [sirius, polymarket, paper_results, /root/sirius]
    max_chars: 5000
    
  - path: ~/.hermes/context/clawhub.md
    keywords: [clawhub, simmer, ClawHub]
    max_chars: 5000

# Global limits
context_anchors_max_total_chars: 20000  # cap total injection size
context_anchors_auto_save: true         # enable write-back (default: true)
```

### Read Path: Post-Compression Injection

After `_compress_context()` summarizes middle turns:

1. Load all configured anchor files from disk
2. Truncate each to `max_chars` (head/tail, same algo as AGENTS.md)
3. Inject as a single user message with a distinctive prefix:

```
[ANCHORED PROJECT CONTEXT - These files contain persistent project state 
that survives compression. They reflect the current ground truth. Do NOT 
repeat work described here. Do NOT re-debug issues marked as fixed.]

## ~/.hermes/context/eclatauto.md
[file contents]

## ~/.hermes/context/sirius.md
[file contents]
```

4. Total injection capped at `context_anchors_max_total_chars`
5. Anchors are also referenced in the compression summary prompt so the summarizer knows NOT to re-summarize information that will be re-injected from files

### Write Path: Auto-Save Project State

When the agent detects it's working on a project with a configured anchor:

**Detection heuristics (any match triggers):**
- Tool calls reference paths matching anchor keywords (file reads/writes, terminal commands)
- User message contains anchor keywords
- Web requests target URLs matching anchor keywords

**Save triggers:**
- Pre-compression flush (alongside `flush_memories()`)
- End of a multi-step task (after 5+ tool calls on the same project)
- Explicit user request

**Save mechanism:**
A dedicated flush prompt asks the model:

```
[System: You've been working on project "{project_name}" (anchor: {path}).
Update the anchor file with any new state: what changed, what's fixed, 
what's running, what NOT to touch. Keep it factual and concise. 
Read the current file first, then patch only what changed.]
```

The model gets the `read_file` and `patch` tools (not `write_file`, to avoid clobbering) and updates the anchor file incrementally.

### Integration Points

#### In `_compress_context()` (run_agent.py)

```python
def _compress_context(self, messages, system_message, *, approx_tokens=None, task_id="default"):
    # 1. Pre-compression memory flush (existing)
    self.flush_memories(messages, min_turns=0)
    
    # 2. NEW: Pre-compression anchor save
    self._flush_anchor_state(messages)
    
    # 3. Compress (existing)
    compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens)
    
    # 4. Re-inject todo snapshot (existing)
    # 5. Re-inject read files list (existing)
    
    # 6. NEW: Re-inject context anchors
    anchor_content = self._load_context_anchors()
    if anchor_content:
        compressed.append({"role": "user", "content": anchor_content})
    
    # ... rest of existing logic
```

#### In the summary prompt (context_compressor.py)

Add to the compression prompt:
```
Note: The following project context files will be re-injected after compression 
and do NOT need to be included in your summary: {anchor_paths}
Focus your summary on actions taken, decisions made, and conversation flow 
rather than project state that's already persisted in anchor files.
```

This prevents the summary from duplicating anchor content, saving tokens.

#### New helper methods on AIAgent

```python
def _load_context_anchors(self) -> Optional[str]:
    """Load all configured anchor files for post-compression injection."""
    
def _flush_anchor_state(self, messages: list):
    """Auto-save project state to the relevant anchor file."""
    
def _detect_active_project(self, messages: list) -> Optional[dict]:
    """Detect which project anchor (if any) the recent work relates to."""
```

### Compression Summary Enhancement

The `_generate_summary()` prompt in `context_compressor.py` is enhanced to be anchor-aware:

```
Create a concise handoff summary...

IMPORTANT: The following project files will be automatically re-injected 
after compression. Do NOT duplicate their content in your summary:
{list of anchor file paths}

Instead, reference them: "See {path} for current project state."
Focus on: actions taken, user preferences expressed, decisions made, 
and any task progress not captured in the anchor files.
```

## Backwards Compatibility

- Feature is opt-in via `config.yaml`. No anchors configured = zero behavior change.
- No changes to MEMORY.md, USER.md, or AGENTS.md behavior.
- No new tools required (uses existing `read_file` and `patch` for auto-save).
- No new dependencies.

## Token Budget

Worst case with 4 anchors at 5000 chars each = ~7000 tokens injected post-compression. This is comparable to a large AGENTS.md (20K char limit = ~7000 tokens). The `context_anchors_max_total_chars` cap prevents runaway injection.

The summary prompt enhancement SAVES tokens by telling the compressor not to duplicate anchor content. Net token impact should be neutral or positive.

## Security

- Anchor files are subject to the same injection scanning as AGENTS.md (`_scan_context_content()`)
- Auto-save uses `patch` (not `write_file`) to prevent clobbering
- Auto-save only triggers for configured paths, never arbitrary files
- Keywords are user-configured, not model-chosen (prevents prompt injection expanding scope)

## Testing Plan

1. **Unit: anchor loading** - config parsing, file reading, truncation, injection format
2. **Unit: project detection** - keyword matching against tool calls and messages
3. **Unit: summary prompt** - verify anchor paths are referenced in compression prompt
4. **Integration: compression cycle** - verify anchors survive multiple compression rounds
5. **Integration: auto-save** - verify anchor file is updated after multi-step work
6. **Edge: missing files** - anchor configured but file doesn't exist yet (create on first save)
7. **Edge: file too large** - truncation works correctly
8. **Edge: no anchors configured** - zero behavior change

## Implementation Estimate

- config parsing: ~50 lines (hermes_cli/config.py)
- anchor loading + injection: ~80 lines (run_agent.py)
- project detection: ~60 lines (run_agent.py)
- anchor auto-save: ~100 lines (run_agent.py)
- compression prompt enhancement: ~20 lines (context_compressor.py)
- tests: ~300 lines
- docs: ~100 lines

Total: ~710 lines of code, ~400 lines of tests/docs
