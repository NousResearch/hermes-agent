# Infographic: PR #27784 — convert_messages_to_anthropic refactor

## Hero metric
**−61 cyclomatic complexity** in `agent/anthropic_adapter.py` (79 → 18 max).
**−4 LOC** net file-wide. **77% drop** in single-function complexity ceiling.

## Title
ANTHROPIC ADAPTER · 1-INTO-7 EXTRACTION
PR #27784 · agent/anthropic_adapter.py · @kshitijk4poor

## Section 1: BEFORE (left side)
**convert_messages_to_anthropic**
- 185 statements
- 90 branches
- Cyclomatic: 79
- Did 7 jobs in one function

Inline responsibilities mixed together:
1. Walk + dispatch by role
2. Tool-result conversion
3. Orphan tool-use stripping
4. Same-role merging
5. Thinking-signature management
6. Screenshot eviction
7. Final assembly

## Section 2: AFTER (right side)
**convert_messages_to_anthropic** — now 63 lines, C<10
Plus 7 single-responsibility helpers:

| Helper | C | Role |
|---|---|---|
| _convert_assistant_message | <10 | Assistant msg → content blocks |
| _convert_tool_message_to_result | 12 | Tool msg → tool_result + merge |
| _convert_user_message | <10 | User msg validation + conversion |
| _strip_orphaned_tool_blocks | 15 | Strip orphan tool_use + tool_result |
| _merge_consecutive_roles | 13 | Anthropic role-alternation enforce |
| _manage_thinking_signatures | 18 | Strip/preserve/downgrade by endpoint |
| _evict_old_screenshots | <10 | Keep most recent 3 images |

## Section 3: METRICS
| Metric | Before | After | Δ |
|---|---:|---:|---:|
| Max function complexity | 79 | 18 | −77% |
| Max statements/function | 185 | ~70 | −62% |
| LOC (file-wide) | — | — | **−4** |
| C901 violations | 3 | 8 | +5 (intentional split) |

## Section 4: ZERO BEHAVIOR CHANGE
- Pure code motion — no logic edits
- Mutating helpers update `result` in place (same as inline)
- `_merge_consecutive_roles` returns new list — caller rebinds
- Anthropic / Kimi / DeepSeek / MiniMax / Azure Foundry / Bedrock semantics preserved
- Thinking-signature handling identical to pre-refactor

## Section 5: TEST VALIDATION
- tests/agent/test_anthropic_adapter.py — **152 / 152 pass**
- tests/agent/test_auxiliary_client.py — **172 / 172 pass**
- tests/agent/test_azure_identity_adapter.py — **39 / 39 pass**
- tests/agent/test_bedrock_1m_context.py — **2 / 2 pass**

## Footer
File: agent/anthropic_adapter.py
Original PR: #27784 (cherry-pick of #23968)
Salvage commit: 9c102b937 (kshitijk4poor authorship preserved)
Repo: NousResearch/hermes-agent
