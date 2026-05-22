Create a professional infographic following these specifications:

## Image Specifications

- **Type**: Infographic
- **Layout**: bento-grid
- **Style**: technical-schematic (engineering blueprint variant)
- **Aspect Ratio**: 1:1 (square)
- **Language**: English

## Core Principles

- Follow the bento-grid layout precisely with varied cell sizes
- Apply technical-schematic aesthetics consistently throughout
- Keep information concise, highlight keywords and core concepts
- Use ample whitespace for visual clarity
- Maintain clear visual hierarchy with a hero cell for the headline metric

## Style Guidelines (technical-schematic blueprint)

- Color palette: deep blue background (#1E3A5F), white lines and text, amber accent (#F59E0B) ONLY on the hero metric and critical deltas, cyan callouts for measurement annotations
- Grid pattern overlay across the entire canvas — fine white grid lines on the deep blue background
- All-caps technical stencil typography for headers; clean sans-serif for body
- Dimension lines with arrowheads connecting metrics to their cells
- Technical symbols where appropriate (gear icons, flow arrows, modular block diagrams)
- Consistent stroke weights — bold for cell borders, thin for grid, medium for connector lines
- Engineering spec-sheet aesthetic: feels like a printed architectural blueprint, austere and precise

## Layout Guidelines (bento-grid)

- Hero cell (TOP-CENTER or LEFT, occupying ~40% of canvas): "−61 COMPLEXITY · 79 → 18" headline metric in massive amber-on-blue, with subtitle "convert_messages_to_anthropic refactored"
- 7 helper cells in a 2x4 or 3x3 grid showing each extracted helper as its own modular block — each cell has the helper name in all-caps, its complexity number, and one-line role
- Metrics strip cell: BEFORE/AFTER table with deltas (185 statements → ~70, 79 C → 18 C, +5 violations intentional)
- Test validation cell: "152/152 + 213/213 PASS" with checkmark stencil
- Footer strip across bottom: "PR #27784 · agent/anthropic_adapter.py · @kshitijk4poor · NousResearch/hermes-agent"

## Content to render

**Main title (top of canvas, all caps):** "ANTHROPIC ADAPTER · 1-INTO-7 EXTRACTION"
**Subtitle:** "PR #27784 — convert_messages_to_anthropic refactor"

**Hero cell (largest, amber accent):**
- "−61"
- "CYCLOMATIC COMPLEXITY"
- "79 → 18 MAX (−77%)"
- Subtext: "convert_messages_to_anthropic · pure code motion · zero behavior change"

**7 helper cells (one per helper, each its own modular block):**

1. _convert_assistant_message · C<10 · "Assistant msg → content blocks"
2. _convert_tool_message_to_result · C=12 · "Tool msg → tool_result + merge"
3. _convert_user_message · C<10 · "User msg validation"
4. _strip_orphaned_tool_blocks · C=15 · "Orphan tool_use removal"
5. _merge_consecutive_roles · C=13 · "Anthropic role-alternation"
6. _manage_thinking_signatures · C=18 · "Strip/preserve by endpoint"
7. _evict_old_screenshots · C<10 · "Keep most recent 3 images"

**Metrics cell (table format with arrows):**
- MAX FUNCTION COMPLEXITY: 79 → 18 (−77%)
- MAX STATEMENTS/FUNCTION: 185 → ~70 (−62%)
- LOC FILE-WIDE: −4
- MAIN FUNCTION LOC: 395 → 63

**Test validation cell (checkmark stencil):**
- test_anthropic_adapter.py: 152/152 PASS
- test_auxiliary_client.py: 172/172 PASS
- test_azure_identity_adapter.py: 39/39 PASS
- test_bedrock_1m_context.py: 2/2 PASS

**Behavior preservation cell:**
"ZERO LOGIC CHANGES · ANTHROPIC + KIMI + DEEPSEEK + MINIMAX + AZURE FOUNDRY + BEDROCK SEMANTICS PRESERVED"

**Footer strip:**
"PR #27784 · agent/anthropic_adapter.py · cherry-picked from #23968 · @kshitijk4poor · NousResearch/hermes-agent"

## Text Requirements

- All text in English, all-caps for headers
- Hero metric "−61" in amber (#F59E0B), oversized, with thick blueprint stencil treatment
- Helper names in white technical stencil
- Complexity numbers (C=12, C=18, etc.) in cyan callouts
- "BEFORE" labels in white-on-blue, "AFTER" labels in amber-on-blue
- Footer in small white stencil

Generate the infographic now as a square engineering blueprint.
