---
name: figma-integration
description: Code-to-Figma and Figma-to-code workflows for design review and visual iteration
---

# Figma Integration

**Goal:** Send code implementations to Figma for design review, documentation, and visual iteration

**Your Role:** Guide the agent through specification-driven Figma export using html.to.design MCP Server

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## WHEN TO USE

Send code to Figma when:
- Component states need visual documentation (hover, active, disabled, etc.)
- Design system components require Figma library representation
- Prototype pages need designer review and feedback
- Visual adjustments are easier to iterate in Figma than code
- Design-code parity documentation is needed

---

## STEP PROCESSING RULES

1. **READ COMPLETELY**: Always read the entire step file before action
2. **FOLLOW SEQUENCE**: Execute all sections in order
3. **WAIT FOR INPUT**: Halt at menus and wait for user selection

---

## STEPS

Execute steps in `./steps-f/`:

| Step | File | Purpose | Time |
|------|------|---------|------|
| 01 | step-01-connection-check.md | Verify MCP connection, guide setup | ~5-10 min |
| 02 | step-02-identify-export-type.md | Determine export type | ~2-3 min |
| 03 | step-03-prepare-specifications.md | Find/create specs with OBJECT IDs | ~5-15 min |
| 04 | step-04-generate-validate.md | Generate Figma-compatible HTML | ~5-10 min |
| 05 | step-05-execute-export.md | Execute export and verify | ~2-5 min |

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/figma-plugin-setup.md` | Plugin installation and MCP verification |
| `data/figma-spec-preparation.md` | Code analysis and OBJECT ID generation |
| `data/figma-integration-guide.md` | Figma-to-code workflow guide |
| `data/figma-integration-summary.md` | Integration overview and concepts |
| `data/figma-designer-guide.md` | Guide for designers in Figma |
| `data/figma-mcp-integration.md` | MCP integration technical docs |
| `data/mcp-server-integration.md` | MCP server setup and configuration |
| `data/tools-reference.md` | Figma MCP tools and parameters |
| `data/when-to-extract-decision-guide.md` | Decision tree for when to use Figma integration |
| `data/prototype-to-figma-workflow.md` | Iterative refinement workflow |

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action (visual refinement, design system update, or re-render)
