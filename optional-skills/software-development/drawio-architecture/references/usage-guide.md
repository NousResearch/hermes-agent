# Using the Skill

How to drive `drawio-architecture` once it's configured. This skill is
**proactive**: an agent should load it whenever a request implies a diagram
with 3+ components, complex data flow, or a visual explanation of a system.

## When to invoke it

- "Draw an architecture / system / service / network / cloud diagram"
- "Make a flowchart / ER diagram / sequence / UML class / C4 model"
- "Show me a diagram of how X talks to Y"
- Any visualization that should be editable in draw.io/diagrams.net

## Two delivery paths

| Path | What happens | Best for |
|------|--------------|----------|
| **A. MCP** | Agent calls `open_drawio_xml`/`_mermaid`/`_csv` → draw.io editor opens in a browser tab. No file on disk. | Chat agents (Claude, OpenCode, Devin, AGY, Copilot, Cursor). "Show me the diagram" flows. |
| **B. Local CLI** | Agent writes a `.drawio` file, then `drawio -x` exports PNG/SVG/PDF. | Headless/CI, image deliverables, no MCP available. |

Both paths share the **same XML authoring rules** below — only delivery differs.

## Recommended workflow

1. **Clarify** (1–3 questions if missing): diagram type, delivery mode
   (MCP open vs file+CLI vs image), scope/fidelity, specific technologies.
   Skip if the request is already specific.
2. **Configure** (path A) or resolve the draw.io CLI binary (path B).
3. **Plan** shapes, relationships, layout (LR/TB), grouping (tier/container),
   icon needs.
4. **Author** the XML (or write Mermaid for a standard typed diagram).
5. **Deliver**: MCP → call the tool; CLI → write `.drawio`, export, optionally
   vision self-check.
6. **Iterate** with targeted XML edits until approved.

## Authoring rules (must-follow)

- `id="0"` and `id="1"` are required root cells; user shapes start at `id="2"`
  with unique ids.
- Every edge `mxCell` needs `<mxGeometry relative="1" as="geometry" />` — never
  self-close edge cells.
- Use the rigid grid: `x = col*180+40`, `y = row*120+40`; rect `140×60`,
  diamond `140×80`, cylinder `100×70`.
- Don't hand-route edges — declare `source`/`target` and let the router work,
  or set `routing:"libavoid"` (MCP) for obstacle-avoiding wires.
- **Never** put XML comments (`<!-- -->`) in diagram output.
- Always include `html=1` in cell styles; use `&#xa;` for line breaks.
- Escape `&` `<` `>` `"` in attribute values.

## Iteration cheatsheet

| Request | XML edit |
|---------|----------|
| Change color of X | find `mxCell` by `value`, update `fillColor`/`strokeColor` |
| Add node | append `mxCell` vertex with next id near related nodes |
| Remove node | delete vertex + edges with matching `source`/`target` |
| Move/resize | update `x`/`y` or `width`/`height` in `mxGeometry` |
| Add arrow A→B | append edge `mxCell` with `source`/`target` |
| Change label | update `value` |
| Swap LR↔TB | full regeneration |

## Validate before delivery

```bash
python3 scripts/validate_drawio.py diagram.drawio --score
```

Full authoring reference: `SKILL.md` and `references/style-guide.md`,
`references/architecture-patterns.md`, `references/cloud-icons.md`.
