# Clean-room ASCII canvas/docs-renderer prototype

Status: local spike prototype for Kanban task `t_1e7d54b3`.

## Provenance

Source-spike artifact read: `/home/filip/spearhead-execution/20260528-source-spikes/ascii-draw/closure-summary.md`.

Behavior-level observations used:

- fixed 2D character grid;
- semantic line glyph palette with corners, crossings, and T-junctions;
- collision-aware line merging;
- simple arrows;
- structured table rendering;
- indentation/tree rendering;
- optional FIGlet title as presentation sugar.

No qindapao/ascii-draw or Nokse22/ascii-draw source text, structure, files, or symbols were copied. The referenced project is GPL-licensed, so this prototype is clean-room and behavior-only.

## Unicode width decision

Canvas coordinates are display-cell based. The prototype uses a deterministic stdlib-only width policy:

- combining marks: 0 cells;
- East Asian Wide, Full-width, and Ambiguous characters: 2 cells;
- all other code points: 1 cell.

This may differ from terminals configured to render Ambiguous characters as single-width. The choice is intentional so golden snapshots remain stable without adding an external `wcwidth` dependency.

Wide characters reserve the following display cell with an internal zero-length placeholder, so rendered strings preserve the configured display width without adding an extra visible space. Text that exceeds the canvas width is truncated at the cell boundary.

## Non-goals

- No interactive editor.
- No package install or dependency change.
- No broad Hermes integration.
- No terminal-specific width probing.
- No GPL code import.

## Verification

Golden snapshot tests live in `tests/agent/test_ascii_canvas.py` and cover rectangle/crossing lines, arrows, table rendering, tree rendering, Unicode width policy, and guarded FIGlet fallback.
