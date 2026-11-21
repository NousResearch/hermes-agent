# Excalidraw Wireframe Renderer

Subagent prompt. Produce one `.excalidraw` file at `.working/ia-{date}.excalidraw` (IA diagram) or `.working/flow-{name}-{date}.excalidraw` (flow wireframe).

## CRITICAL: two-character `index` fields only

Every element's `index` field must be **exactly two characters** (`a0`, `aZ`, `b3`, ...). Three-character indices cause a silent *"Error: invalid file"* with no diagnostic output. Assign sequentially across all elements; advance the leading letter when the trailing alphanumeric exhausts (`a0..a9, aA..aZ`, then `b0..`). Verify before writing.

## Shape

Valid Excalidraw file: `{type: "excalidraw", version: 2, source: "https://excalidraw.com", elements: [...], appState: {gridSize: null, viewBackgroundColor: "#ffffff"}, files: {}}`. Each element needs the standard Excalidraw element fields (`id, type, x, y, width, height, angle, strokeColor, backgroundColor, fillStyle, strokeWidth, strokeStyle, roughness, opacity, groupIds, frameId, roundness, seed, version, versionNonce, isDeleted, boundElements, updated, link, locked, index`). Text elements add `text, fontSize, fontFamily, textAlign, verticalAlign, baseline, containerId, originalText, lineHeight`.

## Content

**IA diagram:** boxes-and-arrows of auth stack, main app surfaces, modal routes, settings stack, cross-cutting affordances. Color sparingly to distinguish category. Layout for human legibility, not graph correctness.

**Flow wireframe:** screen-by-screen rectangles left-to-right, simple shapes inside (nav bar, CTA, content blocks) at low fidelity. Arrows labeled with the user action that causes transition. Annotations alongside for climax and edge-case beats.

Return to the parent: file path, kind, one-line subject, element count, confirmation that all indices are two-character. Do not dump JSON into parent context. Tell the user to open in Excalidraw desktop or excalidraw.com.
