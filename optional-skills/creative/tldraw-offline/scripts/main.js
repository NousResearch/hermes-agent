// tldraw offline — document script (script/main.js)
//
// A document script's default export receives a ctx object and runs whenever the
// document loads (and reruns when you save the script). Contract, verified against
// the app's bundled script-context.d.ts:
//
//   export default function ({ editor, helpers, signal }) { ... }
//
//   editor   — the live tldraw Editor for this document
//   helpers  — editor-bound conveniences: createShapeIfMissing, createShapesIfMissing,
//              createArrowBetweenShapes, translateShapes, onShapeTranslate,
//              richTextToPlainText, boxShapes, getLints
//   signal   — an AbortSignal fired before the script reruns and when the board
//              closes. Register ALL cleanup on it (this is how you avoid leaks).
//
// Pure tldraw primitives (createShapeId, toRichText, Vec, ...) are imported from
// the `tldraw` app module — NOT globals. react / react-dom are also importable.
// It is not a Node project; only those modules are available.

import { createShapeId, toRichText } from 'tldraw'

export default function ({ editor, helpers, signal }) {
	const { createShapeIfMissing, createArrowBetweenShapes } = helpers

	// --- 1. Build durable "furniture" idempotently (stable ids, create-if-missing).
	// Re-running the script must NOT duplicate or clobber user edits.
	const nodes = [
		{ id: createShapeId('node-ui'), x: 0, y: 0, color: 'blue', label: 'CLI / Gateway' },
		{ id: createShapeId('node-core'), x: 280, y: 0, color: 'violet', label: 'Agent Core' },
		{ id: createShapeId('node-tools'), x: 560, y: 0, color: 'green', label: 'Tools' },
	]

	editor.run(() => {
		for (const n of nodes) {
			createShapeIfMissing({
				id: n.id,
				type: 'geo',
				x: n.x,
				y: n.y,
				props: {
					geo: 'rectangle',
					w: 220,
					h: 110,
					color: n.color,
					fill: 'solid',
					richText: toRichText(n.label),
				},
			})
		}
	})

	// Connect them (arrows bind to the shapes, so they follow when moved).
	createArrowBetweenShapes(nodes[0].id, nodes[1].id, { arrowheadEnd: 'arrow' })
	createArrowBetweenShapes(nodes[1].id, nodes[2].id, { arrowheadEnd: 'arrow' })

	// --- 2. Add reactive behavior: recolor the last node based on arrow count.
	// store.listen fires on the tick AFTER a commit — never read state you just
	// wrote synchronously and expect the listener to have run yet.
	const targetId = nodes[2].id
	function update() {
		const hasArrows = editor.getCurrentPageShapes().some((s) => s.type === 'arrow')
		editor.run(
			() =>
				editor.updateShape({
					id: targetId,
					type: 'geo',
					props: { fill: hasArrows ? 'solid' : 'none' },
				}),
			{ history: 'ignore' } // keep script-owned writes out of the user's undo stack
		)
	}

	const stop = editor.store.listen(update)
	signal.addEventListener('abort', () => stop()) // <-- the one required cleanup
	update() // run once on load

	editor.zoomToFit({ animation: { duration: 200 } })
}
