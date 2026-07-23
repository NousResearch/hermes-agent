// Interactive Counter — a tldraw offline document script.
//
// HOW TO RUN IT ON YOUR MACHINE:
//   1. Open tldraw offline, create or open a document.
//   2. Develop → Reveal Script…  (creates script/main.js + a workspace folder)
//   3. Replace the contents of script/main.js with THIS file, and save.
//   4. The app reruns the script automatically. You'll see a "Counter" panel
//      with MINUS / RESET / PLUS buttons — click them; the number updates live.
//   5. File → Save to persist the script into the .tldraw file. Now the file
//      *is* a little app: reopen it anywhere and the buttons still work.
//
// This is the document-script contract (from the app's script-context.d.ts):
//   export default function ({ editor, helpers, signal }) { ... }
//     editor  — the live tldraw Editor
//     helpers — editor-bound conveniences (richTextToPlainText, etc.)
//     signal  — an AbortSignal fired before the script reruns / on close;
//               register ALL cleanup on it so re-saving never leaks listeners.

import { createShapeId, toRichText } from 'tldraw'

export default function ({ editor, helpers, signal }) {
	// Stable ids => idempotent: re-running reuses shapes instead of duplicating.
	const IDS = {
		title: createShapeId('counter-title'),
		display: createShapeId('counter-display'),
		dec: createShapeId('counter-btn-dec'),
		reset: createShapeId('counter-btn-reset'),
		inc: createShapeId('counter-btn-inc'),
	}

	// Create-if-missing helper (leaves user edits intact on rerun).
	function ensure(partial) {
		if (editor.getShape(partial.id)) return
		editor.createShape(partial)
	}

	editor.run(() => {
		ensure({
			id: IDS.title, type: 'text', x: 40, y: 20,
			props: { richText: toRichText('Counter'), size: 'xl', font: 'draw', color: 'black' },
		})
		ensure({
			id: IDS.display, type: 'geo', x: 40, y: 80,
			props: { geo: 'rectangle', w: 360, h: 160, color: 'black', fill: 'none', richText: toRichText('0'), size: 'xl' },
			meta: { ui: 'display', count: 0 },
		})
		// Button labels are load-bearing — the click handler finds buttons by text.
		ensure({
			id: IDS.dec, type: 'geo', x: 40, y: 270,
			props: { geo: 'rectangle', w: 100, h: 80, color: 'red', fill: 'solid', richText: toRichText('MINUS'), size: 'l' },
			meta: { ui: 'button', action: 'MINUS' },
		})
		ensure({
			id: IDS.reset, type: 'geo', x: 170, y: 270,
			props: { geo: 'rectangle', w: 100, h: 80, color: 'grey', fill: 'solid', richText: toRichText('RESET'), size: 'l' },
			meta: { ui: 'button', action: 'RESET' },
		})
		ensure({
			id: IDS.inc, type: 'geo', x: 300, y: 270,
			props: { geo: 'rectangle', w: 100, h: 80, color: 'green', fill: 'solid', richText: toRichText('PLUS'), size: 'l' },
			meta: { ui: 'button', action: 'PLUS' },
		})
	})

	const STEP = { MINUS: -1, PLUS: +1 }

	function displayShape() {
		return editor.getCurrentPageShapes().find((s) => s.meta && s.meta.ui === 'display')
	}
	function setCount(n) {
		const d = displayShape()
		editor.run(
			() =>
				editor.updateShape({
					id: d.id, type: 'geo',
					props: { richText: toRichText(String(n)) },
					meta: { ...d.meta, count: n },
				}),
			{ history: 'ignore' } // keep script writes out of the user's undo stack
		)
	}
	function runAction(label) {
		const d = displayShape()
		const cur = d.meta && typeof d.meta.count === 'number' ? d.meta.count : 0
		if (label === 'RESET') setCount(0)
		else if (label in STEP) setCount(cur + STEP[label])
	}

	function bounds(s) {
		return { x: s.x, y: s.y, w: s.props.w ?? 0, h: s.props.h ?? 0 }
	}
	function inside(b, p) {
		return p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h
	}

	function onEvent(info) {
		if (!info || info.name !== 'pointer_down') return
		let p = null
		try {
			if (info.point && editor.screenToPage) p = editor.screenToPage(info.point)
		} catch {}
		p = p ?? editor.inputs?.currentPagePoint
		if (!p) return
		const hit = editor
			.getCurrentPageShapes()
			.find((s) => s.meta && s.meta.ui === 'button' && inside(bounds(s), p))
		if (hit) runAction(hit.meta.action)
	}

	editor.on('event', onEvent)
	signal.addEventListener('abort', () => editor.off('event', onEvent)) // required cleanup

	editor.zoomToFit({ animation: { duration: 200 } })
}
