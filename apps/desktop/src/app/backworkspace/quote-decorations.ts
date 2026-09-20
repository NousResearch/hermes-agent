import type { Range } from '@codemirror/state'
import { Decoration, type DecorationSet, EditorView, ViewPlugin, type ViewUpdate } from '@codemirror/view'

/**
 * An agent's reply reads as someone else's voice.
 *
 * Replies are written as markdown quotes (`reply.ts`), so the rule is simply
 * "a quoted line": the page stays a plain file, and a quote the user types by
 * hand gets the same treatment, which is honest — it is a quote either way.
 */
const QUOTE_LINE = Decoration.line({ class: 'cm-bw-quote' })
// The first line of a reply — `name · time` — is a label, not prose.
const QUOTE_HEAD = Decoration.line({ class: 'cm-bw-quote cm-bw-quote-head' })
const ATTRIBUTION = /^>\s*\S+\s+·\s/
// The `>` that makes a line a quote: kept, because the file is markdown and
// stays one, but quieted — the bar down the left already carries the meaning.
const QUOTE_MARK = Decoration.mark({ class: 'cm-bw-quote-mark' })
const LEADING_MARK = /^>\s?/

function quoteDecorations(view: EditorView): DecorationSet {
  const ranges: Range<Decoration>[] = []

  for (const { from, to } of view.visibleRanges) {
    for (let pos = from; pos <= to;) {
      const line = view.state.doc.lineAt(pos)

      if (line.text.startsWith('>')) {
        ranges.push((ATTRIBUTION.test(line.text) ? QUOTE_HEAD : QUOTE_LINE).range(line.from))

        const marker = LEADING_MARK.exec(line.text)?.[0].length ?? 0

        ranges.push(QUOTE_MARK.range(line.from, line.from + marker))
      }

      pos = line.to + 1
    }
  }

  // Sorted on the way in: the line decoration and the marker share an offset.
  return Decoration.set(ranges, true)
}

export const quoteTheme = EditorView.theme({
  // `.cm-line` is named too, for the weight rather than the meaning: the page's
  // own theme flattens every line's padding, and style modules are mounted in
  // reverse, so an equally specific rule here would be the one that loses.
  '.cm-line.cm-bw-quote': {
    borderLeft: '2px solid var(--ui-stroke-secondary)',
    color: 'var(--ui-text-secondary)',
    paddingLeft: '0.75rem'
  },
  '.cm-bw-quote-head': {
    color: 'var(--ui-text-tertiary)',
    fontSize: '0.8125rem'
  },
  '.cm-bw-quote-mark': { opacity: '0.35' }
})

export const quoteDecorationPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet

    constructor(view: EditorView) {
      this.decorations = quoteDecorations(view)
    }

    update(update: ViewUpdate) {
      if (update.docChanged || update.viewportChanged) {
        this.decorations = quoteDecorations(update.view)
      }
    }
  },
  { decorations: plugin => plugin.decorations }
)
