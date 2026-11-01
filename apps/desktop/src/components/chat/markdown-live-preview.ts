import { syntaxTree } from '@codemirror/language'
import { type EditorState, type Extension, RangeSetBuilder } from '@codemirror/state'
import { Decoration, type DecorationSet, EditorView, ViewPlugin, type ViewUpdate, WidgetType } from '@codemirror/view'

// Obsidian-style "live preview" for Markdown in the spot editor: the document
// stays plain Markdown (what gets saved is byte-identical to what was typed),
// but headings, emphasis, inline code, links, quotes, list bullets and rules
// render in place. Syntax marks hide everywhere EXCEPT around the selection, so
// the caret always lands in real source and editing never fights a widget.
//
// Driven by the Lezer Markdown tree (`@codemirror/lang-markdown`, loaded by the
// editor's language compartment). Before that grammar arrives the tree is
// empty and this decorates nothing — the editor simply shows source.

const HEADING = /^(?:ATX|Setext)Heading([1-6])$/

class BulletWidget extends WidgetType {
  eq() {
    return true
  }

  toDOM() {
    const el = document.createElement('span')
    el.className = 'cm-md-bullet'
    el.textContent = '•'

    return el
  }
}

class RuleWidget extends WidgetType {
  eq() {
    return true
  }

  toDOM() {
    const el = document.createElement('span')
    el.className = 'cm-md-hr'
    el.setAttribute('aria-hidden', 'true')

    return el
  }
}

const hide = Decoration.replace({})
const bullet = Decoration.replace({ widget: new BulletWidget() })
const rule = Decoration.replace({ widget: new RuleWidget() })
const strong = Decoration.mark({ class: 'cm-md-strong' })
const emphasis = Decoration.mark({ class: 'cm-md-em' })
const strike = Decoration.mark({ class: 'cm-md-strike' })
const inlineCode = Decoration.mark({ class: 'cm-md-code' })
const link = Decoration.mark({ class: 'cm-md-link' })
const quoteLine = Decoration.line({ class: 'cm-md-quote' })
const codeLine = Decoration.line({ class: 'cm-md-codeblock' })
const headingLine = [1, 2, 3, 4, 5, 6].map(level => Decoration.line({ class: `cm-md-h cm-md-h${level}` }))
const LINE_DECOS = new Set<Decoration>([quoteLine, codeLine, ...headingLine])

/** True when any selection range touches [from, to] (inclusive ends, so a caret
 *  sitting right after `**bold**` still reveals its marks). */
function selectionTouches(state: EditorState, from: number, to: number): boolean {
  return state.selection.ranges.some(range => range.from <= to && range.to >= from)
}

/** Same, widened to whole lines — headings/quotes/rules reveal per line, the
 *  way Obsidian does, so the marks don't flicker as the caret moves inside. */
function selectionTouchesLines(state: EditorState, from: number, to: number): boolean {
  return selectionTouches(state, state.doc.lineAt(from).from, state.doc.lineAt(to).to)
}

interface PendingDecoration {
  deco: Decoration
  from: number
  to: number
}

export function buildLivePreviewDecorations(state: EditorState, ranges: readonly { from: number; to: number }[]) {
  const pending: PendingDecoration[] = []
  const add = (from: number, to: number, deco: Decoration) => pending.push({ deco, from, to })
  const tree = syntaxTree(state)

  for (const { from, to } of ranges) {
    tree.iterate({
      from,
      to,
      enter: node => {
        const name = node.name
        const heading = HEADING.exec(name)

        if (heading) {
          const level = Number(heading[1])

          for (let pos = node.from; pos <= node.to;) {
            const line = state.doc.lineAt(pos)
            add(line.from, line.from, headingLine[level - 1])
            pos = line.to + 1
          }

          return
        }

        switch (name) {
          case 'HeaderMark': {
            const parent = node.node.parent

            if (!parent || selectionTouchesLines(state, parent.from, parent.to)) {
              return
            }

            // `## Title` — hide the hashes plus the single space after them.
            // Setext underlines (`===`) hide whole.
            const next = state.doc.sliceString(node.to, node.to + 1)
            add(node.from, next === ' ' ? node.to + 1 : node.to, hide)

            return
          }

          case 'StrongEmphasis':
            add(node.from, node.to, strong)

            return

          case 'Emphasis':
            add(node.from, node.to, emphasis)

            return

          case 'Strikethrough':
            add(node.from, node.to, strike)

            return

          case 'InlineCode':
            add(node.from, node.to, inlineCode)

            return

          case 'Link':
            add(node.from, node.to, link)

            return

          case 'EmphasisMark':
          case 'StrikethroughMark': {
            const parent = node.node.parent

            if (parent && !selectionTouches(state, parent.from, parent.to)) {
              add(node.from, node.to, hide)
            }

            return
          }

          case 'CodeMark': {
            // Only inline code; fenced-code fences stay visible (they carry the
            // language and delimit a block you edit as source anyway).
            const parent = node.node.parent

            if (parent?.name === 'InlineCode' && !selectionTouches(state, parent.from, parent.to)) {
              add(node.from, node.to, hide)
            }

            return
          }

          case 'LinkMark':

          case 'URL':
          case 'LinkTitle': {
            const parent = node.node.parent

            // `[label](url "title")` → just `label`. Bare autolinks keep their URL.
            if (parent?.name === 'Link' && !selectionTouches(state, parent.from, parent.to)) {
              add(node.from, node.to, hide)
            }

            return
          }

          case 'ListMark': {
            const parent = node.node.parent?.parent
            const text = state.doc.sliceString(node.from, node.to)

            if (parent?.name === 'BulletList' && /^[-*+]$/.test(text) && !selectionTouches(state, node.from, node.to)) {
              add(node.from, node.to, bullet)
            }

            return
          }

          case 'Blockquote': {
            for (let pos = node.from; pos <= node.to;) {
              const line = state.doc.lineAt(pos)
              add(line.from, line.from, quoteLine)
              pos = line.to + 1
            }

            return
          }

          case 'QuoteMark': {
            if (!selectionTouchesLines(state, node.from, node.to)) {
              const next = state.doc.sliceString(node.to, node.to + 1)
              add(node.from, next === ' ' ? node.to + 1 : node.to, hide)
            }

            return
          }

          case 'FencedCode':
          case 'CodeBlock': {
            for (let pos = node.from; pos <= node.to;) {
              const line = state.doc.lineAt(pos)
              add(line.from, line.from, codeLine)
              pos = line.to + 1
            }

            // Nothing inside a code block is Markdown — don't descend.
            return false
          }

          case 'HorizontalRule': {
            if (!selectionTouchesLines(state, node.from, node.to)) {
              add(node.from, node.to, rule)
            }

            return
          }
        }

        return
      }
    })
  }

  // RangeSetBuilder needs ascending `from`, then ascending `startSide` (line
  // decorations sort ahead of marks/replacements at the same position).
  pending.sort((a, b) => a.from - b.from || a.deco.startSide - b.deco.startSide || a.to - b.to)
  const builder = new RangeSetBuilder<Decoration>()
  const linesDecorated = new Set<number>()

  for (const { deco, from, to } of pending) {
    // A heading inside a quote (or overlapping visible ranges) can emit two
    // line decorations for one line; keep the first.
    if (LINE_DECOS.has(deco)) {
      if (linesDecorated.has(from)) {
        continue
      }

      linesDecorated.add(from)
    }

    builder.add(from, to, deco)
  }

  return builder.finish()
}

const livePreviewPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet

    constructor(view: EditorView) {
      this.decorations = buildLivePreviewDecorations(view.state, view.visibleRanges)
    }

    update(update: ViewUpdate) {
      if (
        update.docChanged ||
        update.viewportChanged ||
        update.selectionSet ||
        syntaxTree(update.startState) !== syntaxTree(update.state)
      ) {
        this.decorations = buildLivePreviewDecorations(update.state, update.view.visibleRanges)
      }
    }
  },
  { decorations: plugin => plugin.decorations }
)

// Prose typography: the app's sans font at reading size, headings scaled like
// the rendered preview. Colors come from theme vars so light/dark/skins follow.
const livePreviewTheme = EditorView.theme({
  '.cm-content': {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.875rem',
    lineHeight: '1.6'
  },
  '.cm-line': {
    fontFamily: 'var(--font-sans)',
    fontSize: '0.875rem',
    lineHeight: '1.6',
    padding: '0 1rem'
  },
  '.cm-scroller': {
    fontFamily: 'var(--font-sans)',
    lineHeight: '1.6'
  },
  '.cm-md-h': { fontWeight: '700', lineHeight: '1.3' },
  '.cm-md-h1': { fontSize: '1.6em', paddingTop: '0.4em' },
  '.cm-md-h2': { fontSize: '1.35em', paddingTop: '0.35em' },
  '.cm-md-h3': { fontSize: '1.15em', paddingTop: '0.3em' },
  '.cm-md-h4, .cm-md-h5, .cm-md-h6': { fontSize: '1em' },
  '.cm-md-strong': { fontWeight: '700' },
  '.cm-md-em': { fontStyle: 'italic' },
  '.cm-md-strike': { textDecoration: 'line-through' },
  '.cm-md-code': {
    backgroundColor: 'color-mix(in srgb, var(--dt-foreground, currentColor) 8%, transparent)',
    borderRadius: '3px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.85em',
    padding: '0.1em 0.25em'
  },
  '.cm-md-link': {
    color: 'var(--ui-accent, var(--primary))',
    textDecoration: 'underline',
    textUnderlineOffset: '2px'
  },
  '.cm-md-quote': {
    borderLeft: '3px solid color-mix(in srgb, var(--dt-foreground, currentColor) 20%, transparent)',
    color: 'var(--muted-foreground)',
    paddingLeft: 'calc(1rem - 3px + 0.75rem)'
  },
  '.cm-md-codeblock': {
    backgroundColor: 'color-mix(in srgb, var(--dt-foreground, currentColor) 5%, transparent)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8rem'
  },
  '.cm-md-bullet': { color: 'var(--muted-foreground)', padding: '0 0.15em' },
  '.cm-md-hr': {
    borderTop: '1px solid color-mix(in srgb, var(--dt-foreground, currentColor) 20%, transparent)',
    display: 'inline-block',
    verticalAlign: 'middle',
    width: '100%'
  }
})

/** Live-preview Markdown editing: rendered styling with source revealed at the
 *  caret. Pair with a Markdown language extension and line wrapping. */
export function markdownLivePreview(): Extension {
  return [livePreviewPlugin, livePreviewTheme]
}
