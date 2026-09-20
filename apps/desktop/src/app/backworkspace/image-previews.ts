import { type Extension, type Range, StateField, type Text } from '@codemirror/state'
import { Decoration, type DecorationSet, EditorView, WidgetType } from '@codemirror/view'

import { resolveMediaDisplaySrc } from '@/lib/media'

// Every image link on a line, wherever it sits: pasting puts one at the caret,
// which is as often mid-sentence as on a line of its own.
const IMAGE_LINK = /!\[[^\]]*\]\(([^)\s]+)\)/g

/** The images a line links, in the order they appear on it. */
export function imageLinksIn(text: string): string[] {
  return [...text.matchAll(IMAGE_LINK)].map(match => match[1])
}

/** Where the page's own folder is, so `assets/…` resolves against it. The
 *  backend may be a Windows host, which names its folders the other way. */
function pageDirectory(pagePath: null | string): null | string {
  const at = Math.max(pagePath?.lastIndexOf('/') ?? -1, pagePath?.lastIndexOf('\\') ?? -1)

  return at > 0 ? pagePath!.slice(0, at) : null
}

/** What a link points at: a web source as written, a page-relative file against the page. */
export function imageTarget(pageDir: null | string, href: string): null | string {
  if (/^(?:https?|data):/i.test(href) || href.startsWith('/')) {
    return href
  }

  return pageDir ? `${pageDir}/${href}` : null
}

// Where the backend put a picture the page has just stored. A page keeps its
// file only from its first save, so a page written for the first time has no
// folder for `assets/…` to resolve against — but the answer that made the link
// already said where the picture is.
const attachments = new Map<string, string>()

// The picture a paste has just stored. A block widget is placed AFTER its line,
// so the caret the paste leaves behind sits above it and keeping the caret in
// view keeps the picture out of it — paste at the foot of a long page and the
// thing that just arrived is below the fold. Held here so the one that arrived
// can be brought into view, and cleared the moment it is: every other picture
// on the page belongs where it is, and pulling the page to one that happened to
// finish loading would take the page away from whoever was reading it.
let arriving: null | string = null

/** Remember where a freshly stored picture went, so it shows before the first save. */
export function rememberAttachment(href: string, path: string) {
  attachments.set(href, path)
  arriving = path
}

// A file on the backend host is not something an <img> can load by path: the
// app's own resolver turns it into a data URL through Electron, and fetches it
// over the connection when the backend is remote. Resolved once per target —
// the decorations are rebuilt as the document changes, and a note scrolls past
// the same picture many times. Bounded, because a data URL holds the whole
// file: a page of screenshots would otherwise keep every one of them in memory
// for as long as the window lives.
const SOURCE_CACHE_MAX = 24
const sources = new Map<string, string>()

function cacheSource(target: string, src: string) {
  const oldest = sources.size < SOURCE_CACHE_MAX ? undefined : sources.keys().next().value

  if (oldest !== undefined) {
    sources.delete(oldest)
  }

  sources.set(target, src)
}

class ImageWidget extends WidgetType {
  constructor(private readonly target: string) {
    super()
  }

  eq(other: ImageWidget) {
    return other.target === this.target
  }

  // What the editor assumes a picture it has not drawn yet is worth. Without a
  // figure it assumes a line of text, and the scrollbar of a long page lurches
  // as each picture scrolls into view and turns out to be taller.
  get estimatedHeight() {
    return 240
  }

  // A picture is part of the page to click on like anything else: left alone,
  // the editor ignores the mouse over a widget and the caret stays behind.
  ignoreEvent() {
    return false
  }

  toDOM(view: EditorView) {
    // The space above the picture is the block's padding rather than the
    // image's margin: the editor learns a block's height from its box, and a
    // margin is outside the box — so every picture would push the page down by
    // more than the editor had written down, and the caret and the clicks with it.
    const block = document.createElement('div')
    const image = document.createElement('img')
    const known = sources.get(this.target)

    block.className = 'cm-bw-image'
    block.append(image)
    // The picture arrives after its line has been measured, and it is nearly
    // all of that line's height. Asking for another measurement is what keeps
    // the caret, a click and the scroll position right below it.
    image.addEventListener('load', () => {
      view.requestMeasure()

      if (this.target !== arriving) {
        return
      }

      arriving = null
      // In the write phase, after the measure: the block's height is the
      // picture's, and that is only known once the picture has decoded.
      // Scrolling any earlier aims at the height the editor guessed.
      view.requestMeasure({ read: () => null, write: () => block.scrollIntoView({ block: 'nearest' }) })
    })
    // One that cannot be read leaves the line's text, which still says what it
    // points at, rather than a broken-image box. Hidden, not removed: the
    // element belongs to the editor, which syncs its own DOM against it.
    image.addEventListener('error', () => {
      if (this.target === arriving) {
        arriving = null
      }

      block.style.display = 'none'
      view.requestMeasure()
    })

    if (known) {
      image.src = known

      return block
    }

    void resolveMediaDisplaySrc(this.target)
      .then(src => {
        cacheSource(this.target, src)
        image.src = src
      })
      .catch(() => {
        block.style.display = 'none'
        // No `error` event follows a source that never arrived, so the line's
        // guessed height stands until something asks for a measurement.
        view.requestMeasure()
      })

    return block
  }
}

function imageDecorations(doc: Text, pageDir: null | string): DecorationSet {
  const decorations: Range<Decoration>[] = []

  for (let number = 1; number <= doc.lines; number += 1) {
    const line = doc.line(number)

    for (const href of imageLinksIn(line.text)) {
      const target = attachments.get(href) ?? imageTarget(pageDir, href)

      if (target) {
        // Under the line, not instead of the link: the text stays editable
        // and deletable, which is the whole contract of a plain-file page.
        decorations.push(Decoration.widget({ block: true, side: 1, widget: new ImageWidget(target) }).range(line.to))
      }
    }
  }

  return Decoration.set(decorations, true)
}

export const imageTheme = EditorView.theme({
  '.cm-bw-image': { display: 'block', paddingTop: '0.5rem' },
  '.cm-bw-image img': {
    borderRadius: '0.375rem',
    display: 'block',
    maxHeight: '24rem',
    maxWidth: '100%'
  }
})

/**
 * Shows the pictures a page links, under the lines that link them.
 *
 * A state field rather than a view plugin: a picture changes how tall the page
 * is, and the editor only accepts that from a source it can consult *before* it
 * works out which part of the document is on screen — a plugin that returns one
 * is rejected outright ("Block decorations may not be specified via plugins"),
 * which breaks the view mid-update and leaves the caret and every click behind.
 * So the whole document is read rather than the visible part: a note is a page,
 * and a widget that compares equal keeps the picture it has already loaded.
 */
export function imagePreviews(pagePath: null | string): Extension {
  const pageDir = pageDirectory(pagePath)

  return StateField.define<DecorationSet>({
    create: state => imageDecorations(state.doc, pageDir),
    provide: field => EditorView.decorations.from(field),
    update: (decorations, transaction) =>
      transaction.docChanged ? imageDecorations(transaction.state.doc, pageDir) : decorations
  })
}
