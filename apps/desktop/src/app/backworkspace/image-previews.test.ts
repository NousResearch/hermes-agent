import { EditorState } from '@codemirror/state'
import { EditorView } from '@codemirror/view'
import { describe, expect, it, vi } from 'vitest'

import { imageLinksIn, imagePreviews, imageTarget, rememberAttachment } from './image-previews'

vi.mock('@/lib/media', () => ({ resolveMediaDisplaySrc: () => new Promise<string>(() => {}) }))

const PAGE_DIR = '/home/u/.hermes/backworkspace'

describe('imageTarget', () => {
  it('resolves a page-relative image against the page folder', () => {
    expect(imageTarget(PAGE_DIR, 'assets/shot.png')).toBe(`${PAGE_DIR}/assets/shot.png`)
  })

  it('passes web and data sources through untouched', () => {
    expect(imageTarget(PAGE_DIR, 'https://example.com/a.png')).toBe('https://example.com/a.png')
    expect(imageTarget(PAGE_DIR, 'data:image/png;base64,AAA')).toBe('data:image/png;base64,AAA')
  })

  it('shows nothing for a relative link when the page has no file yet', () => {
    expect(imageTarget(null, 'assets/shot.png')).toBeNull()
  })
})

describe('imageLinksIn', () => {
  it('finds a pasted image wherever it landed, including mid-sentence', () => {
    expect(imageLinksIn('![](assets/a.png) what is this? ![](assets/b.png)')).toEqual(['assets/a.png', 'assets/b.png'])
    expect(imageLinksIn('![](assets/a.png)')).toEqual(['assets/a.png'])
  })

  it('leaves ordinary links and prose alone', () => {
    expect(imageLinksIn('see [the notes](assets/a.png) for more')).toEqual([])
    expect(imageLinksIn('nothing here')).toEqual([])
  })
})

describe('imagePreviews', () => {
  const editorOver = (doc: string) => {
    const parent = document.createElement('div')

    document.body.append(parent)

    return new EditorView({
      parent,
      state: EditorState.create({ doc, extensions: [imagePreviews(`${PAGE_DIR}/20260920_030000_aaaaaa.md`)] })
    })
  }

  // A picture is a block, and the editor takes a block only from a source it
  // can read before it lays the page out. Provided the wrong way it throws
  // mid-update, which leaves the page drawn from a document it no longer has:
  // the caret sits away from the writing and a click lands on another line.
  it('builds a working editor over a page that links a picture', () => {
    const view = editorOver('a note\n![](assets/shot.png)\nmore')

    expect(view.state.doc.line(3).text).toBe('more')
    expect(view.contentDOM.textContent).toContain('more')

    view.destroy()
  })

  it('keeps working when a picture is pasted into a page that had none', () => {
    const view = editorOver('a note')

    view.dispatch({ changes: { from: view.state.doc.length, insert: ' ![](assets/shot.png)' } })

    expect(view.state.doc.toString()).toBe('a note ![](assets/shot.png)')
    expect(view.contentDOM.textContent).toContain('a note ![](assets/shot.png)')

    view.destroy()
  })

  // A widget block sits AFTER the line that links it, so the caret a paste
  // leaves behind is above the picture and keeping the caret in view keeps the
  // picture out of it. Paste at the foot of a long page and the thing that just
  // arrived is below the fold — which is what happened, on a real page, to the
  // first picture anyone pasted.
  describe('the picture a paste has just stored', () => {
    const loadTheImage = async (view: EditorView) => {
      view.contentDOM.querySelector('.cm-bw-image img')?.dispatchEvent(new Event('load'))

      // CodeMirror measures on a frame, and the scroll is asked for in that
      // measurement's write phase.
      for (let frame = 0; frame < 3; frame += 1) {
        await new Promise(resolve => requestAnimationFrame(() => resolve(null)))
      }
    }

    const withStubbedScroll = async (run: (scrollIntoView: ReturnType<typeof vi.fn>) => Promise<void>) => {
      // jsdom has no scrolling, so the method the page calls has to be lent to it.
      const original = Element.prototype.scrollIntoView
      const scrollIntoView = vi.fn()

      Element.prototype.scrollIntoView = scrollIntoView

      try {
        await run(scrollIntoView)
      } finally {
        Element.prototype.scrollIntoView = original
      }
    }

    it('is brought into view once it has a height', async () => {
      await withStubbedScroll(async scrollIntoView => {
        rememberAttachment('assets/new.png', `${PAGE_DIR}/assets/new.png`)

        const view = editorOver('a note\n![](assets/new.png)')

        await loadTheImage(view)

        expect(scrollIntoView).toHaveBeenCalled()

        view.destroy()
      })
    })

    // The other half of the promise: every picture already on the page belongs
    // where it is. One finishing its load while the reader is somewhere else
    // must not take the page away from them.
    it('does not move the page for any other picture that finishes loading', async () => {
      await withStubbedScroll(async scrollIntoView => {
        rememberAttachment('assets/new.png', `${PAGE_DIR}/assets/new.png`)

        const view = editorOver('a note\n![](assets/already-here.png)')

        await loadTheImage(view)

        expect(scrollIntoView).not.toHaveBeenCalled()

        view.destroy()
      })
    })
  })
})
