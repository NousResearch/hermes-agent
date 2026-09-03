import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { PreviewTarget } from '@/store/preview'

import { anchorNearLine, changedLineSpans, lineForCrossViewPrefix, LocalFilePreview, MarkdownPreview, normalizeForCrossView, selectionTextWithMath } from './preview-file'

// Behavior tests for the .md file preview renderer: input markdown goes
// through normalizeFilePreviewMath -> Streamdown (+ KaTeX math plugin) and must
// come out as real rendered elements, matching what the chat transcript
// renderer produces. Guards the regression where the preview was a bare
// Streamdown pass with no math plugin and no table/img/a components.
describe('MarkdownPreview', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders block and inline math through KaTeX', () => {
    // KaTeX marks its output; raw "$" delimiters must be gone.
    const { container } = render(
      <MarkdownPreview
        text={'Formula:\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\n$$\n\nInline $a^2 + b^2 = c^2$ too.'}
      />
    )

    expect(container.querySelector('.katex')).not.toBeNull()
    expect(screen.queryByText(/\$\$/)).toBeNull()
  })

  it('renders GFM tables with header and body cells', () => {
    const { container } = render(<MarkdownPreview text={'| h1 | h2 |\n| --- | --- |\n| a | b |'} />)

    const table = container.querySelector('table')
    expect(table).not.toBeNull()
    expect(table?.querySelector('thead th')?.textContent).toBe('h1')
    expect(table?.querySelector('tbody td')?.textContent).toBe('a')
  })

  it('renders images with alt text', () => {
    const { container } = render(<MarkdownPreview text={'![a chart](https://example.com/chart.png)'} />)

    const img = container.querySelector('img')
    expect(img?.getAttribute('alt')).toBe('a chart')
    expect(img?.getAttribute('src')).toBe('https://example.com/chart.png')
  })

  it('renders external links to open in a new tab safely', () => {
    const { container } = render(<MarkdownPreview text={'[docs](https://example.com/docs)'} />)

    const anchor = container.querySelector('a')
    expect(anchor?.getAttribute('href')).toBe('https://example.com/docs')
    expect(anchor?.getAttribute('target')).toBe('_blank')
    expect(anchor?.getAttribute('rel')).toBe('noopener noreferrer')
  })
})

// Regression tests for the "jump to change" affordance. The bug this guards:
// a reset effect keyed on [filePath, reloadKey] wiped lastTextRef on every
// external reload, racing ahead of the load effect's old-vs-new text compare,
// so the jump button could never appear after the file watcher refreshed the
// preview.
describe('LocalFilePreview jump-to-change', () => {
  beforeEach(() => {
    // Spreading the jsdom window into a stub drops prototype-Chain methods
    // (rAF etc.) - re-stub them like preview-pane.test does.
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(Date.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  function makeFileTarget(): PreviewTarget {
    return {
      kind: 'file',
      label: 'notes.md',
      path: 'C:/tmp/notes.md',
      previewKind: 'text',
      source: 'C:/tmp/notes.md',
      url: 'file:///C:/tmp/notes.md'
    }
  }

  function stubDesktopFs(texts: string[]) {
    let readIndex = 0

    // Assign onto the real jsdom window (NOT a spread-stub - a spread drops
    // prototype methods like addEventListener and breaks the component). The
    // test window has no hermesDesktop bridge, so a plain assignment installs
    // the mock without fighting typings.
    ;(window as never as Record<string, unknown>).hermesDesktop = {
      readFileText: vi.fn(async () => {
        const text = texts[Math.min(readIndex, texts.length - 1)]
        readIndex += 1

        return {
          binary: false,
          byteSize: text.length,
          language: 'markdown',
          path: 'C:/tmp/notes.md',
          text
        }
      }),
      writeClipboard: vi.fn(async () => true)
    }
  }

  it('shows the jump button when the file changes between reloads', async () => {
    stubDesktopFs([
      '# Title\n\nfirst version\n',
      '# Title\n\nfirst version\n\nnew paragraph from the agent\n'
    ])

    let rendered!: ReturnType<typeof render>

    await act(async () => {
      rendered = render(<LocalFilePreview reloadKey={0} target={makeFileTarget()} />)
    })

    // First load: nothing changed yet, no button.
    await waitFor(() => {
      expect(screen.queryByText('first version')).not.toBeNull()
    })
    expect(screen.queryByText('改动') ?? screen.queryByText('Change')).toBeNull()

    // External change: same file, reloadKey bumped by the file watcher.
    rendered.rerender(<LocalFilePreview reloadKey={1} target={makeFileTarget()} />)

    await waitFor(() => {
      expect(screen.queryByText('new paragraph from the agent')).not.toBeNull()
    })
    // The label now carries its cycle counter (改动 1/1 / Change 1/1).
    expect(screen.queryByText(/^改动 1\/1$/) ?? screen.queryByText(/^Change 1\/1$/)).not.toBeNull()
  })

  it('keeps the baseline across reloads but resets it on a file switch', async () => {
    stubDesktopFs(['alpha\n\n', 'alpha\n\nbeta\n\n'])

    const otherTarget: PreviewTarget = {
      ...makeFileTarget(),
      label: 'other.md',
      path: 'C:/tmp/other.md',
      source: 'C:/tmp/other.md',
      url: 'file:///C:/tmp/other.md'
    }

    let rendered!: ReturnType<typeof render>

    await act(async () => {
      rendered = render(<LocalFilePreview reloadKey={0} target={makeFileTarget()} />)
    })

    await waitFor(() => {
      expect(screen.queryByText('alpha')).not.toBeNull()
    })

    // Switch to a different file (same reloadKey): the compare baseline must
    // reset, so no jump button even though the new file's text differs.
    await act(async () => {
      rendered.rerender(<LocalFilePreview reloadKey={0} target={otherTarget} />)
    })

    await waitFor(() => {
      expect(screen.queryByText('beta')).not.toBeNull()
    })
    expect(screen.queryByText('改动') ?? screen.queryByText('Change')).toBeNull()
  })
})

// Pure-function tests for the cross-view (rendered preview <-> source)
// position mapping: the two directions must be consistent inverses - what
// `anchorNearLine` produces must be found back by `lineForCrossViewPrefix`.
describe('cross-view mapping', () => {
  const doc = [
    '# Title',           // 1
    '',                  // 2
    'First paragraph',   // 3  (anchor candidate: has >= 8 chars? no - 15 chars ok)
    '',                  // 4
    'Second paragraph with more text', // 5
    '',                  // 6
    '- list item one',   // 7
    '- list item two',   // 8
    '',                  // 9
    '| a | b |',         // 10 (table - normalizes to "a b", too short after usable check? "a b" < 8 -> unusable)
    '',                  // 11
    'Final paragraph here' // 12
  ].join('\n')

  it('normalizeForCrossView strips markdown markers', () => {
    expect(normalizeForCrossView('## Hello world')).toBe('Hello world')
    expect(normalizeForCrossView('- item text here')).toBe('item text here')
    expect(normalizeForCrossView('> quoted words here')).toBe('quoted words here')
    expect(normalizeForCrossView('**bold** and _em_ text')).toBe('bold and em text')
  })

  it('anchorNearLine returns the line itself when usable', () => {
    const anchor = anchorNearLine(doc.split('\n'), 5)
    expect(anchor.prefix).toBe('Second paragraph with more text')
    expect(anchor.fraction).toBeGreaterThan(0)
  })

  it('anchorNearLine walks outward past blank/short lines', () => {
    // Line 11 is blank; the nearest usable line is 10 ("a b" - too short) or 12.
    const anchor = anchorNearLine(doc.split('\n'), 11)
    expect(anchor.prefix).toBe('Final paragraph here')
  })

  it('anchorNearLine degrades to fraction when nothing usable is nearby', () => {
    const sparse = ['', '', '', '']
    const anchor = anchorNearLine(sparse, 2)
    expect(anchor.prefix).toBeNull()
    expect(anchor.fraction).toBeGreaterThanOrEqual(0)
  })

  it('lineForCrossViewPrefix finds the line an anchor came from (inverse)', () => {
    const lines = doc.split('\n')

    for (const line of [3, 5, 7, 12]) {
      const anchor = anchorNearLine(lines, line)

      expect(anchor.prefix).not.toBeNull()
      expect(lineForCrossViewPrefix(lines, anchor.prefix!)).toBe(line)
    }
  })

  it('lineForCrossViewPrefix matches a rendered block against its source line', () => {
    const lines = doc.split('\n')

    // A rendered list block's text "list item one" should find source line 7.
    expect(lineForCrossViewPrefix(lines, 'list item one')).toBe(7)
    // A rendered heading "Title" is too short to anchor (< 8 chars).
    expect(lineForCrossViewPrefix(lines, 'Title')).toBeNull()
  })
})

// changedLineSpans: the multi-change walk behind the cycling jump button.
describe('changedLineSpans', () => {
  it('returns one span per changed region, split on paragraph breaks', () => {
    const old = 'para one\n\npara two\n\npara three'
    // Two separate insertions in different paragraphs.
    const now = 'para one\n\npara two NEW\n\npara three\n\npara four'

    const spans = changedLineSpans(old, now)

    // The middle is lines 3-7; unchanged 'para three' inside it is not
    // jumpable, so the cycle = 'para two NEW' (3) and 'para four' (7).
    const jumpable = spans.filter(span => span.line !== null)

    expect(jumpable.length).toBe(2)
    expect(jumpable[0].line).toBe(3)
    expect(jumpable[1].line).toBe(7)
    expect(jumpable.every(span => span.hasNew)).toBe(true)
  })

  it('merges a single edit into one span', () => {
    const old = 'a\n\nb\n\nc'
    const now = 'a\n\nB!\n\nc'

    const spans = changedLineSpans(old, now)

    expect(spans.length).toBe(1)
    expect(spans[0].line).toBe(3)
  })

  it('returns empty for identical texts and pure suffix deletions have no new lines', () => {
    expect(changedLineSpans('same\ntext', 'same\ntext')).toEqual([])

    // Deleting trailing lines: the middle has no new-text lines at all.
    const spans = changedLineSpans('a\n\nb\n\nc\n\nd', 'a\n\nb')

    expect(spans.every(span => span.hasNew === false || span.endLine > 0)).toBe(true)
  })

  it('keeps firstChangedLine consistent: the first span starts at the first differing line', () => {
    const old = 'x\n\ny\n\nz'
    const now = 'x\n\ny2\n\nz\n\nw'

    const spans = changedLineSpans(old, now)

    expect(spans[0].line).toBe(3)
  })
})

// Behavior tests for selectionTextWithMath: the copy pipeline must slice
// plain text precisely to the selection offsets, emit any touched equation
// whole as bare LaTeX (no $ delimiters), and squeeze everything onto one
// line (a copied selection is a hint for the agent, not a formatted doc).
describe('selectionTextWithMath', () => {
  afterEach(() => {
    cleanup()
    document.body.innerHTML = ''
    document.getSelection()?.removeAllRanges()
  })

  function mountFixture(html: string): HTMLElement {
    const root = document.createElement('div')
    root.innerHTML = html
    document.body.appendChild(root)

    return root
  }

  function selectRange(startNode: Node, startOffset: number, endNode: Node, endOffset: number) {
    const range = document.createRange()
    range.setStart(startNode, startOffset)
    range.setEnd(endNode, endOffset)
    const selection = document.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)

    return range
  }

  function textNodeIn(root: HTMLElement, selector: string, contains: string): Text {
    return Array.from(root.querySelector(selector)!.childNodes).find(
      n => n.nodeType === Node.TEXT_NODE && (n as Text).data.includes(contains)
    ) as Text
  }

  it('slices plain text precisely and emits touched math as whole bare LaTeX', () => {
    const root = mountFixture(
      '<p>Before formula. <span class="katex"><span>a^2+b^2</span><annotation encoding="application/x-tex">a^2 + b^2 = c^2</annotation></span> after text.</p>'
    )

    // Select from the middle of the leading text through the boundary of the
    // trailing text: text slices must be exact, the equation must come out
    // whole with no $ / $$ delimiters.
    const leading = textNodeIn(root, 'p', 'Before')
    const trailing = root.querySelector('span.katex')!.nextSibling as Text

    selectRange(leading, 3, trailing, 6)

    const text = selectionTextWithMath(root)

    expect(text).toContain('formula.')
    expect(text.startsWith('ore formula.')).toBe(true)
    expect(text).toContain('a^2 + b^2 = c^2')
    expect(text.endsWith('after')).toBe(true)
    expect(text).not.toContain('$')
    expect(text).not.toContain('\\n')
  })

  it('squeezes a cross-block selection onto one line with block-boundary spaces', () => {
    const root = mountFixture(
      '<p>first paragraph</p><p>second paragraph <span class="katex"><annotation encoding="application/x-tex">x=1</annotation></span></p><p>third</p>'
    )

    const first = root.querySelectorAll('p')[0].firstChild as Text
    const last = root.querySelectorAll('p')[2].firstChild as Text

    selectRange(first, 0, last, 5)

    const text = selectionTextWithMath(root)

    expect(text).toBe('first paragraph second paragraph x=1 third')
  })

  it('squeezes whitespace even when the selection touches no math', () => {
    const root = mountFixture('<p>alpha\n\nbeta</p><p>gamma</p>')

    const first = root.querySelectorAll('p')[0].firstChild as Text
    const last = root.querySelectorAll('p')[1].firstChild as Text

    selectRange(first, 0, last, 5)

    const text = selectionTextWithMath(root)

    expect(text).toBe('alpha beta gamma')
  })
})
