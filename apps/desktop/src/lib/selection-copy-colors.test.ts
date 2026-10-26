import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  installSelectionCopyColorGuard,
  selectionInkLuma,
  serializeSelectionStructure,
  textColorLuma
} from './selection-copy-colors'

function makeCopyEvent(): {
  event: ClipboardEvent
  setData: ReturnType<typeof vi.fn>
  preventDefault: ReturnType<typeof vi.fn>
} {
  const event = new Event('copy', { bubbles: true, cancelable: true }) as ClipboardEvent
  const setData = vi.fn()
  const preventDefault = vi.fn()

  Object.defineProperty(event, 'clipboardData', { value: { setData } })
  Object.defineProperty(event, 'preventDefault', { value: preventDefault })

  return { event, setData, preventDefault }
}

/** Stage styled text and arm a real DOM selection over it. */
function armSelection(html: string): HTMLDivElement {
  const host = document.createElement('div')

  host.innerHTML = html
  document.body.append(host)

  const range = document.createRange()

  range.selectNodeContents(host)

  const sel = window.getSelection()

  sel?.removeAllRanges()
  sel?.addRange(range)

  return host
}

describe('textColorLuma', () => {
  it('scores white near the top of the range', () => {
    expect(textColorLuma('rgb(255, 255, 255)')).toBeGreaterThan(240)
  })

  it('scores black near the bottom of the range', () => {
    expect(textColorLuma('rgb(10, 10, 10)')).toBeLessThan(20)
  })

  it('composites partial alpha over mid-gray', () => {
    // White at 50% alpha lands halfway between white and gray.
    const halfWhite = textColorLuma('rgba(255, 255, 255, 0.5)')

    expect(halfWhite).not.toBeNull()
    expect(halfWhite!).toBeGreaterThan(185)
    expect(halfWhite!).toBeLessThan(200)
  })

  it('treats fully transparent paint as invisible', () => {
    expect(textColorLuma('rgba(255, 255, 255, 0)')).toBeNull()
    expect(textColorLuma('rgba(255, 255, 255, 0.03)')).toBeNull()
  })

  it('returns null for unparsable colors', () => {
    expect(textColorLuma('var(--foreground)')).toBeNull()
    expect(textColorLuma('inherit')).toBeNull()
    expect(textColorLuma('')).toBeNull()
  })

  it('parses the color(srgb …) computed form Chromium serializes', () => {
    // The app's dark-theme ink: color-mix(in srgb, #e6edf3 94%, transparent)
    // computes to exactly this serialization.
    const luma = textColorLuma('color(srgb 0.901961 0.929412 0.952941 / 0.94)')

    expect(luma).not.toBeNull()
    expect(luma!).toBeGreaterThan(220)
  })

  it('parses color(srgb …) with percentage components and alpha', () => {
    const luma = textColorLuma('color(srgb 100% 100% 100% / 50%)')

    expect(luma).not.toBeNull()
    expect(luma!).toBeCloseTo(191.5, 0)
  })

  it('parses hex colors including alpha', () => {
    expect(textColorLuma('#ffffff')).toBeGreaterThan(240)
    expect(textColorLuma('#111111')).toBeLessThan(20)
    expect(textColorLuma('#ffffff80')).not.toBeNull()
  })
})

describe('selectionInkLuma', () => {
  afterEach(() => {
    window.getSelection()?.removeAllRanges()
  })

  it('scores the live computed ink of the selected text', () => {
    const host = armSelection('<p style="color: rgb(230, 237, 243)">bright transcript ink</p>')
    const luma = selectionInkLuma(window.getSelection()!, document)

    expect(luma).not.toBeNull()
    expect(luma!).toBeGreaterThan(220)

    host.remove()
  })

  it('scores dark ink as dark', () => {
    const host = armSelection('<p style="color: rgb(17, 17, 17)">dim transcript ink</p>')
    const luma = selectionInkLuma(window.getSelection()!, document)

    expect(luma).not.toBeNull()
    expect(luma!).toBeLessThan(20)

    host.remove()
  })

  it('returns null for a collapsed selection', () => {
    const host = armSelection('<p style="color: rgb(255, 255, 255)">text</p>')

    window.getSelection()?.collapseToEnd()

    expect(selectionInkLuma(window.getSelection()!, document)).toBeNull()

    host.remove()
  })
})

describe('serializeSelectionStructure', () => {
  afterEach(() => {
    window.getSelection()?.removeAllRanges()
  })

  it('keeps semantic structure and hrefs while dropping paint and classes', () => {
    const host = armSelection(
      '<p class="aui-md" style="color: rgb(230, 237, 243)">see <a href="https://example.com">the doc</a> and <strong>this</strong></p>'
    )

    const html = serializeSelectionStructure(window.getSelection()!, document)

    expect(html).toContain('<strong>this</strong>')
    expect(html).toContain('href="https://example.com"')
    expect(html).toContain('the doc')
    // The only style attribute allowed is the wrapper's generic font anchor.
    expect(html.replace('<div style="font-family: sans-serif;">', '')).not.toContain('style=')
    expect(html).not.toContain('class=')
    expect(html).not.toContain('rgb(230, 237, 243)')

    host.remove()
  })

  it('keeps list and code structure', () => {
    const host = armSelection('<ul><li>alpha</li><li><em>beta</em></li></ul><pre>code line</pre>')
    const html = serializeSelectionStructure(window.getSelection()!, document)

    expect(html).toContain('<li>alpha</li>')
    expect(html).toContain('<em>beta</em>')
    expect(html).toContain('code line')

    host.remove()
  })

  it('anchors a generic sans family so rich-text receivers keep their own font', () => {
    const host = armSelection(
      '<p style="color: rgb(230, 237, 243); font-family: -apple-system, sans-serif">plain prose</p>'
    )

    const html = serializeSelectionStructure(window.getSelection()!, document)

    // Generic family on the wrapper: resolves to each platform's own face,
    // never the Times browser default, and carries no vendor font names.
    expect(html).toContain('sans-serif')
    expect(html).not.toContain('-apple-system')
    expect(html).not.toContain('color')

    host.remove()
  })

  it('pins monospace inside code elements', () => {
    const host = armSelection('<pre><code>npm run build</code></pre>')
    const html = serializeSelectionStructure(window.getSelection()!, document)

    expect(html).toContain('monospace')

    host.remove()
  })
})

describe('installSelectionCopyColorGuard', () => {
  beforeEach(() => {
    document.documentElement.dataset.hermesMode = 'dark'
  })

  afterEach(() => {
    delete document.documentElement.dataset.hermesMode
    window.getSelection()?.removeAllRanges()
  })

  it('owns the payload when a light-ink selection is copied under a dark theme', () => {
    const dispose = installSelectionCopyColorGuard(document)
    const host = armSelection('<p style="color: rgb(230, 237, 243)">bright transcript ink</p>')

    try {
      const { event, setData, preventDefault } = makeCopyEvent()

      document.body.dispatchEvent(event)

      expect(preventDefault).toHaveBeenCalled()

      const plainCall = setData.mock.calls.find(([type]) => type === 'text/plain')
      const htmlCall = setData.mock.calls.find(([type]) => type === 'text/html')

      expect(plainCall?.[1]).toContain('bright transcript ink')
      expect(htmlCall?.[1]).toContain('bright transcript ink')
      expect(htmlCall?.[1]).not.toContain('rgb(230, 237, 243)')
    } finally {
      dispose()
      host.remove()
    }
  })

  it('leaves dark-ink selections to Chromium under a dark theme', () => {
    const dispose = installSelectionCopyColorGuard(document)
    const host = armSelection('<p style="color: rgb(17, 17, 17)">dim transcript ink</p>')

    try {
      const { event, setData, preventDefault } = makeCopyEvent()

      document.body.dispatchEvent(event)

      expect(preventDefault).not.toHaveBeenCalled()
      expect(setData).not.toHaveBeenCalled()
    } finally {
      dispose()
      host.remove()
    }
  })

  it('leaves light-ink selections alone when the theme is light', () => {
    document.documentElement.dataset.hermesMode = 'light'

    const dispose = installSelectionCopyColorGuard(document)
    const host = armSelection('<p style="color: rgb(230, 237, 243)">bright transcript ink</p>')

    try {
      const { event, setData, preventDefault } = makeCopyEvent()

      document.body.dispatchEvent(event)

      expect(preventDefault).not.toHaveBeenCalled()
      expect(setData).not.toHaveBeenCalled()
    } finally {
      dispose()
      host.remove()
    }
  })

  it('skips selections that start inside an editable field', () => {
    const dispose = installSelectionCopyColorGuard(document)
    const host = document.createElement('div')

    host.setAttribute('contenteditable', 'true')
    host.innerHTML = '<span style="color: rgb(230, 237, 243)">composer text</span>'
    document.body.append(host)

    const range = document.createRange()

    range.selectNodeContents(host)

    const sel = window.getSelection()

    sel?.removeAllRanges()
    sel?.addRange(range)

    try {
      const { event, setData, preventDefault } = makeCopyEvent()

      document.body.dispatchEvent(event)

      expect(preventDefault).not.toHaveBeenCalled()
      expect(setData).not.toHaveBeenCalled()
    } finally {
      dispose()
      host.remove()
    }
  })

  it('stops intercepting after disposal', () => {
    const dispose = installSelectionCopyColorGuard(document)
    const host = armSelection('<p style="color: rgb(230, 237, 243)">bright transcript ink</p>')

    dispose()

    try {
      const { event, setData, preventDefault } = makeCopyEvent()

      document.body.dispatchEvent(event)

      expect(preventDefault).not.toHaveBeenCalled()
      expect(setData).not.toHaveBeenCalled()
    } finally {
      host.remove()
    }
  })
})
