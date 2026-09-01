import { describe, expect, it, vi } from 'vitest'

import { flushAnnotateStack } from './flush'
import { compactIdentity } from './identity'
import { annotateFlushPrompt, packageAnnotatePin, packageAnnotateStack } from './pack'
import { addAnnotatePin, type AnnotatePin, emptyAnnotateStack } from './stack'

const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

function pin(partial: Partial<AnnotatePin> = {}): AnnotatePin {
  return {
    id: 'annotate-1',
    imageDataUrl: png,
    kind: 'element',
    note: 'This button overflows on mobile.',
    number: 1,
    pageTitle: 'Pricing',
    pageUrl: 'http://127.0.0.1:4173/',
    rect: { height: 40, width: 120, x: 8, y: 8 },
    identity: {
      css: { color: 'rgb(24, 24, 24)', 'font-size': '14px' },
      selector: 'button.plan',
      tag: 'button',
      text: 'Select plan'
    },
    ...partial
  }
}

describe('packageAnnotatePin', () => {
  it('describes text in a generic container without exposing DOM and style internals', () => {
    const text = 'גם בקיבוץ חולית הקטן יש ילד שעושה את הצעד הראשון במערכת החינוך'

    const packed = packageAnnotatePin(
      pin({
        identity: {
          css: { color: 'rgb(0, 0, 0)', 'font-family': 'Moses, NarkisBlock', 'font-size': '18px' },
          selector:
            'div.DraftEditor-editorContainer>div.public-DraftEditor-content>div>div.text_editor_paragraph.rtl:nth-of-type(9)',
          tag: 'div',
          text
        },
        note: 'תסכם את זה'
      })
    )

    expect(packed.prompt).toContain(`Target: "${text}"`)
    expect(packed.prompt).toContain('Note: תסכם את זה')
    expect(packed.prompt).not.toContain('div')
    expect(packed.prompt).not.toContain('DraftEditor')
    expect(packed.prompt).not.toContain('font-size')
  })

  it('packs a numbered crop, compact identity, and the note', () => {
    const packed = packageAnnotatePin(pin())

    expect(packed.number).toBe(1)
    expect(packed.imageDataUrl).toBe(png)
    expect(packed.note).toContain('overflows')
    expect(packed.prompt).toContain('Comment 1')
    expect(packed.prompt).toContain('Target: button "Select plan"')
    expect(packed.prompt).toContain('Image 1 marks the target in blue.')
    expect(packed.prompt).toContain('Note: This button overflows on mobile.')
    expect(packed.prompt).not.toContain('button.plan')
    expect(packed.prompt).not.toContain('font-size')
    expect(packed.prompt).not.toContain('<html')
  })

  it('keeps a selector when an element has no readable text', () => {
    const packed = packageAnnotatePin(
      pin({
        identity: {
          css: { display: 'block' },
          selector: '#sales-chart',
          tag: 'div',
          text: ''
        },
        note: 'Use the same scale as the chart above.'
      })
    )

    expect(packed.prompt).toContain('Target: #sales-chart')
    expect(packed.prompt).not.toContain('display: block')
  })

  it('packages an area pin without pretending it has a selector', () => {
    const packed = packageAnnotatePin(pin({ identity: undefined, kind: 'area', note: 'too tight' }))

    expect(packed.prompt).toContain('area')
    expect(packed.prompt).toContain('120×40px')
    expect(packed.prompt).toContain('too tight')
  })
})

describe('compactIdentity', () => {
  it('never keeps the whole document and clips long text', () => {
    const compact = compactIdentity({
      css: { color: 'red', display: 'none', margin: '0px' },
      selector: 'html>body>div>div>button.submit',
      tag: 'BUTTON',
      text: 'x'.repeat(200)
    })

    expect(compact.tag).toBe('button')
    expect(compact.text.endsWith('…')).toBe(true)
    expect(compact.text.length).toBeLessThanOrEqual(80)
    expect(compact.css.display).toBeUndefined()
    expect(compact.css.margin).toBeUndefined()
    expect(compact.css.color).toBe('red')
  })
})

describe('flushAnnotateStack', () => {
  it('attaches one composer item per pin and does not invoke send', async () => {
    const first = pin()
    const second = pin({ id: 'annotate-2', kind: 'area', identity: undefined, note: 'align the chart', number: 2 })
    let stack = emptyAnnotateStack()
    stack = addAnnotatePin(stack, {
      imageDataUrl: first.imageDataUrl,
      identity: first.identity,
      kind: first.kind,
      note: first.note,
      pageTitle: first.pageTitle,
      pageUrl: first.pageUrl,
      rect: first.rect
    })
    stack = addAnnotatePin(stack, {
      imageDataUrl: second.imageDataUrl,
      kind: 'area',
      note: second.note,
      pageTitle: second.pageTitle,
      pageUrl: second.pageUrl,
      rect: second.rect
    })

    const attachImage = vi.fn()
    const insertText = vi.fn()
    const send = vi.fn()
    const result = await flushAnnotateStack(stack.pins, { attachImage, insertText, send }, 'http://127.0.0.1:4173/')

    expect(result.sent).toBe(false)
    expect(result.count).toBe(2)
    expect(send).not.toHaveBeenCalled()
    expect(attachImage).toHaveBeenCalledTimes(2)
    expect(attachImage.mock.calls[0]?.[0]).toBeInstanceOf(File)
    expect((attachImage.mock.calls[0]?.[0] as File).name).toBe('Comment_1.png')
    expect((attachImage.mock.calls[1]?.[0] as File).name).toBe('Comment_2.png')
    expect(insertText).toHaveBeenCalledOnce()
    expect(insertText.mock.calls[0]?.[0]).toContain('I left 2 comments')
    expect(insertText.mock.calls[0]?.[0]).toContain('Comment 1')
    expect(insertText.mock.calls[0]?.[0]).toContain('Comment 2')
  })

  it('a pin save is stacking, not flushing', () => {
    const stacked = packageAnnotateStack([pin(), pin({ id: 'annotate-2', number: 2, note: 'second' })])

    expect(stacked).toHaveLength(2)
    expect(annotateFlushPrompt(stacked)).toContain('2 comments')
  })
})
