import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

// Render-level contract for copying rendered math: the transcript emits display
// math as a <span class="katex-display"> between paragraphs, so only the real
// pipeline proves the copied source keeps its own lines. latex-copy.test.ts
// covers the serializer in isolation.
afterEach(cleanup)

function copyEvent(): ClipboardEvent & { written: Map<string, string> } {
  const written = new Map<string, string>()
  const event = new Event('copy', { bubbles: true, cancelable: true }) as ClipboardEvent
  Object.defineProperty(event, 'clipboardData', {
    value: { setData: (type: string, data: string) => written.set(type, data) },
    configurable: true
  })

  return Object.assign(event, { written })
}

describe('MarkdownTextContent LaTeX copy', () => {
  it('copies prose around display math with the formula on its own lines', () => {
    const { container } = render(
      <MarkdownTextContent isRunning={false} text={'Before\n\n$$\n\\int_0^T f(t)\\,dt\n$$\n\nAfter'} />
    )

    expect(container.querySelector('.katex-display'), 'display math must have rendered').not.toBeNull()

    // The copy handler lives on the Markdown container (`aui-md`), the node that
    // carries `containerProps`. Dispatching anywhere else proves nothing.
    const host = container.querySelector<HTMLElement>('[class*="aui-md"]')
    expect(host, 'markdown container must exist').not.toBeNull()

    const range = host!.ownerDocument.createRange()
    range.selectNodeContents(host!)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)

    const event = copyEvent()
    host!.dispatchEvent(event)

    expect(event.written.get('text/plain')).toBe('Before\n\n$$ \\int_0^T f(t)\\,dt $$\n\nAfter')
  })
})
