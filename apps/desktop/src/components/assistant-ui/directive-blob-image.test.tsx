// @vitest-environment jsdom
import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { DirectiveContent } from './directive-text'

afterEach(cleanup)

// The optimistic attachment ref for an OS-dropped image is a Markdown image
// wrapping the renderer-local object URL. The sent-message surface has no
// generic Markdown-image parser, so this exact producer form must be
// recognized and painted as a thumbnail — raw Markdown and the blob URL
// leaking into visible text is the bug being pinned here.
describe('blob markdown image ref in a sent user message', () => {
  it('paints the object URL as an image instead of raw Markdown', () => {
    const { container } = render(
      <DirectiveContent text="![upload_1.png](blob:file:///07aa165b-55f6-4167-96c0-68f45ce7de27)" />
    )

    const img = container.querySelector('span[data-slot="aui_directive-image"] img')
    expect(img?.getAttribute('src')).toBe('blob:file:///07aa165b-55f6-4167-96c0-68f45ce7de27')
    expect(img?.getAttribute('alt')).toBe('upload_1.png')
    expect(container.textContent).not.toContain('![')
    expect(container.textContent).not.toContain('blob:')
  })

  it('covers the https-origin blob form too', () => {
    const { container } = render(
      <DirectiveContent text="see this ![shot](blob:https://desktop/preview-1) thanks" />
    )

    const img = container.querySelector('span[data-slot="aui_directive-image"] img')
    expect(img?.getAttribute('src')).toBe('blob:https://desktop/preview-1')
    expect(container.textContent).not.toContain('blob:')
    // surrounding prose survives
    expect(container.textContent).toContain('see this')
    expect(container.textContent).toContain('thanks')
  })

  it('keeps multiple dropped images as separate thumbnails', () => {
    const { container } = render(
      <DirectiveContent
        text={
          '![a.png](blob:file:///11111111-1111-1111-1111-111111111111) ' +
          '![b.png](blob:file:///22222222-2222-2222-2222-222222222222)'
        }
      />
    )

    const srcs = Array.from(container.querySelectorAll('span[data-slot="aui_directive-image"] img')).map(el =>
      el.getAttribute('src')
    )
    expect(srcs).toEqual([
      'blob:file:///11111111-1111-1111-1111-111111111111',
      'blob:file:///22222222-2222-2222-2222-222222222222'
    ])
    expect(container.textContent).not.toContain('![')
  })

  it('leaves ordinary markdown images as plain text', () => {
    const { container } = render(<DirectiveContent text="![logo](https://example.com/logo.png)" />)

    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toContain('![logo](https://example.com/logo.png)')
  })
})
