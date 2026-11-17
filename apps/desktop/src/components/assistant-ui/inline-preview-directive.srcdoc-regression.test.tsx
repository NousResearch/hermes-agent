import { createHash } from 'node:crypto'

import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { InlinePreviewDirective } from './inline-preview-directive'

/**
 * Local `::preview` files keep their srcdoc frame exactly as it was before
 * server-hosted artifacts gained a framed origin of their own: same document
 * bytes, same opaque-origin sandbox, never a `src`. The digest pins the whole
 * srcdoc (theme prelude, measurement and intent scripts, the page) for a fixed
 * page and token.
 */

const PAGE = '<html><head><title>Widget</title></head><body><button data-hermes-send="hi">Go</button></body></html>'
const SRCDOC_SHA256 = '6c84c8d317d5bfe638f3d68479425d4afb099408bb6803c77a57a14b65edc0c5'

const original = window.hermesDesktop

beforeEach(() => {
  // The per-mount token comes from Math.random; pin it so the bytes are stable.
  vi.spyOn(Math, 'random').mockReturnValue(0.5)
  window.hermesDesktop = { ...original, readFileText: async (path: string) => ({ path, text: PAGE }) }
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.hermesDesktop = original
})

describe('::preview srcdoc frame', () => {
  it('renders the same bytes in the same opaque-origin sandbox', async () => {
    const { container } = render(<InlinePreviewDirective attrs={{ file: '/work/widget.html' }} streaming={false} />)

    const iframe = await waitFor(() => {
      const node = container.querySelector('iframe')

      if (!node) {
        throw new Error('preview frame not mounted yet')
      }

      return node
    })

    const srcdoc = iframe.getAttribute('srcdoc') ?? ''

    expect(iframe.getAttribute('sandbox')).toBe('allow-scripts')
    expect(iframe.hasAttribute('src')).toBe(false)
    expect(srcdoc).toContain(PAGE.slice('<html><head>'.length, -'</body></html>'.length).slice(0, 20))
    expect(createHash('sha256').update(srcdoc).digest('hex')).toBe(SRCDOC_SHA256)
  })
})
